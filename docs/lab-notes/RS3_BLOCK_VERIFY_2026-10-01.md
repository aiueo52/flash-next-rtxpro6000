# RS3 (2026-10-01): block verification for RS2's draft chain

Status (23:55): **offline estimate on the RS2d dump (`runs/rs2d/accept_offline.log`): BV gains +0.59% pooled
tok/step at RS2's q (paired SE 0.04%), below the +2% gate.** Agent-loop +1.26%, code-edit +0.12%, prose-en +1.28%,
prose-ja +0.85%. E_tok matches the realised accept_len (pooled difference 0.008, SE 0.004). Best pooled pair under
BV by importance weights: (0.7, 0.9), 4.771 tok/step vs RS2's E_tok 4.721 (+1.05%), ESS 18,933 of 24,723. BV still
goes into the package ABBA against target-only (RS2d spec 4, step 3 as replaced 23:55): it costs at most 8.5 us
per verify and adds in expectation.
Status (23:15): **vectorized BV (section 5) committed (sglang-rs3 a78b1a5ebd) and passes everything: CPU tests
B0-B4 and B6, and B5 on the GPU (same as the sequential BV on 768 rows; accept; timing +8.5 us max at slots 16,
budget 10 us; the sequential version cost +58 us).** Details: `bench/rs3/PROGRESS.md`. The offline estimate on
the RS2d dump (section 3, about 00:00) decides whether BV goes into an ABBA.

## 0. Why

RS2 checks the draft chain one token at a time and stops at the first rejection (`_chain_sampling_sparse_kernel`).
Block verification (Sun et al., "Block Verification Accelerates Speculative Decoding", arXiv 2403.10444) decides on
the whole chain at once. A failed coin at position i no longer ends the chain if a later position passes its own
test. The output is still drawn exactly from the target. In expectation it accepts at least as many tokens as
token-by-token checking: the paper proves it optimal for a single chain. It needs no extra model work. The verify
kernel adds one sum per position over data of the size it already loads.

Production drafts chains of 3, 7 or 15 tokens (adaptive). The gain grows with chain length.

Checked before any code (`bench/rs3/bv_exact.py`), by full enumeration of every draft path and output:
- vocabularies of 3-4 tokens, chains of 1-4, 360 random cases;
- the cases include one-hot targets (greedy rows), one-hot drafts (RS2d's θ), truncated supports (top-k zeros) and
  positions with p = q.

With section 1's edge rules, BV's output distribution equals the target's to 7e-16 in every case. It never
accepts fewer tokens than token-level in expectation, and with one-hot targets it equals token-level exactly.

## 1. The rule (per row; a chain of G = slots - 1 draft tokens)

Draft token i (1..G) is `candidates[:, i]`. It was drawn from q at slot i-1 (`draft_support_probs/tokens[:, i-1]`)
and is checked against the target p at slot i-1 (`target_probs/index[:, i-1]`). pt and qt are p and q summed over
the entries whose id equals the token, as the kernel does now.

1. pi_0 = 1. For i = 1..G: pi_i = 0 if pt <= 0. Otherwise pi_i = min(pi_{i-1} * pt / qt, 1), or 1 if qt == 0 (a
   drawn token has qt > 0).
2. For i = 1..G-1, with p and q at slot i (the distributions for draft token i+1):
   - n_i = sum over p's entries x of max(pi_i * p(x) - q(x), 0), where q(x) is q summed over entries with id x
     (today's `matched`);
   - h_i = n_i / (n_i + 1 - pi_i), and h_i = 1 when n_i + 1 - pi_i <= 0.

   h_G = pi_G.
3. tau = the largest i in 1..G with coin_i < h_i, where coin_i = `Coins[i - 1]` (today's per-position coins). tau
   is 0 if there is no such i.
4. Accept draft tokens 1..tau. predicts, accept_index and accept_token_num are written exactly as today for tau
   accepts.
5. Final token at slot tau:
   - tau == G: draw from p (the bonus, as today).
   - Otherwise: draw from w = max(pi_tau * p - q, 0) over p's entries at slot tau, with q at slot tau matched by
     id. Use today's zero-residual rule (if w is 0 everywhere, draw p) and today's draw code (CoinsFinal, id
     order). With tau == 0 this is today's residual, since pi_0 = 1.

Once pi_i = 0, every later pi and h is 0, so the loop may stop there.

## 2. Wiring (worktree `~/tools/sglang-rs3`, branch `opus/rs3-bv` from the G1 commit of `opus/rs2d-g1`)

- `sparse_rs.py`: `_chain_sampling_sparse_kernel` gains the constexpr `BLOCK_VERIFY`; False compiles today's code.
  The wrapper becomes `chain_speculative_sampling_sparse(..., block_verify: bool = False)`.
- `environ.py`: `SGLANG_RS_BLOCK_VERIFY = EnvBool(False)`. `spec_info.py`: ValueError if it is set without
  SGLANG_OPT_SPEC_SPARSE_RS.
- `eagle_utils.py`: read the flag once at import (as `RS_DUMP_DIR` is) and pass `block_verify=RS_BLOCK_VERIFY` at
  the one `chain_speculative_sampling_sparse` call. Show it in the "SGLANG_OPT_SPEC_SPARSE_RS on" log line.

With the flag unset, behaviour is byte-identical to the G1 commit.

## 3. Offline estimate (`bench/rs2d/accept_offline.py`, on the RS2d dump)

The dump's chains were drawn from RS2's q. At RS2's q, a path-based estimate is therefore exact in expectation:
- token-level: E_tok(X) = sum_{i=1..G} prod_{j<=i} min(1, pt_j / qt_j). Its mean must match the realised
  accept_len; it is a second sanity line and replaces nothing.
- block: E_bv(X) = sum_t t * P(tau = t | X), with P(tau = t) = h_t * prod_{j>t} (1 - h_j), h_0 = 1, and pi and h
  as in section 1.

Print per workload and pooled: realised, E_tok and E_bv as tok/step (1 + E), and E_bv's gain over E_tok in % with
a paired standard error.

Other (s, θ) pairs, approximate, by importance weights:
- w(X) = prod_i q'(X_i) / q(X_i) over the chain, with q' from the existing transform;
- self-normalised means of E_tok' and E_bv', both computed with q';
- ESS = (sum w)^2 / sum w^2.

Add these as CSV columns, and print the best pooled pair under BV with its ESS.

Decision: BV goes into the next ABBA if its pooled gain at RS2's q is >= +2% tok/step. BV can only add in
expectation, so code-edit cannot fall.

## 4. Tests (CPU, TRITON_INTERPRET=1; new `bench/rs3/`)

- B0: `block_verify=False` is bit-identical to the G1 worktree's module on G1c's batches.
- B1: the kernel against a per-row Python reference of section 1 with the same coins. predicts, accept_index and
  accept_token_num must be identical on random batches:
  - slots 2/4/8/16, kp 16/64, K 64;
  - row kinds: general rows, one-hot targets, one-hot q, p zeros on q's support, NaN in q, p = q at some slots,
    and a draft token outside p's support.
- B2: exactness by sampling. Small vocab (V 4-6, kp = V, K = V), chains 2-3, all paths:
  - chi2 of the output (accepted prefix plus final token) against the target's exact distribution, Fisher-combined
    p > 0.01 as in test D;
  - mean accept within 3 SE of bv_exact's expectation.
- B3: with one-hot targets, BV outputs are identical to `block_verify=False` on the same coins.
- B4: integration (from `check_integration_rs2d.py`):
  - flag unset: call kwargs unchanged;
  - flag set: `block_verify=True` at the call site;
  - flag without SPARSE_RS: ValueError.
- B5 (CUDA, parent): verify kernel timing with BV on and off, at bs 1/4 and slots 4/8/16. Budget: <= +10 us per
  verify.

## 5. Vectorized BV branch (added 22:45, after B5's timing WARN)

B5: BV adds +7.4/+23.2/+57.8 us per verify at slots 4/8/16, about 3.8 us per chain position, because each
position waits for its own loads and reductions. Only pi is sequential in section 1, and it is a clamped running
product, so a scan computes it. Rewrite the `BLOCK_VERIFY` branch to handle all positions at once.

Index u = 0..SP-1, with SP = next power of 2 >= NUM_SLOTS (a new constexpr) and masks; G = NUM_SLOTS - 1.
1. Position tiles, valid for 1 <= u <= G (draft token u, checked at slot u-1):
   - P/PI at slot u-1 [SP, KP] and Q/QI at slot u-1 [SP, KQ] (NaN q -> 0 as now);
   - token = Candidates[base + u], coin = Coins[base + u - 1];
   - pt[u], qt[u] row-wise, as now.
2. pi[u] by `tl.associative_scan` over maps pi -> min(a*pi + c, b):
   - per position: (pt/qt, 0, 1) normally, (0, 1, 1) if qt == 0 and pt > 0, (0, 0, 0) if pt <= 0;
   - u = 0 is (1, 0, 1); positions u > G are (0, 0, 0);
   - combine (a1, c1, b1) then (a2, c2, b2) -> (a2*a1, a2*c1 + c2, min(a2*b1 + c2, b2));
   - pi[u] = min(A[u] + C[u], B[u]) (the composed map applied to pi_0 = 1);
   - today's loop stops at the first pi == 0, so set pi = 0 from the first zero on.
3. Slot tiles for n, valid for 1 <= u <= G-1 (slot u):
   - P/PI and Q/QI at slot u;
   - matched[u, x] = sum_k q[u, k] * [qi[u, k] == pi[u, x]], with the k axis in chunks so the compare fits in
     registers;
   - n[u] = sum_x max(pi[u] * p[u, x] - matched[u, x], 0);
   - h[u] = n / (n + 1 - pi[u]), or 1 where the denominator is <= 0.

   h[G] = pi[G]; h = 0 at u = 0 and u > G.
4. tau = max over u of (u if coin[u] < h[u] else 0); final_pi = pi[tau], 1 when tau = 0.
5. Accepted outputs for u = 1..tau, as masked vector stores:
   - Predicts[RetriveIndex[base + u - 1]] = Candidates[base + u];
   - AcceptIndex[base + u] = RetriveIndex[base + u];
   - last = RetriveIndex[base + tau].

   The final draw (the shared tail) stays as it is.

Constraints:
- block_verify=False keeps today's code and launch (num_warps 4). BV may choose its own num_warps.
- fp32, as now.
- Outputs equal the sequential BV kernel's, except where a coin lies within rounding of h: the scan multiplies in
  a different order.

Tests (CPU):
- B0-B4 rerun unchanged and pass.
- New B6: the new branch against the sequential BV kernel of fb1e77a05a (its sparse_rs.py from `git show`,
  loaded as a module):
  - shapes: slots 4/8/16, kp 64, K 64, 512 rows each, built like `b5_timing.make_batch`, plus B1's row kinds;
  - outputs identical on every row, except rows with a coin within 1e-5 of a float64 reference h; count them.
- Parent: B5 again on the GPU. Budget: BV adds <= 10 us per verify at slots 16.
