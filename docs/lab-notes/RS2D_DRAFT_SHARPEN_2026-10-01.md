# RS2d (2026-10-01): a sharper draft proposal for RS2, and a dump to choose it offline

Status (2026-10-02 06:40): **package ABBA done (02:19-06:19, 8 arms, all clean: rs2 enabled lines 1 in every B
arm and 0 in every A arm, error lines 0). RS2 §6: not adopted, read as "code-edit unresolved" (4 step 5).**
- LM Studio t/s, package vs stack target-only (request CI; arm CI):
  - pooled **+6.53 [+4.60..+8.51]** PASS; arm [+1.36..+11.97].
  - agent-loop +6.17 [+2.72..+9.73] PASS; arm [-0.38..+13.16].
  - code-edit +1.63 [**-2.49**..+5.90] **FAIL** (lower bound under -2%); arm [-6.12..+10.00]. tok/step +0.23,
    ms/step|a -1.44.
  - prose-en +8.52 [+5.78..+11.33]; prose-ja +10.01 [+5.33..+14.90].
- Greedy: ms/step|a -2.18 [-3.30..-1.04] PASS (<= +1%); arm [-8.05..+4.12]. t/s pooled +1.87 [-0.39..+4.19]; every
  workload's CI crosses 0. Errors 0 PASS.
- The FAIL comes from code-edit alone and its point estimate is +1.6%, so 4 step 5 applies: code-edit is
  unresolved, not measured slower. The rule still says not adopted. Shipping RS as an LM Studio option is the
  user's call with this result.
- Against 4 step 5's expectation: pooled +6.5 (about +5), prose-en +8.5 (+8), prose-ja +10.0 (+7), agent-loop +6.2
  (+6), code-edit +1.6 (-0.7), greedy ms/step|a -2.2 (+0.1..+0.3). Nothing in the package makes a greedy step
  2% cheaper (G1 about breaks even with target-only, and the RS glue adds about 10 us per step in the traces), and
  the arm means drift (A1, B1 and B2 fast, later arms slow). Read the greedy -2.2 and the code-edit +1.6 with that
  drift in mind; the arm CIs carry it.
- Clock refit (4 step 4, secondary): B - A clock is about 0 in every row (lmstudio pooled -0.06 [-0.71..+0.61],
  greedy -0.22 [-1.01..+0.58]). Adjusted lmstudio t/s: pooled +6.36 [+4.57..+8.18], code-edit +0.70 [-2.46..+3.97],
  agent-loop +6.67, prose-en +8.71, prose-ja +8.59; greedy ms/step|a -1.89. The tok/step placebo is not 0
  (elasticity -1.98, se 0.55) and ms/step|a rises with the clock (+1.72): the clock follows the load instead of
  setting it (likely an idler step lets it rise), so the refit is no cleaner than the primary.
- Logs: `runs/rs2d-abba/abba.log`, `post-summary.txt`, `post.log`, `clocks.csv`.
- RS4-lite: K = 16 passes its rule and joins the package (section 6, result 06:40).
Status (23:55): **dump analysed (`runs/rs2d/accept_offline.log`, D1 23:34-23:45, 24,723 verifies from 12
requests; 527 probe/trace rows outside every request window, 0 length mismatches): the knobs fail the 4.2 gate.**
- Pooled E-based tok/step over RS2: best product-grid pair (0.8, 0.9) +0.20%, best s alone (0.8) +0.19%, gate
  +2%. One-hot alone (s = 1) loses on prose at every θ below 0.95.
- Code-edit: RS2 10.148 vs target-only 10.237 E-based (-0.9%); the best code-edit pair (0.3, 0.6) reaches
  10.247. The per-position E is 0.10 below code-edit's realised accept_len (4.8 SE; acceptances along a chain are
  correlated), while the path estimate E_tok matches it (difference 0.011, SE 0.008; RS3 spec 3).
- Importance weights under BV (RS3 spec 3) rank (0.7, 0.9) first: +1.05% pooled over RS2's E_tok (code-edit +1.2%,
  agent-loop +1.7%, prose-en +0.3%, prose-ja -0.1%), ESS 18,933 of 24,723.
- Next: section 4 step 3 as replaced below (package ABBA against target-only).
Status (21:30): part 1 (sections 1-4) committed: sglang-rs2d a8c9cb5f89, tests in bench/rs2d (f36d8a6), all
CPU tests PASS in the parent's reruns. The dump arm waits for the RS2 ABBA's end (bench/rs2d/dump_after_rs2.sh).
G1 (section 5) and the time-window join (21:55): implemented in worktree sglang-rs2g (branch opus/rs2d-g1),
all CPU tests PASS in the parent's reruns (G1a-c, K1-K6, test_sparse_rs A-D, integration, accept_offline
self-test; compat with sglang-rs2d), commit 124840a5c3. G1d PASS (22:02, `bench/rs2d/g1d_timing.log`, CUDA graph,
all rows top_k 1): fp32 n=1 full 21.9 us, greedy_fast 4.43 us, target-only top-1 3.52 us per draft step (bf16 and
n=2/4 alike), so 7 steps cost about 31 us against target-only's 32.6 us (6 top-1 + one argmax). Next: block
verification, RS3 spec.

## 0. Why

RS2's first ABBA pair (A1 target-only, B1 RS2; lmstudio sampling, per prompt) shows the acceptance gain is uneven:
agent-loop tok/step +15%, prose-ja +7.6%, prose-en +3.5%, but code-edit −0.6% with +3.3% ms/step, so code-edit
t/s fell 3.9% (RS2 §6 wants code-edit's lower bound > −2%).

Any proposal q keeps the output exact, as long as the same q both draws the draft token and verifies it. q only
sets the acceptance. Target-only verify is RS with a one-hot q on the draft's argmax: it accepts with p(x_d). RS2
accepts with Σ min(p, q). When the target is sure and the draft is right but less sure (p = 0.9/0.1,
q = 0.6/0.4), RS2 accepts less (0.7 against 0.9). Code-edit looks like this. A sharper q keeps RS2's gain where p
is spread and gives back target-only's acceptance where p is peaked. Two knobs, both exact for any value:

- **temp scale s** (`SGLANG_RS_DRAFT_TEMP_SCALE`, float, default 1.0): q uses temperature `T * s` (s < 1 sharpens);
- **one-hot above θ** (`SGLANG_RS_DRAFT_ONEHOT_ABOVE`, float, default 0.0 = off): if the filtered q's top
  probability is ≥ θ, q becomes one-hot on rank 0 (that position is then exactly target-only).

The dump records, per verify, the target's filtered p and RS2's q at every position. Offline, it gives
Σ min(p, q') for any (s, θ) at the same prefixes, so one GPU run chooses the values; an ABBA then confirms them.

## 1. Changes (worktree `~/tools/sglang-rs2d`, branch `opus/rs2d-sharpen` from `opus/rs2-sparse` cf1e515722)

1. `python/sglang/kernels/ops/speculative/sparse_rs.py`
   - `_draft_finalize_kernel` gains `temp_scale` (runtime fp32), `onehot_above` (runtime fp32) and constexprs
     `HAS_TEMP_SCALE`, `HAS_ONEHOT`. With both constexprs False, the code compiled is today's.
     - HAS_TEMP_SCALE: the row temperature T > 0 becomes `T * temp_scale` (T <= 0 still means 1.0).
     - HAS_ONEHOT: after the final renormalisation, if `probs[rank 0] >= onehot_above`, then probs = 1.0 at rank 0
       and 0.0 elsewhere, and the pick is rank 0. Everything after that (tokens, TopkP, TopkIndex, chain column,
       positions) follows from the new probs and pick as today.
   - `rs_draft_proposal_sparse(..., temp_scale: float = 1.0, onehot_above: float = 0.0)`: assert
     `temp_scale > 0` and `0 <= onehot_above <= 1`; HAS_TEMP_SCALE = `temp_scale != 1.0`,
     HAS_ONEHOT = `onehot_above > 0`.
2. `python/sglang/srt/environ.py`: `SGLANG_RS_DRAFT_TEMP_SCALE = EnvFloat(1.0)`,
   `SGLANG_RS_DRAFT_ONEHOT_ABOVE = EnvFloat(0.0)`, `SGLANG_RS_DUMP_DIR = EnvStr("")`, next to `SGLANG_RS_DRAFT_TOPK`.
   `spec_info.py` (~267, with the other SPARSE_RS checks): ValueError unless `temp_scale > 0` and
   `0 <= onehot_above <= 1`.
3. `python/sglang/srt/speculative/spec_utils.py` / `eagle_worker_v2.py`: read both knobs once at import, as
   `RS_DRAFT_TOPK` is; pass `temp_scale=` and `onehot_above=` at both `rs_draft_proposal_sparse` calls
   (`_rs_sparse_proposal` ~936 and `draft_forward` ~1101); add them to the "SGLANG_OPT_SPEC_SPARSE_RS on" log line.
4. Dump, new `python/sglang/srt/speculative/rs_dump.py` plus one guarded call in `eagle_utils.py eagle_sample`,
   right after `chain_speculative_sampling_sparse(...)` in the SPEC_SPARSE_RS branch, when `SGLANG_RS_DUMP_DIR`
   is set (read once at import; TP = 1 only, documented in the module docstring).
   - One record per verify: `time` (time.time()), `rid` and `input_len` (`len(req.origin_input_ids)`) per row,
     `candidates` (int32), `target_probs` (fp32) and `target_index` (int32) as (bs, slots, kp),
     `draft_support_probs` (fp32) and `draft_support_tokens` (int32) as (bs, slots - 1, K), `accept_len`
     (`num_correct_drafts`, int32), and the rows' `temperatures`, `top_ks`, `top_ps`, `min_ps` - all on CPU.
   - Every 50 records: `torch.save(list, f"{dir}/rs-dump-{os.getpid()}-{chunk:05d}.pt")` and clear. The dump only
     reads tensors after the verify kernel; it changes no output.

Knobs unset and dump off: byte-identical behaviour to cf1e515722.

## 2. Offline analysis (`bench/rs2d/accept_offline.py <dump dir> <fnbench.jsonl>`)

Per verify row with slots S (= candidates.shape[1]); for draft position j = 0..S-2 (candidate j+1):
- p_j: `target_probs[r, j]` over ids `target_index[r, j]` (sum duplicates; zeros allowed).
- q_j: `draft_support_probs[r, j]` over ids `draft_support_tokens[r, j]`, rank order (rank 0 = draft argmax).
- RS2 as run: a_j = Σ_x min(p_j(x), q_j(x)). Target-only: a_j = p_j(id of rank 0).
- Temp scale s: on q_j's kept support (q > 0), w = q^(1/s), normalised; then top-p in rank order (keep where the
  exclusive prefix < top_p, rank 0 always) and min-p (w >= min_p * w[0]), renormalised; a_j = Σ min(p_j, w).
  Approximation: the kernel normalises over all 64 ranks before top-p, and the ranks RS2 cut are not in the dump,
  so for s < 1 this cuts slightly more than the kernel would (tail mass only). Exact for s = 1.
- One-hot θ (applied after s): if max(w) >= θ, w = one-hot rank 0.
- Expected accepted drafts per verify: E = Σ_{i=1}^{S-1} Π_{j<i} a_j; tok/step = 1 + E.
- Sanity: for RS2 as run, mean E over verifies must match the mean realised `accept_len` (it is the exact
  expectation at the realised candidates); print both with a standard error.
- Workload (changed 21:30): a row belongs to the fnbench request whose window [timestamp, timestamp + ttft +
  decode + 2 s] holds its `time`, and only if `input_len` == `usage.prompt_tokens`. The arm's probes and
  profiler traces fall outside every window and are left out (counts printed). Matching on length alone could
  mix them in. A "pooled" block averages the workloads equally (the 4.2 rule).
- Output: per workload: verifies, realised tok/step, E-based tok/step for RS2, target-only, the s grid
  (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3), the θ grid (0.5, 0.6, 0.7, 0.8, 0.9, 0.95) at s = 1, and the best
  (s, θ) pair over the product grid; mean a_j for j = 0..7 for RS2 and target-only. Also a CSV of the grid.
- Caveat to print: positions j >= 1 use prefixes drawn from RS2's q, not from the alternative q.

## 3. Tests (CPU, TRITON_INTERPRET=1; new `bench/rs2d/`)

- `bench/rs2/test_sparse_rs.py` with `WT=~/tools/sglang-rs2d`: A-D pass unchanged.
- `bench/rs2d/test_rs2d.py`:
  - K1: knobs at defaults (explicit kwargs and omitted) bit-identical to the RS2 worktree's module (loaded by path
    from `~/tools/sglang-rs2`, read-only) on test A's grid (fp32 and bf16 ties, n 1/2/4/7, hot on/off, min_p
    None/0.05/0.2, positions and chain on).
  - K2: `temp_scale=s` (0.5, 0.7) bit-identical to the reference called with `temperatures * s`.
  - K3: `onehot_above=θ` (0.5, 0.9): rows whose reference `probs[0] >= θ` give one-hot probs, TopkP 1.0 and the
    rank-0 token; other rows bit-identical to the reference. Include rows on both sides of θ.
  - K4 (exactness with knobs on): accept@0 within 3 SE of Σ min(p, q), and the chain's output distribution at
    positions 0 and 1 passes chi2 (Fisher-combined p > 0.01), for (s, θ) = (0.5, 0), (1.0, 0.6), (0.7, 0.8),
    using test D's machinery.
  - K5: `rs_dump` round trip: synthetic tensors with the real shapes, flush, load, fields and dtypes.
  - K6: `accept_offline.py` on a synthetic dump with hand values: p = (0.9, 0.1), q = (0.6, 0.4) on the same two
    ids (top_p 0.95, min_p 0.05) → RS2 a = 0.7, target-only 0.9, s = 0.5: w = (0.36, 0.16)/0.52 →
    a = 0.692308 + 0.1 = 0.792308, θ = 0.6 at s = 1 → 0.9; a two-position row checks E = a0 + a0 * a1.
- `bench/rs2d/check_integration_rs2d.py` (from `bench/rs2/check_integration_cpu.py`): knobs unset → the call
  kwargs equal cf1e515722's; knobs set → both call sites get them; dump off → `eagle_sample` never imports
  `rs_dump`.

## 4. Measurement and decision

1. GPU dump run (parent, after the evening queue, under the GPU lock):
   `WT=~/tools/sglang-rs2d PORT=8031 OUTDIR=runs/rs2d PROMPT_LIMIT=3 SERVER_ENV=SGLANG_RS_DUMP_DIR=<abs>/runs/rs2d/dump
   bash bench/rs2/arm_rs2.sh D1 rs lmstudio` (12 requests, ~15 min with the start-up).
2. `accept_offline.py`: choose (s, θ) by the E-based tok/step, weighting the four workloads equally. Go on only if
   the best pair gains >= 2% tok/step over RS2 on the pooled set and code-edit reaches target-only's level.
3. ABBA (8 starts) of RS2 against RS2 + chosen knobs, lmstudio and greedy, same §6 rule as RS2.
   **Replaced 23:55** (step 2 failed, RS2 failed its own §6): one 8-start ABBA of the whole RS package against
   target-only instead, `bench/rs2d/abba_rs2d.sh` after the night queue (`bench/rs2d/abba_after_queue.sh`):
   A = stack target-only, B = stack + RS + RS2 + G1 + RS3 BV + (s, θ) = (0.7, 0.9), same §6 rule as RS2. This is
   the measurement for the user's decision on RS as an LM Studio option, not a test of the knobs on their own.
4. Secondary analysis (fixed at 23:55, before the package ABBA's first arm). The RS2 ABBA's greedy verify span moved
   up to 35% between arms on the same code, more than any effect under test. The launcher logs the SM and memory
   clocks, power, pstate and the power-cap counter every 5 s (`runs/rs2d-abba/clocks.csv`).
   - `bench/rs2d/ancova_clock.py` refits each ANCOVA with the log of each request's mean SM clock over its decode
     window as one more covariate. It prints the clock elasticity, a tok/step placebo (expected ~0) and the B - A
     clock difference.
   - `bench/rs2d/clock_phase.py` prints the clocks per arm phase.
   - The §6 decision stays on the primary ANCOVA. The clock-adjusted lines go next to it in the report. If RS
     itself moves the clock (B - A clock CI excludes 0), the adjusted effect leaves out that path and the report
     says so.
5. Expected result (written 02:45, while A1 ran and before any B arm started). Model: RS2's 8-start ABBA per
   workload, plus the package's acceptance over RS2 (IS on the D1 dump, RS3 spec 3), plus kernel timings.
   - LM Studio t/s, package vs target-only: prose-en about +8%, prose-ja about +7%, agent-loop about +6%, pooled
     about +5%.
   - Code-edit about -0.7%. Acceptance is +0.3% against target-only (dump, E-based: RS2 -0.9%, the package +1.2%
     over RS2). The step costs about +1%: 7 proposals of about 20 us each (smoke traces 8.4 + 12.1 us).
   - Greedy ms/step|a about +0.1..+0.3%. G1's proposal is 1.8 + 4.5 us per draft step in the smoke traces.
   - Power: RS2's code-edit t/s CI was ±4.1%, ST1's ±5.9%. At a true -0.7% the code-edit lower bound lands near
     -5%, below §6's -2%, in most runs (about 80-90%). A §6 FAIL from code-edit alone, with its point estimate
     within about 2% of 0, therefore reads as "code-edit unresolved", not as a measured slowdown. The rule does not
     change: it still means "not adopted".

## 5. G1 (added 21:10): greedy fast path in the proposal

Why. The greedy traces (20 profiled steps, code-edit; kernel medians, which the display stalls do not move) show
what RS2 adds per verify step when every row is greedy. All-greedy batches already verify with
`verify_tree_greedy`, so only the proposal is left:
- RS2 runs its 64-way proposal 7 times per verify: `_draft_partial_topk_kernel` 9.0 us and `_draft_finalize_kernel`
  12.4 us.
- Target-only runs the top-1 kernels 6 times (1.9 + 1.8 us) and one argmax (10.4 us).
- Net: about +121 us per verify step, or +1.07% of an 11.3 ms step. RS2 §6 wants greedy ms/step <= +1%.

Change (`sparse_rs.py`). Both proposal kernels gain the constexpr `GREEDY_FAST`; `_draft_partial_topk_kernel` also
gains `TopKs`. With GREEDY_FAST on, a row whose `top_k == 1` takes a fast path (SGLang stores a temperature-0
request as top_k 1):
- partial: store the block's max key in all KB slots (one `tl.max` in place of `tl.topk`);
- finalize: `best` = the row's max candidate key, broadcast to all KB ranks (no `tl.topk`). The rest of the kernel
  is today's code.

Result for those rows:
- Probs, TopkP, TopkIndex, the chain column and positions are bit-identical to the full path.
- Tokens ranks >= 1 repeat rank 0's id with probability 0. The full path has the true ranks 1..63 there, also with
  probability 0.
- The chain verify sums q over matching ids (`qt` and the residual's `matched`), so zero-probability entries never
  change its output.

Other rows (top_k != 1) run today's code.

Wiring:
- env `SGLANG_RS_GREEDY_FAST = EnvBool(False)`;
- wrapper kwarg `greedy_fast: bool = False`;
- passed at both call sites and shown in the "SGLANG_OPT_SPEC_SPARSE_RS on" line.

Tests (CPU, in `bench/rs2d/test_rs2d.py`):
- G1a: `greedy_fast=False` is bit-identical to the RS2 module (as K1).
- G1b: `greedy_fast=True` on test A's grid with mixed rows. Include top_k 1 rows with bf16 ties at the maximum, so
  the smallest index must win.
  - top_k != 1 rows: every output bit-identical.
  - top_k == 1 rows: Probs, TopkP, TopkIndex, chain column and positions bit-identical; Tokens[:, 0] identical;
    Tokens[:, 1:] equal to Tokens[:, :1]; Probs[:, 1:] all 0.
- G1c: `chain_speculative_sampling_sparse` gets the draft support built with `greedy_fast` True and then False, with
  the same uniforms. Use mixed batches and random target probs, including rows where the draft token is rejected.
  predicts, accept_index and accept_token_num must be identical.
- G1d (CUDA, parent): an E timing row for "all rows top_k 1, greedy_fast": target <= 6 us at n=1, fp32 (the
  target-only top-1 pair is 3.7 us).

Measurement: GREEDY_FAST goes into the RS2d ABBA build (B = RS2 + chosen knobs + GREEDY_FAST). In the B greedy
traces, the two proposal kernels' medians should drop to about 2-4 us.

## 6. RS4-lite (added 2026-10-02 01:30): a smaller draft support K

Why. Both proposal kernels sort with `tl.topk(k=K)`, K = `SGLANG_RS_DRAFT_TOPK` (64 in every run so far).
- Triton 3.7.1's `tl.topk` is a bitonic sort. A smaller K means fewer sort passes in `_draft_partial_topk_kernel`
  (24 splits of 2048), and `_draft_finalize_kernel` sorts 24 x K candidates (a 2048 block at K 64, 512 at K 16).
- G1d (fp32, n=1): the full proposal costs 21.9 us per draft step, about 153 us per verify at 7 steps. Target-only
  pays 3.5 us per step. For scale, BV's whole verify costs 9.7 us at 8 slots (RS3 B5, vectorized).
- After LM Studio's filters (top_k 40, top_p 0.95, min_p 0.05) and the (0.7, 0.9) knobs, q has 6.7-15.1 nonzero
  ranks per verify, summed over its 7 positions (D1 dump). Most of the 64 sorted ranks end with probability 0.
- `SGLANG_RS_DRAFT_TOPK` already sizes the support buffers (`eagle_info.py`, the draft graph runner,
  `eagle_worker_v2.py`), so K = 16 needs no code change. Any K keeps the output exact: it only changes q, and the
  same q draws and verifies.

Offline (`bench/rs2d/support_sweep.py`, `runs/rs2d/support_sweep.log`). On the D1 dump at (0.7, 0.9), q is cut to
its first K' ranks before the sharpening and the filters, as the kernel does, and the BV path estimate is
re-weighted (IS E_bv', as in RS3 spec 3). Change in tok/step against K' = 64, paired on the same verifies:

| K' | pooled | agent-loop | code-edit | prose-en | prose-ja |
|---|---|---|---|---|---|
| 32 | -0.000% | +0.000% | +0.000% | +0.000% | -0.001% |
| 24 | +0.001% | -0.000% | +0.000% | -0.001% | +0.009% |
| 16 | -0.005% | -0.020% | -0.001% | +0.008% | -0.013% |
| 12 | -0.002% | +0.007% | +0.007% | -0.014% | -0.044% |
| 8 | -0.038% | -0.012% | -0.033% | -0.036% | -0.106% |
| 4 | +0.001% | +0.178% | +0.014% | -0.137% | -0.218% |

K' = 16 keeps the acceptance; prose starts to lose at 8.

CPU tests (`bench/rs2/test_sparse_rs.py` with `RS_K`; the implementer's run in `bench/rs2d/PROGRESS_K.md`):
- A: the proposal against the torch reference, bit-identical outputs (max delta 6e-8).
- D: losslessness with top_k 8, so the cut does not bind.
- D2 (new): top_k off and top_p 0.95 over 24 hot tokens, so the cut binds at K < 24. Support <= min(K, 24) ranks,
  q sums to 1, chi2 per position and Fisher for token-level and block verification, accept@0 within 6 SE.
- All PASS at K = 16 and K = 8. Parent rerun at K = 16 with another seed (20261002): A and D2 PASS
  (`bench/rs2d/k16_rerun.log`).

Next (GPU, after the package ABBA, about 20 min):
1. `WT=~/tools/sglang-rs3 python bench/rs2d/support_timing.py` under the lock (2-3 min): us per proposal call at
   K 64/32/16/8 (sampling rows, package settings) and 64/16 (greedy rows), fp32 and bf16, n 1 and 2.
2. If K = 16 saves >= 3 us per call: one smoke arm, B_ENV + `SGLANG_RS_DRAFT_TOPK=16` (PROMPT_LIMIT=1, traces,
   lmstudio and greedy).

Decision (fixed 02:40, before any K = 16 GPU run). K = 16 joins the RS package if:
- `support_timing.py` shows >= 3 us saved per call at n = 1 (fp32 and bf16, sampling rows);
- the smoke's traces show both proposal kernels' medians dropping by about as much against the package ABBA's B
  traces, the "draft support K=16" log line, and 0 error lines.
No ABBA for this. About 7 x 3 = 21 us or more per verify is 0.2-0.5% of a step, below what 8 starts resolve, and
the acceptance change is fixed offline (-0.005%, paired). Shipping stays the user's call with the rest of RS.

Result (2026-10-02 06:40): **K = 16 joins the RS package**; all four conditions hold.
- `runs/rs4/support_timing.log` (06:18), sampling rows, n = 1: fp32 21.10 -> 8.22 us per call (-12.89), bf16
  18.16 -> 8.34 (-9.82); n = 2 saves 14.98 and 10.94. Greedy rows: fp32 n = 1 +0.05, bf16 -0.78.
- Smoke K16 (`runs/rs4/smoke-k16.log`, 06:19-06:29, sglang-rs3 a78b1a5ebd): lmstudio and greedy rc 0, error lines
  0, rs2 enabled lines 1, start-up line "draft support K=16".
- Kernel medians against the package ABBA's B traces (`runs/rs4/proposal_kernels.log`, us per call):

| trace | partial B -> K16 | finalize B -> K16 | sum | chain verify B -> K16 |
|---|---|---|---|---|
| code-edit | 8.58 -> 5.04 | 12.29 -> 4.10 | -11.7 | 8.74 -> 6.58 |
| prose-en | 8.54 -> 4.93 | 12.22 -> 4.03 | -11.8 | 9.57 -> 6.08 |
| g-code-edit (G1 path) | 1.82 -> 1.50 | 4.58 -> 2.94 | -2.0 | - |

- The -11.7 us per draft step lies between the fp32 and bf16 timings. The calls per verify differ (K16's traces
  mostly ran 15 draft steps, B's mostly 7): that is the adaptive step controller (steps 3/7/15). Its inputs are the
  draft's own top-1 probability (`top1_prob` of the draft logits, `eagle_worker_v2.py`) and the accept lengths, not
  q, so K does not steer it, and the per-call medians compare like with like (same grid).
- Per verify step this saves about 82 us at 7 draft steps and 176 us at 15, roughly 1% of a step (the smoke's
  unprofiled steps: 10.1 ms prose-en, 17.2 ms code-edit); greedy about 14 us. No ABBA, by design.
- Projection, not a measurement: about 1% off code-edit's step would move its ABBA result from +1.6 [-2.5..+5.9]
  to about +2.6 [-1.5..+6.9]. The section-4 decision stays on the K = 64 ABBA as measured.
- Shipping: `SGLANG_RS_DRAFT_TOPK=16` with the rest of the package, the user's call. The shipping-candidate smoke
  (`bench/cand/smoke_cand.sh 16`) did not run: the GPU plan allowed it only if the package passed RS2 §6.
