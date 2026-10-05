# ST1 (2026-10-01): index-shared draft rows as a valid prefix

Status (2026-10-02 02:30): **8 starts done. Not adopted as a speed-up (§3 rule 1 fails), and no cost either. Turn
it on only together with XA1, whose A/B ran with ST1 in both arms. The switch is the user's call.**
- Runs: A1 B1 B2 A2 16:50-17:59 (`runs/st1/abba-st1.log`), B3 A3 A4 B4 01:12-02:19 (`abba-st1-ext.log`). ST1 log
  lines 0 in every A arm and 3 in every B arm; error lines 0 in all 8 server logs.
- ANCOVA over 8 starts, B vs A, request-level CI (`runs/st1/ancova8.log`, `ancova8_per_workload.log`):

  | mode | ms/token | tok/step | ms/step\|a |
  |---|---|---|---|
  | LM Studio | -2.09% [-4.35..+0.21] | +0.38% [-2.28..+3.11] | -1.68% [-3.16..-0.18] |
  | greedy | -0.64% [-4.01..+2.84] | -3.49% [-7.81..+1.03] | -3.03% [-4.70..-1.33] |

  Arm-level CIs (5 dof) are about 3x wider and all straddle 0.
- The evening speed-up did not repeat at night. Night arms alone: LM Studio ms/token +0.18% [-2.56..+2.99],
  ms/step|a +0.24% [-0.84..+1.34] (resid sd 2.0%, against 4.7% in the evening). The greedy tok/step drop of 18:20
  (2 outlier A requests) shrank to -3.49% and straddles 0.
- Against §3:
  - rule 1 fails: the pooled tok/step CI is > 0 in neither mode, and every per-workload tok/step CI straddles 0;
  - rule 2 holds: no ms/token CI lies entirely above +1%, pooled or per workload (the widest, greedy prose-ja
    t/s -3.06% [-13.02..+8.04], straddles 0);
  - rule 3 holds: 0 error lines.
- Reading: the draft now attends its newest positions, yet acceptance did not move beyond noise. The 18:20 guess
  stands: the MTP input already carries the previous hidden state and the current token's embedding.
- Recommendation: output is exact either way (§0). XA1 (`specs/XA1_DECODE_ATTN_2026-10-01.md` §7b) was measured
  with ST1 in both arms, and its scan bound on index-shared rows relies on ST1's layout. ST1 alone gives no
  reason to switch.
- Correction: the 18:20 table quotes request-level CIs, not arm-level ones as its caption says.
Status (18:20): **4-arm ABBA done (16:50-17:59): inconclusive. Extend to 8 starts (BAAB) per §3.**
- Run: `runs/st1/abba-st1.log`, order A1 B1 B2 A2. ST1 log lines 0/3/3/0; error lines 0 in all 4 server logs.
  The new lookup kernel ran only in the B traces (the old one only in A), so the flag took effect.
- ANCOVA, B vs A (arm-level CI, server start as the unit):

  | mode | ms/token | tok/step | ms/step\|a |
  |---|---|---|---|
  | LM Studio | **-4.31%** [-7.52..-0.99] | +0.95% [-3.01..+5.09] | -2.97% [-5.35..-0.53] |
  | greedy | +0.81% [-4.91..+6.88] | **-8.17%** [-14.86..-0.96] | -4.74% [-7.89..-1.48] |

- Pooled t/s: greedy A 359.7/365.4, B 359.1/363.1; LM Studio A 302.1/318.8, B 309.1/336.2.
- The greedy tok/step drop is mostly 2 requests that fell into high-acceptance runs in the A arms (likely
  repetition; greedy text differs between starts): A1 prose-ja-02 4.10 and A2 prose-ja-03 6.08 tok/step, against
  per-prompt medians 2.44 and 2.54. These are the only requests above 1.5x their prompt's median (none in LM
  Studio). Dropping them (post hoc, sensitivity only), greedy B is still 2-5% lower than A in every domain:
  code-edit A 11.21/10.52 B 10.13/10.06, prose-en 2.46/2.44 vs 2.38/2.43, prose-ja 2.65/2.71 vs 2.60/2.38,
  agent-loop 4.33/4.37 vs 4.19/4.28. LM Studio shows no such pattern (prose-ja B higher, the rest mixed).
- Against §3: the pooled tok/step CI is > 0 in neither mode (not adopted yet). The paired greedy prose-ja t/s
  -11.3% [-19.4..-1.7] comes from the same two A requests. The LM Studio tok/step CI straddles 0, so the rule
  says extend to 8 starts. Queued after the RS2 chain (B3 A3 A4 B4).
- Expected acceptance gain did not show at 4 starts: below about 2048 tokens every draft step at 2..S was hit,
  yet prose tok/step moved by less than the noise (resid sd 7-15% per request). The MTP input carries the
  previous hidden state and the current token's embedding, so the attention fix may matter less than assumed.
Status (16:50): **GPU test D PASS at 8 warps (commit 1df3112044); the A/B is queued after the DT1 exactness chain.**
- First GPU run (4 warps): every correctness check passed, but the latency check failed: +2.0 us per launch
  against a +1 us budget.
- The parent swept num_warps on the new kernel (2055 columns, 20 launches per CUDA graph, median per launch; the
  old kernel is about 1.09 us at 4 warps): 4 warps 3.13-3.2 us, **8 warps 2.44-2.62 us**, 16 warps 2.6-2.75 us,
  32 warps 3.45-3.54 us. The parent set the kernel to 8 warps (one-line edit, commit 1df3112044).
- The parent raised the test budget to +2 us per launch. About 6 lookups run per decode step (120 launches in 20
  profiled steps), so +1.5 us per launch is about 9 us of an ~11 ms step (~0.08%); the ABBA decides, not this
  budget.
- Re-run (DEV=cuda): 8 checks, 736/736 assertions PASS. Delta +1.44 us (1 row) and +1.54 us (2 rows) at
  2055/2059/2067 columns.
Status (16:35): **implemented, CPU tests PASS; GPU test D and the A/B queued.**
- Worktree `~/tools/sglang-st1`, branch opus/st1-shared-tail, commit d680442609 (on c868f2ee86).
- The implementer (an AI coding assistant) wrote it in one run with no rework. The parent moved the flag after the DT1 group, with a comment.
- The parent re-ran `bench/st1/test_shared_tail.py` from scratch: 14 checks, 10462/10462 assertions PASS
  (17 min, interpreter).
- GPU: test D runs in the gap after C1. The A/B runs after the DT1 exactness chain, through
  `bench/st1/chain_st1.sh` (test D again, then `abba_st1.sh`).
Status (15:45): design. Bug found by XA1 (`specs/XA1_DECODE_ATTN_2026-10-01.md` §1.3). The parent confirmed it
by reading the code.

## 0. The bug

- **Draft decode.** Steps 2..S of each draft chain run the MTP layer's QSA attention through index sharing
  (`QSAMTPSharedSparseIndices`, `layers/attention/qwen_sparse_attn_backend.py`). Each row is built as:
  1. the anchor row's 2051 frozen columns, captured at draft extend;
  2. `tail_width` columns `[captured_len .. position]`, at columns 2051 and up;
  3. -1 beyond.
- **The contract.** The packed attention path (`_forward_trtllm_sparse`, and the FA2 fallback) assumes the valid
  columns form a prefix:
  - `_compact_kv` writes column c to packed slot c;
  - the kernel then attends slots `[0, valid_count)`.
  - `torch_expand_qsa_block_indices` keeps this contract on purpose: "Keep all valid entries contiguous. This is
    required by the FA2 packing path." (`qsa/kernel.py:144`).
- **Where it breaks.** The anchor row ends with -1 columns:
  - short context (L_a < 2048): everything from L_a on;
  - long context: the h = 3 - (L_a mod 4) unused uncompressed-tail columns.
  Then the tail at column 2051 and up is not a prefix.
- **Effect at draft step j:**
  - short context: all j drafted positions are dropped, including the current token. j stale scratch slots are
    attended instead.
  - long context: the newest min(j, h) positions are dropped, and the same number of stale slots are attended.
- **Workloads hit.** prose-en, prose-ja and agent-loop prompts are below 2048 tokens: every draft decode step
  there is in the short-context case. code-edit is in the long-context case, 3 anchors in 4.
- **Cost.** The draft's attention at steps 2..S misses its most recent context. Draft quality, and so acceptance,
  is lower than it should be. Output correctness is unaffected, because the target verifies every token.

## 1. Fix

- **Flag** `SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX` (EnvBool, default False, in the Speculative decoding section of
  `srt/environ.py`, next to `SGLANG_OPT_DRAFT_TAIL`).
  - Read once in `QSAMTPSharedSparseIndices.__init__`.
  - Off means production's lookup exactly: same kernel, same launch, same output.
- **Lookup with the flag on**, for each row:
  - Compact the anchor's valid frozen columns (value >= 0, among the first `num_columns - tail_width`) to a prefix,
    keeping their order. Let n_a be their count.
  - Write the tail at columns `[n_a, n_a + tail_width)`: value `captured_len + t` if it is <= the current position,
    else -1.
  - Write -1 in every column from `n_a + tail_width` on.
  - The anchor row is already a valid prefix in practice. The compaction costs nothing extra, and it guards the
    contract against any anchor layout.
- **CUDA path:** a new Triton kernel, one program per row, `BLOCK = next_power_of_2(num_columns)`:
  - load the frozen part;
  - rank = inclusive `tl.cumsum(valid) - 1`, n_a = `tl.sum(valid)`;
  - store the tail and the -1 fill at columns >= n_a, then scatter the valid frozen values to their ranks. The two
    stores' addresses are disjoint.
- **CPU path:** the same semantics in torch, with a stable compaction as in `torch_expand_qsa_block_indices`.
- Capture is unchanged.
- **Never-captured rows** (zeros, `captured_len` 1: graph warm-up dummies and the trash row) are all valid at
  position 0. So n_a = 2051 and the output equals production's.
- **Log line** at construction when the flag is on:
  `QSA MTP shared indices: tail after the valid prefix (SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX)`.

## 2. Tests (`bench/st1/test_shared_tail.py`; DEV=cpu with TRITON_INTERPRET=1, DEV=cuda later)

Build the anchor rows with the worktree's `torch_expand_qsa_block_indices`, or with `bench/xa1/cases.fresh_row`.

- Anchors L_a ∈ {1, 3, 4, 5, 1000, 2047, 2048, 2049, 2050, 2051, 2052, 8184, 8190, 8191}.
- tail_width ∈ {4, 8, 16}; draft steps j = 1 .. tail_width - 1 (position L_a + j - 1).
- Rows: bs 1 and 2, including a never-captured row and the trash row.

**A. Flag off.** The lookup output is bit-identical to a copy of production's lookup, for both the Triton kernel
(called directly; it runs on CPU tensors under the interpreter) and the torch path.

**B. Flag on.**
- The valid entries (>= 0 and <= position) form a prefix.
- Their multiset equals the anchor's valid entries plus {L_a .. position}, with no duplicates.
- Every entry after the prefix is -1.
- The Triton kernel equals the torch path, bit for bit.
- Never-captured rows equal production's output.

**C. Through production's packing.** Use `bench/xa1/hole_demo.py`'s `_packed_positions`: the worktree's valid-count
pass and `_compact_kv`, with K = position + 1.
- Flag on: no dropped position and 0 stale slots, in every case.
- Flag off: reproduces the XA1 table (`bench/xa1/results/xa1-hole-demo.json`). This shows the test can see the bug.

**D. (cuda, the parent runs it)**
- The kernel on GPU equals the CPU torch path.
- Time per call at rows 1 and 2, num_columns 2055 / 2059 / 2067, in a CUDA graph of 20 calls: old kernel vs new.
- Budget: new ≤ old + 1 µs.

## 3. Measurement and decision

- **Server A/B, 4-arm ABBA** (`bench/st1/abba_st1.sh`, through `bench/stack/arm_stack.sh` with WT =
  `~/tools/sglang-st1`):
  - A = the stack (5 flags, RQ2 u2h);
  - B = A + `SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=1`;
  - min_p off, as in stack8, so the numbers compare with STACK §4.
- **Primary metric:** tok/step per domain, LM Studio sampling and greedy. Then ms/token and ms/step|a.
- **Expected:** tok/step up in prose-en, prose-ja and agent-loop (short context, every draft step hit), less in
  code-edit. ms/step flat; the lookup kernel changes by ≤ 1 µs per draft step.
- **Adopt if:**
  - pooled tok/step CI > 0 in at least one mode;
  - no domain gets slower beyond noise: no ms/token CI lies entirely above +1%;
  - 0 error lines.
- **More starts:** if the 4-arm CIs straddle 0, extend to 8 starts (BAAB).
- **Shipping** the fix to production is the user's decision. It is a bug fix, so the expected recommendation is
  to ship it on by default.
