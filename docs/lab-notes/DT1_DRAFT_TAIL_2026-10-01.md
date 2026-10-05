# DT1 — draft-phase tail glue (NEXTN topk=1, 2026-10-01)

Worktree `~/tools/sglang-dt`, branch `opus/draft-tail`, based on production 7b4d539f9b. Code is prototyped and
checked with GPU microbenchmarks. The server A/B scripts in `bench/dt1/` are ready but have NOT been run (§6).
Labels: **measured** = from a trace or a GPU microbenchmark in this note; **est.** = measured per-call cost ×
an assumed count.

Status (parent, 16:55): **the w4 exactness check cannot judge DT1 either.** `runs/stack8/exact/`, chain log
`runs/stack/chain-exact.log`. Even with `--disable-flashinfer-autotune`, w4 greedy text is not reproducible:
- X1 vs X2 (same config, DT off): 0/8 identical.
- Two repeats inside one server: 0/4 identical in each of X1, Y1 and X2.
- Same-config pairs diverge as early as pairs with DT on (first differing char):
  | pair | first differing char (8 prompt-repeats) |
  |---|---|
  | X1 vs X2 (same config) | 667, 511, 121, 30, 73, 598, 122, 144 |
  | X1 vs Y1 (DT on) | 466, 16, 108, 30, 73, 71, 122, 144 |
  | X2 vs Y1 (DT on) | 466, 16, 108, 448, 466, 71, 183, 216 |
- So some kernel in the untuned w4 path is still nondeterministic (not the tuned MoE finalize). This was not
  investigated further.
- DT1's exactness therefore rests on the kernel checks (§3: DT-t 16/16 exact against the reference tree kernel,
  DT-a/b/d bit-exact on synthetic inputs). C1 adds the server evidence: no errors, the enable line, and the trace
  B1 to C1: kernels per forward 49.83 to 42.5, epilogue 4 to 2, lmhead_to_partial 1 to 0, de_tail 17 to 15.
- The same limit applies to ST1 and RS2: their server checks drop the greedy-text gate; correctness comes from
  their unit tests (GPU) and the ABBA's acceptance numbers.

Status (parent, 15:05; replaces the fingerprint pass of the 14:25 block): wa's greedy text is not reproducible
across server starts (stack8 A1 vs A2 and B1 vs B2: 0/16 prompts with equal completion tokens and verify calls), so
the fingerprint cannot judge DT1. Instead, after C1, a w4 exactness check (`bench/stack/chain_exact_w4.sh`,
`runs/stack/chain-exact.log`): three w4 starts from `~/tools/sglang-stack-dt` with the stack flags and RQ2 u2h,
X1 (DT off), Y1 (DT on), X2 (DT off), 4 workloads x 2 repeats x 2048 tokens, full greedy text compared
(`bench/stack/exact_client.py`), all with `--disable-flashinfer-autotune` (the tuned MoE tactics fuse the finalize
with atomics, so even two production runs differ; the untuned ones use a separate finalize kernel). X1 == X2 shows the check can pass at all; Y1 == X1 then shows DT1 keeps every
greedy token. C1 still gives the enable line, the §6 trace evidence and a tok/verify sanity check.

Status (parent, 14:25): the DT1 ABBA (§6) is replaced: one server start moves ms/step by 1-3%, so no ABBA
resolves −0.4%. Instead, DT-a/b/d/t ride on the stack: worktree `~/tools/sglang-stack-dt` (branch `opus/stack-dt`,
stack 73e379d725 + ee132e0e53 + c868f2ee86, jitprobe hit=15 miss=3, CPU import OK). One arm C1 (stack +
`SGLANG_OPT_DRAFT_TAIL=1`) runs after the stack ABBA on its grid (`bench/stack/chain_dt_after8.sh`,
`runs/stack/chain-dt.log`). Pass: the enable line, the §6 trace evidence against B1, and the greedy fingerprint
(completion tokens, verify calls per prompt; `bench/stack/fingerprint.py`) equal to the B arms wherever the four
B arms agree. DT-p0 (not bit-exact, `..._FUSED_CONF`) stays out of this round.

## 0. Summary

| item | change | per-call saving (measured) | calls per step | bit-exact |
|---|---|---|---|---|
| **DT-a** (= DG-6) | no `EagerRunner` registry input copy captured into the draft decode graph (5 `memcpy32_post` + 1 `direct_copy` per draft forward) | −4.76..−4.93 µs per draft forward | F | yes |
| **DT-b** | fp32 `next_token_logits_buffer` for the draft graph: the draft lm_head writes fp32, so the bf16→fp32 cast after it is gone | −2.06..−2.11 µs per draft forward | F | yes |
| **DT-d** | draft_extend tail at topk=1: one Triton select pass (rows from `accept_lens` inside the kernel, split argmax, hidden row copy) replaces `select_index`, two gathers, `argmax` and `ones_like` | −16.2..−16.5 µs per step (in a graph), −17.3..−23.1 (eager, queued) | 1 | yes |
| **DT-t** (= DG-7) | draft-phase epilogue at topk=1: one Triton kernel builds the chain tree (positions, retrieve ×3, tokens, mask cells) instead of `cat` + retrieve `torch.full` + `build_tree_efficient` | −4.80 / −8.12 / −21.18 µs per step in a graph at 3 / 7 / 15 draft steps (eager, queued: −4.82 / −8.60 / −28.95) | 1 | yes |
| **DT-p0** | wa only: position-0 confidence comes from the same select pass instead of torch `top1_prob` (12 kernels) | further −27.0 µs per step (in a graph: 30.7 → 3.7) | 1 | **no**: p0 max rel. diff 2.0e-6 |

F = draft forwards inside the draft graph per step (= draft steps − 1): W4 2, W16 14; wa 4.28 at the width mix
0.67/0.21/0.12 for 3/7/15 steps (est.), 6.0 in the sv code-edit trace and 9.2 in the rq1 code-edit trace (measured).

Expected change in ms/step (est. = the measured per-call savings above × counts; the region is GPU-bound, CPU
launches run ~18 ms ahead, so GPU time saved is wall time saved):

| mode | `SGLANG_OPT_DRAFT_TAIL=1` | + `SGLANG_OPT_DRAFT_TAIL_FUSED_CONF=1` |
|---|---|---|
| W4 (F=2, no confidence channel) | −0.035 | same (no confidence channel) |
| wa, default mix (F=4.28) | −0.053..−0.055 | −0.080..−0.082 |
| wa, code-edit (F=6) | −0.065..−0.067 | −0.093..−0.094 |
| W16 (F=14) | −0.133..−0.144 | same |

That is 0.4-0.7% of a wa code-edit step (13.64 ms, measured trace step period), below what one ABBA resolves on
ms/step (FG1: per-request CI about ±5%). The A/B decision therefore rests on deterministic trace evidence plus a
no-regression check (§6). The user estimate was −0.10..−0.15 ms at W16/wa and −0.02 at W4: W16 and W4 agree;
wa is lower because the default mix spends 67% of steps at 3 draft steps.

Flags (`python/sglang/srt/environ.py`, both EnvBool default False):

- `SGLANG_OPT_DRAFT_TAIL`: DT-a, DT-b, DT-d, DT-t (all bit-exact).
- `SGLANG_OPT_DRAFT_TAIL_FUSED_CONF`: DT-p0; needs the first flag; not bit-exact.

With both off every call site takes the production branch: the flags are read once at init into attributes
(`_draft_capture_no_copy`, `draft_logits_buffer = None`, `_draft_tail_select`, `_draft_tail_fused_conf`,
`_draft_tail_chain`), and `build_tree_kernel_efficient(chain_topk1=False)` runs its old body.

Commits (worktree only, not pushed): **a545bc48c9** (DT-a, DT-b, DT-d, DT-p0), **471acb5c12** (DT-t).
Cherry-pick onto `~/tools/sglang-stack` 714eba5ab6: both commits apply without conflicts, onto 714eba5ab6 and onto the current stack head 73e379d725 (throwaway branch in this worktree, deleted afterwards), and the merged tree imports on CPU. On 73e379d725 the stack moved the p0 recording into `_record_position0_confidence`; the DT-d early return records p0 itself, the same way, and rejection sampling is excluded from the flag, so the merge keeps the behaviour.

DG-2 (fc_hidden half) gave no bit-exact win, see §5. DG-3 and DG-8 were not touched (superseded).

## 1. Where the glue is (CPU, production traces)

Source: `runs/sv/traces/A1-code-edit` and `A1-prose-en` (production `serve-fast.sh wa`, greedy, 20 profiled
steps, 6 draft forwards per step in both), read with `bench/dt1/dt_trace.py`,
`prof/trimmed_step.py` and a per-phase dump. All µs below are measured kernel durations from the trace.

### 1.1 One draft forward (inside the draft graph; 51.8 kernels per forward)

| kernels (trace) | µs | call site | item |
|---|---|---|---|
| 5 × `memcpy32_post` [1,1,1] + 1 `direct_copy` (`elementwise_kernel<128,2>`), right after `_draft_topk1_finalize` | 0.80-0.96 each, 1.28 | `EagerRunner.load_batch` → registry `fill_from`, captured into the draft graph because the graph runner calls the eager decode path during capture | **DT-a** |
| `direct_copy` [96,1,1] between the lm_head `_w8a16_gemv` and `_draft_topk1_partial_argmax` | 2.50 | logits processor writing the bf16 lm_head output into the fp32 logits buffer | **DT-b** |
| RMSNorm 2.40 + cuBLAS `cutlass_80_wmma ... 32x32_64x1` [8,10,20] 10.88 + `splitKreduce` 1.60 + add 2.08 | 16.96 | MTP entry: `fc_hidden(norm(h))` + table embedding | DG-2, §5 |
| `indexSelectSmallIndex` [20,1,1] | 1.89 | R3 embed table lookup (`SGLANG_MTP_EMBED_TABLE`) | DG-2 token half (done by R3) |
| `_mtp_shared_sparse_indices_lookup` on side stream s169 | 5.12 | DG-4 | not done (overlaps the main-stream GEMV) |
| `_fa2_valid_counts` | 1.18 | DG-9 | not done (attention metadata, XA1/DG-1c area) |
| `memcpy32_post` [5,1,1] inside the MoE block | 0.83 | MoE path | not touched (DG-3 area) |

`graph_copies_per_forward` (registry copies + casts launched by the graph, per forward) is 8.17 in production;
DT-a removes 6 and DT-b 1, so B should read about 1.2. The finalize kernel already advances `positions`; the
graph ends with one `CUDAFunctorOnSelf_add<long>` (1.47 µs per step), the `positions.sub_(steps - 1)` that
restores the buffer at the end of the captured `run_once`.

### 1.2 draft_extend (once per step)

- Eager head before the graph: 15 one-block elementwise kernels, 2 `multi_tensor_apply` and 1 DtoD, about 31 µs
  wall. Not touched (next candidate, §7).
- Graph: about 405 µs.
- Eager tail: 17 kernels, 58.2 µs summed, 69.8 µs wall (medians over 20 steps): gather of the hidden rows (7.2)
  and of the logits rows (6.2), `argmax` (11.4), `ones_like` (1.25); then, wa only, `top1_prob` (max,
  logsumexp, abs, isinf, masked_fill, sub, exp, sum, log, add, add, exp: 12 kernels) and the p0 DtoH.
  DT-d replaces the first four with two kernels; DT-p0 also replaces the 12 confidence kernels.

### 1.3 Draft-phase epilogue (once per step, eager, after the draft graph)

`cat` of bonus and draft tokens 1.70, verify-mask `torch.full` (1025 blocks, so `seq_lens_sum` ≈ 65.6 k) 1.25,
retrieve `torch.full(-1)` 0.93, `build_tree_efficient` (1 block of 8 threads) 7.97: 4 kernels, 14.8 µs wall
(median, 7 draft tokens). DT-t replaces the cat, the retrieve fill and the tree build with one kernel; the mask
fill stays (QSA verify probably never reads this mask, but that is not proven, so it is left alone).

### 1.4 Ranking at wa (per step)

| rank | item | µs/step at F=4.28 (est.) | at F=6 (est.) | bit-exact | status |
|---|---|---|---|---|---|
| 1 | DT-p0 | −27.0 | −27.0 | no | done, own flag |
| 2 | DT-a | −20.4..−21.1 | −28.6..−29.6 | yes | done |
| 3 | draft_extend eager head (15 tiny kernels) | −20..−25 if fused into 2 kernels | same | yes | not done |
| 4 | DT-d | −16.4..−16.5 | same | yes | done |
| 5 | DT-b | −8.8..−9.0 | −12.4..−12.7 | yes | done |
| 6 | DT-t | −7.5..−8.5 | −8.1..−8.6 | yes | done |
| 7 | DG-2 `SGLANG_MTP_FC_GEMV` | −6.5..−14 | | no | existing flag, not part of DT |
| 8 | graph-end `positions.sub_` folded into the last finalize | −1.5 | −1.5 | yes | not done |
| 9 | DG-4 | ~0 on the critical path (side stream) | | yes | not done |

## 2. Design (worktree files)

| file | change |
|---|---|
| `srt/environ.py` | the two EnvBools |
| `srt/model_executor/runner/eager_runner.py` | DT-a: `load_batch` returns `replace(forward_batch)` (as `SGLANG_EAGER_INPUT_NO_COPY` does) when the flag is on, the runner is the draft worker's, the batch is decode, and the current stream is capturing |
| `srt/speculative/eagle_draft_cuda_graph_runner.py` | DT-b: private fp32 `draft_logits_buffer` [max tokens, vocab or hot vocab], passed as `next_token_logits_buffer` to the captured forward |
| `kernels/ops/speculative/topk1.py` | DT-d: `_draft_extend_select_partial_kernel`, `_draft_extend_select_finalize_kernel`, `draft_extend_select_topk1`; DT-t: `_chain_tree_topk1_kernel`, `build_chain_tree_topk1` |
| `kernels/ops/speculative/__init__.py` | inventory entries for the two wrappers |
| `srt/speculative/eagle_worker_v2.py` | DT-d/DT-p0 in `_draft_extend_for_decode` (select index no longer built when the flag is on); DT-t switch passed to `build_eagle_verify_input` |
| `srt/speculative/eagle_worker_common.py` | `chain_topk1` pass-through |
| `srt/speculative/eagle_utils.py` | `build_tree_kernel_efficient(chain_topk1=...)`: the chain branch runs on CUDA for FULL_MASK and QLEN_ONLY with a bool mask; other modes and platforms keep the old body |
| `test/registered/kernels/ops/speculative/test_spec_topk1.py` | GPU unit tests: select pass vs torch (accept_lens path, ties, empty batch), chain tree vs the reference Triton tree kernel |

Invariants the code relies on:

- topk=1 chain: `num_draft_tokens == steps + 1`, `parent_list = arange(-1, steps - 1)`,
  `top_scores_index = arange(steps)` (`_rebuild_topk1_chain_buffers`). DT-t asserts the token count.
- The mask cells DT-t writes are exactly the ones `build_tree_efficient` writes (row x of a request's tree block
  attends to draft tokens 0..x); the prefix columns still come from the unchanged fill.
- DT-d reads `accept_lens` straight from the verify output; nothing modifies it between verify and the tail.
- DT-a: inside the draft graph the forward reads the graph runner's static buffers through `forward_batch`
  instead of a captured copy of them; outputs were equal on replay (§3), and the in-server greedy fingerprint
  (§6) is the end-to-end check.

## 3. Correctness (GPU microchecks, synthetic inputs)

| check | result (measured) |
|---|---|
| DT-a: graph of 3 draft-style forwards, registry copy vs no copy, bs 1 and 4 | replay outputs equal; the eager path still copies |
| DT-b: lm_head GEMV into fp32 buffer vs bf16 + cast, M = 1/2/4 | bit-equal |
| DT-d: select pass vs production torch tail, accept_lens path, 1200 + 1202 trials incl. 295 rows with top-1 ties | 0 mismatches (tokens, hidden rows, logits rows) |
| DT-p0: fused p0 vs `top1_prob` | max abs 1.8e-6, max rel 2.0e-6, median abs 1.1e-8 |
| DT-t: chain kernel vs the reference Triton tree kernel, CPU (`TRITON_INTERPRET=1`), 16 cases | 16/16 exact |
| DT-t: vs production `build_tree_kernel_efficient` (sgl_kernel CUDA), bs 1/2/4 × steps 1/3/7/15 × FULL_MASK/QLEN_ONLY × preallocated/new buffer × prefix fill on/off | 84 cases, 0 mismatches in every output (values, shapes and dtypes) |
| CUDA graph capture/replay | DT-a/DT-b/DT-d/DT-p0 timed inside captured graphs (§4); DT-t captured and replayed in a graph (§4.3) |
| unit tests `test_spec_topk1.py` (GPU) | 7 passed (job 6, with the DT-t test; 6 passed in job 3 before it) |

## 4. Timing (GPU microchecks, µs, medians)

"graph" = the code captured in a CUDA graph and replayed; "queued" = eager launches behind a GPU sleep, so the
GPU runs them back to back as it does in the server.

### 4.1 Per draft forward

| item | production | DT | saving |
|---|---|---|---|
| DT-a: 3 forwards in a graph, bs 1 | 40.70 | 25.92 | −4.93 per forward |
| DT-a: same, bs 4 | 40.27 | 26.00 | −4.76 per forward |
| DT-b: cast alone in a graph, M = 1/2/4 | 2.11 / 2.06 / 2.10 | 0 | −2.06..−2.11 |
| DT-b: GEMV + cast vs GEMV into fp32, M = 1/2/4 | 55.03 / 57.03 / 56.04 | 51.61 / 51.65 / 52.65 | −3.39..−5.39 |

### 4.2 draft_extend tail per step (draft steps 3 / 7 / 15)

| variant | production graph | DT graph | production queued | DT queued |
|---|---|---|---|---|
| no confidence channel (W4, W16) | 19.82 / 19.71 / 19.98 | 3.53 / 3.56 / 3.55 | 20.78 / 26.18 / 26.51 | 3.42 / 3.44 / 3.44 |
| wa, exact confidence | 47.12 / 47.16 / 47.15 | 30.65 / 30.72 / 30.70 | 55.72 / 55.06 / 56.05 | 37.72 / 37.80 / 37.90 |
| wa, fused confidence (DT-p0) | 47.12 / 47.16 / 47.15 | 3.7 | 55.72 / 55.06 / 56.05 | 3.72 / 3.67 / 3.67 |

### 4.3 Draft-phase epilogue per step (DT-t, FULL_MASK, `seq_lens_sum` 65592 as in the trace)

| draft steps (tokens) | production queued | DT queued | production graph | DT graph | saving (queued / graph) |
|---|---|---|---|---|---|
| 3 (4) | 7.03 | 2.21 | 6.98 | 2.18 | −4.82 / −4.80 |
| 7 (8) | 10.84 | 2.24 | 10.36 | 2.24 | −8.60 / −8.12 |
| 15 (16) | 31.22 | 2.27 | 23.46 | 2.28 | −28.95 / −21.18 |

Both sides include the unchanged prefix mask fill. In the trace the production epilogue takes 14.8 µs wall at 7
draft steps (`build_tree_efficient` alone 7.97), a little more than the 10.4-10.8 here. CPU launch cost per call
drops from 26.8-28.8 to 20.0-22.3 µs (not on the critical path).

### 4.4 Per-step totals (est.)

Per draft forward: DT-a + DT-b = 6.82..7.04 µs. Per step: DT-d 16.15..16.43 (no confidence channel) or 16.44..16.47
(wa, exact p0), 43.40..43.50 with DT-p0; DT-t from §4.3 (lower bound from the graph column, upper from the queued one).

| mode | F | DT-a + DT-b | DT-d | DT-t | total `on` | total `conf` |
|---|---|---|---|---|---|---|
| W4 | 2 | 13.6..14.1 | 16.3 | 4.8 | **−0.035 ms** | n/a |
| wa, default mix 0.67/0.21/0.12 | 4.28 | 29.2..30.1 | 16.4..16.5 | 7.5..8.5 | **−0.053..−0.055 ms** | **−0.080..−0.082 ms** |
| wa, code-edit (7 steps) | 6 | 40.9..42.2 | 16.4..16.5 | 8.1..8.6 | **−0.065..−0.067 ms** | **−0.093..−0.094 ms** |
| W16 | 14 | 95.5..98.6 | 16.4 | 21.2..29.0 | **−0.133..−0.144 ms** | n/a |

The rq1 code-edit trace ran 9.2 draft forwards per step; there `on` would be about −0.09 ms (est.).

## 5. DG-2 (MTP entry, fc_hidden half)

- `SGLANG_MTP_ENTRY_FUSED=1` does nothing once `SGLANG_MTP_EMBED_TABLE=1` (R3) is on: the fused entry path is
  not reached with the table.
- `SGLANG_MTP_FC_GEMV=1` still applies to `fc_hidden` at decode (M = 4 rows with HC = 4) but also changes the R3
  table build (`bf16_gemv`), and its outputs differ from cuBLAS in 8e-5 of the elements (max abs 0.0625, up to
  64 ulp at M = 4 rows). Saving (measured, synthetic, 16 rotating weight copies so each call streams from DRAM):
  −1.52 µs per forward at 1 token, −1.93 at 4 tokens (warm L2: −1.63..−2.20). In the trace the cuBLAS GEMM +
  splitK reduce take 12.5 µs, matching the cold microbenchmark (12.55-12.90). Est. −6.5..−14 µs/step at wa,
  −21 at W16, −3 at W4. Not bit-exact, so it is not part of `SGLANG_OPT_DRAFT_TAIL`; it can be A/B'd as its own
  arm (`SERVER_ENV=SGLANG_MTP_FC_GEMV=1`).
- `mtp_dense` FP8 never reaches `fc_*` (they stay plain `nn.Linear`); E0's `SGLANG_MTP_FC_FP8` failed its gate.
- Next candidate (est., not implemented): one kernel for RMSNorm + GEMV + embedding add would also remove the
  norm (2.40) and the add (2.08), about −4..−5 µs per forward more; not bit-exact either.

## 6. Server A/B (scripts ready; NOT run)

Setup, done (copies only; production files and the shared FlashInfer caches untouched):

```bash
WT=~/tools/sglang-dt; P=~/tools/sglang-rtxpro6000
cp $P/serve-fast.sh $WT/
sed "s|\$REPO/.venv/bin|$P/.venv/bin|" $P/serve-local.sh > $WT/serve-local.sh && chmod +x $WT/serve-local.sh
mkdir -p $WT/.cache && cp -a $P/.cache/{jit,triton,sglang,xdg} $WT/.cache/   # 2.4 GB
bash ~/tools/flash-next-bench/bench/jitcache/jitprobe.sh $WT                 # reported hit=15 miss=3
```

`serve-fast.sh` and `serve-local.sh` are untracked in the worktree on purpose.

| item | choice |
|---|---|
| arms | A = both flags off (production code path), B = `SGLANG_OPT_DRAFT_TAIL=1` (`on`); optional second ABBA with B = `on` + `SGLANG_OPT_DRAFT_TAIL_FUSED_CONF=1` (`conf`); same worktree and commit, `serve-fast.sh wa`, port 8023 |
| order | ABBA (`bench/dt1/abba_dt.sh`: A1 B1 B2 A2), each arm a fresh server in its own GPU-lock hold (`arm_dt.sh`, MemoryMax=110G, memcheck ≥ 110G); 30 s between arms for other lock waiters |
| grid | code-edit, prose-en, prose-ja, agent-loop × greedy and sampling (LM Studio parameters) × N=4 requests × 600 tokens (`prof/fc_sampling_probe.py`) |
| primary metric | ms per output token per request (`bench/dt1/dt_table.py`) |
| deterministic evidence | per arm, from the 20-step greedy code-edit trace (`bench/dt1/dt_trace.py`): `graph_copies_per_forward` (A 8.17 → B ≈ 1.2), `lmhead_to_partial_kernels` (A 1 → B 0), `draft_extend_select_*` count = 20 (A 0), `chain_tree_topk1_kernel` count = 20 and `build_tree_efficient` 0 (A 20), `epilogue_kernels` (A 4 → B 2), `de_tail_kernels` (A 17 → B ≤ 15 `on`, ≤ 4 `conf`), `de_tail_dur_sum` (A 58.2 µs → at least 15 µs lower `on`, at least 45 µs lower `conf`) |
| greedy fingerprint | `dt_table.py` last line: where A1 and A2 give the same (chunks, completion tokens) for a greedy request, B1 and B2 must give the same too (`on` is bit-exact) |
| server log | 0 Traceback/Error lines; `SGLANG_OPT_DRAFT_TAIL on: draft-extend select True, fused conf <False/True>, chain tree True` once |

Commands (from `~/tools/flash-next-bench`):

```bash
bash bench/dt1/smoke_dt.sh on        # one B server, no grid, then the CLEAN check (about 15 min)
bash bench/dt1/abba_dt.sh "" on      # A1 B1 B2 A2 -> runs/dt1/{A1,B1,B2,A2}-probe.jsonl + traces (about 1.5 h)
bash bench/dt1/smoke_dt.sh conf      # optional
bash bench/dt1/abba_dt.sh -conf conf # optional: labels A1-conf ... A2-conf
```

Decision rule. Turn `SGLANG_OPT_DRAFT_TAIL=1` on in `serve-fast.sh` only if all of these hold:

1. the deterministic evidence row holds in B1 and B2, and A1 and A2 show the production values;
2. greedy fingerprint: every request where A1 == A2 also has B1 == B2 == A (a single mismatch means the change
   is not bit-exact in the server: stop and investigate, most likely DT-a);
3. pooled ms/token is not worse in either mode: the upper bound of the 95% CI is below +0.5%;
4. no workload × mode loses more than 2% tok/step with a CI excluding 0;
5. drift |A2/A1 − 1| on ms/step is smaller than 1%; otherwise rerun the ABBA (the expected effect is about
   −0.4%, so a lower pooled ms/token is welcome but not required).

For `SGLANG_OPT_DRAFT_TAIL_FUSED_CONF=1` (not bit-exact): rules 1, 3, 4, 5 as above (evidence with the
`conf` values), and instead of rule 2, tok/step in each workload × mode within ±2% of A; it changes
p0 by about 2e-6 relative, which can flip a rare adaptive width decision but should not move acceptance.

Effort: about 1.5 h of GPU lock per ABBA (4 server starts of 8-12 min plus the grid), about 15 min per smoke.

## 7. Verified vs assumed; open risks

- Verified (measured): per-call savings in §4, bit-exactness in §3 on synthetic inputs, production kernel
  counts and durations in §1, CPU imports and unit tests.
- Assumed: per-step totals (counts × per-call savings), the wa width mix, and that the in-server effect equals the
  microbenchmark (the trace's 12.5 µs GEMM vs the 12.55-12.90 µs cold benchmark supports this for DG-2).
- DT-a: the draft graph now feeds the model the graph runner's live static buffers instead of a snapshot. If a
  kernel read `positions` after the in-graph `positions += 1`, outputs would change; the replay check passed and
  the greedy fingerprint is the end-to-end test.
- DT-d assumes `accept_lens` is not modified between verify and the tail (true in this code path).
- DT-t assumes the topk=1 chain buffers (asserted on the token count only).
- DT-p0 may flip rare adaptive width decisions (p0 changes by ~2e-6 relative).
- First calls of the new Triton kernels JIT-compile; server warm-up absorbs this (the A/B grid starts after the
  warm-up requests).
- The effect (~0.4-0.7%) is below ABBA resolution on ms/step; a no-regression ABBA plus trace evidence is the
  realistic bar. Greedy outputs differ across fresh servers (A2/A1 tok/step drift up to +22% in earlier ABBAs),
  which is why the fingerprint only uses requests where A1 and A2 agree.
- Not done, ranked: draft_extend eager head fusion (~31 µs wall today), a fused RMSNorm + GEMV + add for
  fc_hidden (not bit-exact, ~4-5 µs per forward), folding the graph-end `positions.sub_` into the last finalize
  (~1.5 µs per step), DG-4 (likely hidden on the side stream), DG-1c/DG-9 (attention metadata; XA1 works there).
