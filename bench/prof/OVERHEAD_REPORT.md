# Where the per-step time actually goes (W4 / W8 / W16, bs=1, NEXTN topk=1)

Date: 2026-09-05. CPU-only analysis of the existing chrome traces in
`$HOME/tools/flash-next-bench/prof/traces/` plus the fnbench run records in
`$HOME/tools/flash-next-bench/runs/`. No GPU used, no server started, no repo file changed.

## TL;DR

**The "wall exceeds kernel time, and the gap grows with the number of speculative steps"
observation does not survive measurement. There is no growing host-side overhead.**

* GPU idle inside a decode step is **0.15–0.21 ms** (median) in *every* config from 3 to 15
  draft steps. It does not grow. It is 1.0–1.3 % of the step.
* The host issues the whole step in ~4.5 ms and then sits blocked for 6–17 ms in a single
  `cudaEventSynchronize`. Host `cpu_op` union is 0.9–1.4 ms per step against a 12–20 ms step.
  The host has 8–19 ms of slack per step at every width. Removing host work saves nothing.
* The entire draft loop is **one** `cudaGraphLaunch`, at every width. There are exactly
  **4 graph replays per step** for S=3, 5, 7, 9 and 15 alike. There are no per-draft-step host
  launches to batch.
* The apparent gap is an artifact of `trimmed_step.py`: its `trimmed_ms` **under-reports GPU
  busy time by 0.8–3.6 ms, and the error grows with kernel count** (§2). Replacing it with the
  interval union of kernel intervals closes the gap: `Σ(phase union) + idle ≈ step wall` to
  within 0.2–0.4 ms in all 14 traces.
* Per-step wall is well described by **`step_ms ≈ 9.74 + 0.700·S`** (S = speculative steps) for
  T = S+1 a power of two, and the 0.700 ms/step splits as **0.322 ms of extra draft loop +
  0.378 ms of extra verify** (one more token through the 48-layer MoE). Both are GPU kernel time.

So the optimisation targets are kernels, not overhead. Ranked list in §6.

---

## 1. Method / trace layout

`profile_decode2.py` starts a streaming request, waits for 30 chunks, then POSTs `/start_profile`
with `num_steps=20`, so each trace holds 20 decode iterations (19 complete step intervals).

Phase markers are `user_annotation` events: `draft`, `step[TARGET_VERIFY bs=1]`, `draft_extend`,
nested inside `scheduler.run_batch` / `scheduler.process_batch_result` / `copy_result_to_cpu`.
`trimmed_step.py` attributes a kernel to a phase via the *launching* `cuda_runtime` event's
timestamp, which is correct; a step is delimited by consecutive `draft` annotation starts.

Two corrections applied here:

1. **Union, not sum.** Kernels land on up to 8 GPU "streams" (tids 13/49/65/69/151/152/168/176 —
   a CUDA graph's internal parallel branches). Summing durations double-counts. `raw_ms`
   over-counts by 10–15 %; the interval union is the real GPU-busy time.
2. **No name-level clipping.** `trimmed_step.py` clips each kernel to `2·median(by name)+5 µs`.
   `_w8a16_gemv_kernel` alone is launched at six different shapes per layer (4/5/6/13/24/55 µs in
   the draft, 4/5/7/8/13/30 µs + a 420 µs target-lm_head instance in verify), so a single median
   per *name* is meaningless and the clip deletes real work.

---

## 2. Measured per step (median of the 19 step intervals per trace)

All values in ms. `Σphases` = sum of per-phase kernel-interval unions. `busy` = union over all
kernels in the step. `wall` = draft-start to draft-start. `trimmed` = what `trimmed_step.py` prints.

| trace | S | T | draft | verify | extend | Σphases | busy | wall | idle | trimmed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prof-base32k-w4 code-edit | 3 | 4 | 0.58 | 10.56 | 0.45 | 11.59 | 11.73 | 11.95 | 0.15 | 10.89 |
| prof-base32k-w4 prose-en | 3 | 4 | 0.58 | 10.31 | 0.44 | 11.33 | 11.50 | 11.68 | 0.16 | 10.64 |
| prof-hot2-49k-w4 code-edit | 3 | 4 | 0.64 | 10.39 | 0.48 | 11.50 | 11.70 | 11.97 | 0.16 | – |
| prof-hot2-49k-w4 prose-en | 3 | 4 | 0.63 | 10.33 | 0.48 | 11.43 | 11.58 | 11.75 | 0.16 | – |
| prof-s5-w4 code-edit | 5 | 6 | 1.44 | 11.80 | 0.53 | 13.77 | 13.81 | 14.14 | 0.18 | 12.71 |
| prof-s5-w4 prose-en | 5 | 6 | 1.25 | 11.19 | 0.52 | 12.95 | 13.02 | 13.30 | 0.19 | 12.11 |
| **prof-s7-w4 code-edit (= W8)** | 7 | 8 | 2.10 | 11.80 | 0.54 | 14.43 | 14.59 | 14.90 | 0.17 | 13.55 |
| **prof-s7-w4 prose-en (= W8)** | 7 | 8 | 2.12 | 11.55 | 0.51 | 14.17 | 14.39 | 14.62 | 0.17 | 13.19 |
| prof-s9-w4 code-edit | 9 | 10 | 3.14 | 14.72 | 0.54 | 18.40 | 18.59 | 18.81 | 0.20 | 15.00 |
| prof-s9-w4 prose-en | 9 | 10 | 2.90 | 13.94 | 0.54 | 17.38 | 17.60 | 17.80 | 0.22 | 14.69 |
| prof-base32k-w16 code-edit | 15 | 16 | 4.44 | 15.10 | 0.55 | 20.09 | 20.28 | 20.47 | 0.21 | 18.62 |
| prof-base32k-w16 prose-en | 15 | 16 | 4.52 | 14.41 | 0.52 | 19.46 | 19.73 | 20.00 | 0.34 | 18.12 |
| prof-hot2-49k-w16 code-edit | 15 | 16 | 4.70 | 14.89 | 0.58 | 20.17 | 20.31 | 20.60 | 0.21 | 18.62 |
| prof-hot2-49k-w16 prose-en | 15 | 16 | 4.66 | 14.19 | 0.55 | 19.39 | 19.55 | 19.77 | 0.22 | 18.06 |

**`wall − busy` = 0.15–0.34 ms everywhere, flat in S.** `wall − trimmed` = 1.06 / 1.43 / 1.35 /
3.81 / 1.85 ms for S = 3 / 5 / 7 / 9 / 15 — that is the reporting error, and it tracks kernel count
(1997 → 2270 → 2200 → 2480 → 2677 kernels/step), not step count.

### T is *not* 4 in the s5/s7/s9 traces

`python/sglang/srt/arg_groups/speculative_hook.py:838-848` forces
`speculative_num_draft_tokens = speculative_num_steps + 1` whenever `speculative_eagle_topk == 1`
(and `base_spec_worker.py:120` asserts it). The `--speculative-num-draft-tokens 4` in the w4
launcher was silently overridden. Confirmed against the server metrics
(`sglang:spec_num_steps=7`, `sglang:spec_num_draft_tokens=8` in `runs/adapt-W8.jsonl`).

Consequence: **`prof-s7-w4-*` is the W8 configuration** (7 steps / 8 draft tokens), not "W8 draft
depth with W4 verify width". The premise "W8 kernel ≈ 13.2 ms vs wall 16.6 ms" compares a `trimmed_ms`
from this trace against a wall from a *different* server session; the trace's own wall is 14.62–14.90 ms.

---

## 3. Where the wall numbers came from, and why they disagreed

`fnbench/runner.py:126-132` computes `effective_forward = decode_tps / accept_length`, i.e. the
premise's "wall = accept_len / tps". Both inputs are unsound for this purpose:

* `accept_length` is read from the Prometheus **gauge** `sglang:spec_accept_length`, sampled once
  after the run. It is a windowed average over the last log interval, not the run average. On
  `code-edit` (repetitive, acceptance rises through the generation) it reads 14.12 at W16 while the
  actual run average is 2400 tokens / 218 chunks = 11.0.
* `decode_tps` uses `usage.completion_tokens`, which for these thinking-enabled runs disagrees with
  the streamed token count (`completion_tokens=2400` vs `reasoning_tokens=2406` in `adapt-W8.jsonl`).

Result: `accept/tps` spreads 17.2–27.9 ms across workloads *within one W16 config*, while the true
step time is flat at 20–22 ms.

A sound client-side per-step wall is available for free: `client.timeline` in every run record is one
entry per token-bearing SSE chunk (`fnbench/http_client.py:170-181`), and at bs=1 there is one chunk
per decode step. `decode_seconds / (len(timeline) − 1)`:

| config | code-edit | prose-en | agent-loop | prose-ja | trace wall (code / prose) |
|---|--:|--:|--:|--:|--:|
| W4 base32k | 12.45 | 11.90 | 12.06 | 11.75 | 11.95 / 11.68 |
| W4 hot2-49k | 13.84 | 13.26 | 13.39 | 11.81 | 11.97 / 11.75 |
| W8 (`adapt-W8`) | 17.88 | 16.59 | 16.51 | 15.01 | 14.90 / 14.62 (s7) |
| W16 base32k | 21.80 | 20.11 | 20.74 | 20.45 | 20.47 / 20.00 |
| W16 hot2-49k | 23.24 | 21.42 | 22.32 | 20.34 | 20.60 / 19.77 |

Distributions are tight (W16 base32k prose-en: p25 20, median 20, p90 21, p99 22, max 23 ms) — there
is no fat tail hiding overhead.

Residual trace-vs-client differences are 0.1–2.6 ms and are **not monotone in S**, so they are
session drift, not step structure. Direct evidence: the base32k→hot2 token-map change makes the
step 1.31 ms *slower* in the client numbers but 0.23 ms *faster* in the traces, while the traces
correctly show the only physically expected difference (draft phase +0.14 ms, from the 49152-entry
draft lm_head vs 32768). In-trace step-wall spread within a single 19-step trace is already
±1.8 ms (W16 base32k: 18.28 … 21.95 ms), which is the same size as the residual. The Max-Q power
cap / desktop-compositor preemption on this box
is the likely source.

---

## 4. Anatomy of one step (W16 base32k code-edit, step #9, 22.58 ms)

Host timeline, offsets from the step start, with the enclosing annotation stack:

```
+ 0.060  0.008ms cudaMemcpyAsync   scheduler.run_batch > draft > aten::repeat_interleave > aten::clone
+ 0.241  0.006ms cudaMemcpyAsync   scheduler.run_batch > draft > aten::copy_
+ 0.780  0.536ms cudaGraphLaunch   scheduler.run_batch > draft            <-- all 15 draft steps
+ 1.648  5 x ~0.004ms cudaMemcpyAsync  scheduler.run_batch > aten::_foreach_copy_
+ 1.839  2 x ~0.004ms cudaMemcpyAsync  ... > step[TARGET_VERIFY bs=1] > aten::copy_
+ 1.859  1.283ms cudaGraphLaunch   scheduler.run_batch > step[TARGET_VERIFY bs=1]
+ 3.260  0.010ms cudaMemsetAsync   scheduler.run_batch > aten::argmax
+ 3.451  0.035ms cudaGraphLaunch   scheduler.run_batch                    <-- sampling
+ 4.046  0.009ms cudaMemcpyAsync   ... > draft_extend > aten::copy_
+ 4.136  0.067ms cudaGraphLaunch   scheduler.run_batch > draft_extend
+ 4.418  2 x cudaMemcpyAsync       ... > copy_result_to_cpu > aten::copy_ (the D2H readbacks)
+ 4.517 17.018ms cudaEventSynchronize   scheduler.process_batch_result    <-- host blocked
+22.156  2 x cudaMemcpyAsync       scheduler.get_next_batch_to_run > aten::to
```

Per-step host call counts, identical (±3 %) for S = 3, 5, 7, 9, 15:

| call | /step | host time/step |
|---|--:|--:|
| `cudaEventSynchronize` | **1** | 6.4 (W4) → 10.4 (W8) → 13.7 (W16) ms *blocked, off critical path* |
| `cudaGraphLaunch` | **4** | 1.2–2.5 ms (queue backpressure) |
| `cudaLaunchKernel` | 54–56 | 0.16–0.25 ms |
| `cuLaunchKernelEx` | 18 (W4) → 26 (W8) → 43 (W16) | 0.06–0.11 ms |
| `cudaMemcpyAsync` | **15** | 0.07–0.10 ms |
| `cudaEventQuery` | 109–113 | 0.04 ms |
| `cudaStreamWaitEvent` | 6 | — |
| `cudaStreamSynchronize` / `cudaDeviceSynchronize` | **0** | — |

The single sync is `result.copy_done.synchronize()` in
`python/sglang/srt/managers/scheduler_components/batch_result_processor.py:863`
(`process_batch_result_decode`); the event is recorded by
`GenerationBatchResult.copy_to_cpu` in `python/sglang/srt/managers/utils.py:120`.

All 15 `cudaMemcpyAsync` are tiny and none of them blocks:

| direction / size | count/step | GPU time/step |
|---|--:|--:|
| DtoD 8 B / 128 B (W16), 8 B / 32 B (W4) | 8.6 | 8.5 µs |
| DtoD 20480 B, 327680 B (W16) / 81920 B (W4) | 1.95 | 2.7 µs |
| **DtoH 64 B (W16) / 16 B (W4) + DtoH 4 B** | **1.9** | **2.3 µs** |
| HtoD pageable 4 B | 1.9 | 1.3 µs |

The two D2H are the accepted token ids and the accept length. **2.3 µs of GPU time and zero
blocking** — the "D2H readback of accept lengths" hypothesis is dead.

The draft loop being a single graph is `eagle_worker_v2.py:655-659`
(`self.cuda_graph_runner.execute(forward_batch)`), with `n_inner = self.speculative_num_steps - 1`
at `eagle_worker_v2.py:643`. Kernel counts confirm one graph body per inner iteration:
125 draft kernels for S=3, 233 for S=5, 341 for S=7, 449 for S=9, 787 for S=15 —
exactly **54 kernels per additional speculative step**, all inside the replay.

---

## 5. The cost model, and what the ms are made of

Fitting the power-of-two-T traces (S = 3, 7, 15, code-edit):

```
draft(S)      = 0.322·S − 0.39   ms      (0.322 ms per extra speculative step)
verify(T)     = 9.05 + 0.378·T   ms      (0.378 ms per extra draft token)
draft_extend  = 0.50 ms   (flat)
gpu idle      = 0.20 ms   (flat)
------------------------------------------------------------
step(S), T=S+1 = 9.74 + 0.700·S  ms
   S=3  -> 11.84 (measured 11.95)
   S=7  -> 14.64 (measured 14.90)
   S=15 -> 20.24 (measured 20.47)
```

S=5 (14.14 vs predicted 13.24) and S=9 (18.81 vs 16.04) sit above the line — see §5.3.

### 5.1 Verify phase, by kernel family (code-edit; ms/step)

| kernel family (48 layers unless noted) | T=4 | T=6 | T=8 | T=10 | T=16 |
|---|--:|--:|--:|--:|--:|
| **MoE grouped GEMM ×2 (cutlass `GemmUniversal` / `GroupProblemShape`)** | **3.76** | **4.64** | **5.08** | **6.40** | **7.90** |
| `_w8a16_gemv_kernel` (dense FP8 proj + target lm_head) | 3.39 | 3.60 | 3.57 | 3.79 | 4.04 |
| `_hc_up`/`_hc_down`/`_hc_branch_stats`/`hc_combine_*` (97+97+97+192) | 1.46 | 1.44 | 1.48 | 1.70 | 1.49 |
| MoE glue (`expandInputRows`, `doActivation`, `blockExpertPrefixSum`, `mergeExpertPrefixSum`, `computeStridesTma`, `_router_triton`) | 1.03 | 1.07 | 1.02 | 1.43 | 1.12 |
| attention (`kernel_mha` ×12, `qsa_index_q_prep` ×12) | 0.35 | 0.41 | 0.42 | 0.40 | 0.43 |
| **GDN / mamba**: `gdn_decode_bf16*` ×36 | 0.165 | 0.168 | 0.183 | 0.181 | 0.164 |
| **GDN / mamba**: `_causal_conv1d_update` ×36 | 0.106 | 0.112 | 0.157 | 0.168 | **0.261** |
| `_compact_kv` ×12 | 0.059 | 0.079 | 0.111 | 0.186 | 0.159 |
| *verify union total* | 10.56 | 11.80 | 11.80 | 14.72 | 15.10 |

**The MoE grouped GEMM is the thing that scales with speculation width**: 3.76 → 7.90 ms from T=4 to
T=16, i.e. 0.345 ms per extra draft token, and 39 % of the whole W16 step. The dense FP8 gemv is
essentially flat (weight-bandwidth-bound; extra tokens are free), and the whole "GDN state handling
for T=8 vs T=16" question resolves to **+0.10 ms on `_causal_conv1d_update` and +0.05 ms on
`_compact_kv`** — 6 % of the growth, not a lever.

### 5.2 Draft phase, per inner iteration (W16 base32k code-edit; 14 iterations × 338 µs = 4.74 ms)

| kernel | calls/iter | µs/iter |
|---|--:|--:|
| `_w8a16_gemv_kernel` (durations 24, 13, 5, 6, 4, **55** µs) | 6.0 | **119** |
| MoE grouped GEMM ×2 | 2.0 | 49 |
| `_hc_up` / `_hc_down` / `_hc_branch_stats` | 3.0 each | 37 |
| `kernel_mha` | 1.0 | 17 |
| `_qsa_graph_row_metadata_kernel` | 1.0 | 14 |
| `gemvx::kernel` | 1.0 | 12.5 |
| `cutlass_80_wmma_tensorop` gemm | 1.0 | 11 |
| `_mtp_shared_sparse_indices_lookup` / `_compact_kv` / `_router_triton` / rmsnorm / rope-gate | 1.0 each | 24 |
| `memcpy32_post` | 4.1 | 3.8 |
| **~30 further glue kernel names (tail)** | ~19 | **32** |

The 55 µs gemv is the draft lm_head over the reduced vocab: 32768 × 2560 FP8 ≈ 84 MB ≈ 1.5 TB/s —
at roofline. Cost per draft iteration is essentially identical at every S (333 µs at S=3, 338 µs at
S=15), so this is a clean per-step price of **0.32–0.34 ms**.

### 5.3 Non-power-of-two draft-token counts are penalised

T=6 and T=10 hit a fallback inside the verify graph that T=4/8/16 do not:

| | T=4 | T=6 | T=8 | T=10 | T=16 |
|---|--:|--:|--:|--:|--:|
| `at::native::index_elementwise_kernel<128,4,…index_kernel_impl<OpaqueType<8>>>` | 1 call, 2 µs | **109 calls, 227 µs** | 1 call, 2 µs | **109 calls, 258 µs** | 1 call, 4 µs |
| `memcpy32_post` | 122 | **193** | 121 | **193** | 121 |
| `blockExpertPrefixSumKernel<512>` | 138 µs | 148 µs | 136 µs | **336 µs** | 121 µs |

Total penalty vs the linear model: **+0.9 ms at T=6, +2.8 ms at T=10.** This matters for the
adaptive-depth work (`adaptive_3_7.json`, `adaptive_two_state.json`): keep S ∈ {1, 3, 7, 15} so that
T ∈ {2, 4, 8, 16}.

---

## 6. Ranked fixes

Expected saving is per decode step. "W8" figures use the `prof-s7-w4-*` traces (which are the W8
config, §2); "W16" uses `prof-base32k-w16-*`.

| # | Fix | W8 | W16 | Effort | Risk |
|---|---|--:|--:|---|---|
| **F0** | **Fix `trimmed_step.py` to report GPU-busy union instead of name-median-clipped sums** — *done 2026-09-05* | 0 | 0 | 15 min | none |
| **F1** | Cut verify-side MoE grouped-GEMM cost (expert dedup across the T chain tokens / smaller verify top-k / dense fallback when few experts are hit) — 20 % would give: | **−1.0** | **−1.6** | high | accuracy |
| **F2** | Token map 49152 → 32768 (`TOKEN_MAP=…/base32k`) — shrinks the draft lm_head gemv | −0.15 | **−0.26** | config only | acceptance |
| **F3** | Hoist per-draft-iteration index bookkeeping (`_qsa_graph_row_metadata`, `_qsa_graph_layout`, `_mtp_shared_sparse_indices_lookup`, `_compact_kv`, `_expand_qsa_block_indices`) out of the draft graph body | −0.10 | **−0.25** | medium | correctness |
| **F4** | Run the GPU headless / stop the compositor during measurement; re-apply `nvidia-smi -pl 325` after reboot | −0.16 | −0.16 | ops only | none |
| **F5** | Fuse the ~30 tail glue kernels in the draft loop body (32 µs/iter) | −0.10 | −0.35 | high grind | correctness |
| **F6** | Keep S ∈ {1,3,7,15} in any adaptive-depth policy (avoid T=6/10) | 0 | 0 | policy | none |
| **F7** | Make the 2 pageable 4-byte HtoD/step pinned or device-resident | ~0 | ~0 | one line | none |

### Notes on each

**F0 — done (2026-09-05).** `prof/trimmed_step.py` now prints `busy_ms` (per-phase kernel-interval
union) as the primary metric, adds `gpu_busy` to the header and `wall_ms`/`idle_ms` to the `TOTAL`
line, and emits a `(check)` line showing `sum(busy)+idle` against `wall` (residual 0.16–0.30 ms in
all ten traces below). The old column is preserved verbatim as `legacy_trimmed_ms`, computed with
the original cuda_runtime-only attribution so it reproduces historical logs bit-for-bit. Kernel
attribution for the new metric also follows `cuda_driver` launch sites (`cuLaunchKernelEx`, i.e.
Triton), which is why `kernels/step` rises slightly (draft 120 → 125 at W4, 2009 → 2017 total).
Companion `prof/sse_wall.py` derives per-step wall from the SSE chunk timeline of an fnbench run
jsonl, replacing the unsound `accept_len / tps` estimator of §3.

Corrected table (both workloads, all widths; ms):

| trace | S | T | draft | verify | extend | busy | wall | idle | busy+idle | resid | legacy_trimmed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prof-base32k-w4 code-edit | 3 | 4 | 0.58 | 10.56 | 0.45 | 11.59 | 11.95 | 0.15 | 11.74 | 0.21 | 10.89 |
| prof-base32k-w4 prose-en | 3 | 4 | 0.58 | 10.31 | 0.44 | 11.33 | 11.68 | 0.16 | 11.49 | 0.19 | 10.64 |
| prof-s5-w4 code-edit | 5 | 6 | 1.44 | 11.80 | 0.53 | 13.77 | 14.14 | 0.18 | 13.95 | 0.19 | 12.71 |
| prof-s5-w4 prose-en | 5 | 6 | 1.25 | 11.19 | 0.52 | 12.95 | 13.30 | 0.19 | 13.14 | 0.16 | 12.11 |
| prof-s7-w4 code-edit (=W8) | 7 | 8 | 2.10 | 11.80 | 0.54 | 14.43 | 14.90 | 0.17 | 14.60 | 0.30 | 13.55 |
| prof-s7-w4 prose-en (=W8) | 7 | 8 | 2.12 | 11.55 | 0.51 | 14.17 | 14.62 | 0.17 | 14.35 | 0.27 | 13.19 |
| prof-s9-w4 code-edit | 9 | 10 | 3.14 | 14.72 | 0.54 | 18.40 | 18.81 | 0.20 | 18.60 | 0.21 | 15.00 |
| prof-s9-w4 prose-en | 9 | 10 | 2.90 | 13.94 | 0.54 | 17.38 | 17.80 | 0.22 | 17.60 | 0.20 | 14.69 |
| prof-base32k-w16 code-edit | 15 | 16 | 4.44 | 15.10 | 0.55 | 20.09 | 20.47 | 0.21 | 20.29 | 0.18 | 18.62 |
| prof-base32k-w16 prose-en | 15 | 16 | 4.52 | 14.41 | 0.52 | 19.46 | 20.00 | 0.34 | 19.79 | 0.20 | 18.12 |

*Original rationale.* `trimmed_ms` under-reports GPU busy time by 0.84 ms
(W4), 1.04 ms (W8), 1.66 ms (W16) and 3.59 ms (S=9), and the error scales with kernel count, which
is exactly the signature that was read as "growing overhead". Two changes in
`prof/trimmed_step.py`: compute the per-phase kernel-interval **union** (as the script already does
for `busy_ivals` at line 33-44) instead of `sum(min(d, 2*med[kn]+5))` at line 57, and drop the
name-level median (line 55) — one kernel *name* covers many shapes. Safe, zero risk, and it stops
the team chasing a phantom.

**F1 — the only large, genuinely step-count-dependent cost.** Two cutlass `GemmUniversal` kernels ×
48 layers, 3.76 ms at T=4 rising to 7.90 ms at T=16 (0.345 ms per draft token). It is 34 % of the W8
step and 39 % of the W16 step. The linear growth in T means the *number of activated experts* is
growing with the chain, not the weight traffic — the classic MoE-plus-speculation problem. Real work
and probably accuracy-visible; nothing here is a one-liner.

**F2** is already a supported knob (`TOKEN_MAP` in `serve-fast.sh:26`). The traces show the price
directly: W16 draft union 4.44 ms (base32k) vs 4.70 ms (hot2-49k), W4 0.58 vs 0.64. The 2026-09-04
note adopted hot2 at "same step time" (its acceptance was compared with base32k only offline on
private data; no result from that comparison is published) — the traces say it actually costs
0.26 ms/step at W16 and 0.06 ms at W4.

**F3** — `_qsa_graph_row_metadata_kernel` (14 µs), `_mtp_shared_sparse_indices_lookup` (5 µs),
`_compact_kv` (4 µs), `_qsa_graph_layout_kernel` run once per draft iteration inside the graph
(0.93 calls/iteration each). For a bs=1 topk=1 chain the block layout is runtime-invariant apart
from a one-token-per-iteration extension — the same reasoning that already justifies
`_rebuild_topk1_chain_buffers` in `base_spec_worker.py:111-122`. Roughly 320 µs/step at W16 is in
this bucket; recovering most of it is plausible.

**F4** — median GPU idle is 0.15–0.21 ms but the *mean* is 0.33–0.55 ms: 3–7 steps out of 19 take a
0.5–1.5 ms hit, always inside `scheduler.run_batch > draft` or `scheduler.process_batch_result`, with
no host call in the gap. That is external preemption. It also causes the ±1.8 ms in-trace step-wall
spread that makes every A/B in this project noisy — worth fixing for measurement quality alone even
though the mean saving is only ~0.16 ms.

**F5** — the ~30 tail kernel names contribute 32 µs of the 338 µs draft iteration (9.5 %). At W16
that is 0.45 ms/step if all of it vanished; realistically half. Long grind, and the fused-kernel
history in this repo (`SGLANG_HC_MIX2`, `SGLANG_SHARED_GATEUP_FUSED`) shows each one is worth
0.1–0.5 ms and takes a day.

**F7** — cosmetic. The host has 8–19 ms of slack; nothing on the host is on the critical path.

### Explicitly ruled out (measured, not guessed)

| hypothesis | measurement |
|---|---|
| per-draft-step host launches not covered by the CUDA graph | 4 `cudaGraphLaunch`/step at S=3,5,7,9,15; the whole draft loop is one replay (`eagle_worker_v2.py:655-659`); 54 kernels per extra step, all inside it |
| D2H readbacks of accept lengths | 2 D2H/step, 4 B + 16..64 B, 2.3 µs GPU, non-blocking |
| python bookkeeping in `eagle_worker_v2.py` / `eagle_utils.py` | host `cpu_op` union 0.91–1.42 ms per 12–20 ms step; host idle-blocked 6.4–13.7 ms/step |
| sampling / finalize | one 35 µs graph replay + one `cudaMemsetAsync` for `aten::argmax` |
| stream syncs | zero `cudaStreamSynchronize` / `cudaDeviceSynchronize`; one `cudaEventSynchronize` (`batch_result_processor.py:863`), which is pure backpressure |
| GDN state handling T=8 vs T=16 | `gdn_decode_bf16*` flat (164–183 µs); only `_causal_conv1d_update` (+0.10 ms) and `_compact_kv` (+0.05 ms) grow |
| growing host bubble | GPU idle 0.15 / 0.17 / 0.21 ms at S=3 / 7 / 15 |
