# X1 — Exclusive (critical-path) time map of the Qwen3.8-Flash-Next decode step

Date: 2026-09-06. CPU-only analysis. No GPU used, no server started, no fork file changed.
Tool: `prof/exclusive_time.py` (added by this task). Traces:
`prof/traces/final0906b-{w4,w16}-{code-edit,prose-en}` — the current production build
(A0 + A3 + G1 pack-only + G2-1 + G2-2, `hot2_49152` token map, `mtpft3` v3 MTP head,
`SGLANG_TRITON_PDL=1` on the W4 path only).

Everything below is **measured** unless a line says *estimated*.

---

## 0. TL;DR

* **The step is a serial chain.** Exactly one kernel is on the GPU for **80 % (W4) / 86 % (W16)**
  of the step. Only 11–17 % of the step has ≥ 2 kernels running, and true idle is 0.08 ms (W4) /
  0.16 ms (W16). There is no bubble left to fill and almost nothing left to hide behind.
* **The MoE grouped GEMMs are 31 % (W4) / 39 % (W16) of the critical path and are 98 % exclusive.**
  They are at the **expert-weight bandwidth roofline**; the trace alone is consistent with that: GEMM2
  (down, exactly half the weight bytes of GEMM1) costs 0.54–0.61 × GEMM1 at both widths, and the
  implied distinct-expert counts (§5.1) come out at **27 (W4) / 70 (W16)** against the independently
  measured **D = 28.41 / 69.39** (`MOE_SMALLM_SPEC.md` §8.2). The only lever is *fewer expert-weight
  bytes*, not a faster kernel — and expert dedup across the chain is **already free** in this kernel.
* **The second-biggest block is dense FP8 gemv, 24 % (W4) / 20 % (W16)**, of which the two lm_heads
  (403 µs target, 81 µs × 14 draft) run at **1577 / 1555 GB/s = 96–98 % of this card's measured
  1615 GB/s pure-read roof** (`G1_LOG.md` §3): done.
* **H1's finding reproduced and extended.** The shared-expert / router chain is 97–99 % hidden
  (`_router_triton` 0–9 % exclusive, `_shared_expert_gate` 0–7 %, `_w8a16_gemv_silu` 0 %,
  `act_and_mul` 0.3 %, `_mtp_shared_sparse_indices_lookup` 0–3 %). So is the async GDN MTP state
  stream (`gdn_decode_bf16state_mtp`: 564 µs raw, **63 µs exclusive** at W16 = 89 % hidden).
  Together **1.0 ms/step of raw kernel time is already free.** Do not re-optimise any of it.
* **What is left is small.** After the two MoE GEMMs, the two lm_heads, the two GDN projections and
  the HC chain, **181–192 families hold 0.78 ms (W4) / 1.17 ms (W16) between them, none over 0.5 %.**
* **Two existing estimates are corrected by exclusive time.** (i) A8 quoted
  "0.19–0.28 ms/step" for `doActivationKernel` from *raw* time; its **exclusive** time is
  115 µs (W4) / 169 µs (W16) — 26–33 % of it already runs under GEMM2. The NO-GO gets stronger.
  (ii) `DRAFT_GLUE_SPEC.md` DG-3 estimates "−0.3…−0.5 ms/step" for a verify-side single-CTA MoE
  metadata kernel; after A0+A3+G2 the *entire* routing prologue holds only **249 µs (W16) /
  207 µs (W4)** of critical path, so that figure is stale by 2×.
* **G2 verified from the outside.** Folding `expandInputRows` into the routing prologue made the
  prologue itself *slower* (W16 5.34 + 4.26 = 9.60 µs for the old pair → **11.30 µs** fused;
  W4 9.53 → 11.87 µs), but the MoE-glue block's **exclusive** time fell
  **654 → 549 µs/step (W16)** and **446 → 377 µs/step (W4)**. The shipped decision was right, and
  raw per-kernel time would have said the opposite.

---

## 1. Method

`exclusive(k)` = the wall time during which kernel `k` is **the only thing running on the GPU**.
This is the deletion metric: it is exactly what the step would shorten by if `k` disappeared.
Time in which N ≥ 2 kernels overlap is charged to **nobody** (deleting any one of them saves
nothing); it is reported only as the aggregate `shared`. Therefore

```
Σ exclusive(family)  +  shared  +  gap(no kernel)  ==  wall
```

is an identity, not a fit — the tables below are a partition of the step, not an estimate.

* Sweep line over the union of all `kernel` + `gpu_memcpy` + `gpu_memset` intervals, over the
  window [first `draft` annotation, last `draft` annotation], divided by 19 steps.
* **Family = (phase, readable kernel label, grid shape).** Grid is in the key on purpose: one
  kernel *name* covers many shapes (`_w8a16_gemv_kernel` runs at 5 distinct shapes per forward),
  and merging them is what made the old `trimmed_ms` meaningless (`prof/OVERHEAD_REPORT.md` §2).
  The two CUTLASS grouped GEMMs share a grid ([1,188,1]) so they are separated by their position
  in the MoE chain instead: GEMM1 = first after the routing prologue (gate/up), GEMM2 = second
  (down).
* Phase = the innermost `draft` / `step[TARGET_VERIFY bs=1]` / `draft_extend` `user_annotation`
  containing the kernel's **launching** `cuda_runtime`/`cuda_driver` event (the rule
  `trimmed_step.py` already uses; CUDA-graph kernels inherit the phase of their `cudaGraphLaunch`).
  86–94 kernels/step are launched under `scheduler.run_batch` outside those three annotations
  (the sampling graph and the async GDN MTP state stream); they are bucketed as `[run_batch]`
  rather than dropped, so the partition stays complete.
* **Validation.** `exclusive + shared` must equal `trimmed_step.py`'s `busy_ms`, and it does:
  W16 code-edit median step 14818 + 2035 = 16853 µs vs `busy_ms` 16.96 ms; W4 8149 + 1792 =
  9941 µs vs 9.99 ms (≤ 0.7 % apart; the residual is the mean/median difference).
* Numbers are the **mean of the two workloads** (`code-edit`, `prose-en`) unless stated. `%` is
  against the **median** step wall, which is the preemption-robust one.

**What is in these traces.** `serve-fast.sh` defaults as of 2026-09-06: A0 + A3 + G2-1 + G2-2 +
G1 pack-only in the FlashInfer csrc; `SGLANG_HC_MIX2(_FP8)`, `SGLANG_ROUTER_GEMV`,
`SGLANG_QSA_META_PAGE_PARALLEL`, `SGLANG_MTP_EMBED_TABLE`, `SGLANG_GDN_*`, `SGLANG_HC_GATE_EARLY=2`,
`SGLANG_HC_LAYER_APPLY_FUSED`, `SGLANG_NORM_INTO_GEMV` on; `TOKEN_MAP=hot2_49152`;
`TARGET_MODEL=…-NVFP4-mtpft3` (v3 MTP head). **Two flags differ between the two profiles and this
matters for any W4↔W16 comparison:** `SGLANG_TRITON_PDL=1` on W4 only, and
`SGLANG_SHARED_GATEUP_FUSED=1` on W4 only (`W16_GATEUP=0`, because it costs long-chain acceptance).
The second is why the shared-expert chain appears as `_w8a16_gemv_silu` at W4 and as
`act_and_mul` at W16.

**Session caveat.** Absolute µs are session-dependent on this box (Max-Q power cap and throttling;
see `docs/measurement.md`): the same W16 code-edit config measures a
22.48 ms step in `final0905` and **17.32 ms** in `final0906b`, a gap far larger than the patches
landed in between. The `final0906b` session agrees with the `ASTRA_REVIEW_2026-09-05` per-shape
GEMV medians to within 1 % (target lm_head 403.3 vs 400.1–404.2 µs; GDN qkvz 29.3 vs 29.4–29.9 µs),
so it is a representative fast session. **Only compare figures taken from the same trace pair**;
the *partition* within one trace is exact regardless.

Reproduce:

```
python3 prof/exclusive_time.py prof/traces/final0906b-w16-code-edit --top 40
python3 prof/exclusive_time.py prof/traces/final0906b-w16-code-edit --phase draft --partners 12
python3 prof/exclusive_time.py prof/traces/final0906b-w4-code-edit --csv /tmp/w4.csv
```

---

## 2. The partition of the step

| | W4 (S=3, T=4) | W16 (S=15, T=16) |
|---|--:|--:|
| median step wall | **9 966 µs** | **17 421 µs** |
| Σ exclusive | 7 786 (78.1 %) | 14 404 (82.7 %) |
| shared (≥ 2 kernels concurrent, charged to nobody) | 1 792 (18.0 %) | 1 997 (11.5 %) |
| gap — **no kernel at all** (median step) | **77 (0.77 %)** | **160 (0.92 %)** |
| of the exclusive time: **a tiny (< 2 µs) kernel alone** | **416 (4.2 %)** | **747 (4.3 %)** |
| kernels/step | 1 519 | 2 104 |
| of which duration < 2 µs | 376 (25 %) | 637 (30 %) |
| distinct families | 217 | 218 |
| families each < 0.5 % of the step | 192 fams = 784 µs (7.9 %) | 181 fams = 1 166 µs (6.7 %) |

Concurrency histogram (fraction of the step with N kernels resident):

| N | 0 | 1 | 2 | 3 | ≥ 4 |
|---|--:|--:|--:|--:|--:|
| W4 code-edit | 2.1 % | **80.6 %** | 15.9 % | 1.3 % | 0.1 % |
| W16 code-edit | 1.4 % | **86.1 %** | 11.4 % | 1.1 % | 0.0 % |

### 2.1 The "gaps" row, in detail

* **No kernel running:** median step **0.077 ms (W4) / 0.160 ms (W16)** — the expected ~0.1 ms.
  The *mean* over the whole trace window is much larger and unstable (W4 code-edit 0.21 ms,
  W4 prose-en 0.49 ms, W16 code-edit 0.23 ms, W16 prose-en **1.01 ms**) because 3–7 steps out of
  19 take a single 0.5–1.5 ms hit with no host call in the gap. That is external preemption
  (desktop compositor / Max-Q power cap), i.e. `OVERHEAD_REPORT.md` **F4**, and it is a
  measurement-quality problem, not a workload property. All `%` figures in this document use the
  median wall for that reason.
* **A sub-2 µs kernel alone on the GPU:** **0.416 ms/step (W4, 4.2 %) / 0.747 ms/step (W16, 4.3 %)**,
  spread over 376 / 637 kernel instances — i.e. **~1.1 µs of critical path per tiny launch**, which
  is the launch floor, not work. For reference `MEGAKERNEL_SPEC.md` §3.2 measured the per-boundary
  cost of a separate Triton kernel in a graph at 0.823 µs, and 0.668 µs with PDL. **This 0.4–0.75 ms
  is the total prize for all launch-deletion work put together**, and it is only reachable by
  deleting launches, not by making the kernels faster.
* Memcpy + memset + `memcpy32_post` together: **18.6 µs/step (W4) / 47.4 µs/step (W16)** exclusive
  out of 67 / 146 µs raw. Not a target.

### 2.2 By phase

| phase | W4 exclusive | W16 exclusive |
|---|--:|--:|
| `step[TARGET_VERIFY bs=1]` | 7 019 µs (70.4 %) | 10 850 µs (62.3 %) |
| `draft` | 498 µs (5.0 %) | 3 280 µs (18.8 %) |
| `draft_extend` | 195 µs (2.0 %) | 173 µs (1.0 %) |
| `[run_batch]` (sampling graph + async GDN MTP state) | 74 µs (0.7 %) | 100 µs (0.6 %) |

Note `draft_extend`: 0.58 ms of raw kernel time but only 0.17–0.20 ms exclusive — **65–70 % of the
draft-extend phase is already overlapped** with the `[run_batch]` stream. It is not a target.

### 2.3 By functional block (all phases)

| block | W4 excl | % | W16 excl | % | excl/raw (W4→W16) |
|---|--:|--:|--:|--:|:--|
| **MoE grouped GEMM (GEMM1+GEMM2)** | 3 085 | **30.9 %** | 6 858 | **39.4 %** | 98 % → 98 % |
| **dense FP8 gemv (`_w8a16_gemv*`)** | 2 374 | **23.8 %** | 3 547 | **20.4 %** | 65 % → 69 % |
| **HC hyper-connections chain** | 1 141 | **11.5 %** | 1 622 | **9.3 %** | 85 % → 94 % |
| MoE routing prologue + activation + glue | 360 | 3.6 % | 542 | 3.1 % | 39 % → 42 % |
| attention + QSA indexer | 169 | 1.7 % | 698 | 4.0 % | 27 % → 68 % |
| GDN / mamba | 253 | 2.5 % | 308 | 1.8 % | 51 % → 40 % |
| torch `at::*` glue | 183 | 1.8 % | 253 | 1.5 % | 66 % → 63 % |
| **shared expert (gate, gate_up gemv, silu, sigmoid·mul)** | **18** | **0.18 %** | **33** | **0.19 %** | **2.6 % → 12 %** |
| memcpy / memset / `memcpy32_post` | 19 | 0.19 % | 47 | 0.27 % | 28 % → 32 % |
| everything else | 185 | 1.9 % | 495 | 2.8 % | |

The `shared expert` row is the headline of the "already hidden" story: **686 µs (W4) / 275 µs (W16)
of raw kernel time costs 18 / 33 µs of critical path.** It runs entirely underneath the MoE routing
prologue and GEMM1 on another graph branch.

---

## 3. Per-phase, per-family exclusive tables

Columns: calls/step · median duration (µs) · **exclusive µs/step** · % of median step wall ·
raw µs/step · exclusive ÷ raw (= what fraction of this family is on the critical path).

### 3.1 W4 — `step[TARGET_VERIFY bs=1]` (7 019 µs = 70.4 %, 77 families)

| family | grid | n/step | med µs | **EXCL** | % | raw | e/r |
|---|---|--:|--:|--:|--:|--:|--:|
| `cutlass_moe_grouped_gemm1` | `[1,188,1]` | 45.5 | 39.65 | **1859.6** | 18.66% | 1877.6 | 99% |
| `cutlass_moe_grouped_gemm2` | `[1,188,1]` | 45.5 | 24.06 | **1069.1** | 10.73% | 1113.2 | 96% |
| `_w8a16_gemv_kernel` (GDN in_proj) | `[512,1,1]` | 34.1 | 29.21 | **895.4** | 8.98% | 1016.1 | 88% |
| `_w8a16_gemv_kernel` (GDN out_proj) | `[40,4,1]` | 34.1 | 14.11 | **520.3** | 5.22% | 533.3 | 98% |
| `_hc_up_kernel` | `[160,1,1]` | 91.9 | 4.96 | 424.0 | 4.25% | 463.8 | 91% |
| `_hc_down_kernel` | `[20,5,1]` | 91.9 | 4.19 | 379.2 | 3.81% | 423.9 | 89% |
| `_w8a16_gemv_kernel` (**target lm_head**) | `[1940,1,1]` | 0.9 | 397.82 | 376.9 | 3.78% | 377.1 | 100% |
| `trtllm::fusedBuildExpertMapsSortFirstToken…` | `[56,1,1]` | 45.5 | 11.84 | 206.8 | 2.07% | 514.1 | **40%** |
| `_w8a16_gemv_kernel` | `[80,6,1]` | 11.4 | 12.83 | 155.7 | 1.56% | 157.9 | 99% |
| `_hc_branch_stats_kernel` | `[16,1,1]` | 91.9 | 1.98 | 147.3 | 1.48% | 185.1 | 80% |
| `flashinfer gdn_decode_bf16_wy_output_only` | `[1,48,1]` | 34.1 | 4.26 | 139.4 | 1.40% | 146.0 | 96% |
| `trtllm::doActivationKernel` | `[40,1,1]` | 45.5 | 3.65 | 115.2 | 1.16% | 166.1 | 69% |
| `sglang::hc_combine_gate_kernel` | `[4,32,1]` | 90.9 | 1.60 | 108.8 | 1.09% | 147.5 | 74% |
| `_w8a16_gemv_kernel` | `[3,1,1]` | 34.1 | 5.60 | 96.4 | 0.97% | 210.6 | **46%** |
| `_causal_conv1d_update_chain_kernel` | `[1,160,1]` | 34.1 | 2.37 | 91.1 | 0.91% | 91.2 | 100% |
| `cutlass80_wmma_bf16_s161616gemm_16x16` | `[8,80,1]` | 0.9 | 36.98 | 60.2 | 0.60% | 60.2 | 100% |
| *(tail: 61 families)* | | | | 373.5 | 3.75% | | |

### 3.2 W4 — `draft` (498 µs = 5.0 %, 47 families) and `draft_extend` (195 µs = 2.0 %, 58 families)

| family | grid | n/step | med µs | **EXCL** | % | raw | e/r |
|---|---|--:|--:|--:|--:|--:|--:|
| `_w8a16_gemv_kernel` (**draft lm_head**, hot2_49152) | `[1536,1,1]` | 1.9 | 81.12 | **157.8** | 1.58% | 158.0 | 100% |
| `cutlass_moe_grouped_gemm2` | `[1,188,1]` | 1.9 | 17.06 | 41.9 | 0.42% | 44.3 | 95% |
| `cutlass_moe_grouped_gemm1` | `[1,188,1]` | 1.9 | 17.89 | 40.7 | 0.41% | 41.2 | 99% |
| `_w8a16_gemv_kernel` | `[416,1,1]` | 2.0 | 24.35 | 38.0 | 0.38% | 47.4 | 80% |
| `_w8a16_gemv_kernel` | `[160,6,1]` | 2.0 | 12.35 | 32.4 | 0.33% | 32.6 | 99% |
| `_hc_up_kernel` | `[160,1,1]` | 5.9 | 4.90 | 27.7 | 0.28% | 29.7 | 93% |
| `_hc_down_kernel` | `[20,5,1]` | 5.9 | 4.16 | 22.4 | 0.22% | 24.6 | 91% |
| `cutlass80_wmma_s161616gemm_32x32_64x1` | `[8,10,20]` | 2.0 | 11.10 | 21.6 | 0.22% | 21.9 | 99% |
| *(tail: 39 families)* | | | | 115.1 | 1.15% | | |
| **`draft_extend`:** `_w8a16_gemv_kernel` (draft lm_head) | `[1536,1,1]` | 0.9 | 83.84 | 76.2 | 0.77% | 79.6 | 96% |
| `cutlass_moe_grouped_gemm1` / `gemm2` | `[1,188,1]` | 0.9 / 0.9 | 46.2 / 35.7 | 70.2 | 0.70% | 77.6 | 90% |
| *(tail: 55 families)* | | | | 48.5 | 0.49% | | |

### 3.3 W16 — `step[TARGET_VERIFY bs=1]` (10 850 µs = 62.3 %, 80 families)

| family | grid | n/step | med µs | **EXCL** | % | raw | e/r |
|---|---|--:|--:|--:|--:|--:|--:|
| `cutlass_moe_grouped_gemm1` | `[1,188,1]` | 45.5 | 88.00 | **4132.2** | 23.72% | 4155.7 | 99% |
| `cutlass_moe_grouped_gemm2` | `[1,188,1]` | 45.5 | 47.42 | **2178.0** | 12.50% | 2223.2 | 98% |
| `_w8a16_gemv_kernel` (GDN in_proj) | `[256,1,1]` | 34.1 | 29.34 | **871.2** | 5.00% | 1024.1 | 85% |
| `_w8a16_gemv_kernel` (GDN out_proj) | `[40,4,1]` | 34.1 | 13.73 | **494.4** | 2.84% | 494.4 | 100% |
| `_hc_up_kernel` | `[160,1,1]` | 91.9 | 4.99 | 450.9 | 2.59% | 467.4 | 96% |
| `_w8a16_gemv_kernel` (**target lm_head**) | `[1940,1,1]` | 0.9 | 403.25 | 392.6 | 2.25% | 392.6 | 100% |
| `_hc_down_kernel` | `[20,5,1]` | 91.9 | 4.06 | 375.6 | 2.16% | 379.0 | 99% |
| `trtllm::fusedBuildExpertMapsSortFirstToken…` | `[176,1,1]` | 45.5 | 11.26 | 248.9 | 1.43% | 512.6 | **49%** |
| `kernel_mha (trtllm-gen)` | `[5,2,16]` | 11.4 | 16.91 | 189.4 | 1.09% | 191.5 | 99% |
| `_hc_branch_stats_kernel` | `[64,1,1]` | 91.9 | 1.92 | 184.5 | 1.06% | 188.6 | 98% |
| `trtllm::doActivationKernel` | `[160,1,1]` | 45.5 | 3.90 | 169.1 | 0.97% | 228.5 | 74% |
| `flashinfer gdn_decode_bf16_wy_output_only` | `[1,48,1]` | 34.1 | 4.42 | 152.2 | 0.87% | 152.2 | 100% |
| `_w8a16_gemv_kernel` | `[80,6,1]` | 11.4 | 12.77 | 145.6 | 0.84% | 145.6 | 100% |
| `sglang::hc_combine_gate_kernel` | `[16,32,1]` | 90.9 | 1.73 | 141.7 | 0.81% | 158.1 | 90% |
| `kernel_kernel` (QSA) | `[16,1024,1]` | 11.4 | 10.69 | 115.1 | 0.66% | 117.0 | 98% |
| `_causal_conv1d_update_chain_kernel` | `[1,160,4]` | 34.1 | 3.36 | 107.1 | 0.62% | 114.6 | 94% |
| *(tail: 64 families)* | | | | 501.2 | 2.88% | | |

### 3.4 W16 — `draft_extend` (173 µs = 1.0 %, 56 families)

Top three are the draft lm_head `[384,1,1]` 61.0 µs (69 % of raw), `grouped_gemm1` 50.2 µs (75 %),
`grouped_gemm2` 44.8 µs (73 %); the other 53 families total 17 µs. **Nothing here is worth touching.**

---

## 4. The W16 `draft` phase on its own (item 5)

The `draft` annotation holds **14 draft forwards per step** (measured: 14 MoE routing prologues
inside the annotation; `n_inner = speculative_num_steps − 1 = 14`, `eagle_worker_v2.py:643` — the
15th forward is the previous step's `draft_extend`). W4 correspondingly measures 2.

**Total 3 280 µs/step = 234.3 µs per draft forward = 18.8 % of the W16 step.** The MTP head is one
layer + its MoE + the hot-vocab lm_head, and the profile below says so exactly: every non-HC family
is 1.0 calls per forward.

| # | family | grid | n/fw | med µs | EXCL µs/step | **µs/forward** | e/r |
|--:|---|---|--:|--:|--:|--:|--:|
| 1 | `_w8a16_gemv_kernel` — **draft lm_head, hot2_49152** | `[1536,1,1]` | 1.0 | 80.90 | 1067.7 | **76.26** | 99% |
| 2 | `cutlass_moe_grouped_gemm1` (MTP MoE gate/up) | `[1,188,1]` | 1.0 | 17.76 | 240.9 | 17.21 | 97% |
| 3 | `_w8a16_gemv_kernel` (MTP in_proj) | `[416,1,1]` | 1.0 | 22.91 | 236.6 | 16.90 | 77% |
| 4 | `kernel_mha (trtllm-gen)` (MTP attention) | `[9,2,1]` | 1.0 | 16.74 | 217.0 | 15.50 | 97% |
| 5 | `cutlass_moe_grouped_gemm2` (MTP MoE down) | `[1,188,1]` | 1.0 | 16.96 | 208.8 | 14.91 | 92% |
| 6 | `_hc_up_kernel` | `[160,1,1]` | 2.9 | 4.86 | 188.6 | 13.47 | 96% |
| 7 | `_w8a16_gemv_kernel` (MTP out_proj) | `[160,6,1]` | 1.0 | 12.19 | 160.7 | 11.48 | 98% |
| 8 | `_hc_down_kernel` | `[20,5,1]` | 2.9 | 3.97 | 157.0 | 11.21 | 98% |
| 9 | `cutlass80_wmma_s161616gemm_32x32_64x1` (QSA q/k) | `[8,10,20]` | 1.0 | 11.04 | 144.6 | 10.33 | 98% |
| 10 | `trtllm::fusedBuildExpertMapsSortFirstToken…` | `[26,1,1]` | 1.0 | 11.81 | 91.3 | 6.52 | 58% |
| 11 | `_hc_branch_stats_kernel` | `[4,1,1]` | 2.9 | 1.54 | 72.7 | 5.20 | 98% |
| 12 | `_compact_kv` | `[1,2,130]` | 1.0 | 3.36 | 38.6 | 2.76 | 92% |
| 13 | `memcpy32_post` | `[1,1,1]` | 2.9 | 0.99 | 37.2 | 2.66 | 97% |
| 14 | `sglang::hc_combine_gate_kernel` | `[1,32,1]` | 1.9 | 1.47 | 34.9 | 2.49 | 86% |
| 15 | `at::unrolled_elementwise_kernel<direct_copy>` | `[96,1,1]` | 1.0 | 2.66 | 32.9 | 2.35 | 94% |
| 16 | `trtllm::doActivationKernel` | `[10,1,1]` | 1.0 | 3.52 | 32.7 | 2.34 | 69% |
| 17 | `flashinfer RMSNormKernel` | `[1,1,1]` | 1.0 | 2.43 | 29.6 | 2.11 | 91% |
| 18 | `_draft_topk1_finalize_kernel` | `[1,1,1]` | 1.0 | 2.14 | 27.3 | 1.95 | 99% |
| | *(tail: 31 families)* | | | | 261.3 | 18.66 | |

**The W16 draft loop is one kernel: the lm_head.** 76.3 of 234.3 µs per forward — **33 % of the
draft phase and 6.1 % of the whole step** — is the reduced-vocabulary output projection, and it is
at 96 % of this card's measured read roof (§5, item 3). Everything else in a draft forward is under 17 µs.
The 31-family tail is 18.7 µs/forward (8 % of the forward, 1.5 % of the step) spread over ~19
launches — i.e. it *is* the launch-floor tax, not work.

Compared with `OVERHEAD_REPORT.md` §5.2 (338 µs/iteration, base32k map, before A0/A3/G1/G2):
the draft forward is now **234 µs**, a 31 % improvement, and the composition changed — the gemv
bucket went from 119 to ~105 µs/forward while the ~30-kernel glue tail went from 32 to 18.7 µs.

---

## 5. The top-15 exclusive families, annotated

Ranked by W16 exclusive µs/step (mean of both workloads). Assessment codes:
**(a)** already at roofline · **(b)** launch-bound, only deletable · **(c)** overlappable with a
named producer · **(d)** reducible by a known technique not yet applied.

| # | family (phase) | W16 µs | W16 % | W4 µs | W4 % | assessment |
|--:|---|--:|--:|--:|--:|---|
| 1 | `cutlass_moe_grouped_gemm1` — MoE gate/up, 48 layers (verify) | 4132 | 23.7 % | 1860 | 18.7 % | **(a)+(d)** at the *expert-weight* roofline; the only lever is fewer distinct experts. Prize below. |
| 2 | `cutlass_moe_grouped_gemm2` — MoE down, 48 layers (verify) | 2178 | 12.5 % | 1069 | 10.7 % | **(a)+(d)** same; exactly half the weight bytes and measured at 0.54 × GEMM1. |
| 3 | `_w8a16_gemv[1536,1,1]` — draft lm_head, 49152 vocab (draft) | 1068 | 6.1 % | 158 | 1.6 % | **(a)** 125.8 MB FP8 in 80.9 µs = **1555 GB/s = 96 % of the measured 1615 GB/s read roof**. (d) only by shrinking the vocabulary → §6 item 3. |
| 4 | `_w8a16_gemv[256/512,1,1]` — GDN in_proj (qkvz), 36 layers (verify) | 871 | 5.0 % | 895 | 9.0 % | **(a)** 40 MB FP8 in 29.3 µs = **1424 GB/s = 88 % of the read roof** (`MEGAKERNEL_SPEC.md` §5, matching this trace's 29.3 µs). Flat in T — extra tokens are already free. 15 % is already hidden behind `_w8a16_gemv[3,1,1]`. |
| 5 | `_w8a16_gemv[40,4,1]` — GDN out_proj, 36 layers (verify) | 494 | 2.8 % | 520 | 5.2 % | **(d)** 15 MB in 13.1–13.7 µs = **1198 GB/s = 74 % of the read roof** (`MEGAKERNEL_SPEC.md` §5). It only gets there *by* split-K (5–10 splits = 400–800 CTAs); the native form is 657 GB/s and an SM-capped persistent grid is 1.8–2.2× worse. ~3.4 µs/call of headroom remains. **Prize −123 µs W16, −130 µs W4.** |
| 6 | `_hc_up_kernel[160,1,1]` — HC mix, 2/layer (verify) | 451 | 2.6 % | 424 | 4.3 % | **(b)** 5.0 µs × 92, of which `MEGAKERNEL_SPEC.md` §3.1 measures **2.63 µs fixed** and 2.27 µs of traffic. `SGLANG_HC_MIX2_FP8` already took the chain 13.7 → 9.9 µs/call, and §5 explicitly corrects the "HC runs at half roofline" idea (`_hc_down` moves 3.20 MB at **844 GB/s**, *above* the tuned GEMV's 674 GB/s at that size) — **the bandwidth headroom is not real, do not chase it**. Barrier-fusing the island measured **18–26 % slower** (§4.2). Only non-barrier launch deletion is left. |
| 7 | `_w8a16_gemv[1940,1,1]` — **target lm_head**, 248320 vocab (verify) | 393 | 2.3 % | 377 | 3.8 % | **(a)** 635.7 MB FP8 in 403 µs = **1577 GB/s = 98 % of the read roof**. Done. Only a hot-vocab target head could change it, and that changes the output distribution. |
| 8 | `_hc_down_kernel[20,5,1]` (verify) | 376 | 2.2 % | 379 | 3.8 % | **(b)** as #6. Depends on `_hc_up` → a fusion needs a grid barrier → NO-GO per MEGAKERNEL §3.2. |
| 9 | `trtllm::fusedBuildExpertMapsSort…` — MoE routing prologue (verify) | 249 | 1.4 % | 207 | 2.1 % | **(c), no known mechanism.** 11.3 µs/call, **independent of T** (11.8 at T=4). **49 % is already hidden** behind the shared-expert gemv + `act_and_mul` (measured partners: `_w8a16_gemv[80,1,1]` 185 µs, `act_and_mul` 65 µs). But `G2_LOG.md` §2 already established that everything it computes is routing-dependent, that inactive experts early-out, and that **"10.34 µs is the floor for this structure"** — it measures 11.3 µs, so ~1 µs/call is all that is left by that route. Hiding the other 51 % would need *more* independent work to hide it behind, and §5.3 says there isn't any. Ceiling **−249 µs W16 / −207 µs W4**, mechanism unknown. |
| 10 | `cutlass_moe_grouped_gemm1` — MTP layer MoE (draft) | 241 | 1.4 % | 41 | 0.4 % | **(a)** same roofline as #1; 1 layer × 14 forwards. |
| 11 | `_w8a16_gemv[416,1,1]` — MTP in_proj (draft) | 237 | 1.4 % | 38 | 0.4 % | **(c)** already 23 % hidden behind `_mtp_shared_sparse_indices_lookup` (66 µs) and the GDN state stream (11 µs). Roofline-adjacent. |
| 12 | `kernel_mha[9,2,1]` — MTP layer attention (draft) | 217 | 1.2 % | 0 | – | **(d)** 16.7 µs on a **grid of 18 CTAs = 9.6 % of the 188 SMs**, for one layer and one token. The verify instance does 16 tokens in 16.9 µs on 160 CTAs. It cannot be bandwidth-bound at 18 CTAs. *Estimated* prize if it reached even 3 × the parallel efficiency: **−145 µs W16**. |
| 13 | `cutlass_moe_grouped_gemm2` — MTP layer MoE (draft) | 209 | 1.2 % | 42 | 0.4 % | **(a)** as #2. |
| 14 | `kernel_mha[5,2,16]` — full attention, 12 layers (verify) | 189 | 1.1 % | ~40 | 0.4 % | **(a)** 160 CTAs, 16.9 µs for 16 tokens × 1 layer. Well parallelised; the QSA indexer keeps the KV read at `indexer_budget=2048`. |
| 15 | `_hc_up_kernel[160,1,1]` — HC mix (draft) | 189 | 1.1 % | 28 | 0.3 % | **(b)** as #6, 2.9 calls/forward. |
| — | `_hc_branch_stats_kernel` (verify) | 185 | 1.1 % | 147 | 1.5 % | **(b)/(d)** 1.9 µs × 92 — i.e. **~1.9 µs of pure launch each**, 98 % exclusive. Folding it into `_hc_up`'s epilogue is a G2-style *deletion*, not a barrier fusion. Prize **−185 µs W16, −147 µs W4**. |
| — | `trtllm::doActivationKernel` (verify) | 169 | 1.0 % | 115 | 1.2 % | **(b)** the A8 target. **A8 quoted 0.19–0.28 ms/step from raw time; the exclusive figure is 0.115–0.169 ms** because 26–33 % of it already runs under GEMM2 (measured partner: GEMM2, 42–46 µs). A8 = NO-GO (4–8 days of bespoke CUTLASS EVT) and the prize is 30 % smaller than believed. A8-lite (deferring `cudaGridDependencySynchronize`) was measured at **+0.01…+0.43 µs, i.e. nothing** — GEMM1's 188 persistent CTAs hold every SM to the end. |
| — | `flashinfer gdn_decode_bf16_wy_output_only` (verify) | 152 | 0.9 % | 139 | 1.4 % | **(d)** 4.4 µs on a **grid of 48 CTAs (26 % of the SMs)**, 100 % exclusive, 36 calls. *Estimated* prize from a 2 × wider grid: **−76 µs**. |

### 5.1 Why the MoE grouped GEMMs are at the *expert-weight* roofline (measured)

Per expert per layer: gate_up = 2 × 640 × 2560 elements, down = 640 × 2560 — **exactly 2 : 1**.
In NVFP4 with FP8 block scales that is 1.843 MB vs 0.922 MB.

| | GEMM1 med | GEMM2 med | GEMM2/GEMM1 |
|---|--:|--:|--:|
| W4 (T=4) | 39.65 µs | 24.06 µs | **0.607** |
| W16 (T=16) | 88.00 µs | 47.42 µs | **0.539** |

A compute-bound kernel would scale with tokens (4 ×) and not with the 2 : 1 weight ratio; a
launch-bound one would not scale at all. The measured ratio sits just above 0.50 at both widths,
i.e. `t = c + b·(weight bytes)` with a **fixed part c ≈ 8.6 µs (W4) / 6.8 µs (W16) per call**.

Two independent checks confirm this is the right model and that the kernel is done:

1. **The fixed part matches G1's fit.** `G1_LOG.md` §3 fitted an 11-point sweep to
   `GEMM1 = 9.4 + 1.036·D µs` (D = distinct experts per layer), straight to ~1 % with **no wave
   staircase**. The trace-only fit here recovers the same intercept from a completely different
   direction (the GEMM2/GEMM1 weight ratio).
2. **The implied expert counts match the measured ones.** Solving the fit at an achieved
   1.6 TB/s gives **≈ 27 distinct experts per layer at T=4 and ≈ 70 at T=16**. The in-server
   census (`MOE_SMALLM_SPEC.md` §8.2, 197 470 + 135 534 calls) measured **D = 28.41 (W4) and
   69.39 (W16)**. So the T=4 → T=16 growth is *exactly* the growth in distinct activated experts
   (5.5 % → 13.6 % of the 512-expert bank) — `OVERHEAD_REPORT.md` **F1**'s hypothesis, now closed.

G1 further measured the mainloop streaming at **1779 GB/s effective / ~1615 GB/s DRAM after ~13 %
L2 hits**, i.e. **~100 % of what this card can read** (the correct roof is the measured pure-read
1612–1621 GB/s, *not* the 1792 GB/s paper number and *not* the 1456 GB/s copy roof).

**Consequences — the kernel is finished, and most of the obvious levers are already spent:**

* **Expert dedup across the chain tokens is not a lever: it is already free.** The grouped GEMM
  reads each distinct expert exactly once no matter how many of the T·10 rows land on it
  (`MOE_SMALLM_SPEC.md` TL;DR #4). D = 69.39 at W16 against 138.57 if the tokens routed
  independently — the dedup is already 50 %.
* Split-K as extra groups: **wash at W4, worse at W16** and not bit-identical (G1 §6). Off.
* Problem-list packing: shipped, −0.125 (W4) / −0.083 (W16) ms/step (G1 §7, `serve-fast.sh:46`).
* Tactic/tile sweep: the autotuner already sits in the only usable corner (narrow-N + `swap_ab`);
  expected 0…−0.2 ms and worth doing for measurement stability, not throughput
  (`MOE_SMALLM_SPEC.md` §8.1).
* A custom expert-parallel NVFP4 kernel has a **−0.51/−0.62 ms ceiling for 2–3 weeks** and a real
  chance of landing slower. NO-GO (`MOE_SMALLM_SPEC.md` §8.4, §8.6).
* **Still untried, and the reason it heads §6: A6 — MXFP4 weights** (4.25 vs 4.5 bits per weight,
  a flat **5.6 %** of grouped-GEMM time because the block is pure weight traffic). Against the
  exclusive figures measured here that is **−372 µs/step at W16 (−2.1 %)** and **−176 µs at W4
  (−1.8 %)**. It needs the checkpoint requantised, which is why nobody has done it.
* **Reducing D itself (A5)** is worth **−0.68 ms/step per 10 % of D at W16** (−0.33 at W4) and is
  the only >2 % item in the whole profile, but `MOE_SMALLM_SPEC.md` §8.6 says **do not pursue**:
  D is already 50 % (W16) / 73 % (W4) of the independent count, so any further reduction is a
  routing change, i.e. an accuracy change.
* The fixed 6.8–8.6 µs × 91 calls = **0.62–0.78 ms/step** of per-call ramp is the largest
  "not real work" item in the step after the tiny-kernel tax. G1 §3 attributes it to a flat
  per-call ramp, not wave quantisation, and no cheap way to remove it is known.

### 5.1a Cross-check: G2 was the right call, and raw time would have said otherwise

Folding `expandInputRows` into the routing prologue (G2-2) made the surviving kernel **slower**:

| | old prologue | + `expandInputRows` | pair | **fused (now)** | Δ raw |
|---|--:|--:|--:|--:|--:|
| W16 code-edit | 5.34 µs `[16,1,1]` | 4.26 µs | 9.60 µs, 2 launches | **11.30 µs `[176,1,1]`, 1 launch** | **+1.70 µs** |
| W4 code-edit | 5.50 µs | 4.03 µs | 9.53 µs, 2 launches | **11.87 µs `[56,1,1]`, 1 launch** | **+2.34 µs** |

(`final0905-*` vs `final0906b-*`; the grid growth `16 → 16 + num_tokens·k` is exactly what
`G2_LOG.md` §8 caveat 3 predicts, and the +1.2–2.3 µs is the extra dependent round trip it warns
about.) Yet the MoE-glue block's **exclusive** time fell:

| MoE glue block (prologue + expand + doActivation + memset + router + `memcpy32_post`) | raw µs/step | **exclusive µs/step** |
|---|--:|--:|
| W16 `final0905` | 1741 | **654** |
| W16 `final0906b` | 1408 | **549** |
| W4 `final0905` | 1154 | **446** |
| W4 `final0906b` | 996 | **377** |

−105 µs (W16) / −70 µs (W4) of critical path, and −68 (W16) / −68 (W4) `memcpy32_post` launches.
**A per-kernel raw-time review would have rejected this change.** That is the case for using
exclusive time as the review metric for any launch-deletion work.

### 5.2 What is already hidden — do not work on it

| family | W16 raw µs/step | W16 **exclusive** | hidden |
|---|--:|--:|--:|
| `_router_triton_kernel` | 340.0 | **0.1** | **100 %** |
| `sglang::act_and_mul_kernel` | 136.8 | **0.4** | **99.7 %** |
| `_mtp_shared_sparse_indices_lookup` | 69.2 | 2.0 | 97 % |
| `_shared_expert_gate_kernel` | 121.8 | 8.7 | 93 % |
| `flashinfer gdn_decode_bf16state_mtp` (async state stream) | 564.3 | 62.7 | 89 % |
| `_w8a16_gemv_silu_kernel` (W4) | 303.5 | 0.1 | **100 %** |
| `trtllm::fusedBuildExpertMapsSort…` | 512.6 | 248.9 | 51 % |
| `trtllm::doActivationKernel` | 252.4 | 169.1 | 33 % |

This confirms and extends H1's result. The shared-expert/router chain and the GDN MTP state stream
together are **1.2 ms/step of raw kernel time for 0.07 ms of critical path.**

### 5.3 How much room is there to hide *more*?

Only 11.4 % (W16) / 15.9 % (W4) of the step has two kernels resident, and only 1.1 % has three.
The total shared time is 2.0 ms (W16) / 1.8 ms (W4). Every large family is 96–100 % exclusive
because it is the *only* thing the graph has to run at that moment — the dependency chain, not the
scheduler, is the constraint.

The fork already runs three deliberate dual-stream sites (QSA indexer ∥ QKV projection; GDN
`in_proj_qkvz` ∥ `in_proj_ba`; shared expert ∥ router + routed experts) and they are exactly the
families that show up as ~100 % hidden in §5.2. **Adding a fourth stream buys nothing unless there
is independent work to put on it, and the profile says there isn't any left.** Verify step *k* ∥
draft step *k+1* is separately ruled out on dependency grounds (`ASTRA_REVIEW_2026-09-05.md`).

**One anomaly worth a cheap experiment.** These W4 traces run with `SGLANG_TRITON_PDL=1` and the
W16 traces do not (`serve-fast.sh:47-48,73`). W4 spends **18.0 %** of the step with ≥ 2 kernels
resident; W16 only **11.5 %**. Inside the verify phase alone, `raw ÷ busy` — how much the graph
overlaps — is **1.176 (W4)** vs **1.119 (W16)**. That is the signature PDL would leave
(`gdc_launch_dependents` lets a successor's prologue start before its producer drains), and it also
explains why the microbench model under-predicts PDL: 1047 W4 glue launches × the measured
0.155 µs/launch saving = 162 µs, against a **measured in-server −0.63/−0.56 ms/step at W4, "all in
verify"** (`MEGAKERNEL_SPEC.md` §6). PDL is evidently buying overlap, not just launch latency.
If W16 verify reached W4's overlap ratio it would save **≈ −0.61 ms/step (−3.5 %, *estimated*)**.
This is confounded — W16's verify kernels are bigger, so a lower overlap *fraction* is partly
structural — and "W16 neutral" was measured once. But the measurement was made against a ±1.8 ms
in-trace step-wall spread, so a headless re-run (`OVERHEAD_REPORT.md` F4) is the cheapest
high-value experiment on the list.

---

## 6. Ranked next 5 work items

Expected saving is **exclusive µs per decode step**, i.e. real wall time. `%` is against the
median step wall (9 966 µs W4 / 17 421 µs W16). "measured" means the number comes from a trace or
an in-server A/B; "estimated" means it is derived from a model stated in the cited section.

| # | item | W16 | W4 | effort | risk | confidence |
|--:|---|--:|--:|---|---|---|
| **1** | **A6 — requantise the expert weights to MXFP4.** 4.25 vs 4.5 bits/weight; the grouped GEMMs are pure weight traffic (§5.1) so the saving is a flat 5.6 % of their exclusive time. | **−372 µs (−2.1 %)** | **−176 µs (−1.8 %)** | high — requantise the checkpoint, verify the sm120 kernel path | accuracy (coarser scale blocks) | *estimated* from `MOE_SMALLM_SPEC.md` §8.5 × exclusive measured here |
| **2** | **Re-measure PDL at W16, headless.** W4 (PDL on) overlaps 18.0 % of the step vs W16's 11.5 %, and PDL's measured W4 gain (−0.6 ms) is 4× what the launch-latency model predicts — so it is buying overlap. Zero code: flip `SGLANG_TRITON_PDL=1` on the w16 profile and A/B with the compositor stopped. | **−610 µs (−3.5 %)** if the W4 overlap ratio is reachable; **0** if the earlier "neutral" holds | n/a (already on) | **ops only, ~1 h** | none — it is a flag | *estimated*, low confidence; §5.3 |
| **3** | **`TOKEN_MAP` 49152 → 32768** (`hot_32768` / `base32k`). The draft lm_head is bandwidth-bound at 96 % of roof, so its time scales with vocabulary: 80.9 → 53.9 µs × 15.1 calls/step. | **−384 µs (−2.2 %)** | −78 µs (−0.8 %) | **config only** (`serve-fast.sh:53`) | acceptance effect not measured on public data: production has served 49152 since 2026-09-04, and the only 32768-vs-49152 comparison was an offline evaluation on private data with no published result → **must be A/B'd end-to-end** | *measured* per-call, *estimated* scaling |
| **4** | **Fold `_hc_branch_stats_kernel` into `_hc_up`'s epilogue.** 1.54–1.92 µs × 92 calls, of which `MEGAKERNEL_SPEC.md` §3.1 measures **1.41 µs as fixed cost**; 98 % exclusive. It is an *independent* stage, so this is a G2-style launch deletion, not a barrier fusion — the MEGAKERNEL NO-GO does not apply. `SGLANG_HC_LAYER_APPLY_FUSED` is the precedent (46/48 boundaries, −42/−47 µs, `torch.equal`). | **−185 µs (−1.1 %)** | **−147 µs (−1.5 %)** | medium, 1–2 days | correctness; bit-exactness is achievable | *measured* exclusive; check `MEGAKERNEL_SPEC.md` §8.2's list of already-written unmeasured fusions first |
| **5** | **Re-tune the GDN `out_proj` GEMV** `[40,4,1]`: 1198 GB/s vs the 1615 GB/s roof and the 1577 GB/s the same kernel reaches on the lm_head. It is a 40-tile shape that only reaches 1198 by split-K 5–10; the remaining gap is a split/tiling choice. | −123 µs (−0.7 %) | **−130 µs (−1.3 %)** | medium — a sweep plus a table entry | low; watch `SGLANG_NORM_INTO_GEMV` (H1-B folded the gated RMSNorm into this GEMV's A-load) | *measured* bandwidth (`MEGAKERNEL_SPEC.md` §5), *estimated* prize |

**Runners-up, and why they are not in the list:**

* **A5 — reduce D (distinct experts per layer).** −0.68 ms/step per 10 % of D at W16, −0.33 at W4.
  **This is the only item in the entire profile worth more than 2.5 %**, and `MOE_SMALLM_SPEC.md`
  §8.6 declines it: D is already 50 % (W16) / 73 % (W4) of the independent count, so cutting it
  further means changing routing, i.e. changing the model's answers. Listed here so nobody
  rediscovers it as free.
* **The routing prologue's remaining 249 µs (W16) / 207 µs (W4)** — real, but `G2_LOG.md` §2 puts
  the structural floor at 10.34 µs against a measured 11.3, and the chain is strictly serial with
  ≤ 0.2 µs gaps. No mechanism.
* **A8 — `doActivationKernel` deletion.** −169 (W16) / −115 (W4), 4–8 days, NO-GO. A8-lite measured
  as nothing.
* **DG-3 verify-side single-CTA MoE metadata**: the spec's "−0.3…−0.5 ms/step" predates A0/A3/G2;
  the whole prologue is now worth at most −0.25 ms.
* **Draft `kernel_mha`'s 18-CTA grid** (*estimated* −145 µs W16, 0 at W4) and
  **`gdn_decode_bf16_wy_output_only`'s 48-CTA grid** (−76 µs). Both are 100 %-exclusive kernels
  running on under a quarter of the SMs; neither has ever been bandwidth-characterised
  (`ASTRA_REVIEW` flags the QSA and GDN kernel bodies as un-audited). Cheapest genuinely *new*
  measurement available.
* **Run the GPU headless** — 0.5–1.5 ms preemption spikes on 3–7 steps of 19
  (`OVERHEAD_REPORT.md` F4). Worth more for measurement quality than throughput, and it is a
  prerequisite for trusting items 2 and 3.

### 6.1 Honest assessment

**Nothing on the actionable list is worth more than 2.2 %, and the two items that are worth more
than 2 % are a checkpoint requantisation and a flag re-measurement.** The one lever that would move
the needle — fewer distinct experts per layer — has already been declined on accuracy grounds.

Below the top five the picture is flat by construction: **181 (W16) / 192 (W4) families hold
1.17 / 0.78 ms between them and not one is over 0.5 % of the step.** That whole tail is the
launch-floor tax — 637 (W16) / 376 (W4) sub-2 µs kernels contributing ~1.1 µs of critical path
each. Harvesting it means deleting ~160 launches to gain 1 % of the step, and both mechanisms are
already measured: barrier-based megakernels are a **regression** (`MEGAKERNEL_SPEC.md` §3.2:
+0.45 ms W4 / +0.62 ms W16 per step), and one-at-a-time non-barrier fusions are worth 0.1–0.5 ms
for about a day each — the `SGLANG_HC_MIX2` / `SGLANG_HC_MIX2_FP8` / `SGLANG_HC_LAYER_APPLY_FUSED`
/ A0 / A3 / G1 / G2 history. There are roughly a dozen such fusions left and they get smaller.

**Recommendation.** Spend an hour on item 2 (it is a flag and an A/B), then decide between item 1
(MXFP4, the only remaining multi-percent kernel-side win) and stopping. The step is 78–83 % a serial
chain of kernels sitting at 88–98 % of this card's measured read roof, with 0.16 ms of idle and
1.2 ms/step of work the graph already hides for free. **There is no structural overhead left to
find** — `OVERHEAD_REPORT.md` said that about the host in September; this says it about the GPU.

## 7. The tool

`prof/exclusive_time.py`. Reuses `trimmed_step.py`'s trace loader and phase-attribution rule.

```
exclusive_time.py <trace.json.gz | trace-dir> [...] [--top N] [--phase draft|verify|draft_extend]
                  [--partners N] [--draft-forwards] [--csv OUT]
```

* `--partners N` — for the top-N exclusive families, which families they overlap with and for how
  long. This is what identifies (c) items: a family with low `excl/raw` is already hidden, and this
  says *behind what*.
* `--draft-forwards` — median number of draft forwards per step, counted by MoE routing prologues
  inside the `draft` annotation.
* `--csv` — one row per (trace, phase, family, grid) with count/step, median duration, exclusive
  and raw µs/step.

The `(check)` invariant to watch: the printed `exclusive + shared + gap` must equal `wall`, and
`exclusive + shared` must equal `trimmed_step.py`'s `busy_ms`.
