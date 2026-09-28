# Roofline and where the step time goes

This page asks two questions: how far the decode step is from the memory
bandwidth limit of one RTX PRO 6000 Blackwell Max-Q, and what the remaining time
consists of. It also explains why the original "1000 t/s" target is out of
reach on a single card.

## 1. The card

| Quantity | Value | Source |
|---|---|---|
| Memory | GDDR7, 512-bit, 28 Gbps (14,001 MHz) | `lab-notes/MOE_SMALLM_SPEC.md` |
| Paper bandwidth | **1792 GB/s** (not an HBM part; 6 TB/s figures do not apply) | same |
| Measured pure-read bandwidth (`sum`/`max`/`norm`) | 1612-1621 GB/s ≈ **1615 GB/s = 90 % of paper** | `lab-notes/G1_LOG.md` |
| Measured copy bandwidth | ~1456-1461 GB/s (the wrong ceiling for weight streaming) | same |
| L2 | 128 MiB; persisting carve-out at most 80 MiB | `lab-notes/L2_LOG.md` |
| SMs | 188 (sm_120) | |
| Power cap | 300-325 W; no cap throttling in 39 decode samples at 100 ms intervals (per-arm median SM 2242-2287 MHz); shorter throttling not ruled out | `lab-notes/X3_ADAPTIVE_COST.md` |

In what follows, "the roof" means the measured read ceiling of about 1.6 TB/s.

## 2. Bytes that must stream every decode step

The model is Qwen3.8-Flash-Next (NVFP4 experts): 48 layers, 512 routed experts
with top-10 routing plus a shared expert, hybrid GDN linear attention and QSA
sparse attention, hyper-connections (HC), and one native MTP layer used as the
draft. At bs = 1 every weight read in a step is paid once per step, whatever
the number of tokens verified.

### 2.1 Routed experts (NVFP4)

* One expert = **2,764,800 B** (gate_up 1,843,200 + down 921,600; 4.5
  bits/weight at block size 16). One layer's 512 experts = 1.416 GB.
* A verify of T tokens reads **D** distinct experts per layer. Independent
  routing would give E[D] = 512·(1 − (1 − 10/512)^T), i.e. 38.8 at T = 4 and
  138.6 at T = 16.
* Measured D (330k logged calls, 2026-09-05) is much lower, because tokens
  along a speculative chain route in a correlated way: **28.4 at W4 (73 % of
  independent), 69.4 at W16 (50 %)**. The correlation is diffuse (only 0.12
  experts shared by all 16 tokens; 35 of 69 reached by a single token).
* After singleton pruning (τ = 0.08, see `optimizations.md` §E), the implied D
  is **18.8 (W4)** and **52.3 (W16)** (`lab-notes/EXCLUSIVE_TIME_MAP_0907.md`).

Expert bytes per verify, which is D × 2.7648 MB × 48 layers:

| | D | GB/step | ms at 1.615 TB/s |
|---|--:|--:|--:|
| W4, no pruning | 28.4 | 3.77 | 2.33 |
| W4, τ = 0.08 | 18.8 | 2.50 | 1.55 |
| W16, no pruning | 69.4 | 9.21 | 5.70 |
| W16, τ = 0.08 | 52.3 | 6.94 | 4.30 |

### 2.2 Dense weights (FP8 after this work)

In the checkpoint the always-on dense modules are BF16: the shared expert,
attention, linear attention, HC mixers, gates and lm_head. That is about 7.2 GB
of the ~9.8 GB read per decoded token by the stock path. After this work:

* Dense projections plus the target lm_head as FP8 with per-row scales: 198
  tensors, **3.73 GB** in total, of which the target lm_head (248,320 × 2,560)
  is **0.636 GB** (`lab-notes/FP8_CODING_SURVEY.md`).
* HC low-rank mix weights: two 6.5 MB BF16 matrices per mix, about 97 mixes per
  target forward. That is about 1.26 GB in BF16 and about **0.64 GB** in FP8
  (estimate: 97 × 6.55 MB).
* The router gate (512 × 2560) stays BF16. It is small and runs on a side
  stream.

### 2.3 Draft (MTP) forwards

* About **247 MB** per draft forward, of which the hot-vocabulary FP8 head
  (49,152 rows) is 126 MB (`lab-notes/DRAFT_V2_SPEC.md`, estimate).
* W4 runs 2 recursive draft forwards per step plus a draft_extend. W16 runs 14
  plus the draft_extend. (The draft loop executes S − 1 forwards; see the
  09-05 design review.)

### 2.4 Totals (estimates)

| | experts | dense + lm_head | HC mix | draft | **total** | floor at 1.615 TB/s | floor at 1.792 TB/s | measured step |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| W4 (τ = 0.08) | 2.50 | 3.7 | 0.64 | ~0.7 | **~7.5 GB** | **~4.7 ms** | 4.2 ms | **8.92 ms** |
| W16 (τ = 0.08) | 6.94 | 3.7 | 0.64 | ~3.7 | **~15.0 GB** | **~9.3 ms** | 8.4 ms | **16.08 ms** |

Measured steps are medians from `lab-notes/EXCLUSIVE_TIME_MAP_0907.md`
(2026-09-07, monitor off). These are rough byte budgets, not a performance
model. Some small weights stay resident in L2 between draft steps. For
example, the MTP `fc` matrices, 13 MB × 2, are reused across the draft loop.

Historical versions of the same estimate:

* **2026-09-02, stock W4 (BF16 dense)**: a bandwidth floor of about 5.6
  ms/step for the dense weights alone, against about 15.6 ms of GPU work, i.e.
  about 36 % efficiency. The rest was hundreds of 1-4 µs kernels.
* **2026-09-02, W16 after the first round**: a theoretical ~**13.5 ms/step**
  (expert reads estimated at ~14 GB, plus 3.5 GB dense, plus heads) against
  **25-26 ms measured**. The difference was attributed to about 2,500 verify
  plus about 1,000 draft kernels of glue. The ~14 GB expert figure predates the
  D measurement and is an over-estimate.

So after the work described here the step runs at roughly **50-60 % of the
pure weight-streaming floor**. The individual heavy kernels already run at 88-98 % of the
roof (§4). The remaining gap is the serial chain of about 1,500-2,100 kernels
per step, each with its own ramp.

## 3. Step-time model

On 2026-09-05, fitting the power-of-two widths (S = 3, 7, 15; T = S + 1)
gave:

```
step(S) ≈ 9.74 + 0.700·S  ms          (2026-09-05, v3 head, before P1/P2)
        = draft 0.322 ms/step + verify 0.378 ms/extra token
```

It predicted S = 3/7/15 as 11.84/14.64/20.24 ms, against 11.95/14.90/20.47 ms
measured. The marginal verify cost is almost entirely the MoE grouped GEMM,
because D grows with T. From T = 4 to T = 16 the GEMM went 3.76 -> 7.90
ms/step, which is 0.345 ms per extra token and 39 % of the W16 step. The dense
GEMV is nearly flat in T.

On 2026-09-07 the model was refitted on the width sweep, on the full production
stack with pruning and PDL, from 10 fixed-width trimmed points:

```
step(S) ≈ 7.943 + 0.5554·S  ms        (RMSE 0.33 ms; W4 9.61, W8 11.83, W16 16.27)
```

The width-sweep wall times were W4 10.36-10.51, W6 12.70, W8 12.82, W12 16.18
and W16 17.43 ms (code-edit, monitor on). W6 and W8 cost the same.

**Non-power-of-two widths are penalised.** At T = 6 and T = 10 the verify
graph hits an indexing fallback: 109 extra `index_elementwise` calls, more
memcpys, and a slower prefix sum. That costs **+0.9 ms at T = 6 and +2.8 ms at
T = 10** over the linear model. All width policies therefore use S ∈ {3, 7, 15}.

The host is not the bottleneck at any width. The whole draft loop is one CUDA
graph replay, and each step has 4 graph launches. GPU idle inside a step is
0.15-0.21 ms at every width. The host issues a step in about 4.5 ms and then
blocks on one event.

## 4. How the heavy kernels compare with the roof

| Kernel (per call) | Bytes | Time | Achieved | % of 1.615 TB/s |
|---|--:|--:|--:|--:|
| Target lm_head FP8 GEMV (M = 4-16) | 635.7 MB | 398-404 µs | ~1577 GB/s | 98 % |
| Draft head FP8 GEMV, 49,152 rows (M = 1) | 125.8 MB | 80.9 µs | 1555 GB/s | 96 % |
| GDN in_proj qkvz FP8 GEMV | 40 MB | 29.3 µs | 1424 GB/s | 88 % |
| GDN out_proj FP8 GEMV | 15 MB | 13.1-13.7 µs | 1198 GB/s | 74 % |
| MoE GEMM1 main loop (W4, D = 28) | 52.3 MB | 42.4 µs per call, of which ~32.4 µs is streaming | ~100 % in the main loop | ramp ≈ 9.4 µs per call |
| HC `_hc_down` | 3.2 MB | ~4.1 µs | 844 GB/s | (small, latency-bound) |

The MoE grouped GEMM fits `GEMM1 ≈ 9.4 + 1.036·D µs` (tactic pinned, warm
clocks). The earlier claim of a "wave-quantisation staircase" came from a
cold-clock data point and was withdrawn (`lab-notes/G1_LOG.md`). The whole
recoverable loss in the expert GEMM is the per-call ramp of about 9-10 µs, and
most of that cannot be removed (§5).

## 5. Exclusive-time map (critical path)

"Exclusive" time is the wall time during which a kernel family is the only
thing running on the GPU. It is computed with a sweep line over all kernel,
memcpy and memset intervals. See `measurement.md` for the caveats.

| Block | 09-06 W4 | 09-06 W16 | **09-07 W4** | **09-07 W16** |
|---|--:|--:|--:|--:|
| Median step wall | 9.97 ms | 17.42 ms | **8.92 ms** | **16.08 ms** |
| MoE grouped GEMM | 3085 µs (30.9 %) | 6858 (39.4 %) | 2093 (23.5 %) | 4827 (30.0 %) |
| Dense FP8 GEMV (projections + lm_heads) | 2374 (23.8 %) | 3547 (20.4 %) | 2385 (26.7 %) | 3656 (22.7 %) |
| HC chain (stats / down / up / combine) | 1141 (11.5 %) | 1622 (9.3 %) | 1082 (12.1 %) | 1649 (10.3 %) |
| MoE glue (routing prologue, activation) | 360 (3.6 %) | 542 (3.1 %) | 314 | 413 |
| Attention + QSA indexer | 169 (1.7 %) | 698 (4.0 %) | 213 | 470 |
| GDN recurrent / conv | 253 | 308 | 237 | 305 |
| Shared expert (hidden on the alt stream) | 18 | 33 | | |
| Time with ≥ 2 kernels resident ("shared") | 18.0 % | 11.5 % | 19.9 % | 16.4 % |
| Gap with no kernel | 0.8 % | 0.9 % | 0.8 % | 0.6 % |
| Kernels per step | 1519 | 2104 | 1516-1525 | 2104-2109 |

By phase, W16 on 09-07: verify 9.26 ms, draft 3.13 ms (14 forwards × about
220-228 µs; the draft head is 81 µs of each), draft_extend 0.12 ms.

What changed between 09-06 and 09-07:

* P2 pruning cut the verify GEMMs by 836 µs (W4) and 1484 µs (W16).
* PDL at W16 hid the attention kernel almost entirely (exclusive 189 + 217 µs
  -> 16 + 10 µs).

Structure of the step:

* **The step is a serial dependency chain.** For 78-86 % of the step exactly
  one kernel is resident. Only three deliberate dual-stream sites exist:
  * QSA indexer ∥ QKV;
  * GDN qkvz ∥ ba;
  * shared expert ∥ router + routed experts.
  
  They hide about 1.2 ms of raw kernel time per step behind 0.07 ms of
  critical path.
* **Tiny kernels.** Kernels under 2 µs are 25-30 % of all launches, and alone
  on the GPU they account for about 0.4 ms (W4) and 0.6-0.75 ms (W16). Deleting
  a kernel saves at most about 1.1 µs, and only if its work does not have to be
  redone elsewhere. J1 is the counterexample.
* **Barriers are dearer than launches.** A dependent stage costs 0.82 µs as a
  separate kernel, 0.67 µs with PDL, and 1.26 µs as a grid-wide barrier inside a
  persistent kernel (`lab-notes/MEGAKERNEL_SPEC.md`). That is why megakernel
  fusion was rejected.

## 6. Kernel count over time

| Date | Event | W4 kernels/step | W16 kernels/step |
|---|---|--:|--:|
| 09-02 | stock fork, steps 3 | ~2500 | — |
| 09-03 am | trimmed-metric baseline | 2299 (prose) | 3175 (code) |
| 09-03 | seq_lens table, direct top-k chain, fused index lookup | | 3175 -> 2937 |
| 09-03 | + A_log / zero-bias / KV store / MTP dense FP8 | 2113 | 2713 |
| 09-04 | final4 (GEMV v3, glue-E, HC FP8) | 2219 -> 2009 | 2855 -> 2633 |
| 09-05 | FlashInfer A0 + A3 | 2017 -> 1864 | 2728 -> 2539 |
| 09-05 | R3 / R4 / R2 + R5 | 1857 | 2508 |
| 09-05 | R1 / R6 / R7 HC boundary | 1806 | 2445 |
| 09-06 | G2-1 / G2-2 (MoE glue) | 1698 -> 1596 | 2337 -> 2211 |
| 09-06 | H2 + H1-B (final0906b) | 1514 | 2129 |
| 09-07 | X2 map (P2 pruning, PDL everywhere) | 1516-1525 | 2104-2109 |

Rows from different days are not strictly comparable, for two reasons:

* The counter changed on 09-05. From then on, kernels launched through
  `cuLaunchKernelEx` (Triton) are attributed to phases, which added about 8 per
  step.
* Each row is a separate server session, with different warmup, tactics and
  kernels present.

The overall trend is about **−40 % kernels per step at W4 and −34 % at W16**.

## 7. Why 1000 t/s is not reachable on one GPU

t/s = (tokens accepted per verify) / (step time). Both factors are bounded.

1. **Byte floor.** From §2.4, a W4 step cannot fall below about 4.2 ms even at
   the paper bandwidth with zero overhead. W16 cannot fall below about 8.4 ms.
2. **Acceptance ceiling per width.** With the best head used here (v5):
   * prose-en saturates near 3.0-3.2 tokens per verify and prose-ja near 2.5-2.6,
     already by W6-W8;
   * agent-loop reaches about 4.8-5.3;
   * code-edit reaches 11.7 at W16.
3. Combining the two gives an idealised ceiling (zero overhead, 1792 GB/s):
   * prose-en at W4 ≈ 2.6 / 4.2 ms ≈ **620 t/s**;
   * prose-ja ≈ **560 t/s**;
   * code-edit at W16 ≈ 11.7 / 8.4 ms ≈ **1,390 t/s**.
   
   These are upper bounds. Real steps are about twice the byte floor, because
   of the per-kernel ramps in a serial chain of 1,500-2,100 kernels.
4. **Wider verify does not scale for free.** Every extra verify token pulls in
   more distinct experts. At W16, D = 52-69 per layer against 19-28 at W4. So
   step time grows by about 0.56-0.70 ms per extra draft step, while acceptance
   on prose stops growing.
5. **Tensor parallelism is not available.** It would split the expert bytes
   over two cards. The early estimate for two cards was about 650 t/s on mixed
   workloads, but only one card was available.

Final production numbers (2026-09-08, quiet night, adaptive width): code-edit
659-669, prose-en 259-287 and agent-loop 362-430 t/s. The byte floor still
allows at most about 2× on prose and agent work, and only if all fixed
per-kernel costs vanished. 1000 t/s across workloads would need either a
different memory system or a draft that accepts far more tokens per verify on
prose.

## 8. What dominates now

At the 2026-09-07 state, per the exclusive-time map:

* **MoE grouped GEMM**: 23.5 % (W4) and 30 % (W16) of the step. The main loop
  is at the read roof. What remains is the per-call ramp of about 9 µs,
  multiplied by about 91 calls. D can only be reduced further by changing the
  model's routing (pruning), which is quality-gated.
* **Dense FP8 GEMV**: 27 % and 23 %. The large shapes run at 88-98 % of the roof,
  so the only remaining lever is fewer bytes: NVFP4 or lossless coding. Both
  were tried and rejected; see `rejected.md`.
* **HC chain**: 12 % and 10 %. This is a stats -> down -> up dependency with
  latency-bound small matrices. Reducing the rank would need retraining.
* **Draft phase**: about 19.5 % of W16 (3.13 ms). The draft head is 37 % of each
  draft forward.
* **Everything under 2.2 % each.** By 2026-09-06 no single remaining kernel-side
  item was worth more than about 2.2 %.

The 09-07 map ranked the remaining levers. After corrections from the 09-07
external review and from J1, the realistic low-risk kernel total is about
1-2.5 %. The only item above 3 % was raising the pruning threshold (τ 0.08 ->
0.10). It later failed its quality/behaviour gate (T10).
