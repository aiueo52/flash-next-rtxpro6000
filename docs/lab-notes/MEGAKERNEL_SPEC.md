# K2 — Persistent "glue island" kernels for the Qwen3.8-Flash-Next decode step

Feasibility study, 2026-09-05. RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs, 1792 GB/s
paper, 128 MiB L2), shared with other agents throughout: every GPU command under
`flock ~/.gpu.lock`, clocks not pinnable without sudo, so every A/B is interleaved
round-by-round and reported as a median.

Bench: `bench/megakernel/` — `harness.py`, `bench_barrier.py`, `bench_gemv_stage.py`,
`bench_hc_island.py`, `islands.py`, `run_all.sh`. Raw JSON in `bench/megakernel/results/`.
Trace analysis is CPU-only against `prof/traces/final0905-{w4,w16}-code-edit`.

---

## 0. Verdict: **NO-GO** on barrier-based glue islands

Measured on this GPU, one dependent stage costs **more inside a persistent kernel than as
its own kernel**, and much more than as a kernel launched with PDL:

| one dependent stage, 188 CTAs | µs | source |
|---|--:|---|
| as a separate Triton kernel in a CUDA graph | **0.823** | §3.2 |
| as a separate Triton kernel **with PDL** (K1) | **0.668** | §3.2 |
| as a stage behind a grid-wide barrier | **1.264** | §3.2 |
| as a stage behind a per-CTA dependency flag | **1.256** | §3.2 |
| *(hypothetical free barrier: arrive, never wait)* | *0.531* | §3.2 |

Fusing a boundary away therefore **costs** 0.44 µs against today's baseline and 0.60 µs
against a PDL'd one. Across the 1011 (W4) / 1404 (W16) strictly-serial island boundaries
per step (§2) that is a **+0.45 / +0.62 ms per step regression** today and **+0.60 /
+0.84 ms after K1 lands** — before the 11-13 % of GEMV bandwidth a 188-CTA persistent grid
costs (§5) and the 1.04 / 0.85 ms the CUDA graph already hides by running glue kernels
concurrently (§2). Even the unattainable ceiling is small: with a *free* barrier
(arrive-only, no wait) a stage would cost 0.531 µs against a PDL'd launch's 0.668, i.e. at
most **0.14 µs per boundary = −0.14 ms/step at W4 (−1.2 %), −0.20 ms at W16 (−1.0 %)**.

Measured end-to-end on the real HC island, the fork's existing megakernel deletes 4 of 5
launches and is **18-26 % slower** than the equivalent split (§4.2) — the same verdict the
fork already reached twice on this island (§4.1). **Recommended day spend on megakernels:
0.** The 1-3 days should go to K1's PDL and to the already-written, still-unmeasured
non-barrier fusions (§8.2).

---

## 1. Method

Three primitives, each reported as the **slope** of total time against stage count, so
every fixed cost (graph-replay entry, kernel ramp, the counter reset) cancels:
`launch` (N chained trivial Triton kernels in one graph, each reading the previous one's
output; also with `launch_pdl=True` + `gdc_wait`/`gdc_launch_dependents`); `barrier` (ONE
persistent kernel doing N grid-wide barriers, in three implementations — `atomic_add` RMW
poll = the fork's `_grid_barrier`, release-add + volatile-load poll = the fork's
`hc_combine_fused_triton`, and arrive-only as a lower bound on the atomic alone — swept
over grid 8 / 94 / 188); and `cta_flag` (the same kernel where a stage waits on **one**
neighbouring CTA's flag: fine-grained per-tile counters instead of a barrier).

Every stage stores its own 1 KB slice and reads its producer's, so a barrier stage pays
the same L2 round trip a launch-chain stage pays and the comparison isolates the boundary
mechanism. Each captured graph holds 32 copies of the construct (a graph holding one 2 µs
kernel is host-launch-bound on replay). Everything runs under graph replay after a
spin-up, ≥ 0.3 s per measurement, 5 interleaved rounds, weights rotated over ≥ 4× L2.

---

## 2. The islands: what can and cannot be fused

`bench/megakernel/islands.py` cuts one phase's kernel stream at every kernel this project
cannot call from inside another — the CUTLASS NVFP4 grouped GEMMs, the FlashInfer GDN WY
kernel, the TRT-LLM MoE prologue/activation/`memset32`, trtllm-gen attention
(`kernel_mha`), the QSA indexer (`qsa_index_q_prep`, `qsa_index_k_compress`, `fast_topk`),
cuBLAS — leaving the **glue islands**. Per whole decode step
(`final0905-*-code-edit`, occurrence 5, draft + verify + extend):

| | islands | glue kernels | intra-island boundaries | of those, strictly serial | glue kernel-µs | step wall |
|---|--:|--:|--:|--:|--:|--:|
| W4 | 162 | 1401 | 1239 | **1011 (82 %)** | 8.98 ms | 11.95 ms |
| W16 | 191 | 1920 | 1729 | **1404 (81 %)** | 13.17 ms | 20.47 ms |

The other 18-19 % of boundaries sit between kernels that **already overlap** in the
captured graph, hiding 1.04 ms (W4) / 0.85 ms (W16); fusing across them serialises work the
graph currently hides. Most glue kernels are also far below the machine's width, which
matters because a persistent island grid is one fixed grid for all its stages (W4 medians):

| kernel | CTAs | n/step | med µs | µs/step |
|---|--:|--:|--:|--:|
| `_w8a16_gemv_kernel` qkvz / out+o_proj / attn qkv | 512 / 480 / 416 | 36 / 49 / 15 | 29.7 / 13.2 / 26.3 | 1070 / 644 / 395 |
| `_hc_up` / `_hc_down` / `_w8a16_gemv_silu` | 160 / 100 / 100 | 106 / 106 / 49 | 4.90 / 3.98 / 8.96 | 519 / 422 / 439 |
| `_w8a16_gemv_kernel` (**GDN in_proj_ba**) / `_router_triton` | **3** / **4** | 36 / 49 | 5.73 / 4.16 | 206 / 204 |
| `hc_combine_gate` / `_hc_branch_stats` / `hc_combine_apply` | 128 / 16 / 32 | 98 / 100 / 98 | 2.00 / 1.54 / 1.15 | 196 / 154 / 113 |
| `_causal_conv1d_update` / `qkvzba_split_reshape` | 160 / 64 | 36 / 36 | 2.66 / 2.88 | 96 / 104 |

A 3-CTA 5.7 µs kernel and a 4-CTA 4.2 µs kernel are neither launch- nor bandwidth-bound;
they are latency-bound with 1.6 % of the machine busy. A 188-CTA persistent grid does not
shorten them — it makes 185 CTAs spin at a barrier while they run.

### 2.1 Island inventory per layer (W4 verify medians; `*` = ≤ 1.5 µs, i.e. boundary-dominated)

**GDN layer, ×36 — 2 islands, 24 kernels, ~110 µs, 22 boundaries**

| island | between | stages (µs) | total |
|---|---|---|--:|
| **I1** | MoE GEMM2 … GDN WY | `fused_gate_sigmoid_mul_add` 1.7, `hc_combine_gate` 2.1, `hc_combine_apply` 1.1\*, `hc_branch_stats` 1.5, `hc_down` 4.0, `hc_up` 5.0, `gemv qkvz` 30.4, `gemv ba` 6.1, `qkvzba_split_reshape` 2.9, `causal_conv1d` 2.7, `memcpy32` 0.8\*×2 | 59.5 |
| **I2** | GDN WY … MoE prologue | `layer_norm` 1.8, `gemv out_proj` 13.2, `hc_combine_gate` 1.6, `hc_combine_apply` 1.2\*, `hc_branch_stats` 1.5, `hc_down` 3.9, `hc_up` 4.9, `memcpy32` 0.9\*, `gemv_silu gate_up` 9.0, `gemv shared_down` 3.9, `gemv` 4.3, `router` 4.2 | 50.5 |

**QSA layer, ×12 — 3 islands, 23 kernels, ~102 µs, 20 boundaries**

| island | between | stages (µs) | total |
|---|---|---|--:|
| **Q1** | MoE GEMM2 … indexer GEMM | `hc_combine_gate` 2.2, `hc_combine_apply` 1.2\*, `hc_branch_stats` 1.6, `hc_down` 4.2, `hc_up` 4.9 | 13.9 |
| **Q2** | `fast_topk` … `kernel_mha` | `expand_qsa_block_indices` 1.8, `gemv attn qkv` 26.3, `fused_qk_rmsnorm_rope_gate` 2.5, `fp8_kv_store` 1.1\*, `fa2_valid_counts` 1.2\*, `compact_kv` 5.0 | 38.0 |
| **Q3** | `kernel_mha` … MoE prologue | `fused_sigmoid_mul` 1.0\*, `gemv o_proj` 12.9, `hc_combine_gate` 1.6, `hc_combine_apply` 1.2\*, `hc_branch_stats` 1.5, `hc_down` 4.0, `hc_up` 4.8, `memcpy32` 1.0\*, `gemv_silu` 8.9, `gemv` 3.9, `gemv` 4.4, `router` 4.2 | 49.5 |

The W16 draft loop repeats a 16-kernel island (110.9 µs, of which the 81.2 µs draft
lm_head GEMV is one stage) once per inner step: 46 islands, 560 glue kernels, 514
boundaries per step.

By duration, 809 (W4) / 1088 (W16) glue kernels are at or below 3 µs and carry
1.24 / 1.73 ms of the step; that is the entire population a boundary-cost change can act
on. The 178 / 296 kernels above 6 µs carry 5.90 / 9.06 ms and are bandwidth work.

---

## 3. What a dependent stage boundary costs

### 3.1 From the production trace: the fixed part is 1.0-2.0 µs and flat in M

Bytes moved per HC-boundary kernel against its measured median; "traffic" is
bytes / 1500 GB/s, the rate the CUTLASS MoE GEMM sustains in-wave here.

| kernel | W4 bytes | traffic | measured | **fixed** | W16 fixed |
|---|--:|--:|--:|--:|--:|
| `hc_combine_gate` | 160 KB | 0.11 | 2.00 | **1.89** | 1.99 |
| `hc_combine_apply` | 180 KB | 0.12 | 1.15 | **1.03** | 0.73 |
| `_hc_branch_stats` | 185 KB | 0.13 | 1.54 | **1.41** | 1.08 |
| `_hc_down` / `_hc_up` (fp8 w) | 3.20 / 3.25 MB | 2.24 / 2.27 | 3.98 / 4.90 | 1.74 / 2.63 | 1.66 / 2.36 |
| `_fused_gate_sigmoid_mul_add` | 60 KB | 0.04 | 1.73 | **1.69** | 1.98 |
| `memcpy32_post` | 4 KB | 0.00 | 0.82 | **0.82** | 0.82 |

`hc_combine_gate` and `hc_combine_apply` are sgl-kernel CUDA ops that **already launch
with PDL** (`PDLWaitPrimary`/`PDLTriggerSecondary` in `jit/csrc/elementwise/hc_combine.cuh`),
so their 0.7-2.0 µs is the residual *after* PDL — the ceiling fusion is competing for.

### 3.2 Measured: launch vs barrier vs per-CTA flag

`bench/megakernel/results/barrier.json`, 2026-09-05 20:34, SM 2.3 GHz under load, 5
interleaved rounds. µs per dependent stage (slope over 1/2/4/8/16 stages).

**A dependent kernel launch in a graph**

| grid | plain | with PDL | PDL saves |
|--:|--:|--:|--:|
| 1 | 0.762 | 0.551 | 0.211 |
| 188 | **0.823** | **0.668** | 0.155 |

**A stage boundary inside one persistent kernel**

| grid | `atomic_add` RMW poll | release + volatile-load poll | per-CTA flag | *arrive-only (no wait)* |
|--:|--:|--:|--:|--:|
| 8 | 1.067 | 0.782 | 0.715 | *0.454* |
| 94 | 1.084 | 1.115 | 0.751 | *0.459* |
| **188** | **1.264** | 2.067 | **1.256** | *0.531* |

Four things this says:

1. **At full width a barrier costs 1.26 µs — more than the 0.82 µs kernel launch it
   replaces, and nearly double a PDL'd launch's 0.67 µs.** Regression: 0.44 / 0.60 µs.
2. **Fine-grained per-tile counters do not rescue it.** Waiting on one neighbouring CTA
   rather than all 188 is cheaper at small grids (0.715 vs 1.067 at 8) but identical at 188
   (1.256 vs 1.264): what dominates is waiting for the slowest producer plus the dependent
   read's L2 round trip, not the number of arrivals.
3. **The atomic is cheap; the wait is not.** Arrive-only is 0.45-0.53 µs and flat in grid,
   so even a *perfect* barrier leaves at most 0.668 − 0.531 = 0.14 µs per boundary against
   a PDL'd launch — the hard ceiling on the whole idea.
4. **The poll implementation inverts with grid.** The volatile-load poll the fork chose
   (correctly, at ≤ 94 CTAs) is best at 8 and *worst* at 188 (2.07 µs); an island kernel
   would need its barrier re-picked per grid size. The persistent kernel's own ramp (the
   fitted intercept) is 0.94-1.38 µs, consistent with the 0.82 µs launch.

---

## 4. The HC island: separate kernels vs one persistent kernel

### 4.1 This was already built in the fork, twice, and lost both times

| experiment | commit / file | measured |
|---|---|---|
| Persistent megakernel: HC combine + per-branch RMSNorm + low-rank down + **grid barrier** + up, one launch; then a round 4 with weight prefetch hoisted above the barriers, end-of-kernel clear and a pipelined up-projection | `4ddfa5a229`, `aeef3d2ab5`, `layers/hc_fused_triton.py` | **parity** with the 3-4 kernel chain both times: ~18-22 µs, then 19.5-24 vs 19.4-22.4 |
| Replacing that chain with **three separate** kernels (K0 stats/normalise/zero, K1 split-K down, K2 up+gate+mean) | `b20ce6eebb`, `layers/hc_mix2_triton.py` | 19.5-20.7 → **14.6-16.4 µs**. Splitting beat the megakernel by 25 % and is what ships (`SGLANG_HC_MIX2=1`) |
| Two CUDA combine kernels → one Triton kernel with a grid barrier | `layers/hc_combine_fused_triton.py` | **regression**: 2.14 / 2.85 µs (1 / 16 rows) → 3.26 / 3.74. With the barrier deleted in a diagnostic build, 2.53 / 2.88 — barrier ≈ 0.7-0.9 µs at that grid, matching §3.2 |
| Fusing by **merging grids** instead (apply folded into the next mix's K0; gate computed at mix time) | `67218c0bad`, `a13ef3e469` | kept as wins, flag-gated |
| Fusing an elementwise stage into a GEMV **epilogue** (`silu(g)*u` inside the gate_up GEMV) | `081e8d9b09` | −1.50 / −1.76 / −2.11 µs at M = 1 / 4 / 16 |

`hc_mix2_triton.py`'s docstring gives the mechanism: *"The persistent variant in
`hc_fused_triton` serialises the down projection, a grid barrier, and the up projection
inside one launch, so at M ≤ 16 neither weight stream ever runs at full rate."*
`serve-fast.sh` records the operational conclusion: *"HC fusion (`SGLANG_HC_FUSED=1`) …
slower -> off."*

**The pattern across six experiments plus §3.2: deleting a launch by merging grids or by
an epilogue wins; deleting a launch by putting a grid-wide barrier in its place does not.**

### 4.2 Re-measured here, on the real island

`bench/megakernel/results/hc_island.json`. The production chain (`hc_combine_split` +
`hc_norm_mix2`, 5 launches) against the fork's existing megakernel
`hc_fused_combine_norm_mix` (1 launch), at the production shapes (hidden 2560, hc 4,
K 10240, low rank 320), activations and mix weights both rotated over >= 4x L2, 5
interleaved rounds. µs of CUDA-graph wall per island call:

| | M=4 | M=16 | launches |
|---|--:|--:|--:|
| `prod` — combine + mix, FP8 mix weights (what ships) | **12.71** | **13.54** | 5 |
| `bf16_split` — the same, BF16 weights (apples-to-apples baseline) | 16.41 | 16.73 | 5 |
| `bf16_fused` — the persistent megakernel | **19.68** | **19.82** | 1 |
| `mix_split_bf16` — mix only, three kernels | 13.85 | 14.25 | 3 |
| `mix_fused_bf16` — mix only, persistent | **17.40** | **18.02** | 1 |

**The megakernel deletes 4 of 5 launches and is 18-26 % slower**: +3.27 µs (M=4) / +3.09
(M=16) on the full island, +3.55 / +3.77 on the mix alone — **+0.8 to +1.3 µs of loss per
deleted boundary**, of which §3.2 accounts for 0.44 µs as the barrier and the rest is the
bandwidth loss `hc_mix2_triton.py` names (the mix's two weight stages cost 12.1 µs as two
kernels against 16.7 µs of device time inside the fused one).

Also from this run: the inter-kernel *gap* is negligible (`prod`'s graph wall is 12.71 µs
against 12.14 µs of device time, 0.14 µs per boundary), so the launch cost lives inside the
kernel durations, as `OVERHEAD_REPORT.md` found.

---

## 5. Can a W8A16 GEMV be a stage of a persistent kernel? No

Every island worth fusing contains a GEMV (§2.1). `bench_gemv_stage.py` runs **one**
Triton GEMV under three launch shapes — identical inner-loop machine code, only the grid
and the tile-loop trip count change — plus a `fork` row calling the production
`w8a16_gemv` with its tuned tile as a calibration gate.

µs, M=4 / M=16:

| shape (fp8 weight) | tiles | fork | native | persistent (188) | island (+barrier) |
|---|--:|--:|--:|--:|--:|
| qkvz `16384×2560`, 40 MB | 256 | 29.45 / 29.95 (1424 GB/s) | 29.77 / 29.98 | **33.01 / 34.29 (1271 / 1223)** | 34.38 / 35.47 |
| out_proj `2560×6144`, 15 MB | 40 | **13.13 / 13.39 (1198)** | 23.94 / 24.91 (657) | 23.96 / 24.80 | 25.54 / 26.45 |
| gate_up `1280×2560`, 3.1 MB | 20 | **4.86 / 5.48 (674 / 598)** | 10.86 / 11.28 (302) | 10.83 / 11.26 | 12.48 / 12.90 |

* **Calibration**: at qkvz the bench kernel matches the production kernel to 1 %.
* **A persistent 188-CTA grid costs qkvz 11-13 % of its bandwidth** (+3.2 µs at M=4, +4.3
  at M=16): 256 tiles over 188 CTAs is 1.36 waves, so a fixed grid gives 68 CTAs two tiles
  and 120 CTAs one, and the tail is a whole extra wave.
* **The narrow GEMVs cannot be a stage at all.** `out_proj` and `gate_up` have 40 and 20
  tiles; production reaches 1198 and 674 GB/s by *split-K* (`SPLITS` 5-10), i.e. 400-800
  CTAs. An island grid is capped at the SM count, so those stages give up split-K — a
  1.8-2.2× regression, dwarfing any boundary saving.
* **`island − persistent` is +1.18…+1.65 µs at every shape**, independently confirming
  §3.2's 1.26 µs barrier.
* **By-product**: the tuned production GEMV moves a 3.1 MB fp8 weight at 674 GB/s (M=4)
  while `_hc_down` already moves 3.20 MB at 844 GB/s, so the apparent "the HC mix runs at
  half roofline" headroom is not real and should not be chased.

---

## 6. Composition with PDL (K1)

K1 (`~/tools/sglang-pdl`, branch `opus/pdl-sweep`) adds `SGLANG_TRITON_PDL` plus
`gdc_wait`/`gdc_launch_dependents` to `w8a16_gemv`(+`_silu`), `hc_mix2` K0/K1/K2,
`fused_qkvzba_split_reshape_cat`, the fla layer-norm and `_fused_sigmoid_mul`. Much of the
glue already had PDL and is not K1's to win:

Of the glue kernels per step, 1047 (W4) / 1370 (W16) are this fork's Triton kernels with
no PDL — K1's target; 207 / 318 are sgl-kernel JIT CUDA ops that **already** launch with
it; 142 / 227 are torch/aten and unreachable.

K1's measurements (read from its logs): a 100-kernel toy chain costs 1.137 / 1.953 /
3.113 µs per kernel at grid 4 / 16 / 128, the launch attribute taking 0.34-0.41 µs off;
and the in-server W4 A/B, `SGLANG_TRITON_PDL` 0 → 1:

| | step wall | GPU busy | verify busy | idle |
|---|--:|--:|--:|--:|
| off, code-edit / prose-en | 11.26 / 10.86 | 11.00 / 10.39 | 9.99 / 9.38 | 0.14 |
| **on**, code-edit / prose-en | **10.63 / 10.30** | **10.39 / 10.07** | **9.38 / 9.06** | 0.09 |

**−0.63 / −0.56 ms per step (−5.6 % / −5.2 %), all in verify, kernel count unchanged.**

So PDL and fusion do not compose — they compete for the same microseconds, and PDL wins on
every axis: one flag, no arithmetic change, no capture-shape or occupancy analysis, and
already implemented. It also *raises* the bar fusion must clear, from 0.823 to 0.668 µs
per boundary against a barrier's 1.264. (The PDL arm in `bench_barrier.py` duplicates
K1's toy only so that launch-with-PDL and barrier-inside-a-kernel are measured in one
process at the same clocks, which is what a composition claim needs here.)

---

## 7. Engineering constraints (why even the ceiling is out of reach)

**C1 — Co-residency is not guaranteed inside the model's CUDA graph.** A grid-wide barrier
deadlocks unless every CTA is resident. `hc_combine_fused_triton` keeps `rows*Q ≤ SM count`
and argues "a CUDA-graph replay serialises the stream"; that is **false for this graph**,
where 30-33 % of verify kernels start before their predecessor ends (§2), so a concurrent
branch can hold SMs the island needs. A safe implementation needs a cooperative launch
(`launch_cooperative_grid=True`, if the runtime accepts it under stream capture), or a
smaller grid and less throughput, or islands restricted to positions with no concurrent
branch — which excludes exactly the GEMV-bearing islands, since the qkvz/ba pair and the
shared-expert / router / routed-expert group are concurrency the fork built on purpose.

**C2 — Counters must be self-restoring, and the obvious way deadlocks.** "One more grid
barrier, then CTA 0 zeroes the counters" is wrong: CTA 0 can store the zero while another
CTA is still polling, and that CTA spins forever. This study hit it — the first GPU window
produced no output in 40 minutes and had to be killed, holding the shared lock throughout.
The correct pattern is the fork's ticket (`hc_fused_triton.py:461`):
`ticket = atomic_add(c,1); if ticket == num_ctas-1: reset`. That is the third independent
way a grid barrier hangs, after C1 and a poll the compiler hoists out of the loop (which
is why both fork kernels mark the poll `volatile`).

**C3 — Registers, shared memory and compile time are taken by the union of the stages.**
A persistent kernel compiles once, for its worst stage: an island holding a GEMV stage
(16×64 fp32 accumulator + double-buffered `[64,128]` fp8 tile) and an `_hc_up` stage
compiles at the maximum of both, which can drop occupancy below the 1 CTA/SM the barrier
requires and discards the per-shape tile tuning in `_BY_SHAPE`/`_BY_SHAPE_M16` and
`HCMix2Config`. Capture shapes are fine (`ROWS`/`M_PAD` are already `tl.constexpr`) but
each island multiplies the JIT matrix (rows × dtype × fp8/bf16) on an already noticeable
start-up compile cost. The fork's round-4 megakernel needed hand-placed prefetches and an
`inline_asm_elementwise` barrier to stop Triton hoisting loads — ~400 lines of fragile
scheduling, for parity.

**Numerics.** A barrier-based fusion that keeps each stage's tiling is bit-exact by
construction — only the launch boundary moves. That is its one attractive property, and it
is exactly the version that measures at parity or worse; every variant that *won* in the
fork's history changed the tiling to merge grids. (The shipped path is not bit-reproducible
today anyway: `_hc_down_kernel` accumulates `t_raw` with device-scope `atomic_add`.)

---

## 8. Effort, ordering, go/no-go

### 8.1 The arithmetic

With `L` = 0.823, `Lp` = 0.668, `B` = 1.264 µs (§3.2) and `n` = strictly-serial boundaries
per step (1011 W4 / 1404 W16, §2):

| | per boundary | W4 / step | W16 / step |
|---|--:|--:|--:|
| fuse against today's baseline (`L − B`) | −0.441 µs | **+0.45 ms (+3.7 %)** | **+0.62 ms (+3.0 %)** |
| fuse against a PDL'd baseline (`Lp − B`) | −0.596 µs | **+0.60 ms (+5.0 %)** | **+0.84 ms (+4.1 %)** |
| *ceiling: free barrier vs PDL (`Lp − B_arrive`)* | *+0.137 µs* | *−0.14 ms (−1.2 %)* | *−0.20 ms (−1.0 %)* |

Positive = slower. And `n` is already optimistic: it assumes every serial boundary is
absorbable, which §5 refutes for the GEMV stages (11-13 % of bandwidth on qkvz, 1.8-2.2×
on the split-K GEMVs) and §2 refutes for the 3-16 CTA stages.

**The go threshold set before measuring was `Lp − B ≥ 0.5 µs`; the no-go threshold
`Lp − B ≤ 0.2 µs`. Measured: −0.60 µs. The result is not marginal.**

What would change this: only a case where the *dataflow itself* cannot cross a kernel
boundary — a producer whose output feeds a differently-tiled consumer without a round trip
to memory (a true GEMM→GEMM fusion holding accumulators in registers). No island in §2.1
has that shape; every stage already publishes to global memory and the next reads it back,
so a kernel boundary costs nothing the barrier does not also cost.

### 8.2 Where the 1-3 days should go instead

| # | lever | evidence | state | est. |
|---|---|---|---|---|
| **A1** | **Land K1's PDL.** −0.63 / −0.56 ms per step measured at W4; W16 and the MODE=full acceptance/needle validation still to run. Zero numerics risk — with the flag off the PTX and launch config are byte-identical. | §6 | implemented, W4 measured | 0.5 d |
| **A2** | **Validate the already-written non-barrier HC-boundary fusions**: `SGLANG_HC_GATE_EARLY=2` (move the gate launch; bit-identical values) measured −12..−20 µs/step; `SGLANG_SHARED_GATE_EARLY=1` and `SGLANG_HC_APPLY_MIX_FUSED=1` (fold the combine apply into the next mix's K0 — a *grid merge*, no barrier) still unmeasured. | `opus/hc-boundary` | implemented, validation running | 0.5 d |
| **A3** | More **epilogue** fusions inside the fork's own GEMVs — the shape that already pays (`SGLANG_SHARED_GATEUP_FUSED` bought −1.5..−2.1 µs/call by making an activation launch free). Candidates are the elementwise kernels adjacent to a GEMV in §2.1: `_fused_sigmoid_mul` before o_proj, `_fused_gate_sigmoid_mul_add` after GEMM2. | `081e8d9b09` | not attempted | 1-2 d |
| ~~A4~~ | ~~Barrier-based glue islands~~ | this document | **no-go** | 0 d |

Three things this study also rules **out**, with measurements: **fusing work into a
narrow-grid stage** (`SGLANG_HC_GATE_EARLY=1` folds the combine gate into K0's 16 CTAs:
+114 µs/step at W4, K0 1.5 → 4.3 µs, commit `375c8b12b4`); **making one latency-bound glue
kernel faster** (A8 recovered 0.2-0.5 µs of `doActivationKernel`'s 3.6 µs — the rest is the
cost of having a launch in the chain, `A8_FEASIBILITY.md` §4); and **"the HC mix runs at
half roofline"** (§5: 3.1 MB of fp8 weights stream at 674 GB/s even from the tuned
production GEMV, below `_hc_down`'s 844 GB/s).

---

## 9. Reproduction

```bash
cd ~/tools/flash-next-bench
. bench/megakernel/env.sh          # fork venv + CUDA + PYTHONPATH
bash bench/megakernel/run_all.sh   # takes ~/.gpu.lock per bench; ~5 min each

# trace-side island enumeration (CPU only)
python bench/megakernel/islands.py prof/traces/final0905-w4-code-edit/*.gz \
    "step[TARGET_VERIFY bs=1]"
python bench/megakernel/islands.py prof/traces/final0905-w16-code-edit/*.gz "draft"
```

Results: `bench/megakernel/results/*.json`, console logs in `bench/megakernel/logs/`. Run
when the GPU is otherwise idle: a persistent kernel with `grid = SM count` is exactly the
workload another process's kernels perturb.
