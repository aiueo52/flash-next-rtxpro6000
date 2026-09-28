# G1 — recover the MoE grouped-GEMM tail-wave loss on sm_120

Running log. Newest entries at the bottom of each section.
Task: custom CUTLASS/CuTe variant for the sm120 NVFP4 grouped GEMM so the CTA grid is a
whole number of waves. Baseline facts: `MOE_SMALLM_SPEC.md` §8.

## Environment

* Private FlashInfer copy: `$HOME/tools/flashinfer-g1` (`cp -a` of the production
  venv package on 2026-09-05, i.e. **A0+A3 already applied**).
* Private JIT/autotune cache: `$HOME/.cache/sglang-g1` (`cp -a` of `~/.cache/sglang`).
* `bench/moe_smallm/g1env.sh` sources both plus the CUDA-13 toolchain.
* Production venv, `~/tools/flashinfer-a8`, `~/tools/flashinfer-g2`: never touched.
* Every GPU command under `flock -w 21600 $HOME/.gpu.lock`.

## 0. Plan / decision tree

The task's option (a) is split-K by the **problem-shape trick** (duplicate each expert into
`s` K-slices as extra grouped-GEMM groups). Reading the sources first (below) determined the
shape of that patch before any GPU time was spent.

## 1. Source reading (no GPU) — what a split-K patch would have to touch

Read against `~/tools/flashinfer-g1` (= production, i.e. **A0+A3 already applied**).

* `TmaWarpSpecializedGroupedGemmInput` (`.../cutlass_kernels/include/moe_gemm_kernels.h:66`)
  is **fully per-group**: `shape_info.problem_shapes[g]`, `ptr_act/ptr_weight/ptr_c/ptr_d[g]`,
  `stride_act/stride_weight/stride_d[g]`, `alpha_scale_ptr_array[g]`,
  `fpX_block_scaling_factors_{act,weight}[g]` and their per-group **LayoutSF**.
  Every one of these is written by `computeStridesTmaWarpSpecializedForExpert`
  (`cutlass_fused_moe_kernels.cuh:1325`) at index `out_idx == expert`.
* The sm120 array collective replaces the TMA descriptor's **dims *and* strides** per group
  (`cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp:1053
  tensormaps_replace_global_tensor_properties`), building the tensor from
  `make_shape(M,K,1)` + `mainloop_params.dA[group]` and from
  `mainloop_params.layout_SFA[group]`. **A K-slice with a full-K stride is therefore a legal
  per-group tensor** — no CUTLASS change needed, only different numbers in the arrays the
  TRT-LLM strides kernel already writes. This is what makes "split-K as extra groups" a
  host/prologue-side patch rather than a kernel rewrite.
* The one non-obvious piece is the NVFP4 block-scale layout. `Sm1xxBlockScaledConfig::
  tile_atom_to_shape_SFA` is `tile_to_shape(SfAtom(128x64), (M,K,L), Step<_2,_1,_3>)`, i.e.
  **K-tile-fastest**, so a K range is *not* a contiguous sub-buffer and a plain pointer offset
  is wrong for any operand with more than one 128-row tile (the GEMM1 weight SF has 10).
  Verified by a standalone cute program
  (a throwaway host program, not published): taking the packed layout for `K_slice` and overwriting the
  single M-tile stride `cute::get<0,1>(layout.stride())` with the full-K value gives a layout
  that matches the full one *exactly* under a constant byte offset
  `k_offset * Blk_MN / SFVecSize` (= 10240 B for k0=1280) — 0 mismatches over both SFA
  (M=1280, 10 tiles) and SFB (rows<=16, 1 tile).

Consequence: the split-K patch is confined to `cutlass_fused_moe_kernels.cuh` (the strides /
fused-prologue kernel + `doActivationKernel` reduction + one workspace buffer). No CUTLASS
header, no new template instantiation, no TU explosion — unlike A8.

Also learned: **a private FlashInfer copy forces a full 97-TU JIT rebuild** (~7 min here),
because ninja keys on the absolute source paths. Subsequent csrc edits are incremental.

## 2. Measurement 1 — the wave staircase is NOT in the data

The premise of the whole task (spec 8.3) is `t = ceil(D*10/188) * wave`, from four points at
T=4 (D=10/20/28/40). An 11-point sweep with the tactic pinned (`--force-tactics 17,57`,
synthetic weights, rotation 16, clocks steady at 2272-2317 MHz) says otherwise:

| D | 10 | 14 | 18 | 19 | 22 | 26 | 28 | 32 | 36 | 38 | 40 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GEMM1 us | 19.75 | 23.01 | 27.00 | 27.78 | 31.10 | 35.32 | 36.82 | 41.88 | 46.07 | 48.57 | 50.82 |

Wave boundaries are at D = 18.8 and 37.6. **There is no step at either.** A straight line
`9.4 + 1.036*D` fits every point to ~1 %. The persistent kernel has no barrier between
"waves": CTAs pick up tiles asynchronously, so a partly-filled last wave costs its own bytes,
not a whole wave.

Spec 8.3's D=40 point (74.67 us) is 47 % above the line and is what created the apparent
staircase; the 2026-09-05 morning runs were taken on a cold/low-clock card (the same harness
reports `sm_clock_burst = 1402 MHz` on the first point after idle).

> **GPU-sharing note (2026-09-05 22:26).** `probe2` (real weights, same sweep) was started
> under `flock $HOME/.gpu.lock` and then another agent's `scripts/g3_sweep.py`
> (38.9 GB resident) appeared on the device *without* taking the lock. probe2 was killed
> rather than record contaminated numbers. probe1 (22:15-22:21) ran with the device to itself
> (6.7 GB total) and its numbers stand.

## 3. Measurement 2 — the DRAM read roofline on this box

`bench/moe_smallm/roofline.py`, 6 GB buffer, device idle:

| reader | GB/s |
|---|--:|
| `sum` fp32 / bf16 / `max` / `norm` | **1612 - 1621** |
| `copy` (read+write) | 1456 |

So the achievable **pure-read** rate is **~1615 GB/s = 90 % of the 1792 GB/s data sheet**, and
the 1461 GB/s figure the spec quotes is the *copy* roof (confirmed at 1456) — the wrong
comparison for a weight-streaming GEMM.

### What that does to the GEMM1 model

The synthetic sweep's slope is 1.036 us per distinct expert = 1.8432 MB / 1.036 us =
**1779 GB/s**, i.e. *above* the measured read roof. The excess is L2: the rotation's working
set is ~825 MB of GEMM1 weights against a 128 MiB L2, and real (non-LRU) replacement gives
roughly `C/W = 15 %` hits; 13 % hits reconcile 1779 GB/s effective with 1615 GB/s from DRAM
exactly. **The GEMM1 mainloop is therefore streaming at ~100 % of what this card can read.**

Re-reading the production point with that in hand (real weights, D=28.4, T=4,
GEMM1 = 42.4 us in the server trace):

```
weight bytes   28.4 x 1.8432 MB = 52.3 MB
at 1615 GB/s                    = 32.4 us     <- irreducible
measured                          42.4 us
difference                        10.0 us     <- the whole prize
```

That 10 us matches spec 8.3's "11.0 us/call of wave quantisation" almost exactly — but the
sweep above shows it is **not** wave quantisation (no staircase), it is a **flat per-call
intercept** (the fit's constant term is 9.4 us). Its components are the persistent kernel's
ramp: launch, the tile scheduler's initial scan over the 512-entry group list, the per-CTA
TMA tensormap init + first descriptor replace/fence, the mainloop pipeline fill and the
epilogue drain.

**This redirects the task.** Split-K cannot touch a per-call intercept — it multiplies the
number of tile ramps and can only make it worse. What *can* touch it is shortening the group
list the scheduler walks. Hence the patch implements two independent switches.

## 4. The patch (in `~/tools/flashinfer-g1`, two files, no CUTLASS change)

Three independent switches, all defaulting to upstream behaviour, all confined to
`csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` plus a params struct in
`csrc/nv_internal/.../include/moe_kernels.h`. They only ever engage when the A0/A3 fused
routing prologue engages (`num_tokens <= 32`), i.e. exactly the decode/verify regime.

1. **`FLASHINFER_MOE_PACK_GROUPS=1` — pack + shorten the group list.**
   The prologue already has every expert's token offset in shared memory, so each block
   warp-computes `rank(e)` = the number of active experts below `e` and `D` = the total (two
   32-wide reductions over the offsets, ~0.1 us), and writes expert `e`'s CUTLASS group at
   index `rank(e)` instead of `e`. Leftover slots get a zero-token problem shape, each written
   by exactly one thread, so no races. On top of that the *number of groups* drops from
   `num_experts_per_node` (512) to the host-side bound `expanded_num_rows` (40 at W4, 160 at
   W16) — an active expert owns at least one permuted row, so `D <= expanded_num_rows`.
   This is aimed at the tile scheduler's linear group scan
   (`sm90_tile_scheduler_group.hpp: get_work_idx_m_and_n`, 32 groups per iteration of
   dependent global loads), which is part of the per-call intercept section 3 identified.
   Everything CUTLASS indexes per group -- `alpha_scale_ptr_array`, the SF pointers and
   layouts, `ptr_act/weight/d`, and the FINALIZE epilogue's `ptr_source_token_index` /
   `ptr_router_scales` / `ptr_bias` (verified: `sm90_visitor_scatter.hpp:268` uses
   `ptr_index[l]`, and the launcher gives alpha/scale an `int64_t` L-stride of 1) -- moved
   from `expert` to the group index.
   **Must be bit-identical to upstream.**

2. **`FLASHINFER_MOE_SPLITK_GEMM1=1` (or `=n` to cap at n) — split-K as extra groups.**
   Requires (1). Each active expert emits `s` groups, slice `j` covering a contiguous K range
   of whole 256-element units, with:
   * problem shape K = the slice's K, but `make_cute_packed_stride(..., K_full)` for A and B
     (both are K-major, verified: `StrideA(M,K) = (K,_1,_0)`), and `ptr += k0`;
   * the block-scale layout taken packed for the slice K and then patched in one place --
     `cute::get<0,1>(layout.stride()) = <full-K value>` -- with the pointer advanced by
     `k0 * Blk_MN / SFVecSize` bytes. Verified exhaustively on the host for s = 1..5 over
     K=2560 including the uneven splits (1024/768/768), 0 mismatches
     (the same throwaway host programs).
   * slice 0 writes the normal GEMM1 output, slices 1.. write into a small extra workspace
     (`splitk_partials`, 7 buffers x 32*top_k rows, ~5.7 MB, only allocated when the env var
     is set), and `doActivationKernel` sums them in FP32 before SwiGLU. The slice count is
     published in a device scalar the activation kernel loads *before* its
     `cudaGridDependencySynchronize()`, so the load hides behind GEMM1's drain.
   `s` is chosen on the device from the measured `D` by
   `chooseKSplits`: minimise `ceil(D*tiles*s/SMs) * (ramp + 1/s)` where `ramp` is
   `FLASHINFER_MOE_SPLITK_RAMP_PCT` (default 25 %).
   **Not bit-identical**: partial sums are rounded to BF16 before the FP32 reduction.

3. **`FLASHINFER_MOE_SPLITK_RAMP_PCT`** — the one number that decides whether splitting is
   worth it; 0 makes the planner purely wave-count driven (the spec 8.4 model).

No CUTLASS header is touched and no new kernel is instantiated: the sm120 array collective
re-derives each group's TMA descriptor from `dA/dB/layout_SFA/layout_SFB` + the problem shape,
so all of this is data, not code, from CUTLASS's point of view. Incremental rebuild is one TU
(~95 s).

### Geometry verified on the host before any GPU time

A throwaway host program (not published; compiles against the vendored cute; CPU only):

```
s=1: (k0=0,ks=2560)                                          total_K=2560 OK
s=2: (k0=0,ks=1280) (k0=1280,ks=1280)                        total_K=2560 OK
s=3: (k0=0,ks=1024) (k0=1024,ks=768) (k0=1792,ks=768)        total_K=2560 OK
s=4: (k0=0,ks=768) (k0=768,ks=768) (k0=1536,ks=512) (k0=2048,ks=512)  OK
s=5: 5 x 512                                                 total_K=2560 OK
StrideA(M=1280,K=2560) = (2560,_1,_0)     <- K-major
StrideB(N=3,   K=2560) = (2560,_1,_0)     <- K-major
```

"OK" = for every (row, k) in the slice, `full_layout(row, k0+k) - sliced_layout(row, k)` is the
same constant, and that constant equals the predicted `k0 * Blk_MN / SFVecSize`.

## 5. Re-reading the "intercept": it is soft wave quantisation, not a fixed cost

The D=10 point is the tell. Achieved GEMM1 bandwidth (bytes / measured us, synthetic sweep,
tactic 17, T=4) is **not** flat:

| D | tiles | tiles/CTA (188 CTAs) | us | GB/s |
|--:|--:|--:|--:|--:|
| 10 | 100 | 0.53 | 19.75 | 932 |
| 19 | 190 | 1.01 | 27.78 | 1261 |
| 28 | 280 | 1.49 | 36.82 | 1401 |
| 40 | 400 | 2.13 | 50.82 | 1450 |

(the last two are inflated ~13 % by L2, section 3; the ordering is the point).

So the kernel does not reach the DRAM roof until there are ~2 tiles per CTA. A CTA streams
one 184 KB tile through a 3-6 stage TMA pipeline; **one CTA cannot keep enough requests in
flight to saturate its share of DRAM**, so a half-empty tail wave is slow not because the SMs
idle but because 92 streaming CTAs move less bandwidth than 188 do.

That *is* the wave-quantisation loss the task is about — but its shape is a soft bandwidth
roll-off, not the hard `ceil(tiles/188)` staircase of spec 8.4, and the fitted straight line
(section 2) hides it inside a 9.4 us "intercept". It also means the two levers are:

* raise the tail wave's CTA count -> **split-K** (same bytes, twice the concurrent CTAs);
* cut whatever fixed work every CTA does before it starts streaming -> **group packing**.

Predicted from the probe-1 pair (equal bytes, K=2560/D=28 vs K=1280/D=56, the exact split-K=2
geometry): **-2.2 us on GEMM1 at W4**, i.e. -0.11 ms/step, against the task's -9..-11 us
target. Group packing is worth about 16 -> 2 scheduler iterations, ~1.8 us by a latency
estimate. Both are real but an order of magnitude below the brief's expectation, because the
brief's expectation came from a hard-staircase model that the data does not support.

## 6. Measurement 3 — the patch, measured (probe 4)

Real weights, tactic pinned, rotation 16, `--mode profile`; the sweep's first and last point
repeat the production D so drift is visible. Deltas are the mean over every D in the sweep,
+- the standard error.

### Correctness first

| comparison | result |
|---|---|
| **pack vs upstream, `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`** | **bit-identical**, all 12 tensors, T = 1/4/16 |
| pack vs upstream, production fused-finalize | max_abs 2.441e-04 — *exactly* the off-vs-off control (the epilogue's scatter-add is order-dependent) |
| split-K vs upstream | max_abs 1.038e-03, 4.3x the control |

Packing is therefore proved correct: the group remapping (alpha scales, SF pointers *and*
layouts, act/weight/D pointers, and the FINALIZE epilogue's per-group arrays) reproduces
upstream bit for bit once the one genuinely non-deterministic epilogue is removed.

### Timing

| | GEMM1 | GEMM2 | glue | total | ms/step |
|---|--:|--:|--:|--:|--:|
| **W4 (T=4), pack** | −0.73 ±0.46 | −0.99 ±0.26 | +0.87 ±0.23 | **−1.47 ±0.51** | −0.072 |
| W4, pack + split-K | −1.14 ±0.29 | −0.66 ±0.37 | +2.24 ±0.43 | −0.60 ±0.50 | −0.029 |
| **W16 (T=16), pack** | +0.43 ±0.49 | −1.26 ±0.39 | +0.73 ±0.21 | **−0.30 ±0.23** | −0.015 |
| W16, pack + split-K | +1.39 ±0.91 | −1.14 ±0.55 | +2.38 ±0.36 | **+2.32 ±0.59** | **+0.114** |

Per-glue-kernel attribution of the cost:

| | prologue | doActivation |
|---|--:|--:|
| W4 pack | +0.73 ±0.11 | −0.01 |
| W4 split-K | +1.52 ±0.43 | +0.77 ±0.17 |
| W16 pack | +0.33 ±0.15 | +0.23 ±0.11 |
| W16 split-K | +0.97 ±0.23 | +1.39 ±0.27 |

### Split-K: measured NO-GO

* At W4 it does what it was supposed to: **GEMM1 −1.14 us**, i.e. the tail wave really does
  stream faster with twice the CTAs. But the two latency-bound glue kernels give it all back
  (+1.5 us prologue, +0.8 us doActivation), so the call is a wash.
* At W16 the planner picks `s = 1` (D=69: 690 tiles = 3.7 waves, and no `s` improves
  `ceil(690 s/188)/s`), so it is *supposed* to be identical to packing — and it still costs
  **+1.39 us in doActivation**, because merely making that kernel load one device scalar
  (and carry the reduction's registers) is expensive when it is pure dependent latency
  (`A8_FEASIBILITY.md` section 3 measured the same effect from the other direction).
* And it is not bit-identical.

So: **split-K is off by default and should stay off.** The measurement is the interesting
part — it supports the mechanism (more concurrent CTAs in the tail *is* worth ~1 us of GEMM1
at W4) and shows the mechanism is worth less than the plumbing needed to exploit it.

## 7. Measurement 4 — packing, drift-cancelled and in-server (probes 6-8)

probe 5 (10 repeats per side, run back to back) showed **+10 us of thermal drift across three
consecutive W16 runs**, which is larger than the effect being measured. probe 6 therefore
alternates `off, pack, off, pack, ...` and takes each `pack` against the mean of its
neighbouring `off`s.

| | rounds | GEMM1 | GEMM2 | prologue | **total** | **ms/step** |
|---|--:|--:|--:|--:|--:|--:|
| **W4, D=28** | 3 x 5 | −1.69 | −1.53 | +0.62 | **−2.47 ±0.12 us/call** | **−0.121** |
| **W16, D=69** | 4 x 5 | −0.71 | +0.21 | +0.53 | **+0.14 ±0.24 us/call** | +0.009 |

(the W4 rounds were −2.38 / −2.16 / −2.70, the W16 `off` side 152.27 / 152.71 / 152.30 / 152.92,
so the alternation really did remove the drift.)

**Packing wins at W4 and is a wash at W16**, and the reason is visible in the numbers: the
saving is the tile scheduler's group scan, which shortens 512 -> 40 (12.8x) at W4 but only
512 -> 160 (3.2x) at W16, while the ~0.6 us prologue cost is the same. So the patch now
auto-gates on the shortening ratio (`FLASHINFER_MOE_PACK_MIN_RATIO`, default 4;
`expanded_num_rows * ratio <= num_experts_per_node`). Re-measured with the gate (probe 8,
3 alternating rounds):

| | W4 | W16 (gate declines) | W16 (`MIN_RATIO=1`, forced) |
|---|--:|--:|--:|
| total | **−2.35 us/call = −0.115 ms/step** | +0.08 us/call (no-op) | −0.07 us/call |

Bit-identity re-verified against the final build, gate on and gate forced: **PASS**.

### In-server A/B (probe 7, `validate.sh`, MODE=prof, same build, flag off/on back to back)

Outlier-robust (median over the 20 profiled steps of each call position — the mean is dominated
by the 1-5 preempted MoE GEMM calls per step, spec section 4):

| profile / workload | MoE GEMM off | MoE GEMM pack | delta | MoE total delta |
|---|--:|--:|--:|--:|
| W4 code-edit | 3.3905 | 3.2908 | **−0.100** | −0.079 |
| W4 prose-en | 3.1967 | 3.0421 | **−0.155** | −0.129 |
| W16 code-edit | 7.2661 | 6.4645 | −0.802 | −0.771 |
| W16 prose-en | 6.9190 | 6.8471 | −0.072 | −0.027 |

W4 agrees with the isolated measurement (predicted −0.128 ms/step, measured −0.079/−0.129).
The W16 pair disagrees with itself by 0.7 ms and with the isolated measurement, which is what
you would expect: the fused-finalize epilogue is order-dependent, packing perturbs it at the
1e-4 level, and at W16 that is enough to change which draft tokens are accepted, hence D and
the step composition. **Treat W16 as "no measurable change"** and W4 as −0.10 to −0.13 ms/step.
Step wall from the same runs: W4 10.40/10.32 ms off vs 10.33/10.09 ms pack.

## 8. Verdict

**The task's target cannot be reached, and the reason is in the hardware, not the
implementation.**

| the brief's model | what the GPU does |
|---|---|
| `t = ceil(D*tiles/188) * wave`, a hard staircase | no staircase at all over 11 points at T=4 and 16 at T=16 |
| 11.0 us/call (W4) / 13.8 (W16) of recoverable tail | ~10 us/call of *ramp*, of which ~2 us is the group scan and the rest is TMA/pipeline/launch |
| split-K makes the grid a whole number of waves | the grid is never "a wave"; splitting K does raise the tail's CTA count and *is* worth −1.14 us of GEMM1 at W4, but the reduction costs +2.3 us in two latency-bound glue kernels |
| the mainloop has headroom | the mainloop streams at ~100 % of the 1615 GB/s read roofline this card actually delivers |

Delivered instead: **group packing**, bit-identical, **−0.115 ms/step at W4** (−1.1 % of a
10.3 ms step), auto-gated to a no-op at W16. That is a third of the −0.3 ms/step the brief set
as success, and there is no version of split-K or of a hand-written sm120 CuTe kernel that gets
the rest, because the rest is either at the memory roofline or is per-launch ramp that a
different kernel would also pay.

**Task option (b) (a smaller CTA N tile) is moot for this shape**, and not for the reason the
brief assumed: with `swap_ab` the N axis carries the *tokens* (1-3 rows per expert), so it is
one tile whatever the N-tile size is. The tile count per expert is `gemm1_n / cta_tile_m` = 10,
set by the M axis, and sm120 has no M tile below 128. A smaller N tile would change nothing.

**Task option (c) (a dedicated sm120 CuTe kernel) is a NO-GO on stronger grounds than spec 8.6
gave**: the existing mainloop is at the measured DRAM read roofline, so a new kernel could only
compete on the ~10 us of ramp — and it would pay its own launch, tensormap and pipeline-fill
costs.

## 9. Rebase onto the G2 state — `g1-pack-only.patch` (2026-09-06)

`g1-pack-splitk.patch` fails 10/32 hunks on top of A0+A3+G2-1+G2-2, so packing was re-derived
on that base as a **split-K-free** patch. Private copy `~/tools/flashinfer-g1b` (a `cp -a` of
`~/tools/flashinfer-g2`), private cache `~/.cache/sglang-g1b` (from `~/.cache/sglang-g2`);
neither `flashinfer-g2` nor the production venv was touched.

Chain check: `a0 -> a3 -> g2-1 -> g2-2 -> g1-pack-only` applied to a pristine wheel tree
reproduces `flashinfer-g1b` **byte for byte** in both files.

### Composition with G2-2: no relocation needed

G2-2 turns the prologue grid into `ceil(E/32)` descriptor blocks + `num_tokens*k` row blocks.
Packing stays entirely in the descriptor blocks — which are exactly the blocks whose
`expert = blockIdx.x*32 + threadIdx.x` covers `[0, E)` once, so the "thread `expert` owns
leftover group slot `expert`" argument still gives complete, race-free coverage. The row blocks
and their shared-memory inverse permutation (`shared_inv_row` / `shared_inv_expert`) are
untouched and packing adds no shared memory, so the two are independent.

One bug was found and fixed while porting: the empty-slot write must **not** be restricted to
inactive experts. An *active* expert whose id sits above `d_active` writes its descriptor at
`rank < d_active` and must still empty slot `expert`; on the old base this was invisible because
`num_groups` was 40 and no active expert id lands below 40 at W4, but at W16 (`num_groups = 160`)
it would have left a stale problem shape. Now every thread with `expert >= d_active` empties its
own slot regardless of activity.

### Measured (drift-cancelled, 3 alternating rounds x 5 points)

| | GEMM1 | GEMM2 | prologue | **total** | **ms/step** |
|---|--:|--:|--:|--:|--:|
| **W4 (T=4, D=28)** | −0.72 | −1.52 | +0.10 | **−2.55 ±0.14 us/call** | **−0.125** |
| **W16 (T=16, D=69)** | −0.61 | −0.05 | +0.14 | **−1.32 ±0.11 us/call** | **−0.083** |

Rounds: W4 −2.76 / −2.79 / −2.30; W16 −1.34 / −1.20 / −1.39. Correctness: **bit-identical** to
the G2 state in deterministic mode (finalize fusion off, no autotune cache), with the default
gate and with the gate forced on at every width; in the production config `max_abs = 2.441e-04`,
the same value as the off-vs-off control.

### The gate default changed from 4 to 1

On the A0+A3 base the rank computation cost ~0.6 us on a short latency-bound prologue, which the
3.2x list shortening at W16 did not repay — hence `MIN_RATIO = 4`, which declined at W16. With
G2-2's row blocks in the same launch the prologue is 7.5 us (W4) / 11.4 us (W16) instead of
6.3 / 6.5, and the rank hides inside it (+0.10 / +0.14 us). So on this base packing wins at both
widths and the default is now 1; `FLASHINFER_MOE_PACK_MIN_RATIO=4` restores the old behaviour.

**Combined W16 result is the interesting one**: packing was worth nothing at W16 before G2-2 and
is worth −0.083 ms/step after it. The two patches are not merely additive — G2-2 pays for the
part of packing that had been eating its own benefit.
