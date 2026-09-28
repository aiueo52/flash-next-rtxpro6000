# A8 feasibility: fuse `silu(g)*u` + NVFP4 quant into the sm120 grouped-GEMM1 epilogue

Date: 2026-09-05. Author: A8 agent. All GPU numbers below are **measured from the existing
production traces** (`prof/traces/final0905-*`), no new server was started for this verdict.
Source reading is against the private copy `~/tools/flashinfer-a8/flashinfer` (a `cp -a` of the
production venv package, i.e. FlashInfer 0.6.17 **with A0+A3 already applied**).

---

## Verdict: **NO-GO on A8 as specified.** Technically possible, but clearly > 2 days.

> And -- measured, section 4 -- **the cheap alternative does not exist either**: making
> `doActivationKernel` faster recovers 0.2-0.5 us of its 3.6 us, because the rest is launch
> overhead, not work. Only deleting the launch (i.e. A8 itself) recovers the 0.19-0.28 ms/step.

It is not blocked by a missing CUTLASS capability — the pieces exist. It is blocked by the
*amount* of new, bespoke CUTLASS EVT code required, which is much larger than the A0/A3 patches,
because of one fact that was not known when the A8 row was written:

> **GEMM1 runs with `swap_ab = true`.** The gate/up pairing is therefore along the CUTLASS **M**
> axis, not N — the opposite of the assumption in the task brief ("per-thread register fragments
> hold contiguous column pairs").

Evidence (two independent):
* The production autotune caches (`~/.cache/sglang/flashinfer/autotune/0.6.17/sm120/*/rank_tp0_pp0_dp0.json`)
  select **tactic 16 / 17** for `trtllm::fused_moe::gemm1`. For GEMM1 `supports_finalize_fusion`
  is `false` (`moe_kernels.h:655` — finalize is GEMM2-only), so the TMA-WS tactic list is
  `10 base configs` + `10 identical configs with swap_ab=true`
  (`moe_gemm_template_dispatch.h:631-668`). Indices 16 and 17 are in the second half →
  `swap_ab = true`, tiles `all_tiles[6] = CtaShape128x32x128B` and `all_tiles[7] = CtaShape128x32x64B`
  (`cutlass_heuristic.cpp:622-630`).
* `MOE_SMALLM_SPEC.md` §8.3 independently established the same from a GPU sweep
  ("with `swap_ab`, which puts the ≤16 tokens on the 32-wide N axis").

So the CUTLASS problem for GEMM1 is `M = 1280` (gate_up rows), `N = num_rows` (tokens ≤ 160),
`K = 2560`, CTA tile `128 x 32`, `grid = [1,188,1]` persistent.

---

## 1. What the sm120 TMA-WS epilogue *can* do (the positives)

1. **A custom EVT node that does its own non-TMA global stores is a proven, in-production pattern
   on exactly this kernel.** `ScaledAccPerRowBiasPerColScaleScatter` /
   `Sm90ScatterPtrArray` (`cutlass_extensions/epilogue/fusion/sm90_visitor_scatter.hpp`, 757 lines)
   is instantiated with `ElementD = void` in the CollectiveBuilder
   (`moe_gemm_tma_ws_launcher.inl:~420-440`) so the collective epilogue performs no D store at all,
   and the EVT node writes through `UniversalCopy` / red-global atoms at addresses it computes
   itself. `generate_kernels.py:777-778` emits `epilogue_fusion_finalize` for sm120, and this fork
   runs it in production (see `patches/README.md`, `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE`).
   **Therefore `ElementD = void` + a fully custom store node compiles and runs on sm_120.**
2. **CUTLASS ships sm120 NVFP4-output-with-scale-factor visitors**:
   `cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp` —
   `Sm120BlockScaleFactorRowStore` (SF vector along N) and `Sm120BlockScaleFactorColStore`
   (SF vector along M), wired by `sm120_callbacks_tma_warpspecialized.hpp` for the
   `LinCombBlockScaleFactor` / `LinCombEltActBlockScaleFactor` /
   `LinCombPerColBiasEltAct*BlockScaleFactor` operations (`operations.hpp:505-545`).
   These are the reference for the amax → UE4M3 → reciprocal → convert recipe and for the
   cross-lane reduction.
3. **The register layout is documented and workable.** `Sm120BlockScaleFactorColStore::reduce()`
   states it exactly: `FragmentSize == 4 == RowsPerThreadAccFrag(2) * ColsPerThreadAccFrag(2)`,
   i.e. a thread holds `(row r, col c), (row r, col c+1), (row r+8, col c), (row r+8, col c+1)`
   of a 16x8 MMA tile. Under `swap_ab`, "row" = gate_up index and "col" = token.
   * Adjacent M rows `2j`, `2j+1` live in lanes that differ by `lane ^ 4` (quad = lane/4), so a
     single `__shfl_xor_sync(mask, v, 4)` pairs them. Alternatively an interleave that puts the
     partner at `M + 8` or `M + 16` makes the pair thread-local with zero shuffles.
     Neither is a blocker.
4. `EpilogueFusion::GATED_ACTIVATION` already exists in the enum (`moe_gemm_kernels.h`), and the
   dispatch only needs a third `case` next to `FINALIZE`/`NONE`
   (`moe_gemm_template_dispatch.h:846-859` — today it falls into `TLLM_THROW("Unimplemented fusion")`).

## 2. Why it is still > 2 days (the actual blockers)

**B1 — the half-width store is a genuinely new EVT node, not a parameterisation.**
The output must be `N/2 = 640` FP4 values per token. A full-width store of duplicated values is
excluded (it doubles GEMM2's K traffic), and the "write into a 1280-wide buffer and give GEMM2 a
1280-element row stride with K=640" trick does **not** work either: after any pairing interleave a
CTA tile owning M rows `[128m, 128m+128)` produces the 64 outputs `[64m, 64m+64)`, which is neither
contiguous with a 128-tile boundary nor expressible as a stride. So the node must:
* consume `FragmentSize = 4` accumulators and emit 2 values (2:1 compression along the *contiguous*
  direction of D, since under `swap_ab` D is column-major with M contiguous),
* re-derive its own global addresses (the standard `tiled_copy` partitioning no longer describes
  the output), including FP4 sub-byte packing — two 4-bit values per thread, i.e. **1 byte per
  thread per store**, with the neighbouring thread owning the other nibble pair,
* and run its own cross-lane amax reduction over 16 output values that come from 32 accumulator M
  rows, i.e. a different reduction footprint from either CUTLASS node.
`Sm90ScatterPtrArray` is 300 lines for a *pure index remap* with no compression and no sub-byte
packing. The gated node is strictly harder.

**B2 — the scale-factor layout is TRT-LLM's, not CUTLASS's.**
`doActivationKernel` writes `fc2_act_sf_flat` via
`getOffsetActivationSF(expert_id, num_tokens_before_expert, num_cols, NVFP4)` +
`cvt_quant_get_sf_out_offset<..., QuantizationSFLayout::SWIZZLED_128x4>(token - num_tokens_before_expert, elem_idx, ...)`
(`cutlass_fused_moe_kernels.cuh:1054-1105`, `2535`, `2558`). That is a **per-expert-based**
128x4-swizzled layout keyed on a runtime `expert_first_token_offset[]` read, not
`Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(problem_shape)` which the CUTLASS sm120 nodes
use. So the CUTLASS SF store cannot be reused as-is; its addressing has to be replaced with the
TRT-LLM helpers, and `expert_first_token_offset` has to be threaded into the epilogue arguments.
There is also the K-dim SF zero-padding loop (`MinKDimAlignmentNVFP4`) that the epilogue would have
to reproduce (here `inter_size = 640`, `640 % 64 == 0`, so it happens to be a no-op — but that is a
model-specific accident that would have to be asserted, not assumed).

**B3 — numerics must match `cvt_warp_fp16_to_fp4` bit-for-bit-ish.** The production path goes
through `PackedVec<__nv_bfloat16>` + `cvt_warp_fp16_to_fp4<..., DISABLE_FP4_QUANT_FAST_MATH,
NVFP4_4OVER6_CONFIG>`. Reproducing the same SF quantisation (amax/6 · global_scale → e4m3 → mufu.rcp
→ scale → `cvt.rn.satfinite.e2m1x2`) from FP32 accumulators, including the fast-math and 4-over-6
variants, is a further correctness surface.

**B4 — plumbing fan-out.** A new fusion value has to be threaded through:
`TmaWarpSpecializedGroupedGemmInput` (new pointer fields for `fc2_act_sf_flat`,
`fc2_act_global_scale`, `expert_first_token_offset`), `setupTmaWarpSpecializedInputs`, the
`select_function` switch, `moe_gemm_template_dispatch_tma_ws.h`, the ~250-line
`INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM` macro in `moe_gemm_tma_ws_launcher.inl`,
`moe_tma_warp_specialized_traits.h`, and `jit/gemm/cutlass/generate_kernels.py`
(`generate_sm120_grouped_gemm_operations`, +8-32 new TUs → a **full** fused_moe JIT module rebuild,
not the ~95 s incremental relink that A0/A3 enjoyed), plus the tactic-list construction, the
autotune cache key, and the SGLang-side w13 row + w13-block-scale interleave in
`modelopt_quant.py`.

**B5 — the prize is smaller than the risk.** Measured, not estimated (below): 0.19 ms/step at W4
and 0.23-0.28 ms/step at W16, i.e. **1.6 % / 1.2 %** of the step. And the fused epilogue would not
recover all of it — some of the work moves into GEMM1.

Honest estimate: **4-8 days** of CUTLASS-fluent work with 5-15 min compile cycles, to win ~1.5 %.

---

## 3. What was measured (this is the important part)

`prof/gridstats.py <trace> doActivation 20` on the four production traces:

| trace | grid | n/step | median dur | sum/step |
|---|---|--:|--:|--:|
| final0905-w4-code-edit | [40,1,1] (verify) | 49.0 | **3.6 µs** | 179.4 µs |
| " | [10,1,1] (draft) | 2.0 | 3.6 µs | 7.2 µs |
| final0905-w4-prose-en | [40] / [10] | 49 / 2 | 3.6 / 3.6 µs | 185.8 µs total |
| final0905-w16-code-edit | [160] / [10] | 49 / 14 | 3.8 / 3.6 µs | 277.0 µs total |
| final0905-w16-prose-en | [160] / [10] | 49 / 14 | 3.7 / 3.6 µs | 305.2 µs total |

**The duration is 3.6 µs whether the grid is 10 blocks or 160 blocks, and whether the width is 4 or
16.** The kernel moves ~467 KB at W16 (read 160x1280 BF16, write 160x640 FP4 + 6.4 KB SF); at the
1500 GB/s this box sustains that is **0.31 µs of memory work**. The kernel is running at ~8 % of
its roofline and its cost is **pure dependent-latency, flat in the amount of work**.

The MoE chain really is serial — from a `final0905-w16-code-edit` timeline slice (single stream,
gap = start minus previous end):

```
GEMM1               dur=165.60   gap=-0.51
doActivationKernel  dur=  3.58   gap=+0.22   grid=[160,1,1] blk=[256,1,1]
memset32            dur=  0.96   gap=+0.13
GEMM2               dur= 84.58   gap=+0.13
```

so the 3.6 µs is genuine wall time, not PDL overlap.

### Where the 3.6 µs goes (read from the source, `cutlass_fused_moe_kernels.cuh:2402-2570`)

1. `cudaGridDependencySynchronize()` is called **at the very top**, before anything else (line ~2444).
   Everything after it is serialised behind GEMM1's completion — including work that does not depend
   on GEMM1's output at all.
2. Every one of the 256 threads then runs
   `findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node=512, token+1)` — a
   **binary search, ~10 dependent global loads**, only to recover the expert id. The source itself
   carries the comment `// TODO this is almost certainly faster as a linear scan`. It is unavoidable
   for this model because the `if (bias_ptr || IsNVFP4 || ...)` guard is always true under NVFP4.
3. Then `fc2_act_global_scale[expert]`, `expert_first_token_offset[expert]`,
   `expert_first_token_offset[num_experts_per_node]` — 3 more dependent loads.
4. Then the actual `gemm_result` load (the only one that truly depends on GEMM1) and the store.
5. Occupancy is awful: **1 block per token, 256 threads, of which only 640/8 = 80 are active**
   (`ACTIVATION_THREADS_PER_BLOCK = 256`, `CVT_ELTS_PER_THREAD = 8`, `inter_size = 640`). At W4 the
   grid is 40 blocks on 188 SMs. There is nothing to hide the latency chain with.

A chain of ~14 dependent global accesses at L2 latency (~0.13 µs each at 2.3 GHz) plus launch ramp
accounts for the observed 3.6 µs almost exactly.

---

## 4. The cheap alternative was built and measured -- it does NOT work

The verdict above originally proposed "A8-lite": stop `doActivationKernel` being latency-bound
instead of deleting it. **That was implemented and measured, and it is not a win.** The result is
worth recording because it changes what A8 is worth.

### What was built

In the private FlashInfer copy (`~/tools/flashinfer-a8`, never the production venv), behind
`FLASHINFER_MOE_FAST_ACTIVATION` (0 = upstream exactly, 1 / 2 = variants), two ideas:

* **mode 1** -- stage the 4 KB `expert_first_token_offset` array in shared memory with one
  coalesced pass, then run the *unchanged* binary search against shared memory. Bit-identical by
  construction; kills ~10 dependent L2 probes per row.
* **mode 2** -- mode 1 plus deferring `cudaGridDependencySynchronize()` from the top of the kernel
  to just before the first `gemm_result` read, so the expert lookup and the per-expert scale loads
  issue while GEMM1 is still draining.

(An earlier variant that replaced the binary search with a block-parallel scan + a shared
`atomicMax` was **slower** -- 256 threads contending on one shared address serialises. Discarded.)

The diff is kept at `specs/A8_rejected_fast_activation.patch` for reproduction. It is **not** in
`bench/moe_smallm/patches/` because it should not be applied.

### What was measured

`bench/moe_smallm/bench_moe.py --mode profile --widths 4,16 --sweep-distinct 28 --rotation 16`
with the production autotune cache, one process per mode, 5 alternating rounds,
`--min-seconds 2.0`, GPU otherwise idle, SM clock 2295-2325 MHz in every run.

`us_per_call_sustained` (CUDA-graph wall time per MoE call, the low-noise metric):

| mode | T=4 | T=16 |
|---|--:|--:|
| 0 (upstream) | 83.32 83.40 83.33 83.36 (85.92) | 83.69 84.03 83.36 83.46 (86.48) |
| 1 (smem offsets) | 82.86 83.02 82.56 82.90 (85.71) | 83.53 83.60 83.33 83.32 (85.94) |
| **delta (4 clean rounds)** | **-0.52 us/call** | **-0.19 us/call** |

(The 5th round is parenthesised: both modes jumped together, i.e. an external perturbation.)
An earlier 3-round set including mode 2 gave mode 2 - mode 1 = +0.01 (T=4) / +0.43 (T=16) us, so
deferring the PDL wait buys nothing -- consistent with the trace, where `doActivation` starts
**after** GEMM1 ends (gap +0.22 us) because GEMM1's 188 persistent CTAs hold every SM until the
end, so there is no window for the block to run early.

`doActivationKernel`'s own CUPTI self time stayed at 3.4-4.0 us in every mode; its run-to-run
spread (3.0-4.9 us) is larger than any effect being chased.

**-0.5 / -0.2 us per call is 0.03 / 0.01 ms per step. That is nothing.**

### What that proves (the useful part)

The ~3.6 us is **not** the expert search and **not** the PDL placement: removing both changes
almost nothing. It is the irreducible cost of *having a launch in the chain*. For calibration, the
`memset32` in the same chain -- a kernel that does essentially nothing -- measures **0.96-0.98 us**,
and `doActivation` measures the same 3.6 us at grid=10 as at grid=160.

So there is no cheap way to shrink `doActivationKernel`. **The 0.19 ms/step (W4) / 0.23-0.28
ms/step (W16) can only be had by deleting the launch, which is exactly and only what A8 does.**
That makes A8's estimated prize credible -- and it still costs 4-8 days for ~1.5 %.

## 5. Where the cheap money actually is (measured, not pursued)

Per-MoE-call medians from the production traces (`final0905-*`, verify phase) and from the
standalone bench at D=28:

| kernel | W4 trace | W16 trace | bench T=4 | ms/step W4 (x51) | ms/step W16 (x63) |
|---|--:|--:|--:|--:|--:|
| `fusedBuildExpertMapsSortFirstTokenAndStrides` (A0+A3) | 5.5 | 5.3 | 6.75 | 0.28 | 0.33 |
| `expandInputRowsKernel` | 4.0 | 4.2 | 4.55 | 0.20 | 0.26 |
| `doActivationKernel` | 3.6 | 3.8 | 3.70 | 0.18 | 0.24 |
| `memset32` | 1.0 | 1.0 | 0.98 | 0.05 | 0.06 |
| **glue total** | **14.1** | **14.3** | **15.98** | **0.72** | **0.90** |

Against a 66 us (W4) / 250 us (W16) two-GEMM cost, the glue is **18 % of the MoE call at W4**. Every
one of these four kernels is launch/latency-bound, not bandwidth-bound (the bench's near-null
`memset32` costs 0.98 us, so ~1 us of each is pure launch), and they are strictly serially
dependent. **The structural win is removing launches from the chain, not making any one of them
faster** -- A8 is one instance of that, and the cheapest remaining instance is probably folding
`memset32` (the fused-finalize output zeroing) into the `doActivation` launch, worth ~1 us/call
for a very small change. `expandInputRows` (4-5 us to permute and re-quantise <=160x2560 rows) is
the next-largest and is a bigger, separate question.

## 6. If A8 is ever revisited, this is the shape

* Keep `swap_ab = true` (the autotuner picks it and it is what makes the 32-wide token axis a
  single tile).
* Permute w13 rows so the partner of gate row `r` sits at `r + 8` **within each 16-row MMA group**,
  not at `r + 1`. Then the pair is thread-local -- a thread's `FragmentSize = 4` accumulator holds
  `(r, c), (r, c+1), (r+8, c), (r+8, c+1)`, so `silu(frg[0])*frg[2]` and `silu(frg[1])*frg[3]` need
  **zero shuffles**. (With the naive `r, r+1` interleave the partner is at `lane ^ 4`, which works
  but costs a shuffle and leaves half the lanes idle for the store.) The w13 **block scales** must
  be permuted with exactly the same row permutation -- weight SF is indexed per (N-row, K-block).
* New EVT node `Sm120GatedActNvfp4Store`: take `Sm90ScatterPtrArray`
  (`cutlass_extensions/epilogue/fusion/sm90_visitor_scatter.hpp`) as the model for doing its own
  non-TMA global stores with `ElementD = void` in the CollectiveBuilder, and
  `Sm120BlockScaleFactorColStore`
  (`cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp:514-760`) as the model for
  the amax -> UE4M3 -> `mufu.rcp` -> convert recipe and its cross-lane reduction.
* Address the scale factors with the TRT-LLM helpers, **not** CUTLASS's:
  `getOffsetActivationSF(expert, num_tokens_before_expert, num_cols, NVFP4)` +
  `cvt_quant_get_sf_out_offset<..., QuantizationSFLayout::SWIZZLED_128x4>`. That means threading
  `expert_first_token_offset`, `fc2_act_sf_flat` and `fc2_act_global_scale` into the epilogue
  arguments via `TmaWarpSpecializedGroupedGemmInput`.
* Restrict the candidate tactic list to `128x32x{64,128}B, swap_ab = true` while the fusion is on,
  and change the autotune cache key (the production caches under
  `~/.cache/sglang/flashinfer/autotune/0.6.17/sm120/*/` select tactics 16/17 by index; adding
  configs renumbers them, so a stale cache would silently select a different kernel).
* Numerics: the fused version drops the intermediate BF16 rounding that `doActivation` does today
  (it quantises straight from FP32 accumulators), so it is *more* accurate but **not** bit-identical.
  Keep the `cvt_warp_fp16_to_fp4` scale formula, including the `DISABLE_FP4_QUANT_FAST_MATH` and
  `NVFP4_4OVER6_CONFIG` variants, or the difference will not be explainable.

## 7. Reproducing section 4

Isolation used throughout (the production venv csrc was **never** edited):

```bash
mkdir -p ~/tools/flashinfer-a8
cp -a ~/tools/sglang-rtxpro6000/.venv/lib/python3.12/site-packages/flashinfer ~/tools/flashinfer-a8/
cp -a ~/.cache/sglang ~/.cache/sglang-a8            # pre-warm so only fused_moe rebuilds
source ~/tools/flash-next-bench/bench/moe_smallm/a8env.sh   # PATH/CUDA/PYTHONPATH/SGLANG_CACHE_DIR
python -c "import flashinfer; print(flashinfer.__file__)"    # must print the flashinfer-a8 copy
cd ~/tools/flashinfer-a8/flashinfer/data/csrc/fused_moe/cutlass_backend
cp -p cutlass_fused_moe_kernels.cuh cutlass_fused_moe_kernels.cuh.bak-a8lite
patch -p5 < ~/tools/flash-next-bench/specs/A8_rejected_fast_activation.patch
```

Then, GPU idle, one process per mode (the env is read once per process):

```bash
cd ~/tools/flash-next-bench
C=~/.cache/sglang/flashinfer/autotune/0.6.17/sm120/990c98ce8356ecfe/rank_tp0_pp0_dp0.json
for r in 1 2 3 4 5; do for m in 0 1; do
  FLASHINFER_MOE_FAST_ACTIVATION=$m flock -w 7200 ~/.gpu.lock \
    ~/tools/sglang-rtxpro6000/.venv/bin/python -m moe_smallm.bench_moe --mode profile \
      --widths 4,16 --sweep-distinct 28 --rotation 16 --min-seconds 2.0 --autotune-cache "$C" \
      > /tmp/c${m}_r${r}.log 2>&1
done; done
```

Read `us_per_call_sustained` out of the `{"profile": true, ...}` line -- that is the metric with
usable noise. Do **not** trust `doActivationKernel`'s CUPTI self time for a sub-microsecond effect;
its round-to-round spread is 3.0-4.9 us.

Note: the JIT rebuild after a `.cuh` edit is one TU plus a relink (~2 min here). The **first**
build against a fresh private `SGLANG_CACHE_DIR` rebuilds all 97 TUs (~10 min) because the include
paths change with the package location. Edit the `.cuh` only when no build is running -- a mid-build
edit produces a mixed-state compile (that happened once here: the kernel signature was picked up
but the launch site was not, giving a `cudaLaunchKernelEx ... cannot be called with the given
argument list` error that looks like a real bug and is not).
