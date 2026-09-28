# FlashInfer csrc patches (MoE small-M work, 2026-09-05)

Applied **in place** inside the SGLang fork's venv
(`~/tools/sglang-rtxpro6000/.venv/lib/python3.12/site-packages/`).
Every patched file keeps a pristine `<file>.bak-a0` copy next to it.
Nothing under `python/sglang` is touched.

| patch | item | files | what it does |
|---|---|---|---|
| `a0-fused-routing-prologue.patch` | **A0** | `flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` | enables `fusedBuildExpertMapsSortFirstToken` for top-k = 10 and 512 experts (`expert_log == 10`), replacing the 3-kernel `threeStepBuildExpertMapsSortFirstToken` prologue with one kernel. |
| `a3-fuse-compute-strides.patch` | **A3** | same `.cuh` + `flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h` | folds `computeStridesTmaWarpSpecializedKernel` into that prologue, removing a second launch per MoE call. |

Apply in that order (A3's hunks are on top of A0's).

## Re-applying after a FlashInfer wheel upgrade

```bash
V=~/tools/sglang-rtxpro6000/.venv/lib/python3.12/site-packages
P=~/tools/flash-next-bench/bench/moe_smallm/patches
for f in flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh \
         flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h; do
  cp -p $V/$f $V/$f.bak-a0
done
cd $V && patch -p1 < $P/a0-fused-routing-prologue.patch \
       && patch -p1 < $P/a3-fuse-compute-strides.patch
```

The next `import flashinfer` + MoE call rebuilds the JIT module incrementally (ninja owns
freshness; no cache clearing needed):
`~/.cache/sglang/.cache/flashinfer/<ver>/120f/cached_ops/fused_moe_120/`.
Measured rebuild: **~95 s** for A0 (one TU, `cutlass_fused_moe_instantiation.cu`, + relink),
**~112 s** for A0+A3. Note SGLang redirects FlashInfer's cache under `SGLANG_CACHE_DIR`
(`~/.cache/sglang/.cache/flashinfer/...`), *not* `~/.cache/flashinfer`.

## Runtime switches

| env | default | effect |
|---|---|---|
| `FLASHINFER_MOE_FUSED_PROLOGUE=0` | on | back to the upstream 3-kernel prologue (disables A0 **and** A3) |
| `FLASHINFER_MOE_FUSED_STRIDES=0` | on | keeps `computeStridesTmaWarpSpecialized` as its own launch (disables A3 only) |

## Fail-safe gates (all fall back to the upstream path, none of them throws)

A0 (`fusedBuildExpertMapsSortFirstTokenBlockSize` / `...Dispatch`):
* `num_tokens > 256`;
* `sizeof(BlockRadixRank<BLOCK,LOG2>::TempStorage) >= cudaDevAttrMaxSharedMemoryPerBlockOptin`;
* `experts_per_token` not in `{1,2,4,6,8,10}`; `expert_log > 10`.

A3 (`tryFusedPrologueAndStrides`, plus the `try_fused_strides` gate in `runMoe`):
* everything A0 needs, plus `num_tokens > 32`, `num_experts_per_node + 1 >= 1024`;
* `use_w4_groupwise`, `use_lora`, `min_latency_mode`, or a WFP4AFP8 (`quant_params.fp8_mxfp4`)
  configuration — the last one because that path's scale memset has to stay after
  `expandInputRows`, which the A3 re-ordering would break;
* the GEMM config not being TMA-warp-specialized (then `setupTmaWarpSpecializedInputs` returns
  before the fused launch and `succeeded` stays false).
If the fused launch is declined, `runMoe` runs the standalone prologue and calls
`setupTmaWarpSpecializedInputs` in its original place, i.e. the upstream sequence exactly.

## Implementation notes

* **A0 uses `cub::BlockRadixRankMatch`, not `cub::BlockRadixRank`, for the >=10-bit digit
  space** (`FusedPrologueRadixRank`). The stock ranker's temp storage is
  `COUNTER_LANES x BLOCK_THREADS x PACKING_RATIO` digit counters = **65 696 B** at
  `BLOCK_SIZE=32 / RADIX_BITS=10`; zeroing and raking that with 32 threads measured
  **15.05 us/call**, i.e. *worse* than the 9.4 us 3-kernel path it replaces. The match-based
  ranker keeps one counter per (digit, warp) — **4 256 B** — and measured **3.36 us/call**.
  Configurations with `expert_log <= 9` keep the stock ranker bit-for-bit.
* **A3 re-runs the prologue in every block** (grid = `ceil(E/32)`, block = 32) and publishes
  `expert_first_token_offset` through shared memory, so the descriptor phase stays as parallel
  as the standalone kernel was and no grid-wide barrier is needed. Only block 0 writes the
  routing maps and the offsets to global memory.
* The strides body was factored out of `computeStridesTmaWarpSpecializedKernel` into
  `computeStridesTmaWarpSpecializedForExpert` so both kernels run identical code.

## Verification

`bench/moe_smallm/ab_prologue.py` runs the production path once per process and compares saved
outputs. Under `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 --autotune-cache none` (which removes the
GEMM2 finalize epilogue's non-deterministic scatter-add) the three paths — upstream 3-kernel,
A0, A0+A3 — are **bit-identical** at T = 1, 2, 4, 8, 16 with the real 512-expert layer.
With the production fused-finalize epilogue, T = 8/16 are still bit-identical and T <= 4 differ
by ~1e-4 absolute — the same envelope the *unpatched* path shows against itself run-to-run
(control included in the script's usage).

## Gotcha: the wheel's files are hardlinked across venvs

`uv` installs by hardlinking, so both csrc files started with **3 links**: the SGLang venv,
another project's venv, and `~/.cache/uv/archive-v0/<hash>/`. Editing in place patches all
three. After applying, break the links so only this fork is affected:

```bash
V=~/tools/sglang-rtxpro6000/.venv/lib/python3.12/site-packages/flashinfer/data/csrc
for f in fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh \
         nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h; do
  F=$V/$f; MT=$(stat -c '%y' $F); cp $F /tmp/p.$$
  cat $F.bak-a0 > $F          # restore pristine through the shared inode
  rm $F && cp /tmp/p.$$ $F    # fresh inode, link broken
  touch -d "$MT" $F           # keep ninja from rebuilding
done
```

(Done for the 2026-09-05 install: both files now have `links=1`, the other project's venv and the uv
cache are back to pristine.)

---

# G2: two more launches out of the MoE chain (2026-09-05)

Built and measured in the private copy `~/tools/flashinfer-g2` with
`SGLANG_CACHE_DIR=~/.cache/sglang-g2`.  **Not applied to the production venv** — apply
with the recipe below after A0 and A3.

| patch | item | files | what it does |
|---|---|---|---|
| `g2-1-fold-finalize-memset.patch` | **G2-1** | `cutlass_fused_moe_kernels.cuh` + `moe_kernels.h` | `doActivationKernel` zeroes the fused-finalize output buffer in its own grid, so `gemm2` skips the `cudaMemsetAsync`.  One launch (`memset32`, ~1.0 µs) leaves the serial chain. |
| `g2-2-fold-expand-rows.patch` | **G2-2** | same two files | folds `expandInputRowsKernel` (permute + NVFP4 re-quantise of the routed rows) into the A0/A3 prologue kernel.  One launch (~4.5 µs) leaves the chain. |

Apply order: `a0`, `a3`, `g2-1`, `g2-2`.

## Runtime switches

| env | default | effect |
|---|---|---|
| `FLASHINFER_MOE_FOLD_MEMSET=0` | on | keeps the fused-finalize `cudaMemsetAsync` as its own launch (disables G2-1) |
| `FLASHINFER_MOE_FOLD_EXPAND=0` | on | keeps `expandInputRowsKernel` as its own launch (disables G2-2) |

## Fail-safe gates

G2-1 (`runMoe`, next to the `Self::gemm1` call) — the zeroing is only handed to
`doActivationKernel` when *all* of:
* `foldMemsetEnabled()`, `!min_latency_mode`, `!use_lora`, `!use_w4afp8`,
  no DeepSeek block-scale runner;
* GEMM2 actually uses the `FINALIZE` epilogue fusion (otherwise there is no memset to fold);
* GEMM1 is TMA-warp-specialized and the activation is gated (i.e. `doActivationKernel` runs);
* the byte count is a multiple of 16 and `final_output` is 16-byte aligned.
Otherwise `gemm2` issues the memset exactly where upstream does.  The zeroing is placed
*before* `cudaGridDependencySynchronize()` in `doActivationKernel` (it depends on nothing
GEMM1 produces) and is ordered ahead of GEMM2 by the kernel's existing
`cudaTriggerProgrammaticLaunchCompletion()`, i.e. the same stream position the memset had.

G2-2 (`tryFusedPrologueAndStrides`) — everything A0+A3 need, plus:
* `use_fp4` with a BF16/FP16 unpermuted input (`InputType`), NVFP4 FC1 block scaling,
  a non-null `fc1_fp4_act_scale_`, no pre-quantised `input_sf`;
* `!use_awq`, `!use_w4afp8`, `!use_lora`, `!use_wfp4afp8`, `!use_mxfp8_act_scaling`;
* `hidden_size % (8 * 32) == 0` and `hidden_size / 8 / 32 <= 16`
  (the per-thread staging array; 2560 → 10 chunks);
* the default NVFP4 quantiser tags, i.e. `TRTLLM_NVFP4_USE_4_OVER_6` and
  `TRTLLM_DISABLE_FP4_QUANT_FAST_MATH` unset — the kernel hard-codes
  `<false, std::false_type>` so only one instantiation is emitted.
When the offer is declined, `FusedPrologueRequest::expand_fused` stays false and `runMoe`
launches `expandInputRowsKernelLauncher` in its usual place.

## Implementation notes

* The A3 kernel already re-runs the (cheap) rank in **every** block, so every block holds the
  routing in registers.  G2-2 publishes the *inverse* permutation and the per-permuted-row
  expert id in shared memory (`BLOCK_SIZE * EXPERTS_PER_TOKEN` ints each, 2.5 KB at 32×10), so
  a row block needs neither a global round trip for `permuted_row_to_unpermuted_row` nor the
  `findTotalEltsLessThanTarget` binary search that `expandInputRowsKernel` pays per row.
* The grid becomes `ceil(E/32) + num_tokens*k` blocks of 32 threads: blocks
  `[0, ceil(E/32))` do the TMA descriptors, blocks `[ceil(E/32), ...)` do one row each, so
  the two phases overlap instead of running back to back.
* One warp per row means 10 iterations of 16 B per thread.  Issued as a plain loop those are
  10 *dependent* memory round trips; the row is therefore staged into a fixed-size register
  array first (`G2_EXPAND_MAX_CHUNKS = 16`) so all the loads are in flight at once.
* `hidden_size / 8 % 32 == 0` is required so the trip count is warp-uniform: the NVFP4
  quantiser reduces amax across `lane ^ 1` with a **full** `__shfl_xor_sync` mask.
  With `elem_index = threadIdx.x + 32*chunk` the pairs (lane, lane^1) cover the same
  16-element scale-factor group as the upstream 256-thread layout, so the scale factors and
  the quantised values are bit-identical.

---

## G1 — `g1-pack-only.patch` (grouped-GEMM problem-list packing) — **this is the one to apply**

**Apply order: `a0`, `a3`, `g2-1`, `g2-2`, `g1-pack-only`.**  Verified: that chain applied to a
pristine wheel reproduces `~/tools/flashinfer-g1b` byte for byte.  Built and measured there
with `SGLANG_CACHE_DIR=~/.cache/sglang-g1b`; **not applied to the production venv**, and
`~/tools/flashinfer-g2` was not modified.  Two files, 364 diff lines, no CUTLASS change.

`g1-pack-splitk.patch` below is the earlier version: same packing plus GEMM1 split-K, on the
A0+A3 base only.  Split-K measured as a wash and is not shipped, so that patch is kept only as
the record of the experiment — **it does not apply on top of G2** (10/32 hunks fail).

### What it does

The A0/A3 prologue already holds every expert's token offset in shared memory (the design
re-runs the rank in every block), so each descriptor block warp-computes `rank(e)` — the number
of active experts below `e` — and `D`, then writes expert `e`'s CUTLASS group at index `rank(e)`
instead of `e`.  Slots past the packed range get a zero-token problem shape, each written by
exactly one thread.  The group *count* also drops from `num_experts_per_node` (512) to
`expanded_num_rows` (40 at W4, 160 at W16), a valid host-side bound because an active expert
owns at least one permuted row.

The target is the persistent tile scheduler's linear group scan
(`sm90_tile_scheduler_group.hpp: get_work_idx_m_and_n`, 32 problem shapes per iteration of
dependent global loads) — part of the per-call ramp identified in `specs/G1_LOG.md` §5.
Everything CUTLASS indexes per group moves from `expert` to the group index: alpha scales, the
SF pointers *and* layouts, `ptr_act/weight/d`, and the FINALIZE epilogue's source-token /
router-scale / bias arrays (`sm90_visitor_scatter.hpp:268` reads `ptr_index[l]`; the launcher
gives alpha and router scales an `int64_t` L-stride of 1).

### How it composes with G2-2

G2-2 splits the prologue grid into `ceil(E/32)` descriptor blocks plus `num_tokens*k` row
blocks.  Packing lives **entirely inside the descriptor blocks**, which are exactly the blocks
whose `expert = blockIdx.x*32 + threadIdx.x` covers `[0, E)` once; the row blocks and their
shared-memory inverse permutation (`shared_inv_row` / `shared_inv_expert`) are untouched, and
packing adds no shared memory.  **No relocation was needed.**  The composition is in fact
*better* than on the A0+A3 base: there, the rank computation added ~0.6 µs to a short
latency-bound prologue; with G2-2's row blocks in the same launch it hides completely
(prologue +0.10 µs at W4, +0.14 µs at W16), which is why the `MIN_RATIO` gate now defaults to 1
instead of 4.

### Runtime switches

| env | default | effect |
|---|---|---|
| `FLASHINFER_MOE_PACK_GROUPS=1` | **off** | enable packing |
| `FLASHINFER_MOE_PACK_MIN_RATIO` | 1 | require `expanded_num_rows * ratio <= num_experts_per_node` before packing engages.  4 restores the gate that was needed on the A0+A3 base. |

### Gates (each falls back to the upstream layout, none throws)

Inherits the A0/A3 prologue's gates (`num_tokens <= 32`, `experts_per_token` in
`{1,2,4,6,8,10}`, `num_experts+1 < 1024`, no w4-groupwise / LoRA / min-latency / WFP4AFP8),
plus `swap_ab` on, `fc1_out_size % cta_tile_m == 0` with the tile M read from the chosen sm120
tactic (anything but 128/256 disables it), and the ratio above.  If the fused prologue declines
at runtime the standalone strides kernel runs with `out_idx == expert`, i.e. upstream exactly.

### Verification (probe 10/11, `runs/g1/probe1{0,1}.sh`)

* **Deterministic mode** (`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`, `--autotune-cache none`):
  **bit-identical** to the G2 state, all 12 tensors, T = 1/4/16 — with the default gate and with
  `MIN_RATIO=1` forcing it on at every width.
* **Production config** (fused finalize on, tactics pinned): `max_abs = 2.441e-04`, the *same*
  value as the off-vs-off control — i.e. entirely the finalize epilogue's own order dependence.

### Measured on the G2 base (drift-cancelled: 3 alternating rounds x 5 points)

| | GEMM1 | GEMM2 | prologue | **total** | **ms/step** |
|---|--:|--:|--:|--:|--:|
| **W4 (T=4, D=28)** | −0.72 | −1.52 | +0.10 | **−2.55 ±0.14 µs/call** | **−0.125** (49 calls) |
| **W16 (T=16, D=69)** | −0.61 | −0.05 | +0.14 | **−1.32 ±0.11 µs/call** | **−0.083** (63 calls) |

Round-by-round: W4 −2.76 / −2.79 / −2.30, W16 −1.34 / −1.20 / −1.39.

W4 reproduces the pre-G2 measurement (−2.35 µs/call) exactly, confirming the two changes are
independent.  W16 goes from "a wash" on the A0+A3 base to −1.32 µs/call here, purely because
G2-2 absorbs the prologue cost.

---

## P2 — `p2-prune-in-prologue.patch` (singleton-route pruning inside the fused prologue)

**Apply order: `a0`, `a3`, `g2-1`, `g2-2`, `g1-pack-only`, `p2-prune-in-prologue`.**  Verified: that
chain applied to the production files reproduces `~/tools/flashinfer-p2` byte for byte.  Built and
measured there with `SGLANG_CACHE_DIR=~/.cache/sglang-p2`; **not applied to the production venv**.
Two files, no CUTLASS change.  Log: `specs/P2_LOG.md`.

### What it does

SGLang's P1 (`python/sglang/srt/layers/moe/prune_singleton.py`) drops a route when it is the only
route to its expert within the call, its routing weight is `< tau`, and it is not the row's top-1
(`P1_LOG.md`).  As a separate Triton launch in front of the routing prologue it costs that prologue
its ~54 % overlap with the preceding kernels (exclusive 265 -> 462 us/step at W16), which eats the
GEMM saving.  This patch evaluates the same rule inside
`fusedBuildExpertMapsSortFirstTokenAndStridesKernel`, which already holds the top-k ids in
registers: a block-local histogram over `shared_offsets` (dead until after `RankKeys`) gives the
pre-prune per-expert count, the within-row rank is a k x k register compare with P1's tie rule,
and a dropped route is re-keyed to the `num_experts_per_node` ("not on this node") bucket
**before** the radix rank -- so the permutation, the offsets, the G1 packing and the G2-2 row
expansion all see the pruned set with no further change.  Block 0 writes the dropped routes back
as id -1 / weight 0 (P1's convention) so the unfused finalize kernels skip them.

### Runtime switches

| env | default | effect |
|---|---|---|
| `FLASHINFER_MOE_PRUNE_SINGLETON_TAU` | **0 = off** (upstream byte-identical) | weight threshold; the SGLang hook exports it from `SGLANG_MOE_PRUNE_SINGLETON_TAU` when `SGLANG_MOE_PRUNE_IN_PROLOGUE=1` |
| `FLASHINFER_MOE_PRUNE_MIN_RANK` | 1 | lowest within-row rank eligible (clamped to >= 1) |
| `FLASHINFER_MOE_PRUNE_MIN_ROWS` / `MAX_ROWS` | 2 / 64 | batch sizes touched (the fused prologue itself stops at 32 rows) |

### Gates (fall back to "not pruned", never throw)

Everything the A0/A3 fused prologue needs (`num_tokens <= 32`, top-k in `{1,2,4,6,8,10}`, no
w4-groupwise / LoRA / min-latency / WFP4AFP8, TMA-warp-specialized tactic), plus `ep_size == 1`
(the -1 write-back is per-rank state), a non-null `token_final_scales`, `min_rank < top_k`, and
`MIN_ROWS <= num_tokens <= MAX_ROWS`.  When the fused prologue declines, the standalone prologue
runs on the unpruned routing.

## G1 (superseded) — `g1-pack-splitk.patch` (packing + GEMM1 split-K, A0+A3 base only)

**Superseded by `g1-pack-only.patch`; does not apply on top of G2.** Kept as the record of the
split-K experiment. Developed and measured in the private copy `~/tools/flashinfer-g1` on top of
A0+A3 only. Two files:
`flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` and
`flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h`.
Apply after `a0-fused-routing-prologue.patch` and `a3-fuse-compute-strides.patch`.

### Runtime switches (all default off = byte-for-byte upstream)

| env | effect |
|---|---|
| `FLASHINFER_MOE_PACK_GROUPS=1` | write the D active experts into CUTLASS groups `[0, D)` and shorten the group list from `num_experts_per_node` (512) to `expanded_num_rows` (40 at W4 / 160 at W16) |
| `FLASHINFER_MOE_PACK_MIN_RATIO` | how much the list must shorten for packing to engage (default 4; `expanded_num_rows * ratio <= num_experts_per_node`).  This is what makes it a no-op at W16, where the 3.2x shortening does not cover the fixed ~0.6 us of extra prologue. |
| `FLASHINFER_MOE_SPLITK_GEMM1=1` | additionally split GEMM1's K into `s` slices emitted as extra groups (`=n` caps `s` at n); implies packing |
| `FLASHINFER_MOE_SPLITK_RAMP_PCT` | the planner's per-wave ramp cost as a % of a full-K tile's streaming time (default 25; 0 = pure wave-count model) |

### Gates (each falls back to the upstream layout, none throws)

Everything is inside the A0/A3 fused prologue, so it inherits its gates
(`num_tokens <= 32`, `experts_per_token` in `{1,2,4,6,8,10}`, `num_experts+1 < 1024`, no
w4-groupwise / LoRA / min-latency / WFP4AFP8), plus:
* `swap_ab` must be on and `fc1_out_size % cta_tile_m == 0` (the tile M is read from the chosen
  sm120 tactic; anything but 128/256 disables it);
* split-K additionally needs NVFP4 weights, `hidden_size % 256 == 0`, and the split-K workspace
  (only allocated when the env var was set at process start).
If the fused prologue declines at runtime, `runMoe` passes null partial buffers to
`doActivation`, so the reduction is skipped and nothing stale is read.

### Why it needs no CUTLASS change

The sm120 array collective rebuilds each group's TMA descriptor dims *and* strides from
`dA[g] / dB[g] / layout_SFA[g] / layout_SFB[g]` plus the group's problem shape
(`cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`,
`tensormaps_replace_global_tensor_properties`). A K slice with a full-K stride and a permuted
group order are therefore ordinary inputs. The one trap is the NVFP4 block-scale layout, which
is K-tile-fastest (`tile_to_shape(SfAtom, (MN,K,L), Step<_2,_1,_3>)`) so a K range is not a
contiguous sub-buffer; the patch takes the packed layout for the slice K and overwrites the
single MN-tile stride `cute::get<0,1>(layout.stride())` with the full-K value, then offsets the
pointer by `k0 * Blk_MN / SFVecSize` bytes. Verified exhaustively on the host for s = 1..5 over
K = 2560, including uneven splits.

### Verification

`bench/moe_smallm/ab_prologue.py` (see `runs/g1/probe4.sh`):
* with `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0` (which removes the epilogue's order-dependent
  scatter-add) **packing must be bit-identical** to upstream;
* with the production fused-finalize epilogue, packing changes the scatter-add order, so it is
  compared against a same-config off-vs-off control;
* split-K is never bit-identical (BF16 partials, FP32 reduction) and is compared with a
  tolerance against the same control.

### Measured effect (specs/G1_LOG.md sections 6-8)

| | W4 (T=4) | W16 (T=16) |
|---|--:|--:|
| **pack** (drift-cancelled, isolated, 3-4 alternating rounds) | **−2.35 us/call = −0.115 ms/step** | gate declines; forced, −0.07 us/call |
| pack, in-server robust MoE GEMM | −0.100 / −0.155 ms/step (code-edit / prose-en) | no measurable change |
| pack + split-K | −0.60 us/call | **+2.32 us/call (worse)** |

**Split-K is off by default and should stay off.**  It does what it was designed to do --
GEMM1 −1.14 us at W4, because a tail wave with twice the CTAs streams faster -- but the
reduction costs +1.5 us in the prologue and +0.8 us in `doActivationKernel`, both of which are
pure dependent latency, so the call is a wash at W4 and 2.3 us worse at W16 (where the planner
picks s=1 and the only cost left is *loading one device scalar* in the activation kernel).

Only `FLASHINFER_MOE_PACK_GROUPS=1` is worth applying, and only because it is bit-identical.

## Measured (2026-09-05, private copy, details in `specs/G2_LOG.md`)

Per MoE call, medians over the 1020-1260 MoE calls in four `prof/validate.sh` server runs
(one off/on pair per width, in both orders):

| | folds off | folds on |
|---|--:|--:|
| routing prologue | 5.4-5.6 | 6.6-6.7 |
| `expandInputRowsKernel` | 4.1-4.3 | *(deleted)* |
| `doActivationKernel` | 3.6-3.7 | 3.6-3.8 |
| `memset32` | 0.96-0.99 | *(deleted)* |
| **glue** | **14.16-14.34** | **10.34-10.43** |
| GEMM1 + GEMM2 | 66.3-66.6 (W4) / 115-118 (W16) | 66.7 / 116-120 |

**-3.90 us per MoE call**, GEMMs untouched.  Launches per step: W4 1698 -> 1596 (-102),
W16 2337 -> 2211 (-126) -- exactly `2 x` the MoE calls per step.  That is **-0.19 ms/step (W4)**
and **-0.25 ms/step (W16)**.  In the isolated bench (`us_per_call_sustained`, 4 alternating
rounds, D = 28): 74.92 -> 70.05 (T=4) and 74.91 -> 70.50 (T=16); G2-1 alone -2.22 / -1.91,
G2-2 alone -2.84 / -2.55.

Numerics: with `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 --autotune-cache none` (the cache **must**
go with the flag -- the production cache indexes tactics positionally and gives
`Invalid gemm2 profile id: 57` otherwise) all combinations are **bit-identical** to the control.
With the production fused-finalize epilogue every combination sits inside the control's own
run-to-run envelope (max_abs 2.441e-04, same n_diff), which is the only statement available
there because that epilogue is non-deterministic against itself.
`MODE=full`: acceptance W4 ~3.9 / ~2.5 / ~3.2, W16 ~9 / ~3.0 / ~4.9, needle PASS at both widths.

## P4 contribution-aware pruning candidate (2026-09-08; NOT production enabled)

`p4-contrib-prune.patch` applies after the production chain `a0 -> a3 -> g2-1 -> g2-2 -> g1-pack-only -> p2` from the venv's `site-packages` directory. It changes `flashinfer/fused_moe/core.py`, the CUTLASS typed FFI binding, `QuantParams`/`FusedPrologueRequest`, and the fused routing prologue. Python callers pass an FP32 layer table, row inverse residual norms, threshold, joint flag, and separate effective ids/weights outputs. All CTAs read immutable originals; unfused finalize switches to the effective outputs only after the prologue launch. No P2 in-place race argument is used for the contribution policy.

The matching uncommitted SGLang integration is in `$HOME/tools/sglang-p4`, branch `codex/p4-contrib-prune`; private package `$HOME/tools/flashinfer-p4`, JIT cache `~/.cache/sglang-p4`. `bench/moe_smallm/p4env.sh` configures the overlay. Set `SGLANG_MOE_PRUNE_POLICY=contrib`, `SGLANG_MOE_PRUNE_NORM_MANIFEST` to `prune/p4-manifest.json`, optional `SGLANG_MOE_PRUNE_CONTRIB_W4/W16`, and `SGLANG_MOE_PRUNE_CONTRIB_JOINT=0/1`. Unset policy preserves P2 tau behavior. This candidate needs the ship gates in `specs/P4_CONTRIB_PRUNE_SHIP.md`; patch presence is not shipping approval.

P4 final verdict: **NO-SHIP**. All four W4/W16 traced step-time cells miss the >=3% improvement gate, and seven of eight acceptance cells exceed +/-1%; conditional quality was not run. Keep this patch experimental and do not append it to the production chain. Full numbers and preservation audit: `specs/P4_CONTRIB_PRUNE_SHIP.md`.
