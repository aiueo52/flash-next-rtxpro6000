# Patches

Two layers of changes, both applied on top of pinned upstream versions:

| directory | applies to | what |
|---|---|---|
| `sglang/` | [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) @ `16e5682aad6e3335f38ec8d5711176278f12a086` (branch `pennyroyal-main-sm120-final`, 2026-08-27) | 121 `git format-patch` patches (`series/`), plus two squashed diffs limited to `python/`, `test/`, `docs/`: `perf-1002-core.diff` (all 121 = the 2026-10-02 production build) and `perf-v1-core.diff` (0001–0105 = the 2026-09-08 measured tree that the September numbers refer to) |
| `sglang/mtp-dump/` | branch `flash-next-fast` after the series above | 9 patches adding the training-data dump hook used by `mtp-train/`; `apply.sh` puts them on branch `flash-next-mtp-dump` in a separate worktree. See `docs/train-your-own-mtp-head.md` §4 |
| `flashinfer/` | the installed `flashinfer-python==0.6.17` wheel (in the SGLang venv's `site-packages`) | 8 CUDA/C++ source patches to the CUTLASS fused-MoE path (JIT-compiled), 1 patch to the vendored GDN WY decode wrapper, 1 patch to the JIT loader (`FLASHINFER_P2_NO_NINJA`) |

## SGLang series

```bash
git clone https://github.com/jpezzulli/sglang-rtxpro6000
patches/sglang/apply.sh sglang-rtxpro6000        # creates branch flash-next-fast and git-am's the series
```

- The series is the linearised first-parent history of the production branch; two side-branch
  merges (0040: FP8 HC mix weights + fused shared-expert gate_up; 0121: XA1 on ST1 into the RS
  package) are each folded into a single patch whose commit message lists the side-branch commits.
  After 0105 the tree is the 2026-09-08 measured tree; after 0121 it is the 2026-10-02 production
  tree. Both are byte-identical to the originals, except for publication edits (all in 0001–0105;
  0106–0121 are unedited): home-directory paths inside a few
  experiment scripts were genericised to `/home/user`, private session links and references to the
  author's other projects were removed or genericised in commit messages, docs and code comments
  (comments and docstrings in four `python/` files were reworded; the only executable change is the
  default NVFP4 weight-cache directory in `nvfp4_dense.py`, now `<SGLANG_CACHE_DIR>/nvfp4`), and the five
  `prof-pdl/greedy-*.json` result files added by 0074 keep only token counts and output hashes
  (the generated text was stripped). The development shell scripts that drove the `dbg/` and `prof-pdl/` experiments
  (`dbg/*.sh` in 0061–0066, `prof-pdl/gpu_session*.sh` in 0074) are left out: they managed the
  GPU of the development machine and are not needed to build or serve.
- `MODIFICATIONS.md` lists every upstream file the series (and the FlashInfer patches) modify, as the
  Apache-2.0 §4(b) change notice.
- `SERIES.md` lists every patch with its date and subject. Almost all speed features are behind
  environment flags that default to **off**; `launch/serve-fast.sh` turns on the adopted set.
  Rejected experiments stay in the series (flag-gated, default off) so their numbers can be
  reproduced; see `docs/rejected.md`.
- Besides `python/` and `test/`, the series also adds the experiment records we used while
  developing (`bench/…` micro-benchmarks, `dbg/`, `prof-pdl/`, `adaptive_*.json`). They are not
  needed at runtime. The two `perf-*-core.diff` files exclude them.
- New unit tests are under `test/registered/unit/…` and `test/srt/…`; most CPU tests run with
  `CUDA_VISIBLE_DEVICES=9 python -m pytest …` (a non-existent GPU index; an empty value trips a
  string-indexing path in SGLang's test utils).

## FlashInfer patches

```bash
patches/flashinfer/apply.sh "$SGLANG_DIR/.venv/lib/python3.12/site-packages"
```

The script checks that the four target files are pristine 0.6.17 (sha256 from the wheel's
`RECORD`), keeps `.orig` copies, breaks hardlinks (uv installs wheels as hardlinks, so an in-place
edit would silently patch every other venv sharing the file), then applies `01`–`10` in order (an
optional second argument stops earlier: `07` = the 2026-09-08 measured stack, `09` = the exact
2026-10-02 production sources). Verified on a pristine 0.6.17 wheel: `01`–`07` reproduce the
September files and `01`–`09` the 2026-10-02 production files byte for byte. The next MoE call
triggers a JIT rebuild (~2 minutes).

One caveat about the 2026-10-02 build: its fused-MoE module was not built by the JIT. The RQ2
variant was compiled by hand with `-DG2_EXPAND_QUANT_UNROLL=2 -DG2_EXPAND_SF_HOIST=1` ("u2h") into a
private JIT cache, and the server loads that cache with `FLASHINFER_P2_NO_NINJA=1` (patch 09: use a
cached module as-is, never run ninja; fail if it is missing). The production source keeps
`G2_EXPAND_SF_HOIST` at default 0, so a plain JIT build of `01`–`09` gives "u2" without the hoist.
Patch `10` flips that default so that the normal JIT build compiles the shipped variant. We did not
re-measure a JIT-built module against the hand-built one; the `-D` flags and sources are the same.

| patch | id | effect (measured, per step unless stated) | numerics |
|---|---|---|---|
| `01-a0-fused-routing-prologue` | A0 | enables FlashInfer's one-kernel routing prologue for top-k = 10 / 512 experts (replaces a 3-kernel prologue; uses `BlockRadixRankMatch`, 3.4 µs vs 15 µs for the stock ranker at 10-bit digits) | bit-identical routing |
| `02-a3-fuse-compute-strides` | A3 | folds `computeStridesTmaWarpSpecialized` into that prologue. A0+A3: bookkeeping W4 0.57 → 0.41 ms, W16 0.68 → 0.45 ms; kernels/step W4 2017 → 1864, W16 2728 → 2539 | bit-identical |
| `03-g2-1-fold-finalize-memset` | G2-1 | zeroing of the fused-finalize output moved into `doActivationKernel` (one launch less per MoE call) | bit-identical (deterministic mode) |
| `04-g2-2-fold-expand-rows` | G2-2 | `expandInputRowsKernel` (permute + NVFP4 re-quantise) folded into the prologue. G2-1+G2-2: −3.9 µs per MoE call = −0.19 ms (W4) / −0.25 ms (W16) | bit-identical (deterministic mode) |
| `05-g1-pack-only` | G1 | packs the active experts into CUTLASS groups `[0, D)` and shortens the group list 512 → rows (opt-in: `FLASHINFER_MOE_PACK_GROUPS=1`): −2.55 µs/call W4 (−0.125 ms), −1.32 µs/call W16 (−0.083 ms) | bit-identical (deterministic mode) |
| `06-p2-prune-in-prologue` | P2 | evaluates the singleton-route prune (see docs) inside the prologue, keeping its overlap with the previous kernels. Off unless `FLASHINFER_MOE_PRUNE_SINGLETON_TAU` > 0 (exported by SGLang when `SGLANG_MOE_PRUNE_IN_PROLOGUE=1`) | **changes the model's computation** (drops routes) |
| `07-gdn-wy-T16-strided-qkv` | glue-E 1 | GDN verify reads q/k/v strided at T = 16 instead of three `.contiguous()` copies per layer (`FLASHINFER_GDN_WY_STRIDED_QKV=1`): −200 µs W16, −221 µs W4 | bit-identical |
| `08-rq2-expand-cpasync-stage` | RQ2 | the expand row of the fused prologue (04) is staged in shared memory with `cp.async` instead of registers; the quantize loop runs `G2_EXPAND_QUANT_UNROLL` chunks per trip (default 2) and, with `G2_EXPAND_SF_HOIST=1`, computes the scale-factor address once per row. Microbench (u2h): prologue 7.5/8.4/12.5 → 6.1/7.0/8.2 µs warm and 10.9/12.5/12.3 → 6.8/7.9/7.8 µs cold at T = 1/4/16. 4-start server ABBA: in-server prologue 12.4 → 7.8 µs per call; ms/step at equal acceptance −3.4 % greedy, −3.3 % sampling at request level (no arm-level interval with 4 starts; the mechanism alone is about −1.6 to −2 %) | bit-identical by construction (same `cvt_warp_fp16_to_fp4` on the same values); output check within the production run-to-run envelope |
| `09-jit-no-ninja-frozen-cache` | — | `FLASHINFER_P2_NO_NINJA=1`: `JitSpec.build` loads an already compiled module as-is and raises `MissingJITCacheError` instead of running ninja. A JIT cache copied to another directory keeps the old absolute paths in `build.ninja`, so ninja would recompile every translation unit inside the server (this caused a host out-of-memory once) | no kernel change |
| `10-rq2h-sf-hoist-default-on` | RQ2h | `G2_EXPAND_SF_HOIST` default 0 → 1 (see the caveat above) | — |

Kill switches (default on after patching): `FLASHINFER_MOE_FUSED_PROLOGUE=0` (A0 and A3),
`FLASHINFER_MOE_FUSED_STRIDES=0` (A3), `FLASHINFER_MOE_FOLD_MEMSET=0`, `FLASHINFER_MOE_FOLD_EXPAND=0`.
Every patch has fail-safe gates that fall back to the upstream path (details in
`docs/lab-notes/MOE_SMALLM_SPEC.md`, `G1_LOG.md`, `G2_LOG.md`, `P2_LOG.md`).

"Deterministic mode" = `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 --autotune-cache none`. With the
production fused-finalize epilogue the MoE output is not bit-reproducible run to run even without
any patch (atomic scatter-add order), so the patched and unpatched paths were compared against
that same run-to-run envelope.

`experimental/` keeps patches that were measured and rejected: `g1-pack-splitk` (GEMM1 split-K,
a wash at W4 and +2.3 µs/call at W16), `p4-contrib-prune` (contribution-aware pruning, NO-SHIP),
`a8-rejected-fast-activation` (A8 lite). They are not part of the production chain.
