# Patches

Two layers of changes, both applied on top of pinned upstream versions:

| directory | applies to | what |
|---|---|---|
| `sglang/` | [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) @ `16e5682aad6e3335f38ec8d5711176278f12a086` (branch `pennyroyal-main-sm120-final`, 2026-08-27) | 105 `git format-patch` patches (`series/`), plus `perf-v1-core.diff` = the same changes squashed and limited to `python/`, `test/`, `docs/` |
| `sglang/mtp-dump/` | branch `flash-next-fast` after the series above | 9 patches adding the training-data dump hook used by `mtp-train/`; `apply.sh` puts them on branch `flash-next-mtp-dump` in a separate worktree. See `docs/train-your-own-mtp-head.md` §4 |
| `flashinfer/` | the installed `flashinfer-python==0.6.17` wheel (in the SGLang venv's `site-packages`) | 6 CUDA/C++ source patches to the CUTLASS fused-MoE path (JIT-compiled) + 1 patch to the vendored GDN WY decode wrapper |

## SGLang series

```bash
git clone https://github.com/jpezzulli/sglang-rtxpro6000
patches/sglang/apply.sh sglang-rtxpro6000        # creates branch flash-next-fast and git-am's the series
```

- The series is the linearised first-parent history of the production branch; one side-branch
  merge (FP8 HC mix weights + fused shared-expert gate_up) is folded into a single patch whose
  commit message lists the side-branch commits. The resulting tree is byte-identical to the
  measured production tree, except for publication edits: home-directory paths inside a few
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
  needed at runtime. `perf-v1-core.diff` excludes them.
- New unit tests are under `test/registered/unit/…` and `test/srt/…`; most CPU tests run with
  `CUDA_VISIBLE_DEVICES=9 python -m pytest …` (a non-existent GPU index; an empty value trips a
  string-indexing path in SGLang's test utils).

## FlashInfer patches

```bash
patches/flashinfer/apply.sh "$SGLANG_DIR/.venv/lib/python3.12/site-packages"
```

The script checks that the three target files are pristine 0.6.17 (sha256 from the wheel's
`RECORD`), keeps `.orig` copies, breaks hardlinks (uv installs wheels as hardlinks, so an in-place
edit would silently patch every other venv sharing the file), then applies `01`–`07` in order.
Verified: the chain on a pristine 0.6.17 wheel reproduces the measured production files byte for byte.
The next MoE call triggers a JIT rebuild (~2 minutes).

| patch | id | effect (measured, per step unless stated) | numerics |
|---|---|---|---|
| `01-a0-fused-routing-prologue` | A0 | enables FlashInfer's one-kernel routing prologue for top-k = 10 / 512 experts (replaces a 3-kernel prologue; uses `BlockRadixRankMatch`, 3.4 µs vs 15 µs for the stock ranker at 10-bit digits) | bit-identical routing |
| `02-a3-fuse-compute-strides` | A3 | folds `computeStridesTmaWarpSpecialized` into that prologue. A0+A3: bookkeeping W4 0.57 → 0.41 ms, W16 0.68 → 0.45 ms; kernels/step W4 2017 → 1864, W16 2728 → 2539 | bit-identical |
| `03-g2-1-fold-finalize-memset` | G2-1 | zeroing of the fused-finalize output moved into `doActivationKernel` (one launch less per MoE call) | bit-identical (deterministic mode) |
| `04-g2-2-fold-expand-rows` | G2-2 | `expandInputRowsKernel` (permute + NVFP4 re-quantise) folded into the prologue. G2-1+G2-2: −3.9 µs per MoE call = −0.19 ms (W4) / −0.25 ms (W16) | bit-identical (deterministic mode) |
| `05-g1-pack-only` | G1 | packs the active experts into CUTLASS groups `[0, D)` and shortens the group list 512 → rows (opt-in: `FLASHINFER_MOE_PACK_GROUPS=1`): −2.55 µs/call W4 (−0.125 ms), −1.32 µs/call W16 (−0.083 ms) | bit-identical (deterministic mode) |
| `06-p2-prune-in-prologue` | P2 | evaluates the singleton-route prune (see docs) inside the prologue, keeping its overlap with the previous kernels. Off unless `FLASHINFER_MOE_PRUNE_SINGLETON_TAU` > 0 (exported by SGLang when `SGLANG_MOE_PRUNE_IN_PROLOGUE=1`) | **changes the model's computation** (drops routes) |
| `07-gdn-wy-T16-strided-qkv` | glue-E 1 | GDN verify reads q/k/v strided at T = 16 instead of three `.contiguous()` copies per layer (`FLASHINFER_GDN_WY_STRIDED_QKV=1`): −200 µs W16, −221 µs W4 | bit-identical |

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
