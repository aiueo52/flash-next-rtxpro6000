#!/bin/bash
# Apply the production FlashInfer 0.6.17 patch stack (01..10) to an installed flashinfer-python wheel.
#
#   patches/flashinfer/apply.sh <site-packages dir of the SGLang venv> [last patch number, default 10]
#
# 01-07 = the 2026-09-08 measured stack; 08 (RQ2 expand-row staging), 09 (FLASHINFER_P2_NO_NINJA) and
# 10 (RQ2h default on) were added for the 2026-10-02 production build. Pass 07 as the second argument
# for the September stack, 09 for the exact 2026-10-02 production sources (whose shipped fused-MoE module
# was compiled outside the JIT with -DG2_EXPAND_SF_HOIST=1; 10 only makes the default JIT build do the
# same, see ../README.md).
#
# - Verifies the four target files are pristine 0.6.17 (sha256 from the wheel's RECORD) before patching.
# - Keeps <file>.orig copies and breaks hardlinks first: `uv` installs wheels by hardlinking, so an in-place
#   edit would otherwise also change every other venv (and the uv cache) that shares the inode.
# - The next MoE call rebuilds FlashInfer's JIT module (~2 minutes; SGLang keeps FlashInfer's JIT cache
#   under its own cache dir, e.g. ~/.cache/sglang/.cache/flashinfer/<ver>/120f/).
# Verified: 01-07 on a pristine 0.6.17 wheel reproduce the 2026-09-08 measured files byte for byte, and
# 01-09 reproduce the 2026-10-02 production files byte for byte (01-10: plus the one-line default).
set -euo pipefail
SP="${1:?usage: $0 <site-packages dir> [last patch number]}"
LAST="${2:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$SP"
FILES=(
  flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh
  flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h
  flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py
  flashinfer/jit/core.py
)
REC=$(ls -d flashinfer_python-0.6.17.dist-info 2>/dev/null || true)
[ -n "$REC" ] || { echo "flashinfer_python-0.6.17.dist-info not found in $SP (other versions are untested)" >&2; exit 1; }
for f in "${FILES[@]}"; do
  want=$(grep "^$f," "$REC/RECORD" | cut -d, -f2)
  have="sha256=$(python3 -c 'import base64,hashlib,sys;print(base64.urlsafe_b64encode(hashlib.sha256(open(sys.argv[1],"rb").read()).digest()).rstrip(b"=").decode())' "$f")"
  [ "$want" = "$have" ] || { echo "not pristine: $f (already patched?)" >&2; exit 1; }
done
for f in "${FILES[@]}"; do
  cp -p "$f" "$f.orig"
  cp -p "$f" "$f.tmp.$$" && mv "$f.tmp.$$" "$f"   # new inode: breaks any hardlink before editing
done
for p in "$HERE"/[0-9][0-9]-*.patch; do
  n=$(basename "$p"); [ "$((10#${n%%-*}))" -le "$((10#$LAST))" ] || continue
  echo "applying $n"
  patch -p1 --no-backup-if-mismatch < "$p"
done
echo "done. Restore with: for f in ${FILES[*]}; do mv \$f.orig \$f; done"
