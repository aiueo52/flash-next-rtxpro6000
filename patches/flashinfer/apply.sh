#!/bin/bash
# Apply the production FlashInfer 0.6.17 patch stack (01..07) to an installed flashinfer-python wheel.
#
#   patches/flashinfer/apply.sh <site-packages dir of the SGLang venv>
#
# - Verifies the three target files are pristine 0.6.17 (sha256 from the wheel's RECORD) before patching.
# - Keeps <file>.orig copies and breaks hardlinks first: `uv` installs wheels by hardlinking, so an in-place
#   edit would otherwise also change every other venv (and the uv cache) that shares the inode.
# - The next MoE call rebuilds FlashInfer's JIT module (~2 minutes; SGLang keeps FlashInfer's JIT cache
#   under its own cache dir, e.g. ~/.cache/sglang/.cache/flashinfer/<ver>/120f/).
# Verified: this chain on a pristine 0.6.17 wheel reproduces the measured production files byte for byte.
set -euo pipefail
SP="${1:?usage: $0 <site-packages dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$SP"
FILES=(
  flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh
  flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h
  flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py
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
for p in "$HERE"/0[1-7]-*.patch; do
  echo "applying $(basename "$p")"
  patch -p1 --no-backup-if-mismatch < "$p"
done
echo "done. Restore with: for f in ${FILES[*]}; do mv \$f.orig \$f; done"
