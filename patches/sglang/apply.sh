#!/bin/bash
# Apply the 105-patch series to jpezzulli/sglang-rtxpro6000 at the base commit it was developed on.
#
#   patches/sglang/apply.sh <path to a clone of https://github.com/jpezzulli/sglang-rtxpro6000>
#
# Creates branch `flash-next-fast` at 16e5682aad (branch pennyroyal-main-sm120-final, 2026-08-27) and
# `git am`s the series. The resulting tree equals the measured production tree (codex/perf-v1 @ 7b4d539f9b)
# except for publication edits: genericised home paths and project names in scripts/docs, generated
# text stripped from prof-pdl/greedy-*.json, and the development shell scripts dbg/*.sh and
# prof-pdl/gpu_session*.sh left out (see patches/README.md). Under python/ and test/ only comments and
# docstrings in four files were reworded, and the default NVFP4 weight-cache directory in
# nvfp4_dense.py moved from a path in the author's tools tree to <SGLANG_CACHE_DIR>/nvfp4.
set -euo pipefail
REPO="${1:?usage: $0 <sglang-rtxpro6000 clone>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE=16e5682aad6e3335f38ec8d5711176278f12a086
cd "$REPO"
git cat-file -e "$BASE^{commit}" 2>/dev/null || { echo "base commit $BASE not found; git fetch origin first" >&2; exit 1; }
git checkout -b flash-next-fast "$BASE"
git am --whitespace=nowarn "$HERE"/series/*.patch
echo "applied $(ls "$HERE"/series/*.patch | wc -l) patches on $BASE"
