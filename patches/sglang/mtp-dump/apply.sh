#!/bin/bash
# Apply the MTP training-data dump hook (9 patches) on top of the flash-next-fast branch.
#
#   patches/sglang/mtp-dump/apply.sh <sglang-rtxpro6000 clone with branch flash-next-fast> <worktree dir>
#
# Creates branch `flash-next-mtp-dump` from `flash-next-fast` in a new git worktree at <worktree dir>
# and `git am`s the patches there. The serving checkout itself is not touched: it stays on its
# branch and keeps serving. The dump server reuses the serving checkout's .venv and imports the
# worktree's patched files through PYTHONPATH=<worktree dir>/python; mtp-train/parity/serve.sh does
# this when SGLANG_DUMP_TREE=<worktree dir>. The hook is inert unless SGLANG_MTP_DUMP_DIR (target-side
# dump) or SGLANG_MTP_DRAFT_DUMP_DIR (draft parity dump) is set; see docs/train-your-own-mtp-head.md.
set -euo pipefail
REPO="${1:?usage: $0 <sglang-rtxpro6000 clone> <worktree dir>}"
TREE="${2:?usage: $0 <sglang-rtxpro6000 clone> <worktree dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
git -C "$REPO" rev-parse --verify -q flash-next-fast >/dev/null || { echo "branch flash-next-fast not found; run patches/sglang/apply.sh first" >&2; exit 1; }
git -C "$REPO" worktree add -b flash-next-mtp-dump "$TREE" flash-next-fast
git -C "$TREE" am --whitespace=nowarn "$HERE"/0*.patch
echo "applied $(ls "$HERE"/0*.patch | wc -l) patches on flash-next-fast in $TREE"
