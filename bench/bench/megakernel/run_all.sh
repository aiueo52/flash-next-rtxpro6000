#!/bin/bash
# K2 megakernel feasibility: run every microbenchmark under the shared GPU lock.
# The GPU is shared with validation/training agents, so each bench takes the lock
# for its own (short) run rather than holding it across the suite.
# Every bench runs even if an earlier one fails; failures are listed and make the exit status non-zero.
set -uo pipefail
FAILED=()
cd "$(dirname "$0")/../.."
. bench/megakernel/env.sh
mkdir -p bench/megakernel/results bench/megakernel/logs
LOCK=${GPU_LOCK:-$HOME/.gpu.lock}
R=${ROUNDS:-5}
for b in bench_barrier bench_gemv_stage bench_hc_island; do
  echo "=== $b  $(date -Is) ==="
  flock -w 14400 "$LOCK" timeout -k 20 1500 "$PY" -u "bench/megakernel/$b.py" --rounds "$R" \
      2>&1 | tee "bench/megakernel/logs/$b.log"
  rc=${PIPESTATUS[0]}
  echo "=== $b done rc=$rc $(date -Is) ==="
  [ "$rc" -eq 0 ] || FAILED+=("$b (rc $rc)")
done
if [ ${#FAILED[@]} -gt 0 ]; then
  # a failed bench may not have written its results/*.json; a file there can be from an earlier run
  echo "=== FAILED: ${FAILED[*]} -- their results/ files may be stale or missing; do not use them"; exit 1
fi
