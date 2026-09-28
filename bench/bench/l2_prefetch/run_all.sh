#!/bin/bash
# L2-prefetch feasibility (Phase A): every bench under the shared GPU lock, one lock per bench.
# Every bench runs even if an earlier one fails; failures are listed and make the exit status non-zero.
set -uo pipefail
FAILED=()
cd "$(dirname "$0")/../.."
. bench/l2_prefetch/env.sh
mkdir -p bench/l2_prefetch/results bench/l2_prefetch/logs
LOCK=${GPU_LOCK:-$HOME/.gpu.lock}
for b in ${BENCHES:-bench_hit bench_overlap bench_pipeline}; do
  echo "=== $b  $(date -Is) ==="
  flock -w 28800 "$LOCK" timeout -k 20 2400 "$PY" -u "bench/l2_prefetch/$b.py" ${ARGS:-} \
      2>&1 | tee "bench/l2_prefetch/logs/$b.log"
  rc=${PIPESTATUS[0]}
  echo "=== $b done rc=$rc $(date -Is) ==="
  [ "$rc" -eq 0 ] || FAILED+=("$b (rc $rc)")
done
if [ ${#FAILED[@]} -gt 0 ]; then
  # a failed bench may not have written its results/*.json; a file there can be from an earlier run
  echo "=== FAILED: ${FAILED[*]} -- their results/ files may be stale or missing; do not use them"; exit 1
fi
