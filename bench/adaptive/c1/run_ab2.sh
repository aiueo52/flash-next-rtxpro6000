#!/bin/bash
# Every arm runs even if an earlier one fails, but any failure makes the exit status non-zero.
set -uo pipefail
D=~/tools/flash-next-bench/adaptive/c1
rm -f ~/tools/flash-next-bench/runs/c1ab2-*.jsonl
FAILED=()
arm() { "$D/ab_run.sh" "$@" 2>&1 | tail -40 || FAILED+=("$1 (rc ${PIPESTATUS[0]})"); }
echo "=== C1 A/B round 2 start $(date +%F' '%T)"
arm c1ab2-ema  ~/tools/flash-next-bench/adaptive/w16_3_15.json ema
arm c1ab2-conf ~/tools/flash-next-bench/adaptive/w16_conf.json confidence
if [ ${#FAILED[@]} -gt 0 ]; then echo "=== C1 A/B round 2 FAILED arms: ${FAILED[*]} $(date +%F' '%T)"; exit 1; fi
echo "=== C1 A/B round 2 done $(date +%F' '%T)"
