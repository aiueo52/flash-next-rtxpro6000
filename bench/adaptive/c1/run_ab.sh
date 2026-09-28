#!/bin/bash
# Every arm runs even if an earlier one fails, but any failure makes the exit status non-zero.
set -uo pipefail
D=~/tools/flash-next-bench/adaptive/c1
FAILED=()
arm() { "$D/ab_run.sh" "$@" 2>&1 | tail -40 || FAILED+=("$1 (rc ${PIPESTATUS[0]})"); }
echo "=== C1 A/B start $(date +%F' '%T)"
arm c1ab-ema  ~/tools/flash-next-bench/adaptive/w16_3_15.json ema
arm c1ab-conf ~/tools/flash-next-bench/adaptive/w16_conf.json confidence
if [ ${#FAILED[@]} -gt 0 ]; then echo "=== C1 A/B FAILED arms: ${FAILED[*]} $(date +%F' '%T)"; exit 1; fi
echo "=== C1 A/B done $(date +%F' '%T)"
