#!/bin/bash
# Both profiles run even if the first fails, but any failure makes the exit status non-zero.
set -uo pipefail
D=~/tools/flash-next-bench/adaptive/c1
FAILED=()
ref() { "$D/fixed_run.sh" "$1" 2>&1 | tail -12 || FAILED+=("$1 (rc ${PIPESTATUS[0]})"); }
ref w16
ref w4
if [ ${#FAILED[@]} -gt 0 ]; then echo "=== fixed refs FAILED: ${FAILED[*]} $(date +%F' '%T)"; exit 1; fi
echo "=== fixed refs done $(date +%F' '%T)"
