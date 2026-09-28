#!/bin/bash
# Q1 arms, one flock acquisition per arm so other GPU users can interleave.
# A failed non-prod arm does not stop the remaining arms, but every failure is reported in a summary and
# makes the exit status non-zero; a failed prod arm (the comparison base) stops the sequence.
# Each invocation gets a fresh run id (runs/CURRENT_RUN). run_arm.sh writes runs/<arm>/DONE with that id
# only after a fully successful arm, and analyze.py / write_report.py aggregate only arms whose DONE
# carries the current id, so failed, interrupted or older arms are never mixed into the report.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); LOCK=${GPU_LOCK:-$HOME/.gpu.lock}
LEGACY="SGLANG_MOE_PRUNE_SINGLETON_TAU=0 SGLANG_TRITON_PDL=0 SGLANG_NORM_INTO_GEMV=0 SGLANG_HC_LAYER_APPLY_FUSED=0 SGLANG_HC_APPLY_MIX_FUSED=0 SGLANG_SHARED_GATE_EARLY=0 SGLANG_HC_GATE_EARLY=0 SGLANG_GDN_CONV_CHAIN_PARALLEL=0 SGLANG_GDN_PROJ_DIRECT_LAYOUT=0 SGLANG_GDN_AB_STASH_DIRECT=0 SGLANG_ADAPTIVE_POLICY=ema"
export Q1_RUN_ID="$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$HERE/runs"
printf '%s\n' "$Q1_RUN_ID" > "$HERE/runs/CURRENT_RUN.tmp" && mv "$HERE/runs/CURRENT_RUN.tmp" "$HERE/runs/CURRENT_RUN" \
  || { echo "cannot write runs/CURRENT_RUN" >&2; exit 1; }
echo "=== run $Q1_RUN_ID $(date +%T)"
ARMLIST=(${ARMS:-prod noprune legacy prod2})
OK=(); FAILED=(); NOTRUN=()
done_ok() { grep -qF "\"run_id\": \"$Q1_RUN_ID\"" "$HERE/runs/$1/DONE" 2>/dev/null; }
for i in "${!ARMLIST[@]}"; do
  arm=${ARMLIST[$i]}
  case $arm in
    prod|prod2) E="Q1_ARM=$arm" ;;
    noprune) E="SGLANG_MOE_PRUNE_SINGLETON_TAU=0" ;;
    legacy) E="$LEGACY" ;;
    *) echo "=== $arm: unknown arm"; FAILED+=("$arm (unknown arm)"); continue ;;
  esac
  echo "=== $arm waiting for lock $(date +%T)"
  flock -w 28800 "$LOCK" "$HERE/run_arm.sh" $arm $E; rc=$?
  if [ $rc -eq 0 ] && done_ok "$arm"; then
    OK+=("$arm")
  else
    [ $rc -eq 0 ] && why="exited 0 but no DONE marker for this run" || why="rc $rc"
    echo "=== $arm FAILED ($why)"; FAILED+=("$arm ($why)")
    if [ "$arm" = prod ]; then NOTRUN=("${ARMLIST[@]:$((i + 1))}"); break; fi
  fi
done
echo "=== run $Q1_RUN_ID finished $(date +%T)"
echo "    complete: ${OK[*]:-none}"
if [ ${#FAILED[@]} -gt 0 ] || [ ${#NOTRUN[@]} -gt 0 ]; then
  for f in "${FAILED[@]}"; do echo "    FAILED:   $f"; done
  [ ${#NOTRUN[@]} -gt 0 ] && echo "    not run (prod failed): ${NOTRUN[*]}"
  echo "=== $((${#FAILED[@]} + ${#NOTRUN[@]})) arm(s) incomplete; analyze.py / write_report.py will use only: ${OK[*]:-none}"
  exit 1
fi
echo "=== all arms done"
