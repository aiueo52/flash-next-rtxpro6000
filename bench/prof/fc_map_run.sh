#!/bin/bash
# fc_map_run.sh <label> <profile> -- FC_fc-map §5 for one profile, unattended: start the capped holder
# unit (fc_hold.sh takes the GPU lock itself), wait for the server, record clocks, run fc_measure.sh,
# release, wait for STOPPED. State files stay in prof/logs/hold-<label>-<profile>/.
set -uo pipefail
L="$1"; P="$2"; BENCH=~/tools/flash-next-bench; cd $BENCH
ST=prof/logs/hold-$L-$P
echo "[fc-map $P] start $(date +%T)"
systemd-run --user --collect --unit=fcmap-hold-$L-$P -p MemoryMax=110G -p MemorySwapMax=0 -p WorkingDirectory=$PWD \
    --setenv=HOME=$HOME --setenv=PATH=$PATH bash prof/fc_hold.sh $L $P || { echo "[fc-map $P] holder did not start"; exit 1; }
for i in $(seq 1 720); do   # up to 60 min: lock wait + 8-9 min start-up
  { [ -f $ST/READY ] || [ -f $ST/DIED ] || [ -f $ST/NOMEM ] || [ -f $ST/TIMEOUT ]; } && break
  sleep 5
done
rc=1
if [ -f $ST/READY ]; then
  echo "[fc-map $P] server up $(date +%T)"
  nvidia-smi --query-gpu=power.limit,clocks.mem,clocks.sm --format=csv,noheader > $ST/CLOCKS_AT_READY
  bash prof/fc_measure.sh $L $P; rc=$?
  echo "[fc-map $P] measure rc=$rc $(date +%T)"
else
  echo "[fc-map $P] no server: $(ls $ST)"
fi
touch $ST/DONE
for i in $(seq 1 60); do [ -f $ST/STOPPED ] && break; sleep 3; done
echo "[fc-map $P] released $(date +%T); gpu mem after stop: $(cat $ST/STOPPED 2>/dev/null)"
exit $rc
