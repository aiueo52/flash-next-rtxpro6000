#!/bin/bash
# abba_after_queue.sh -- waits for the night queue ("[queue] done" in runs/queue-1001-eve.log), then runs
# bench/rs2d/abba_rs2d.sh with the caller's B_ENV, logging nvidia-smi pmon (display-load covariate) alongside.
#   B_ENV="..." bash bench/rs2d/abba_after_queue.sh > runs/rs2d-abba/launcher.log 2>&1
set -uo pipefail
cd ~/tools/flash-next-bench
: "${B_ENV:?set B_ENV}"
OUT=runs/rs2d-abba; mkdir -p $OUT
echo "[rs2d-launch] waiting for the night queue $(date +%T); B_ENV='$B_ENV'"
timeout 28800 bash -c 'until LC_ALL=C /usr/bin/grep -q "^\[queue\] done" runs/queue-1001-eve.log; do sleep 10; done' ||
  { echo "[rs2d-launch] queue not done after 8 h; not started $(date +%T)"; exit 1; }
echo "[rs2d-launch] queue done; ABBA starts $(date +%T)"
timeout 21600 nvidia-smi pmon -s u -d 5 -o DT > $OUT/pmon.log 2>&1 < /dev/null &
PMON=$!
# Covariates for the drift seen in the RS2 ABBA (same code, verify span +35% from 19:10 to 23:00)
Q=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,clocks_event_reasons.active
timeout 21600 nvidia-smi --query-gpu=$Q,clocks_event_reasons_counters.sw_power_cap --format=csv -l 5 \
    > $OUT/clocks.csv 2>&1 < /dev/null &
CLOCKS=$!
B_ENV="$B_ENV" bash bench/rs2d/abba_rs2d.sh > $OUT/abba.log 2>&1
echo "[rs2d-launch] abba rc=$? $(date +%T)"
kill $PMON $CLOCKS 2>/dev/null
true
