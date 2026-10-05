#!/bin/bash
# chain_late.sh <pid>... -- after the given processes (chain_rq and the smoke chain): the RT1 ABBA if its smoke
# was clean, then the DG1 ABBA if its smoke was clean. Each arm takes its own GPU-lock hold.
cd ~/tools/flash-next-bench
echo "[chain-late] start $(date +%T); waiting for $*"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
echo "[chain-late] predecessors done $(date +%T)"
R=runs/rt1/smoke-rt.log
if grep -q "error lines in server log: 0$" $R && grep -q "_router_softmax_fast32_kernel: n=[1-9]" $R \
   && grep -q "_router_triton_kernel: n=0 " $R; then
  sleep 30
  echo "[chain-late] rt1 abba start $(date +%T)"
  bash bench/rt1/abba_rt.sh > runs/rt1/abba-rt.log 2>&1
  echo "[chain-late] rt1 abba done $(date +%T)"
else
  echo "[chain-late] RT1 smoke not clean; RT1 ABBA skipped"
fi
if grep -q "\[smoke-dg\] CLEAN" runs/dg1/smoke-dg.out; then
  sleep 30
  echo "[chain-late] dg1 abba start $(date +%T)"
  bash bench/dg1/abba_dg.sh > runs/dg1/abba-dg.log 2>&1
  echo "[chain-late] dg1 abba done $(date +%T)"
else
  echo "[chain-late] DG1 smoke not clean; DG1 ABBA skipped"
fi
