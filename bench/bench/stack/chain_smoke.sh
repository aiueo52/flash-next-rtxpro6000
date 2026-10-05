#!/bin/bash
# chain_smoke.sh <pid>... -- wait for the given processes (the RT1 and DG1 smokes), then run the stacked smoke
# only if both single smokes were clean. Everything on, plus SGLANG_SPEC_MIN_P=1 to exercise min_p in SV1.
cd ~/tools/flash-next-bench
echo "[chain-smoke] start $(date +%T); waiting for $*"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
echo "[chain-smoke] smokes done $(date +%T)"
R=runs/rt1/smoke-rt.log; D=runs/dg1/smoke-dg.out
grep -q "error lines in server log: 0$" $R && grep -q "_router_softmax_fast32_kernel: n=[1-9]" $R \
  && grep -q "_router_triton_kernel: n=0 " $R || { echo "[chain-smoke] RT1 smoke not clean; stop"; exit 1; }
grep -q "\[smoke-dg\] CLEAN" $D || { echo "[chain-smoke] DG1 smoke not clean; stop"; exit 1; }
sleep 30
SERVER_ENV="SGLANG_SPEC_MIN_P=1" bash bench/stack/smoke_stack.sh
