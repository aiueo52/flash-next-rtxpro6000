#!/bin/bash
# chain_rq.sh <variant, e.g. rq2u2h or rq1> -- after chain4 (FG1): smoke of the RQ build in the server (or reuse
# a clean runs/rq1/smoke-<variant>.log), then the server ABBA production vs that build (bench/rq1/abba_rq1.sh,
# FC_fc-moe 1d) if the smoke is clean.
cd ~/tools/flash-next-bench
B=${1:?variant}
mkdir -p runs/rq1
C4=$(ps -eo pid=,args= | grep "[c]hain4\.sh" | awk '{print $1}' | head -1)
echo "[chain-rq] start $(date +%T); B=$B; waiting for chain4 pid=${C4:-none}"
while [ -n "$C4" ] && kill -0 "$C4" 2>/dev/null; do sleep 20; done
echo "[chain-rq] chain4 done $(date +%T)"
sleep 30
# the arm's JIT cache: rq1 runs from ~/.cache/sglang-p2, an RQ2 build from ~/.cache/sglang-<variant>
case $B in rq1) CN=sglang-p2 ;; *) CN=sglang-$B ;; esac
SM=runs/rq1/smoke-$B.log
clean() { grep -q "error lines in server log: 0" $SM && grep -q "mapped: .*/$CN/" $SM; }
if [ -s $SM ] && clean; then
  echo "[chain-rq] reusing the clean smoke $SM"
else
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=runs/rq1 bash bench/rq1/arm_rq1.sh smoke-$B $B 0 > $SM 2>&1
  echo "[chain-rq] smoke done $(date +%T)"
fi
grep -E "server up|mapped|error lines|died|not ready|grid" $SM
if clean; then
  sleep 30
  echo "[chain-rq] abba start $(date +%T)"
  B=$B bash bench/rq1/abba_rq1.sh > runs/rq1/abba-$B.log 2>&1
  echo "[chain-rq] abba done $(date +%T)"
  tail -40 runs/rq1/abba-$B.log
else
  echo "[chain-rq] smoke not clean; ABBA not started"
fi
