#!/bin/bash
# chain_exact_w4.sh <pid>... -- after the given processes (the DT1 arm C1) exit: DT1's in-server exactness check in w4,
# X1 (DT off), Y1 (DT on), X2 (DT off), each its own GPU-lock hold (bench/stack/exact_arm.sh). X1 == X2 shows the
# check can see a difference; Y1 == X1 then shows DT1 keeps every greedy token.
cd ~/tools/flash-next-bench
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
echo "[chain-exact] start $(date +%T); waiting for $*"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 10; done; done
for arm in X1:0 Y1:1 X2:0; do
  sleep 30
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
    bash bench/stack/exact_arm.sh ${arm%%:*} ${arm##*:} || echo "[chain-exact] arm $arm rc=$?"
done
E=runs/stack8/exact
$PY bench/stack/exact_client.py --compare $E/X1.jsonl $E/X2.jsonl $E/Y1.jsonl
echo "[chain-exact] done $(date +%T)"
