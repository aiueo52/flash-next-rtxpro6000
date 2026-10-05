#!/bin/bash
# chain_st1.sh [pid...] -- after the given processes exit: ST1's GPU unit test (specs/ST1_SHARED_TAIL_2026-10-01.md
# 2 D) under the GPU lock, then, if it passed and NO_ABBA is unset, the 4-arm server A/B (bench/st1/abba_st1.sh).
cd ~/tools/flash-next-bench; mkdir -p runs/st1
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
echo "[chain-st1] start $(date +%T); waiting for ${*:-nothing}"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 10; done; done
sleep 30
flock -w 28800 ~/.gpu.lock env DEV=cuda TRITON_CACHE_DIR=$HOME/tools/sglang-st1/.cache/triton \
    $PY bench/st1/test_shared_tail.py > runs/st1/test-d.log 2>&1
rc=$?; tail -30 runs/st1/test-d.log; echo "[chain-st1] test D rc=$rc $(date +%T)"
[ $rc = 0 ] || { echo "[chain-st1] test failed; stop before the A/B"; exit 1; }
[ -n "${NO_ABBA:-}" ] && { echo "[chain-st1] NO_ABBA set; stop"; exit 0; }
bash bench/st1/abba_st1.sh > runs/st1/abba-st1.log 2>&1
tail -5 runs/st1/abba-st1.log
echo "[chain-st1] done $(date +%T)"
