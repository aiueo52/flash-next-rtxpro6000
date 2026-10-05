#!/bin/bash
# chain_stack8.sh <pid>... -- after the given processes (the RQ2 u2h ABBA chain) exit:
#   1. SV2's GPU unit test, FlashInfer and torch top-k (bench/sv/test_sparse_verify.py, one short lock hold);
#   2. the stacked smoke with SV2 (SMOKE_LABEL smoke-stack2; min_p on to exercise it in the sparse verify);
#   3. if both pass, the stack ABBA over 8 server starts (ABBA BAAB, PROMPT_LIMIT 4) into runs/stack8.
# One server start moves ms/step by 1-3% (the 4-start ABBAs could not resolve 1% changes), hence 8 starts.
cd ~/tools/flash-next-bench
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
echo "[chain-stack8] start $(date +%T); waiting for $*"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 10; done; done
sleep 30
echo "[chain-stack8] sv2 unit test, lock wait $(date +%T)"
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
  env TRITON_CACHE_DIR=$HOME/.cache/sv2-triton PY=$PY timeout 1500 bash -c '
    for T in fi torch; do
      PYTHONPATH=$HOME/tools/flashinfer-p3:$HOME/tools/sglang-sv2/python FLASHINFER_WORKSPACE_BASE=$HOME/.cache/sglang-rq2u2h \
        FLASHINFER_P2_NO_NINJA=1 DEV=cuda WT=$HOME/tools/sglang-sv2 TOPK=$T \
        $PY bench/sv/test_sparse_verify.py > runs/sv2/test-cuda-$T.log 2>&1
      echo "[chain-stack8] sv2 test TOPK=$T rc=$? $(date +%T)"
    done'
for T in fi torch; do
  grep -A12 "^D\. " runs/sv2/test-cuda-$T.log | sed "s/^/[$T] /"; tail -1 runs/sv2/test-cuda-$T.log
done
grep -q "RESULT: PASS" runs/sv2/test-cuda-fi.log || { echo "[chain-stack8] SV2 GPU test failed; stop"; exit 1; }
sleep 30
SMOKE_LABEL=smoke-stack2 SERVER_ENV="SGLANG_SPEC_MIN_P=1" bash bench/stack/smoke_stack.sh | tee runs/stack/smoke-stack2.out
grep -q "\[smoke-stack\] CLEAN" runs/stack/smoke-stack2.out || { echo "[chain-stack8] stacked smoke not clean; stop"; exit 1; }
sleep 30
mkdir -p runs/stack8
ORDER="A1:off B1:on B2:on A2:off B3:on A3:off A4:off B4:on" PROMPT_LIMIT=4 OUTDIR=runs/stack8 \
  bash bench/stack/abba_stack.sh > runs/stack8/abba-stack8.log 2>&1
echo "[chain-stack8] done $(date +%T)"
