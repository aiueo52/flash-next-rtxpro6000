#!/bin/bash
# chain_rs2.sh [pid...] -- after the given processes exit: RS2's GPU checks (specs/RS2_SPARSE_RS_2026-10-01.md):
# unit tests A-E on CUDA (stop if A-D fail), F1 smoke + profile (S1, rs, probes only), then, unless NO_ABBA is set,
# the 8-start ABBA (to vs rs). Each step is its own GPU-lock hold. No greedy-text gate: w4 greedy text differs even
# between two runs of one config (specs/DT1_DRAFT_TAIL_2026-10-01.md, 16:55); exact_rs2.sh stays for a later fix.
cd ~/tools/flash-next-bench
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
echo "[chain-rs2] start $(date +%T); waiting for ${*:-nothing}"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 10; done; done
sleep 30
mkdir -p runs/rs2
# FlashInfer as in the server arms (arm_rs2.sh): p3 with the prebuilt rq2u2h cache, never a JIT build (no nvcc here).
flock -w 28800 ~/.gpu.lock env DEV=cuda WT=$HOME/tools/sglang-rs2 TRITON_CACHE_DIR=$HOME/tools/sglang-rs2/.cache/triton \
  PYTHONPATH=$HOME/tools/flashinfer-p3 FLASHINFER_WORKSPACE_BASE=$HOME/.cache/sglang-rq2u2h FLASHINFER_P2_NO_NINJA=1 \
  nice -n 10 $PY bench/rs2/test_sparse_rs.py \
  > runs/rs2/test-cuda.log 2>&1 < /dev/null
rc=$?
# Tests run in order and stop at the first failure, so PASS D means A-D passed; E only times kernels.
LC_ALL=C grep -q '^PASS D losslessness' runs/rs2/test-cuda.log ||
  { echo "[chain-rs2] cuda tests A-D did not pass (rc=$rc); stop"; tail -20 runs/rs2/test-cuda.log; exit 1; }
[ $rc = 0 ] && echo "[chain-rs2] cuda tests PASS $(date +%T)" ||
  { echo "[chain-rs2] cuda E rc=$rc (speed budget), A-D passed; continuing, the ABBA decides"; tail -5 runs/rs2/test-cuda.log; }
sleep 30
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
  env OUTDIR=runs/rs2 bash bench/rs2/arm_rs2.sh S1 rs none || { echo "[chain-rs2] smoke rc=$?; stop"; exit 1; }
echo "[chain-rs2] smoke done $(date +%T)"
[ -n "${NO_ABBA:-}" ] && { echo "[chain-rs2] NO_ABBA set; stop"; exit 0; }
bash bench/rs2/abba_rs2.sh > runs/rs2/abba-rs2.log 2>&1
echo "[chain-rs2] done $(date +%T)"
