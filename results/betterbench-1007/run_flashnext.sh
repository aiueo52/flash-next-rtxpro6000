#!/bin/bash
# BetterBench (GGZ14/BetterBench 7696bf2) against the production Flash-Next SGLang build (wa profile).
# Run inside: flock -w 600 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 bash run_flashnext.sh
set -uo pipefail
export BETTERBENCH_NO_UPDATE_CHECK=1
PORT=30000
LOG=~/tools/betterbench/logs/server-$(date +%m%d-%H%M).log
mkdir -p ~/tools/betterbench/logs
cd ~/tools/sglang-prod-1002
PORT=$PORT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SERVE_DISPLAY_HZ=60 \
  setsid bash ./serve-fast.sh wa > "$LOG" 2>&1 &
SPID=$!
cleanup() { kill -TERM -- -$SPID 2>/dev/null; for i in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -KILL -- -$SPID 2>/dev/null; echo "[bb] server stopped $(date +%T)"; }
trap cleanup EXIT
echo "[bb] server pid $SPID, log $LOG, $(date +%T)"
for i in $(seq 360); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "[bb] server died"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "[bb] server not ready after 30 min"; exit 1; }
echo "[bb] server ready $(date +%T)"
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr -d ' ')
cd ~/tools/betterbench
timeout 13800 .venv/bin/betterbench run --no-update-check \
  --endpoint http://127.0.0.1:$PORT/v1 --model flash-next --name flashnext-wa-prod1002 \
  --note build=sglang-prod-1002 --note profile=wa-adaptive-3/7/15 --note quant=NVFP4-experts+W8A16-dense \
  --note mtp_head=private-finetuned --note token_map=private-49k \
  --note gpu="RTX PRO 6000 Blackwell Max-Q, power limit $PL, also drives a 4K desktop" \
  --note tuned_for=c1-only "$@"
echo "[bb] betterbench rc=$? $(date +%T)"
