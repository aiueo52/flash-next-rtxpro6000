#!/bin/bash
# dump_after_rs2.sh -- RS2d's GPU dump run (specs/RS2D_DRAFT_SHARPEN_2026-10-01.md 4.1), slotted in right after the
# RS2 ABBA's last arm: it waits for the ABBA's "== ancova lmstudio" line (all 8 arms done, GPU lock free), then
# takes the lock before the queue's next step does. Knobs at their defaults (RS2 as run) plus the dump.
set -uo pipefail
cd ~/tools/flash-next-bench
timeout 21600 bash -c 'until LC_ALL=C /usr/bin/grep -q "^== ancova lmstudio" runs/rs2/abba-rs2.log; do sleep 2; done' ||
  { echo "[rs2d-dump] RS2 ABBA did not finish; not run $(date +%T)"; exit 1; }
mkdir -p runs/rs2d/dump
echo "[rs2d-dump] queued $(date +%T)"
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
    env WT=$HOME/tools/sglang-rs2d PORT=8031 OUTDIR=runs/rs2d PROMPT_LIMIT=3 \
        SERVER_ENV="SGLANG_RS_DUMP_DIR=$HOME/tools/flash-next-bench/runs/rs2d/dump" \
        bash bench/rs2/arm_rs2.sh D1 rs lmstudio
echo "[rs2d-dump] arm rc=$? $(date +%T); dump files: $(ls runs/rs2d/dump | wc -l)"
