#!/bin/bash
# chain_dt_after8.sh <pid>... -- after the stack ABBA (chain_stack8.sh) exits: one arm C1 = the stack + DT1
# (SGLANG_OPT_DRAFT_TAIL=1; worktree ~/tools/sglang-stack-dt = opus/stack + the two DT1 commits), on the stack8
# arms' grid; then DT1's evidence: the enable log line, the draft-tail kernels in the greedy trace against B1's
# (bench/dt1/dt_trace.py) and the greedy fingerprint against the B arms (bench/stack/fingerprint.py).
cd ~/tools/flash-next-bench
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
echo "[chain-dt] start $(date +%T); waiting for $*"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 10; done; done
sleep 30
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
  env OUTDIR=runs/stack8 PROMPT_LIMIT=4 WT=$HOME/tools/sglang-stack-dt SERVER_ENV="SGLANG_OPT_DRAFT_TAIL=1" \
  bash bench/stack/arm_stack.sh C1 on lmstudio,greedy || echo "[chain-dt] arm C1 rc=$?"
echo "[C1] draft tail enabled lines: $(grep -c 'SGLANG_OPT_DRAFT_TAIL on: draft-extend select True' runs/stack8/serve-C1.log)"
for L in B1 C1; do echo "== dt_trace $L"; $PY bench/dt1/dt_trace.py runs/stack8/traces/$L-g-code-edit; done
echo "== fingerprint: A arms alone, then the B arms against C1"
python3 bench/stack/fingerprint.py --base runs/stack8/A{1,2,3,4}-greedy.jsonl
python3 bench/stack/fingerprint.py --base runs/stack8/B{1,2,3,4}-greedy.jsonl --test runs/stack8/C1-greedy.jsonl
echo "[chain-dt] done $(date +%T)"
