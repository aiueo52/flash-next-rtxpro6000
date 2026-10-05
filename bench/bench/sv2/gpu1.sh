#!/bin/bash
# Record of the 12:58 run on the worktree with alt_triton_topk.patch applied; that kernel was rejected,
# so to rerun only the FlashInfer comparison, run step 2 (topk_bench.py) alone.
# gpu1.sh -- SV2 on the GPU, one lock window after chain4 (FG1 ABBA) exits, so never inside an ABBA:
#   1. the worktree's sv2_bench/gpu_check.py (the alternative kernel's harness): correctness at V=248320 + timing vs torch.topk;
#   2. bench/sv2/topk_bench.py: torch.topk vs Triton sparse_topk vs FlashInfer radix top_k (p3, rq2u2h cache).
cd ~/tools/flash-next-bench
WT=~/tools/sglang-sv2
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
C4=$(ps -eo pid=,args= | grep "[c]hain4\.sh" | awk '{print $1}' | head -1)
echo "[sv2-gpu1] start $(date +%T); waiting for chain4 pid=${C4:-none}"
while [ -n "$C4" ] && kill -0 "$C4" 2>/dev/null; do sleep 2; done
echo "[sv2-gpu1] lock wait $(date +%T)"
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
  env TRITON_CACHE_DIR=$HOME/.cache/sv2-triton WT=$WT PY=$PY timeout 2400 bash -c '
    echo "[sv2-gpu1] locked $(date +%T)"
    $PY $WT/sv2_bench/gpu_check.py runs/sv2/gpu1.jsonl; echo "[sv2-gpu1] gpu_check rc=$? $(date +%T)"
    PYTHONPATH=$HOME/tools/flashinfer-p3:$WT/python FLASHINFER_WORKSPACE_BASE=$HOME/.cache/sglang-rq2u2h \
      FLASHINFER_P2_NO_NINJA=1 $PY bench/sv2/topk_bench.py runs/sv2/topk1.jsonl > runs/sv2/topk1.log 2>&1
    echo "[sv2-gpu1] topk_bench rc=$? $(date +%T)"'
echo "[sv2-gpu1] unlocked $(date +%T)"
$PY - runs/sv2/gpu1.jsonl <<'PY'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
bad = [r for r in rs if r["type"] == "correctness" and not r["ok"]]
print(f"gpu_check correctness: {sum(r['type'] == 'correctness' for r in rs)} cases, {len(bad)} bad")
for r in bad[:10]:
    print("  BAD", r)
t = {}
for r in rs:
    if r["type"] == "timing" and "median_us" in r:
        t[(r["dtype"], r["n"], r["kp"], r["mode"], r["operator"], r["timing_mode"])] = r["median_us"]
    elif r["type"] == "timing":
        print("  timing error", r)
print(f"{'dtype':8s} {'n':>3s} {'kp':>4s} {'mode':7s} | torch eager / graph | sparse eager / graph")
for (d, n, kp, m, op, tm), v in sorted(t.items()):
    if op != "torch.topk" or tm != "eager" or m not in ("random", "peaked"):
        continue
    g = lambda o, x: t.get((d, n, kp, m, o, x), float("nan"))
    print(f"{d:8s} {n:3d} {kp:4d} {m:7s} | {g('torch.topk','eager'):7.1f} / {g('torch.topk','cuda_graph'):7.1f}"
          f" | {g('sparse_topk','eager'):7.1f} / {g('sparse_topk','cuda_graph'):7.1f}")
PY
$PY - runs/sv2/topk1.jsonl <<'PY'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
bad = [r for r in rs if r["errors"]]
print(f"topk_bench correctness: {len(rs)} cases, {len(bad)} bad")
for r in bad[:12]:
    print("  BAD", r)
print("k   n  mode    variant        graph   queued    sync  (us per call)")
for r in rs:
    if "graph" in r:
        print(f"{r['k']:3d} {r['n']:2d} {r['mode']:7s} {r['variant']:13s} {r['graph']!s:>7} {r['queued']!s:>8} {r['sync']!s:>7}")
PY
