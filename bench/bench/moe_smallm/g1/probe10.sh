#!/bin/bash
# G1b: packing rebased on G2 (A0+A3+G2-1+G2-2).  Correctness, then the drift-cancelled A/B.
# Any failed save or timing run aborts (so a cmp never reads a stale .pt left by an earlier run);
# a cmp that reports a mismatch is recorded and makes the final exit status non-zero.
set -euo pipefail
set -x
CMP_FAILED=()
cd $HOME/tools/flash-next-bench
source bench/moe_smallm/g1benv.sh
touch $MOE_BENCH_MARKER
mkdir -p runs/g1/ab
# (a) deterministic mode: finalize fusion off, no autotune cache -> must be bit-identical
AB="python -u -m moe_smallm.ab_prologue save --widths 1,4,16 --rotation 4 --autotune-cache none"
SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 $AB runs/g1/ab/g1b-off.pt
SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 FLASHINFER_MOE_PACK_GROUPS=1 $AB runs/g1/ab/g1b-pack.pt
SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 FLASHINFER_MOE_PACK_GROUPS=1 FLASHINFER_MOE_PACK_MIN_RATIO=1 \
  $AB runs/g1/ab/g1b-packforce.pt
echo "=== g1b pack (auto-gate) vs G2 state, deterministic ==="
python -u -m moe_smallm.ab_prologue cmp runs/g1/ab/g1b-off.pt runs/g1/ab/g1b-pack.pt || CMP_FAILED+=("g1b-off.pt vs g1b-pack.pt")
echo "=== g1b pack forced at every width vs G2 state, deterministic ==="
python -u -m moe_smallm.ab_prologue cmp runs/g1/ab/g1b-off.pt runs/g1/ab/g1b-packforce.pt || CMP_FAILED+=("g1b-off.pt vs g1b-packforce.pt")
# (b) timing, alternating so thermal drift cancels
P="python -u -m moe_smallm.bench_moe --mode profile --rotation 16 --autotune-cache none --clocks --force-tactics"
for r in 1 2 3; do
  $P 17,57 --widths 4 --sweep-distinct 28,28,28,28,28 --json runs/g1/p10-w4-off-$r.json
  FLASHINFER_MOE_PACK_GROUPS=1 $P 17,57 --widths 4 --sweep-distinct 28,28,28,28,28 --json runs/g1/p10-w4-pack-$r.json
  $P 16,57 --widths 16 --sweep-distinct 69,69,69,69,69 --json runs/g1/p10-w16-off-$r.json
  FLASHINFER_MOE_PACK_GROUPS=1 FLASHINFER_MOE_PACK_MIN_RATIO=1 $P 16,57 --widths 16 --sweep-distinct 69,69,69,69,69 --json runs/g1/p10-w16-packforce-$r.json
done
if [ ${#CMP_FAILED[@]} -gt 0 ]; then echo "G1-PROBE10 FAILED (cmp mismatches): ${CMP_FAILED[*]}"; exit 1; fi
echo "G1-PROBE10-DONE"
