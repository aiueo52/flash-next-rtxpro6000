#!/bin/bash
# G1b final: default gate (always on where legal), both widths, correctness + timing.
# Any failed save or timing run aborts (so a cmp never reads a stale .pt left by an earlier run);
# a cmp that reports a mismatch is recorded and makes the final exit status non-zero.
set -euo pipefail
set -x
CMP_FAILED=()
cd $HOME/tools/flash-next-bench
source bench/moe_smallm/g1benv.sh
touch $MOE_BENCH_MARKER
AB="python -u -m moe_smallm.ab_prologue save --widths 1,4,16 --rotation 4 --autotune-cache none"
SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 $AB runs/g1/ab/f-off.pt
SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 FLASHINFER_MOE_PACK_GROUPS=1 $AB runs/g1/ab/f-pack.pt
echo "=== FINAL g1b: pack (default gate) vs G2 state, deterministic ==="
python -u -m moe_smallm.ab_prologue cmp runs/g1/ab/f-off.pt runs/g1/ab/f-pack.pt || CMP_FAILED+=("f-off.pt vs f-pack.pt")
# production-config control (fused finalize on, tactics pinned)
AB2="python -u -m moe_smallm.ab_prologue save --widths 1,4,16 --rotation 4 --force-tactics 17,57 --autotune-cache none"
$AB2 runs/g1/ab/f-pon-off.pt
$AB2 runs/g1/ab/f-pon-off2.pt
FLASHINFER_MOE_PACK_GROUPS=1 $AB2 runs/g1/ab/f-pon-pack.pt
echo "=== control: off vs off (fused finalize on) ==="
python -u -m moe_smallm.ab_prologue cmp runs/g1/ab/f-pon-off.pt runs/g1/ab/f-pon-off2.pt --tol 1e-2 || CMP_FAILED+=("f-pon-off.pt vs f-pon-off2.pt")
echo "=== pack vs off (fused finalize on) ==="
python -u -m moe_smallm.ab_prologue cmp runs/g1/ab/f-pon-off.pt runs/g1/ab/f-pon-pack.pt --tol 1e-2 || CMP_FAILED+=("f-pon-off.pt vs f-pon-pack.pt")
P="python -u -m moe_smallm.bench_moe --mode profile --rotation 16 --autotune-cache none --clocks --force-tactics"
for r in 1 2 3; do
  $P 17,57 --widths 4 --sweep-distinct 28,28,28,28,28 --json runs/g1/p11-w4-off-$r.json
  FLASHINFER_MOE_PACK_GROUPS=1 $P 17,57 --widths 4 --sweep-distinct 28,28,28,28,28 --json runs/g1/p11-w4-pack-$r.json
  $P 16,57 --widths 16 --sweep-distinct 69,69,69,69,69 --json runs/g1/p11-w16-off-$r.json
  FLASHINFER_MOE_PACK_GROUPS=1 $P 16,57 --widths 16 --sweep-distinct 69,69,69,69,69 --json runs/g1/p11-w16-pack-$r.json
done
if [ ${#CMP_FAILED[@]} -gt 0 ]; then echo "G1-PROBE11 FAILED (cmp mismatches): ${CMP_FAILED[*]}"; exit 1; fi
echo "G1-PROBE11-DONE"
