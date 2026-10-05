#!/bin/bash
# build_rq2.sh [variants, default "1 2"] -- RQ2 (FC_fc-moe 1d): the fused prologue's expand row staged in
# shared memory with cp.async, quantize loop G2_EXPAND_QUANT_UNROLL chunks per trip. A variant is N (the
# unroll) or Nh (N plus G2_EXPAND_SF_HOIST=1: the SF write address computed once per row). Source tree
# ~/tools/flashinfer-p3 (p2 + the RQ2 edit). No ninja: the instantiation TU is compiled by hand with the
# exact command of the RQ1 build (runs/fc_moe/c1-build.log, p2 -> p3 sources, plus the variant's -D flags)
# and linked with p2's other objects (read-only) into a private cache ~/.cache/sglang-rq2u<variant>, which holds
# copies of p2's other modules (byte-identical to production's). Load it with nojit / FLASHINFER_P2_NO_NINJA.
# Run under a capped scope, e.g.
#   systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 bash bench/fc_moe/build_rq2.sh
set -uo pipefail
cd ~/tools/flash-next-bench
VARIANTS=${1:-"1 2"}
REL=.cache/flashinfer/0.6.17/120f/cached_ops/fused_moe_120
P2C=$HOME/.cache/sglang-p2; P2M=$P2C/$REL
LOG=runs/fc_moe/c1-build.log
COMPILE=$(sed -n 's/^\[1\/2\] *//p' $LOG)
LINK=$(sed -n 's/^\[2\/2\] *//p' $LOG)
[ -n "$COMPILE" ] && [ -n "$LINK" ] || { echo "no commands in $LOG"; exit 1; }
INST=cutlass_backend_cutlass_fused_moe_instantiation.cuda.o
echo "start $(date +%T); MemAvailable $(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)G"
pids=()
for V in $VARIANTS; do
  U=${V%h}; FLAGS="-DG2_EXPAND_QUANT_UNROLL=$U"
  [ "$V" != "$U" ] && FLAGS+=" -DG2_EXPAND_SF_HOIST=1"
  C=$HOME/.cache/sglang-rq2u$V; M=$C/$REL
  mkdir -p $C
  rsync -a --exclude "$REL/*.o" --exclude "$REL/*.o.d" --exclude "$REL/fused_moe_120.so" "$P2C/" "$C/"
  cmd=${COMPILE//\/tools\/flashinfer-p2\//\/tools\/flashinfer-p3\/}
  cmd=${cmd//$P2M\//$M\/}
  (
    nice -n 19 bash -c "$cmd $FLAGS" > runs/fc_moe/rq2u$V-build.log 2>&1
    rc=$?; echo "u$V compile rc=$rc ($FLAGS) $(date +%T)"
    [ $rc -eq 0 ] || exit $rc
    lcmd=${LINK//$P2M\/$INST/$M\/$INST}
    lcmd=${lcmd//-o $P2M\/fused_moe_120.so/-o $M\/fused_moe_120.so}
    nice -n 19 bash -c "$lcmd" >> runs/fc_moe/rq2u$V-build.log 2>&1
    echo "u$V link rc=$? $(date +%T)"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
for V in $VARIANTS; do
  M=$HOME/.cache/sglang-rq2u$V/$REL
  ls -la $M/fused_moe_120.so $M/$INST 2>&1 | sed "s/^/u$V: /"
done
echo "done $(date +%T)"
