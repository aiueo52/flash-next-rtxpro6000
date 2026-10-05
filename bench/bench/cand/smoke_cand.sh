#!/bin/bash
# smoke_cand.sh [K, default 64] -- one smoke arm of the shipping candidate (~/tools/sglang-cand = opus/cand-1002) with
# everything on: the stack, ST1, XA1 and the RS package at draft support K (STACK 5.1), 1 prompt per workload, then
# the XA1 evidence that bench/rs2/arm_rs2.sh does not print. Out: runs/cand (C<K>-*). Run under the GPU lock:
#   flock -w 600 ~/.gpu.lock bash bench/cand/smoke_cand.sh 16 > runs/cand/smoke-16.log 2>&1
# Clean = lmstudio and greedy rc=0, error lines 0, rs2 enabled lines 1 (with the K below), ST1 lines >= 1,
# Triton decode fallback warnings 0, and _qsa_decode_attn_kernel plus the RS kernels in every trace.
set -uo pipefail
cd ~/tools/flash-next-bench
K=${1:-64}; L=C$K; OUT=runs/cand; mkdir -p $OUT
B_ENV="SGLANG_RS_DRAFT_TEMP_SCALE=0.7 SGLANG_RS_DRAFT_ONEHOT_ABOVE=0.9 SGLANG_RS_GREEDY_FAST=1 SGLANG_RS_BLOCK_VERIFY=1"
systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 env WT=$HOME/tools/sglang-cand OUTDIR=$OUT \
    PROMPT_LIMIT=1 SERVER_ENV="$B_ENV SGLANG_RS_DRAFT_TOPK=$K SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=1 SGLANG_OPT_TRITON_DECODE_ATTN=1" \
    bash bench/rs2/arm_rs2.sh $L rs lmstudio,greedy
echo "[$L] arm rc=$?"
echo "[$L] startup: $(grep -o 'SGLANG_OPT_SPEC_SPARSE_RS on: .*' $OUT/serve-$L.log | head -1)"
echo "[$L] Triton decode fallback warnings: $(grep -c 'SGLANG_OPT_TRITON_DECODE_ATTN: unsupported' $OUT/serve-$L.log)"
for T in g-code-edit code-edit prose-en; do
  n=$(zcat $OUT/traces/$L-$T/*.trace.json.gz 2>/dev/null | grep -oE '"name": *"_qsa_decode_attn_kernel"' | wc -l)
  echo "[$L] XA1 $T _qsa_decode_attn_kernel in trace: $n"
done
true
