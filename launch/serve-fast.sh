#!/bin/bash
# Fast decode profiles for Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell (SM120), batch size 1.
# Portable version of launch/as-measured/serve-fast.sh: identical flags and defaults, but
# machine-specific paths are variables and the private artefacts (fine-tuned MTP head, token map
# built from private transcripts) are replaced by public / user-built equivalents.
#
#   ./serve-fast.sh w4    # prose / chat / agent: draft width 4 (NEXTN steps 3)
#   ./serve-fast.sh w8    # draft width 8  (agent-loop sweet spot as a fixed width)
#   ./serve-fast.sh w16   # mechanical code edits: draft width 16 (NEXTN steps 15)
#   ./serve-fast.sh wa    # one server, per-request adaptive width {3,7,15} (recommended)
#   ./serve-fast.sh ngram # experiment: NGRAM_CHAIN v0 (W16), not recommended
#
# Required env:  SGLANG_DIR   patched sglang-rtxpro6000 checkout (see docs/reproduce.md)
# Recommended:   TOKEN_MAP    reduced draft vocabulary .pt (tokenmaps/README.md). Default: none
#                              (full-vocab draft head; much slower at W16, see docs/rejected.md)
# Optional:      TARGET_MODEL (default: the public RadixArk/Qwen3.8-Flash-Next-NVFP4 under $HOME/models)
#                ADAPTIVE_CONFIG (default: bench/adaptive/w16_3_7_15_c.json of this repo)
#                FLASHINFER_DIR  a patched FlashInfer package directory to put ahead of the venv's
#                                (default: none = the venv's flashinfer, patched in place by
#                                patches/flashinfer/apply.sh)
#                CONTEXT_LENGTH  above 262144 serve-local.sh turns on factor-2 YaRN (up to 524288) and
#                                the wa profile raises its memory fraction to WA_LONG_MEM_FRACTION (0.96)
# Extra arguments are passed through to `sglang serve`.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PUB="$(cd "$HERE/.." && pwd)"
: "${SGLANG_DIR:?set SGLANG_DIR to the patched sglang-rtxpro6000 checkout}"
export SGLANG_DIR
PROFILE="${1:-w4}"; shift || true

# Allocator: without expandable segments the caching allocator kept ~9 GB of inactive split blocks after
# the draft load and the KV-pool check failed at start-up. Allocator-only change; set it empty to disable.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"
# Triton W8A16 GEMV for the FP8 dense weights, Triton in_proj_ba GEMV
export SGLANG_FP8_W8A16_GEMV=1 SGLANG_GDN_BA_TRITON_GEMV=1
# Fused bf16->fp8 KV scale+cast+store (bit-exact)
export SGLANG_KV_FP8_FUSED_STORE=${SGLANG_KV_FP8_FUSED_STORE:-1}
# Three-kernel HC norm+mix (~14.3 us vs 19.8 us per call)
export SGLANG_HC_MIX2=${SGLANG_HC_MIX2:-1}
# GDN verify reads q/k/v strided at T=16 (needs patches/flashinfer/07-gdn-wy-T16-strided-qkv.patch; bit-identical)
export FLASHINFER_GDN_WY_STRIDED_QKV=${FLASHINFER_GDN_WY_STRIDED_QKV:-1}
# Draft head writes the logits buffer directly (bit-identical)
export SGLANG_DRAFT_LOGITS_OUT=${SGLANG_DRAFT_LOGITS_OUT:-1}
# Router gate via the skinny BF16 Triton GEMV
export SGLANG_ROUTER_GEMV=${SGLANG_ROUTER_GEMV:-1}
# FP8 low-rank HC mix weights (13.7 -> 9.9 us/call, <= 2 ulp)
export SGLANG_HC_MIX2_FP8=${SGLANG_HC_MIX2_FP8:-1}
# Shared-expert gate_up GEMV with a silu*up epilogue (turned off for w16/wa below: lowers long-chain acceptance)
export SGLANG_SHARED_GATEUP_FUSED=${SGLANG_SHARED_GATEUP_FUSED:-1}
# Draft skips its unused 2 x 1.19 GB vocab tensors (frees 2.4 GB)
export SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS=${SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS:-1}
# QSA graph row-metadata kernel, page-parallel (bit-exact)
export SGLANG_QSA_META_PAGE_PARALLEL=${SGLANG_QSA_META_PAGE_PARALLEL:-1}
# Draft entry token term as a precomputed full-vocab BF16 table (1.18 GB, bit-exact at M=1)
export SGLANG_MTP_EMBED_TABLE=${SGLANG_MTP_EMBED_TABLE:-1}
# Chain-parallel verify conv1d (bit-exact)
export SGLANG_GDN_CONV_CHAIN_PARALLEL=${SGLANG_GDN_CONV_CHAIN_PARALLEL:-1}
# GDN projections write their consumers' layouts directly; verify a/b straight into the recovery stash
export SGLANG_GDN_PROJ_DIRECT_LAYOUT=${SGLANG_GDN_PROJ_DIRECT_LAYOUT:-1}
export SGLANG_GDN_AB_STASH_DIRECT=${SGLANG_GDN_AB_STASH_DIRECT:-1}
# HC boundary: combine gate at mix time (=2; =1 is a measured loss), early shared-expert gate, combine apply folded into next K0
export SGLANG_HC_GATE_EARLY=${SGLANG_HC_GATE_EARLY:-2}
export SGLANG_SHARED_GATE_EARLY=${SGLANG_SHARED_GATE_EARLY:-1}
export SGLANG_HC_APPLY_MIX_FUSED=${SGLANG_HC_APPLY_MIX_FUSED:-1}
# Layer->layer HC apply folded into the next layer's K0
export SGLANG_HC_LAYER_APPLY_FUSED=${SGLANG_HC_LAYER_APPLY_FUSED:-1}
# GDN gated RMSNorm folded into the out_proj GEMV A-load (M>=4)
export SGLANG_NORM_INTO_GEMV=${SGLANG_NORM_INTO_GEMV:-1}
# NVFP4 lm_heads (N1): off -- acceptance loss outweighed the step saving in-server
export SGLANG_MTP_LMHEAD_NVFP4=${SGLANG_MTP_LMHEAD_NVFP4:-0}
export SGLANG_LMHEAD_NVFP4=${SGLANG_LMHEAD_NVFP4:-0}
# Singleton-route MoE pruning (P1/P2), evaluated inside FlashInfer's fused routing prologue.
# NOTE: changes the target model's computation (drops low-weight routes to experts used by exactly one
# verify row). Requires patches/flashinfer/06-p2-prune-in-prologue.patch. PRUNE_TAU=0 disables it.
export SGLANG_MOE_PRUNE_SINGLETON_TAU=${SGLANG_MOE_PRUNE_SINGLETON_TAU:-${PRUNE_TAU:-0.08}}
export SGLANG_MOE_PRUNE_IN_PROLOGUE=${SGLANG_MOE_PRUNE_IN_PROLOGUE:-1}
# FlashInfer MoE active-expert group packing (patches/flashinfer/05-g1-pack-only.patch; opt-in in the csrc)
export FLASHINFER_MOE_PACK_GROUPS=${FLASHINFER_MOE_PACK_GROUPS:-1}
# Programmatic dependent launch for the fork's Triton kernels
export SGLANG_TRITON_PDL=${SGLANG_TRITON_PDL:-1}

# ---- 2026-10-02 additions (patches 0106-0121, FlashInfer 08-10); any flag =0 turns it off. ----
# FlashInfer: by default the venv's (patched) package is used. To serve a frozen copy instead, set
# FLASHINFER_DIR to it and FLASHINFER_WORKSPACE_BASE to its pre-built JIT cache, and add
# FLASHINFER_P2_NO_NINJA=1 (patch 09) so that a copied cache is loaded as-is instead of recompiled.
if [ -n "${FLASHINFER_DIR:-}" ]; then export PYTHONPATH="$FLASHINFER_DIR:$SGLANG_DIR/python${PYTHONPATH:+:$PYTHONPATH}"; fi
# Stack (RT1 packed-key router top-k, SV1 sparse sampling verify, SV2 FlashInfer radix top-k for it,
# FG1 GDN verify front overlap, DG1 one-token draft MoE GEMV); with FlashInfer RQ2 (patch 08):
# wa 8-start ABBA ms/token -9.6% LM Studio sampling, -4.6% greedy (docs/optimizations.md).
export SGLANG_ROUTER_FAST_TOPK=${SGLANG_ROUTER_FAST_TOPK:-1} SGLANG_OPT_SPEC_SPARSE_VERIFY=${SGLANG_OPT_SPEC_SPARSE_VERIFY:-1}
export SGLANG_OPT_SPEC_SPARSE_TOPK=${SGLANG_OPT_SPEC_SPARSE_TOPK:-1} SGLANG_OPT_GDN_FRONT_OVERLAP=${SGLANG_OPT_GDN_FRONT_OVERLAP:-1}
export SGLANG_OPT_DRAFT_MOE_GEMV=${SGLANG_OPT_DRAFT_MOE_GEMV:-1}
# min_p honoured in the speculative sampling verify and the RS draft proposal (a fidelity fix: before,
# min_p was ignored on the speculative path); ST1 MTP shared-index tail + XA1 split-KV Triton decode
# attention (adopted together, +2.8%; ST1 alone is not a speed-up).
export SGLANG_SPEC_MIN_P=${SGLANG_SPEC_MIN_P:-1}
export SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=${SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX:-1} SGLANG_OPT_TRITON_DECODE_ATTN=${SGLANG_OPT_TRITON_DECODE_ATTN:-1}
# RS package: sparse chain rejection sampling over a K=16 draft support, draft sharpened (temperature
# x0.7, one-hot above p=0.9), greedy fast path, block verification. LM Studio t/s +6.5% pooled
# (code-edit +1.6%, unresolved). Needs --speculative-use-rejection-sampling (added below).
# DT1 (SGLANG_OPT_DRAFT_TAIL) was never measured with it and stays off.
export SGLANG_OPT_SPEC_SPARSE_RS=${SGLANG_OPT_SPEC_SPARSE_RS:-1} SGLANG_RS_DRAFT_TOPK=${SGLANG_RS_DRAFT_TOPK:-16}
export SGLANG_RS_DRAFT_TEMP_SCALE=${SGLANG_RS_DRAFT_TEMP_SCALE:-0.7} SGLANG_RS_DRAFT_ONEHOT_ABOVE=${SGLANG_RS_DRAFT_ONEHOT_ABOVE:-0.9}
export SGLANG_RS_GREEDY_FAST=${SGLANG_RS_GREEDY_FAST:-1} SGLANG_RS_BLOCK_VERIFY=${SGLANG_RS_BLOCK_VERIFY:-$SGLANG_OPT_SPEC_SPARSE_RS}  # block verify needs the RS package

# Target checkpoint. The measurements used privately fine-tuned MTP heads (mtpft3 / mtpft5) that are NOT
# published; the public checkpoint's original MTP head gives somewhat lower acceptance (README.md, "Limitations and measurement noise").
export TARGET_MODEL="${TARGET_MODEL:-$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}"
# Reduced draft vocabulary (FR-Spec style). Build one with tokenmaps/build_hot_vocab.py.
TOKEN_MAP="${TOKEN_MAP:-none}"
if [ "$TOKEN_MAP" = none ]; then
  echo "[serve-fast] TOKEN_MAP not set: full-vocabulary draft head (slower). See tokenmaps/README.md." >&2
fi
# Optional: switch the primary display to a lower refresh rate while serving (desktop compositor
# preemption on the same GPU costs ~5-7% step time at 4K 160 Hz). Restored when the server exits.
if [ -n "${SERVE_DISPLAY_HZ:-}" ]; then
  _dl=$(xrandr 2>/dev/null | awk '/\*/{print; exit}')
  _out=$(xrandr 2>/dev/null | awk '/ connected primary/{print $1; exit}')
  _mode=$(echo "$_dl" | awk '{print $1}'); _orig=$(echo "$_dl" | grep -oE '[0-9.]+\*' | tr -d '*')
  if [ -n "$_out" ] && [ -n "$_mode" ] && [ -n "$_orig" ] && [ "$_orig" != "$SERVE_DISPLAY_HZ" ] \
     && xrandr --output "$_out" --mode "$_mode" --rate "$SERVE_DISPLAY_HZ" 2>/dev/null; then
    echo "[serve-fast] display $_out $_mode: ${_orig}Hz -> ${SERVE_DISPLAY_HZ}Hz while serving (pid $$)" >&2
    ( setsid bash -c "while kill -0 $$ 2>/dev/null; do sleep 5; done; xrandr --output $_out --mode $_mode --rate $_orig" >/dev/null 2>&1 < /dev/null & )
  fi
fi
# Above 262144 serve-local.sh turns on factor-2 YaRN; wa then needs ~3.5 GB more KV. 524288 passed
# all needles (up to 480k tokens) at 0.96 with ~6.8 GB of VRAM used by the desktop (docs/optimizations.md).
if [ "$PROFILE" = wa ] && [ "${CONTEXT_LENGTH:-0}" -gt 262144 ]; then WA_MEM_FRACTION="${WA_LONG_MEM_FRACTION:-0.96}"; fi
COMMON=()
[ "$TOKEN_MAP" != none ] && COMMON+=(--speculative-token-map "$TOKEN_MAP")
COMMON+=(--qwen4-exp-dense-fp8 shared_expert,attn,linear_attn,lm_head,mtp_dense
        --speculative-draft-model-quantization modelopt_fp4 --enable-metrics)
# RS runs in the rejection-sampling verify; ngram keeps the target-only verify (RS never measured there).
if [ "$PROFILE" = ngram ]; then export SGLANG_OPT_SPEC_SPARSE_RS=0
elif [ "$SGLANG_OPT_SPEC_SPARSE_RS" = 1 ]; then COMMON+=(--speculative-use-rejection-sampling); fi
case "$PROFILE" in
  w4)    MEM_FRACTION="${MEM_FRACTION:-${W4_MEM_FRACTION:-0.935}}" exec "$HERE/serve-local.sh" "${COMMON[@]}" "$@" ;;
  w8)    MAMBA_SLOTS="${MAMBA_SLOTS:-10}" exec "$HERE/serve-local.sh" "${COMMON[@]}" --speculative-num-steps 7 --speculative-num-draft-tokens 8 "$@" ;;
  w16)   MAMBA_SLOTS="${MAMBA_SLOTS:-10}" SGLANG_SHARED_GATEUP_FUSED="${W16_GATEUP:-0}" SGLANG_ROUTER_GEMV="${W16_ROUTER:-1}" exec "$HERE/serve-local.sh" "${COMMON[@]}" --speculative-num-steps 15 --speculative-num-draft-tokens 16 --mem-fraction-static "${W16_MEM_FRACTION:-0.93}" "$@" ;;
  # Adaptive width: confidence-mixture policy over {3,7,15} (C1 + WA5 config c), step-time model
  # 7.943 + 0.5554*S ms, target autotune before capturing the extra candidate graphs (X3 fix).
  # Previous two-state policy: ADAPTIVE_CONFIG=$PUB/bench/adaptive/w16_conf.json SGLANG_ADAPTIVE_STEP_A=9.74 SGLANG_ADAPTIVE_STEP_B=0.70
  wa)    SGLANG_ADAPTIVE_STEP_A="${SGLANG_ADAPTIVE_STEP_A:-7.943}" SGLANG_ADAPTIVE_STEP_B="${SGLANG_ADAPTIVE_STEP_B:-0.5554}" SGLANG_ADAPTIVE_TARGET_AUTOTUNE="${SGLANG_ADAPTIVE_TARGET_AUTOTUNE:-1}" SGLANG_ADAPTIVE_POLICY="${SGLANG_ADAPTIVE_POLICY:-confidence}" MAMBA_SLOTS="${MAMBA_SLOTS:-10}" SGLANG_SHARED_GATEUP_FUSED="${W16_GATEUP:-0}" SGLANG_ROUTER_GEMV="${W16_ROUTER:-1}" exec "$HERE/serve-local.sh" "${COMMON[@]}" --speculative-num-steps 15 --speculative-num-draft-tokens 16 --mem-fraction-static "${WA_MEM_FRACTION:-0.925}" --speculative-adaptive --speculative-adaptive-config "${ADAPTIVE_CONFIG:-$PUB/bench/adaptive/w16_3_7_15_c.json}" "$@" ;;
  ngram) MAMBA_SLOTS="${MAMBA_SLOTS:-10}" exec "$HERE/serve-local.sh" "${COMMON[@]}" --speculative-algorithm NGRAM_CHAIN --speculative-num-steps 15 --speculative-num-draft-tokens 16 "$@" ;;
  *) echo "usage: $0 {w4|w8|w16|wa|ngram} [extra sglang args]" >&2; exit 2 ;;
esac
