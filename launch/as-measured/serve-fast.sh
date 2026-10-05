#!/bin/bash
# flash-next 高速プロファイル起動 (2026-09-02 計測に基づく)
#   ./serve-fast.sh w4    # 散文/チャット/エージェント向け: NEXTN steps3  (prose ~150-165, agent ~200-244 t/s)
#   ./serve-fast.sh w8    # コード寄り:                       NEXTN steps7  (code-edit ~350-370 t/s)
#   ./serve-fast.sh w16   # 機械的コード編集向け:             NEXTN steps15 (code-edit ~430 t/s, prose ~100)
#   ./serve-fast.sh ngram # 実験: NGRAM_CHAIN v0 (W16)
# 共通: 縮小語彙ドラフト(hot_32768) + 密FP8(W8A16 Triton gemv: shared_expert,attn,linear_attn,lm_head)
#       + in_proj_ba Triton + QSAリング修正 + ドラフト/検証glue融合 (branch codex/perf-v1, 2026-09-03)
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
PROFILE="${1:-w4}"; shift || true
# 2026-09-07 00:15-: every start died at the KV-pool check ("Loaded weights leave no GPU memory ... raise --mem-fraction-static")
# because the caching allocator kept ~9 GB of inactive split blocks after the draft load (released 4.2 GB instead of 10.8 GB).
# expandable segments make the pools identical to the known-good runs (Q1 diag, bench/quality/run_arm.sh). Allocator-only;
# override with PYTORCH_CUDA_ALLOC_CONF= (empty) to get the native allocator back.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"
export SGLANG_FP8_W8A16_GEMV=1 SGLANG_GDN_BA_TRITON_GEMV=1
# 2026-09-03 glue fusion: fused bf16->fp8 KV store (bit-exact). HC fusion (SGLANG_HC_FUSED=1)
# and router GEMV / MTP entry (SGLANG_ROUTER_GEMV / SGLANG_MTP_ENTRY_FUSED) are slower -> off.
export SGLANG_KV_FP8_FUSED_STORE=${SGLANG_KV_FP8_FUSED_STORE:-1}
# 2026-09-03: three-kernel HC norm+mix (K0/K1/K2, ~14.3us vs 19.8us per call; -0.5ms/step W16, -0.26 W4)
export SGLANG_HC_MIX2=${SGLANG_HC_MIX2:-1}
export FLASHINFER_GDN_WY_STRIDED_QKV=${FLASHINFER_GDN_WY_STRIDED_QKV:-1}   # glue-e item1: strided q/k/v read in GDN verify (venv file patched, bit-identical)
export SGLANG_DRAFT_LOGITS_OUT=${SGLANG_DRAFT_LOGITS_OUT:-1}          # glue-e item3: draft head writes the logits buffer directly (bit-identical)
export SGLANG_ROUTER_GEMV=${SGLANG_ROUTER_GEMV:-1}               # glue-e item4: router gate via bf16_gemv (gemv3 table)
export SGLANG_HC_MIX2_FP8=${SGLANG_HC_MIX2_FP8:-1}               # hc-fp8 item1: FP8 low-rank HC mix weights (13.7->9.9us/call, <=2 ulp)
export SGLANG_SHARED_GATEUP_FUSED=${SGLANG_SHARED_GATEUP_FUSED:-1}       # hc-fp8 item2: shared-expert gate_up gemv + silu*up epilogue
# 2026-09-05: draft skips its dead 2x1.19GB vocab tensors (ea52699963); frees 2.4GB at start-up. Needs W16 mem fraction <= 0.945.
export SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS=${SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS:-1}
# 2026-09-05: QSA graph row-metadata kernel page-parallel (f21ef2f0f6): 240->49us/step W16, bit-exact
export SGLANG_QSA_META_PAGE_PARALLEL=${SGLANG_QSA_META_PAGE_PARALLEL:-1}
# 2026-09-05: draft entry token-term as a full-vocab BF16 table (R3, 1.18GB, bit-exact at M=1): W16 draft -0.25ms/step. Needs W4 mem fraction <= 0.947.
export SGLANG_MTP_EMBED_TABLE=${SGLANG_MTP_EMBED_TABLE:-1}
# 2026-09-05: chain-parallel verify conv1d (R4, cf12af5ddc, bit-exact): per-call 6.2->3.4us at T=16 = -0.10ms/step W16, -0.007 W4
export SGLANG_GDN_CONV_CHAIN_PARALLEL=${SGLANG_GDN_CONV_CHAIN_PARALLEL:-1}
# 2026-09-05: R5 GDN projections write mixed_qkv/z/b/a directly (no re-layout kernel) + R2 verify a/b straight into the recovery stash (ccf9d2faba, bit-exact): -108 kernels/step, -0.16ms W4 / -0.12ms W16
export SGLANG_GDN_PROJ_DIRECT_LAYOUT=${SGLANG_GDN_PROJ_DIRECT_LAYOUT:-1}
export SGLANG_GDN_AB_STASH_DIRECT=${SGLANG_GDN_AB_STASH_DIRECT:-1}
# 2026-09-05: HC boundary (d723dc04c7): combine gate at mix time (R1, =2 moved kernel, bit-exact; =1 K0-fused is a measured LOSS),
# shared-expert gate early + join folded into HC apply (R6, bit-exact), combine apply folded into next K0 (R7, <=1ulp): -0.10ms/step W4, -0.09 W16
export SGLANG_HC_GATE_EARLY=${SGLANG_HC_GATE_EARLY:-2}
export SGLANG_SHARED_GATE_EARLY=${SGLANG_SHARED_GATE_EARLY:-1}
export SGLANG_HC_APPLY_MIX_FUSED=${SGLANG_HC_APPLY_MIX_FUSED:-1}
# 2026-09-06: H2 layer->layer HC apply folded into the next layer K0 (46/48 boundaries; PLE layer and last layer keep the plain apply): -42/-47us/step W4/W16, torch.equal
export SGLANG_HC_LAYER_APPLY_FUSED=${SGLANG_HC_LAYER_APPLY_FUSED:-1}
# 2026-09-06: H1-B GDN gated RMSNorm folded into the out_proj GEMV A-load (M>=4 only): -1.06us/call x36 = -36us/step; <=6ulp on 0.03% at M=16 (split-K re-partition), acceptance/needle PASS. H1-C (SGLANG_GEMV_BA_SPLIT_GRID) measured ~0 -> off.
export SGLANG_NORM_INTO_GEMV=${SGLANG_NORM_INTO_GEMV:-1}
# 2026-09-06 N1: NVFP4 (W4A16 Triton GEMV, 83-88% of roof) for the draft hot-vocab lm_head (A: 1.61x, -2.15% W16 step; offline acceptance gate run on private data, result not published)
# and the target lm_head (B: 1.70x, -1.4% W4 step; needle PASS). Stage C (GDN qkvz/attn qkv) measured ~0 -> off.
# A in-server (20:44-20:52 vs 20:16-20:23 off): W16 code 11.77->10.05 / agent 5.75->5.20, W4 prose-en 2.58->2.46 -> acceptance loss in-server -> OFF.
export SGLANG_MTP_LMHEAD_NVFP4=${SGLANG_MTP_LMHEAD_NVFP4:-0}
# 2026-09-06 P1/P2: singleton-route pruning (drop routes to experts used by exactly one verify row when weight < tau), evaluated INSIDE
# FlashInfer's fused routing prologue (csrc patch p2-prune-in-prologue, keeps the prologue's overlap): same-hour off->on W16 step -9.5%
# (t/s code 492->543 / prose-en 162->179 / agent 250->310 / prose-ja 147->177), W4 -9..-12%. needle PASS, acceptance flat.
# NOTE: first change that alters the target's computation. PRUNE_TAU=0 disables. Requires the csrc patch (without it the in-prologue path prunes nothing).
export SGLANG_MOE_PRUNE_SINGLETON_TAU=${SGLANG_MOE_PRUNE_SINGLETON_TAU:-${PRUNE_TAU:-0.08}}
export SGLANG_MOE_PRUNE_IN_PROLOGUE=${SGLANG_MOE_PRUNE_IN_PROLOGUE:-1}
# B in-server (20:20 same-hour off/on): acceptance W4 prose-en 2.58->2.38 / agent 3.22->3.05, W16 agent 5.75->5.08 -> net t/s loss -> OFF.
export SGLANG_LMHEAD_NVFP4=${SGLANG_LMHEAD_NVFP4:-0}
# 2026-09-06: FlashInfer csrc patches g2-1/g2-2 (MoE glue: memset+expandInputRows folded, -3.9us/call) + g1-pack-only
# (active-expert group packing): W4 -0.31 / W16 -0.33 ms/step, bit-identical in deterministic mode. Kill switches:
# FLASHINFER_MOE_FOLD_MEMSET=0 FLASHINFER_MOE_FOLD_EXPAND=0 (folds default on); packing is opt-in in the csrc -> enabled here.
export FLASHINFER_MOE_PACK_GROUPS=${FLASHINFER_MOE_PACK_GROUPS:-1}
# 2026-09-06: PDL on the fork's Triton kernels (e58cf349d3): W4 -0.6ms/step (-5.5%); W16 -0.2ms (-1.2%, headless 2x2 A/B 02:35) -> on for all profiles.
export SGLANG_TRITON_PDL=${SGLANG_TRITON_PDL:-1}
# 2026-10-02 shipping: this worktree is opus/cand-1002 (7118260ce3, patch 0121); the flags below are what the wa
# measurements ran (flash-next-bench specs/STACK_2026-10-01.md 5.1, RS2D_DRAFT_SHARPEN_2026-10-01.md 6). Any of them =0 turns it off.
# Frozen copy of the RQ2 u2h FlashInfer (fused MoE prologue + top-k module): a copied cache must never run ninja -> NO_NINJA.
export PYTHONPATH="$HOME/tools/flashinfer-prod-1002:$REPO/python"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME/.cache/sglang-prod-1002}" FLASHINFER_P2_NO_NINJA=1
# Stack (RT1 router top-k, SV1 sparse verify, SV2 FlashInfer top-k, FG1 GDN front overlap, DG1 draft MoE GEMV): wa ABBA -9.6% ms/token LM Studio, -4.6% greedy.
export SGLANG_ROUTER_FAST_TOPK=${SGLANG_ROUTER_FAST_TOPK:-1} SGLANG_OPT_SPEC_SPARSE_VERIFY=${SGLANG_OPT_SPEC_SPARSE_VERIFY:-1}
export SGLANG_OPT_SPEC_SPARSE_TOPK=${SGLANG_OPT_SPEC_SPARSE_TOPK:-1} SGLANG_OPT_GDN_FRONT_OVERLAP=${SGLANG_OPT_GDN_FRONT_OVERLAP:-1}
export SGLANG_OPT_DRAFT_MOE_GEMV=${SGLANG_OPT_DRAFT_MOE_GEMV:-1}
# min_p honored in the speculative verify (LM Studio sends it); ST1 MTP shared-index tail + XA1 Triton decode attention (+2.8% together).
export SGLANG_SPEC_MIN_P=${SGLANG_SPEC_MIN_P:-1}
export SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=${SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX:-1} SGLANG_OPT_TRITON_DECODE_ATTN=${SGLANG_OPT_TRITON_DECODE_ATTN:-1}
# RS package: sparse chain rejection sampling, draft support K=16 sharpened (LM Studio t/s +6.5% pooled, prose-ja +10%, code-edit +1.6% unresolved).
# Needs --speculative-use-rejection-sampling (added to COMMON below); DT1 (SGLANG_OPT_DRAFT_TAIL) was never measured with it -> stays off.
export SGLANG_OPT_SPEC_SPARSE_RS=${SGLANG_OPT_SPEC_SPARSE_RS:-1} SGLANG_RS_DRAFT_TOPK=${SGLANG_RS_DRAFT_TOPK:-16}
export SGLANG_RS_DRAFT_TEMP_SCALE=${SGLANG_RS_DRAFT_TEMP_SCALE:-0.7} SGLANG_RS_DRAFT_ONEHOT_ABOVE=${SGLANG_RS_DRAFT_ONEHOT_ABOVE:-0.9}
export SGLANG_RS_GREEDY_FAST=${SGLANG_RS_GREEDY_FAST:-1} SGLANG_RS_BLOCK_VERIFY=${SGLANG_RS_BLOCK_VERIFY:-1}
# 2026-09-03: MTP draft dense projections FP8 (mtp_dense) + online NVFP4 draft experts
# 2026-09-05: v3 MTP head (soft-target distillation + rollout; W4 prose-en/agent +6-7%, W16 neutral). TARGET_MODEL=<dir> overrides.
# 2026-09-07: v5 MTP head (breadth re-extraction; server A/B vs v3: W4 acc +0.8/+4.5/+4.0/+3.2%, W16 +6.5/+7.5/+3.1/+1.6% code/prose-en/prose-ja/agent, needle PASS). mtpft3 kept as fallback.
export TARGET_MODEL="${TARGET_MODEL:-$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5}"
# TOKEN_MAP=none disables the reduced draft vocabulary (FR-Spec); default hot_32768
TOKEN_MAP="${TOKEN_MAP:-$HOME/tools/flash-next-bench/tokenmaps/hot2_49152.pt}"   # 2026-09-04: blended map (corpora + self-gen argmax), same step time; its acceptance vs hot_32768 was compared only offline on private data (no result published)
# 2026-09-05: the desktop compositor's GPU preemption inflates the decode step ~7.5% (W4) at 4K 160Hz;
# 60Hz recovers most of it (-5.5..6.7% step time, CUPTI A/B). SERVE_DISPLAY_HZ=60 switches the primary
# output to that rate for the server's lifetime and restores the original rate when the server exits.
if [ -n "${SERVE_DISPLAY_HZ:-}" ]; then
  export DISPLAY="${DISPLAY:-:1}"
  _dl=$(xrandr 2>/dev/null | awk '/\*/{print; exit}')
  _out=$(xrandr 2>/dev/null | awk '/ connected primary/{print $1; exit}')
  _mode=$(echo "$_dl" | awk '{print $1}'); _orig=$(echo "$_dl" | grep -oE '[0-9.]+\*' | tr -d '*')
  if [ -n "$_out" ] && [ -n "$_mode" ] && [ -n "$_orig" ] && [ "$_orig" != "$SERVE_DISPLAY_HZ" ] \
     && xrandr --output "$_out" --mode "$_mode" --rate "$SERVE_DISPLAY_HZ" 2>/dev/null; then
    echo "[serve-fast] display $_out $_mode: ${_orig}Hz -> ${SERVE_DISPLAY_HZ}Hz while serving (pid $$)" >&2
    # Double-fork so the restore helper is reparented to init: SGLang's crash path
    # (kill_process_tree by ppid) killed it and left the desktop at 60 Hz (2026-09-09).
    ( setsid bash -c "while kill -0 $$ 2>/dev/null; do sleep 5; done; xrandr --output $_out --mode $_mode --rate $_orig" >/dev/null 2>&1 < /dev/null & )
  fi
fi
# 2026-10-02: above 262144 serve-local.sh turns on factor-2 YaRN; wa then needs ~3.5GB more KV.
# 524288 passed all needles (480k tokens) at 0.96 with a 6.8GB desktop (runs/long/wa-524288-summary.txt).
[ "$PROFILE" = wa ] && [ "${CONTEXT_LENGTH:-0}" -gt 262144 ] && WA_MEM_FRACTION="${WA_LONG_MEM_FRACTION:-0.96}"
COMMON=()
[ "$TOKEN_MAP" != none ] && COMMON+=(--speculative-token-map "$TOKEN_MAP")
COMMON+=(--qwen4-exp-dense-fp8 shared_expert,attn,linear_attn,lm_head,mtp_dense
        --speculative-draft-model-quantization modelopt_fp4 --enable-metrics)
# 2026-10-02: RS runs in the rejection-sampling verify; ngram keeps the target-only verify (RS never measured there).
if [ "$PROFILE" = ngram ]; then export SGLANG_OPT_SPEC_SPARSE_RS=0
elif [ "$SGLANG_OPT_SPEC_SPARSE_RS" = 1 ]; then COMMON+=(--speculative-use-rejection-sampling); fi
case "$PROFILE" in
  # 2026-09-06 P1: drop routes to experts used by exactly one verify row when their routing weight < tau (SGLANG_MOE_PRUNE_SINGLETON_TAU):
  #   W4 step -10% (t/s code +9.9 / prose-en +5.1 / agent +14.5 / prose-ja +13.3%), needle PASS, acceptance flat.
  #   (P2 folded the mask into the prologue -> now on for all profiles via the global export above.)
  #   NOTE: this is the first change that alters the target's computation (low-weight singleton experts skipped). PRUNE_TAU=0 disables.
  w4)    MEM_FRACTION="${MEM_FRACTION:-${W4_MEM_FRACTION:-0.935}}" exec "$REPO/serve-local.sh" "${COMMON[@]}" "$@" ;;
  w8)    MAMBA_SLOTS="${MAMBA_SLOTS:-10}" exec "$REPO/serve-local.sh" "${COMMON[@]}" --speculative-num-steps 7 --speculative-num-draft-tokens 8 "$@" ;;
  # W16: the fused shared-expert gate_up lowers long-chain acceptance (agent-loop 4.5 vs 4.8,
  # 4-repeat A/B 2026-09-04) for ~0.1 ms/step, so it stays off here; the Triton router is harmless.
  w16)   MAMBA_SLOTS="${MAMBA_SLOTS:-10}" SGLANG_SHARED_GATEUP_FUSED="${W16_GATEUP:-0}" SGLANG_ROUTER_GEMV="${W16_ROUTER:-1}" exec "$REPO/serve-local.sh" "${COMMON[@]}" --speculative-num-steps 15 --speculative-num-draft-tokens 16 --mem-fraction-static "${W16_MEM_FRACTION:-0.93}" "$@" ;;
  # WA: one server, per-request adaptive steps {3,15} (926e3661fb): code 463 / prose-en 229 / prose-ja 205 / agent 306 t/s (2026-09-05, vs W4 318/220/208/282, W16 528/156/170/235).
  #     Needs ~0.7GB more than W16 -> mem fraction 0.925 keeps >=4.4GB free for the desktop.
  # 2026-09-06 C1: confidence-mixture policy (real draft top-1 prob on the topk=1 path) + asymmetric hysteresis: code-edit 480->586 t/s, others +-2%, 4 switches/bench.
  # 2026-09-07 X3/WA4: non-initial adaptive states skipped target autotune (slow MoE GEMM + 48 finalize launches/step);
  # SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1 runs the target warmup before capturing candidate graphs (wa only; global default off).
  # 2026-09-08 WA5: three-width confidence policy [3,7,15] (config c: max_grace 80, down_margin, adjacent-only promotion) + step model
  # refit from WIDTH_SWEEP_0907 (7.943 + 0.5554*S). Quiet-night ABAB vs [3,15]: agent-loop +14.4%, code +0.7%, prose +0.2..0.4%, needle PASS.
  # Revert: ADAPTIVE_CONFIG=$HOME/tools/flash-next-bench/adaptive/w16_conf.json SGLANG_ADAPTIVE_STEP_A=9.74 SGLANG_ADAPTIVE_STEP_B=0.70
  wa)    SGLANG_ADAPTIVE_STEP_A="${SGLANG_ADAPTIVE_STEP_A:-7.943}" SGLANG_ADAPTIVE_STEP_B="${SGLANG_ADAPTIVE_STEP_B:-0.5554}" SGLANG_ADAPTIVE_TARGET_AUTOTUNE="${SGLANG_ADAPTIVE_TARGET_AUTOTUNE:-1}" SGLANG_ADAPTIVE_POLICY="${SGLANG_ADAPTIVE_POLICY:-confidence}" MAMBA_SLOTS="${MAMBA_SLOTS:-10}" SGLANG_SHARED_GATEUP_FUSED="${W16_GATEUP:-0}" SGLANG_ROUTER_GEMV="${W16_ROUTER:-1}" exec "$REPO/serve-local.sh" "${COMMON[@]}" --speculative-num-steps 15 --speculative-num-draft-tokens 16 --mem-fraction-static "${WA_MEM_FRACTION:-0.925}" --speculative-adaptive --speculative-adaptive-config "${ADAPTIVE_CONFIG:-$HOME/tools/flash-next-bench/adaptive/w16_3_7_15_c.json}" "$@" ;;
  ngram) MAMBA_SLOTS="${MAMBA_SLOTS:-10}" exec "$REPO/serve-local.sh" "${COMMON[@]}" --speculative-algorithm NGRAM_CHAIN --speculative-num-steps 15 --speculative-num-draft-tokens 16 "$@" ;;
  *) echo "usage: $0 {w4|w8|w16|wa|ngram} [extra sglang args]" >&2; exit 2 ;;
esac
