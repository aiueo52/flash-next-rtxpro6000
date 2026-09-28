#!/bin/bash
# Base SGLang launch for Qwen3.8-Flash-Next (NVFP4) on one RTX PRO 6000 (SM120).
# Portable version of launch/as-measured/serve-local.sh (same server flags; paths are variables).
#
# Required:
#   SGLANG_DIR    checkout of jpezzulli/sglang-rtxpro6000 @ 16e5682aad with patches/sglang applied,
#                 with its virtualenv in $SGLANG_DIR/.venv (see docs/reproduce.md)
#   TARGET_MODEL  local copy of RadixArk/Qwen3.8-Flash-Next-NVFP4 (default: $HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4)
# Optional:
#   CUDA_HOME     CUDA 13.x toolkit used for JIT builds (default /usr/local/cuda)
#   MEM_FRACTION CONTEXT_LENGTH PORT MAMBA_SLOTS MAX_TOTAL_TOKENS  (see below)
# Any extra arguments are passed through to `sglang serve` (later flags override earlier ones).
set -euo pipefail
: "${SGLANG_DIR:?set SGLANG_DIR to the patched sglang-rtxpro6000 checkout}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$SGLANG_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
# conda-packaged CUDA keeps its headers under targets/; harmless for a regular toolkit install
if [ -d "$CUDA_HOME/targets/x86_64-linux/include" ]; then
  export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
export TORCH_CUDA_ARCH_LIST=12.0
CACHE_BASE="${CACHE_BASE:-$SGLANG_DIR/.cache}"
mkdir -p "$CACHE_BASE"
export HF_HOME="$CACHE_BASE/hf" XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TRITON_CACHE_DIR="$CACHE_BASE/triton" SGLANG_JIT_CACHE_DIR="$CACHE_BASE/jit"
TARGET_MODEL="${TARGET_MODEL:-$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}"
MEM_FRACTION="${MEM_FRACTION:-0.96}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
PORT="${PORT:-8001}"
# GDN (mamba) state pool: with speculation one request uses ~9 slots; 24 = two concurrent requests.
# For single-user serving 10 is enough and frees ~0.8 GB for the KV pool (the w8/w16/wa profiles set 10).
MAMBA_SLOTS="${MAMBA_SLOTS:-24}"
# Optional explicit KV pool size in tokens (~13.5 KB/token with FP8 KV). Empty = SGLang's own estimate.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"

EXTRA_ARGS=()
if [ -n "$MAX_TOTAL_TOKENS" ]; then
  EXTRA_ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi
# Beyond the native 262144, extend to 524288 with factor-2 YaRN as in the base fork's qualified config.
if [ "$CONTEXT_LENGTH" -gt 262144 ]; then
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  EXTRA_ARGS+=(--json-model-override-args '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}')
fi

exec "$SGLANG_DIR/.venv/bin/sglang" serve \
  --model-path "$TARGET_MODEL" \
  --load-format safetensors \
  --served-model-name flash-next \
  --host 127.0.0.1 --port "$PORT" --tp 1 \
  --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CONTEXT_LENGTH" \
  --page-size 64 --max-running-requests 4 --sleep-on-idle \
  --chunked-prefill-size 8192 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 \
  --max-mamba-cache-size "$MAMBA_SLOTS" --gdn-mtp-cache-mode none \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --ple-offload-embedding \
  --chat-template "$TARGET_MODEL/chat_template.jinja" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant --watchdog-timeout 1800 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  "$@"
