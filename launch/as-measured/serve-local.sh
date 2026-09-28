#!/bin/bash
# レビュー反映版の起動スクリプト (serve-flash-next.sh ベース)
# 変更点: 127.0.0.1バインド / trust-remote-code除去 / NIXL・HiCache無効 /
#         ctx 262144(YaRN外挿なし) / mem-fraction 0.92 / CUDAはmamba env
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
export CUDA_HOME=$HOME/tools/mamba/envs/cuda13
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="$REPO/.venv/bin:$CUDA_HOME/bin:$PATH"
# conda-layout CUDA: headers live under targets/, not $CUDA_HOME/include
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=12.0
CACHE_BASE="$REPO/.cache"
mkdir -p "$CACHE_BASE"
export HF_HOME="$CACHE_BASE/hf" XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TRITON_CACHE_DIR="$CACHE_BASE/triton" SGLANG_JIT_CACHE_DIR="$CACHE_BASE/jit"
TARGET_MODEL="${TARGET_MODEL:-$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}"
MEM_FRACTION="${MEM_FRACTION:-0.96}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
PORT="${PORT:-8001}"
# mambaステートプール: spec込みで1リクエスト=9スロット(S=5+D=4)。24=2並列分。
# シングルユーザーなら10に絞ると0.8GBをKVプールへ回せる
MAMBA_SLOTS="${MAMBA_SLOTS:-24}"
# KVプールのトークン数直接指定(未指定=SGLang自動見積り。自動値はこの構成では39424と
# 極端に保守的なので、長文が要るときは明示する。1トークン≈13.5KB)
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"

# 262144(ネイティブ上限)超は元フォークのqualified構成どおりfactor-2 YaRNで524288まで拡張
EXTRA_ARGS=()
if [ -n "$MAX_TOTAL_TOKENS" ]; then
  EXTRA_ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi
if [ "$CONTEXT_LENGTH" -gt 262144 ]; then
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  EXTRA_ARGS+=(--json-model-override-args '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}')
fi

exec "$REPO/.venv/bin/sglang" serve \
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
