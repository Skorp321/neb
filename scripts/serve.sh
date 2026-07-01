#!/usr/bin/env bash
# Task 4: launch a vLLM server for one of the four benchmark configurations.
#
# All four configs must be served with identical runtime knobs (prefix caching
# off, fixed seed) so the benchmark comparison is fair. Speculative decoding is
# enabled by passing --draft-head + --num-spec-tokens.
#
# Run inside vllm_venv. Examples:
#   # Baseline BF16
#   bash serve.sh --model Qwen/Qwen3-8B
#   # FP8 quantized
#   bash serve.sh --model Qwen3-8B-FP8-Dynamic
#   # BF16 + EAGLE-3 draft head, 2 speculative tokens
#   bash serve.sh --model Qwen/Qwen3-8B \
#        --draft-head output/checkpoints/best --num-spec-tokens 2
#   # FP8 + EAGLE-3, 1 speculative token
#   bash serve.sh --model Qwen3-8B-FP8-Dynamic \
#        --draft-head output/checkpoints/best --num-spec-tokens 1
#
# The server runs in the foreground; Ctrl-C to stop, or background it and poll
# the /health endpoint before benchmarking.
set -euo pipefail

MODEL=""
DRAFT_HEAD=""
NUM_SPEC_TOKENS=""
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
SEED="${SEED:-0}"
# Empty = use the model's default context length (matches the "out of the box"
# reference benchmark conditions). Set MAX_MODEL_LEN=4096 to cap it explicitly.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --draft-head) DRAFT_HEAD="$2"; shift 2 ;;
    --num-spec-tokens) NUM_SPEC_TOKENS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ERROR: --model is required" >&2
  exit 2
fi

CMD=(vllm serve "$MODEL"
     --host "$HOST" --port "$PORT"
     --seed "$SEED"
     --gpu-memory-utilization "$GPU_MEM_UTIL"
     --no-enable-prefix-caching)   # keep prefix caching OFF for clean benchmarks
if [[ -n "$MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi

# Enable EAGLE-3 speculative decoding when a draft head is provided.
if [[ -n "$DRAFT_HEAD" ]]; then
  : "${NUM_SPEC_TOKENS:?--num-spec-tokens is required when --draft-head is set}"
  SPEC_CONFIG=$(printf '{"method": "eagle3", "model": "%s", "num_speculative_tokens": %s}' \
                "$DRAFT_HEAD" "$NUM_SPEC_TOKENS")
  CMD+=(--speculative-config "$SPEC_CONFIG")
  echo "==> Speculative decoding ON: $SPEC_CONFIG"
else
  echo "==> Speculative decoding OFF (baseline / FP8-only)"
fi

echo "==> ${CMD[*]}"
exec "${CMD[@]}"
