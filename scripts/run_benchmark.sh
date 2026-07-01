#!/usr/bin/env bash
# Task 4: benchmark a running vLLM server with identical settings for every
# configuration. Start a server with serve.sh first (in another shell), then run
# this against it.
#
# Fixed across all four configs: dataset (mt-bench), 80 prompts, concurrency 8,
# fixed seed. Only the served model / speculative config should change.
#
# Run inside vllm_venv:
#   bash run_benchmark.sh Qwen/Qwen3-8B
#   bash run_benchmark.sh Qwen3-8B-FP8-Dynamic
#
# Optionally tee the output into results/ so it can be pasted into the notebook:
#   bash run_benchmark.sh Qwen/Qwen3-8B baseline
set -euo pipefail

MODEL="${1:?usage: run_benchmark.sh <model> [label]}"
LABEL="${2:-}"

# Fixed benchmark knobs -- do NOT vary these between configurations.
DATASET_NAME="${DATASET_NAME:-hf}"
DATASET_PATH="${DATASET_PATH:-philschmid/mt-bench}"
NUM_PROMPTS="${NUM_PROMPTS:-80}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
SEED="${SEED:-0}"
PORT="${PORT:-8000}"

CMD=(vllm bench serve
     --model "$MODEL"
     --dataset-name "$DATASET_NAME"
     --dataset-path "$DATASET_PATH"
     --num-prompts "$NUM_PROMPTS"
     --max-concurrency "$MAX_CONCURRENCY"
     --seed "$SEED"
     --port "$PORT")

echo "==> ${CMD[*]}"

if [[ -n "$LABEL" ]]; then
  mkdir -p results
  OUT="results/${LABEL}.txt"
  "${CMD[@]}" | tee "$OUT"
  echo "==> Saved benchmark output to $OUT (paste into the notebook)."
else
  "${CMD[@]}"
fi
