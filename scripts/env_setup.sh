#!/usr/bin/env bash
# Task 1: create three isolated environments for the HW3 pipeline.
#   speculators_venv - data prep, hidden-state generation, EAGLE-3 training
#   vllm_venv        - vLLM serving + benchmarking
#   comp_venv        - FP8 dynamic quantization (llmcompressor)
#
# The speculators training stack and the vLLM serving stack have conflicting
# dependencies, so they MUST stay in separate venvs. Run this once on the H100 box.
#
# Usage:
#   bash env_setup.sh
#
# Requires: python3.12, git, ~140GB free disk for later hidden-state generation.
set -euo pipefail

# --- Config ---------------------------------------------------------------
PYTHON="${PYTHON:-python3.12}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECULATORS_TAG="${SPECULATORS_TAG:-v0.5.0}"
VLLM_VERSION="${VLLM_VERSION:-0.20.0}"
LLMCOMPRESSOR_VERSION="${LLMCOMPRESSOR_VERSION:-0.12.0}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
# Shared HF cache so the 8B model is downloaded only once for all three venvs.
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/../.hf_home}"

cd "$SCRIPT_DIR"
echo "==> Using python: $($PYTHON --version)"
echo "==> HF_HOME: $HF_HOME"
mkdir -p "$HF_HOME"

# --- 1. speculators_venv (editable install from source, pinned tag) --------
if [[ ! -d speculators_venv ]]; then
  echo "==> Creating speculators_venv"
  "$PYTHON" -m venv speculators_venv
fi
if [[ ! -d speculators ]]; then
  echo "==> Cloning speculators @ $SPECULATORS_TAG"
  git clone https://github.com/vllm-project/speculators.git
fi
git -C speculators fetch --tags --quiet
git -C speculators checkout "$SPECULATORS_TAG"
# shellcheck disable=SC1091
source speculators_venv/bin/activate
pip install --upgrade pip
pip install -e ./speculators
# Extras used by prepare_data.py (dataset download + Qwen3 chat template).
# transformers/torch normally come with speculators; datasets often does not.
# pip install "datasets>=2.18" "transformers>=4.44" huggingface_hub
deactivate

# --- 2. vllm_venv (serving + benchmark runtime) ----------------------------
if [[ ! -d vllm_venv ]]; then
  echo "==> Creating vllm_venv"
  "$PYTHON" -m venv vllm_venv
fi
# shellcheck disable=SC1091
source vllm_venv/bin/activate
pip install --upgrade pip
pip install "vllm==${VLLM_VERSION}" "fastapi<0.137"
# `vllm bench serve --dataset-name hf` (mt-bench) needs datasets + pandas to
# load and sample the benchmark prompts.
pip install "datasets>=2.18" pandas
deactivate

# --- 3. comp_venv (FP8 quantization) ---------------------------------------
if [[ ! -d comp_venv ]]; then
  echo "==> Creating comp_venv"
  "$PYTHON" -m venv comp_venv
fi
# shellcheck disable=SC1091
source comp_venv/bin/activate
pip install --upgrade pip
pip install "llmcompressor==${LLMCOMPRESSOR_VERSION}"
# quantize_fp8.py loads the 8B model with device_map="auto" -> needs accelerate.
pip install accelerate
deactivate

# --- 4. Pre-download the verifier once into the shared cache ---------------
echo "==> Pre-downloading $MODEL into shared HF cache"
# shellcheck disable=SC1091
source vllm_venv/bin/activate
python - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download("${MODEL}", cache_dir=os.environ["HF_HOME"])
print("model cached")
PY
deactivate

echo
echo "==> Done. Environments ready:"
echo "    speculators_venv  (train/data)   -> source $SCRIPT_DIR/speculators_venv/bin/activate"
echo "    vllm_venv         (serve/bench)  -> source $SCRIPT_DIR/vllm_venv/bin/activate"
echo "    comp_venv         (quantize)     -> source $SCRIPT_DIR/comp_venv/bin/activate"
echo "    Remember to 'export HF_HOME=$HF_HOME' in every shell."
