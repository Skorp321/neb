#!/usr/bin/env python3
"""Task 3: FP8 dynamic quantization of the verifier with llmcompressor.

Applies weight+activation FP8 dynamic quantization to the linear layers of
Qwen/Qwen3-8B, keeps lm_head in high precision, and saves a NEW model directory
(never overwrites the base checkpoint). Mirrors the llm-compressor W8A8 FP8
example:
https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/README.md

Run inside comp_venv:

    source comp_venv/bin/activate
    python quantize_fp8.py --model Qwen/Qwen3-8B --out Qwen3-8B-FP8-Dynamic

Why FP8 dynamic on H100? H100 has native FP8 tensor cores, so FP8 roughly
doubles matmul throughput and halves weight memory/bandwidth versus BF16.
"Dynamic" computes activation scales at runtime, so no calibration data is
needed and accuracy stays close to BF16. lm_head is excluded because it shapes
the full-vocabulary logit distribution: quantizing it tends to cost accuracy for
little memory saving.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="Base BF16 model id or path (left untouched).")
    p.add_argument("--out", default="Qwen3-8B-FP8-Dynamic",
                   help="Output directory for the quantized model.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ERROR: {out} already exists and is non-empty. Remove it or "
                 "choose another --out; refusing to overwrite.")

    try:
        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        sys.exit(f"ERROR: import failed ({exc}). Activate comp_venv "
                 "(llmcompressor==0.12.0).")

    print(f"==> Loading base model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # FP8 dynamic: per-channel weights, dynamic per-token activations, no
    # calibration dataset required. Ignore lm_head.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    print("==> Applying FP8_DYNAMIC oneshot quantization (targets=Linear, "
          "ignore=lm_head)")
    oneshot(model=model, recipe=recipe)

    print(f"==> Saving quantized model to {out}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), save_compressed=True)
    tokenizer.save_pretrained(str(out))

    # Verify the saved config actually carries a quantization section.
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    qcfg = cfg.get("quantization_config") or cfg.get("compression_config")
    if not qcfg:
        sys.exit("ERROR: saved config has no quantization/compression section; "
                 "quantization did not persist correctly.")
    print("==> Quantization config found in config.json:")
    print(json.dumps(qcfg, indent=2)[:2000])
    print(f"\n==> Done. Base model {args.model} left untouched; quantized model "
          f"at {out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
