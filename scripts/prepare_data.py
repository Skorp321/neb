#!/usr/bin/env python3
"""Task 1: prepare ShareGPT-style conversations for offline EAGLE-3 training.

Downloads a ShareGPT dataset, keeps only well-formed multi-turn conversations,
renders them with the Qwen3 chat template, drops samples that exceed the target
sequence length after tokenization, and writes a JSONL file that the
hidden-state generator consumes.

Run inside speculators_venv:

    source speculators_venv/bin/activate
    python prepare_data.py --model Qwen/Qwen3-8B \
        --max-samples 3000 --seq-len 2048 \
        --out data/sharegpt_qwen3.jsonl

Notes:
    * More samples helps draft-head quality more than most other knobs, but
      hidden states are huge on disk (~140GB for a few thousand samples), so
      start at 3000 and scale up only if you have disk headroom.
    * The output schema is one conversation per line:
          {"conversations": [{"role": "user"|"assistant", "content": "..."}, ...]}
      Adjust --output-format if your speculators tag expects a different key.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from transformers import AutoTokenizer

# ShareGPT role strings vary across mirrors; normalize them here.
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="Tokenizer/verifier model id used for the chat template.")
    p.add_argument("--dataset", default="anon8231489123/ShareGPT_Vicuna_unfiltered",
                   help="HF dataset id with ShareGPT-style conversations.")
    p.add_argument("--data-file",
                   default="ShareGPT_V3_unfiltered_cleaned_split.json",
                   help="Raw JSON array file inside the dataset repo. Many "
                        "ShareGPT repos ship a single big JSON that load_dataset "
                        "cannot auto-detect; we download this file directly. "
                        "Pass an empty string ('') to use the datasets loader "
                        "instead (for parquet-based repos).")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--conversations-key", default="conversations",
                   help="Field holding the list of turns in each record.")
    p.add_argument("--max-samples", type=int, default=3000,
                   help="Number of conversations to keep after filtering.")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Max tokenized length; longer conversations are skipped.")
    p.add_argument("--min-turns", type=int, default=2)
    p.add_argument("--out", default="data/sharegpt_qwen3.jsonl")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def normalize_turns(raw_turns: list[dict]) -> list[dict] | None:
    """Map arbitrary ShareGPT role/value keys to {"role","content"} turns."""
    turns: list[dict] = []
    for t in raw_turns:
        role_raw = t.get("from") or t.get("role")
        content = t.get("value") if "value" in t else t.get("content")
        if role_raw is None or content is None:
            return None
        role = ROLE_MAP.get(str(role_raw).lower())
        if role is None:
            return None
        turns.append({"role": role, "content": str(content)})
    return turns or None


def iter_records(args: argparse.Namespace):
    """Yield raw conversation records from either a raw JSON file or datasets.

    Raw-JSON ShareGPT repos (e.g. anon8231489123/ShareGPT_Vicuna_unfiltered)
    store one big JSON array in a single file that ``load_dataset`` cannot
    auto-infer, so we download that file with ``hf_hub_download`` and iterate it.
    For parquet-based repos, pass ``--data-file ''`` to use the streaming
    ``datasets`` loader instead.
    """
    if args.data_file:
        from huggingface_hub import hf_hub_download
        print(f"==> Downloading raw JSON {args.dataset}:{args.data_file}")
        local = hf_hub_download(repo_id=args.dataset, filename=args.data_file,
                                repo_type="dataset")
        print(f"==> Loading {local} into memory")
        with open(local, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        random.Random(args.seed).shuffle(data)
        yield from data
        return

    from datasets import load_dataset
    print(f"==> Streaming dataset {args.dataset}:{args.dataset_split}")
    ds = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
    yield from ds


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"==> Loading tokenizer for {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = iter_records(args)

    kept = 0
    scanned = 0
    skipped_len = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for record in ds:
            if kept >= args.max_samples:
                break
            scanned += 1
            raw = record.get(args.conversations_key)
            if not isinstance(raw, list):
                continue
            turns = normalize_turns(raw)
            if turns is None or len(turns) < args.min_turns:
                continue
            # Render with the model's own chat template so token boundaries match
            # exactly what the verifier will see during hidden-state generation.
            try:
                rendered = tok.apply_chat_template(
                    turns, tokenize=False, add_generation_prompt=False)
            except Exception:
                continue
            n_tokens = len(tok(rendered, add_special_tokens=False)["input_ids"])
            if n_tokens > args.seq_len:
                skipped_len += 1
                continue
            fh.write(json.dumps({"conversations": turns}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 250 == 0:
                print(f"    kept={kept} scanned={scanned} skipped_len={skipped_len}")

    size_mb = out_path.stat().st_size / 1e6
    print(f"==> Wrote {kept} conversations to {out_path} ({size_mb:.1f} MB)")
    print(f"    scanned={scanned} skipped_too_long={skipped_len}")
    if kept < args.max_samples:
        print("    WARNING: fewer samples than requested; loosen filters or "
              "pick a larger dataset split.")


if __name__ == "__main__":
    main()
