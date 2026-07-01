#!/usr/bin/env python3
"""Task 1: offline hidden-state generation for EAGLE-3 training.

"Offline" EAGLE-3 precomputes the verifier's hidden states for every training
token *before* the draft head is trained. This wraps the speculators data-
generation entrypoint so the whole thing runs sequentially on a single GPU.

Run inside speculators_venv:

    source speculators_venv/bin/activate
    python generate_hidden_states.py --model Qwen/Qwen3-8B \
        --data data/sharegpt_qwen3.jsonl \
        --out data/hidden_states --min-free-gb 20

Why so much disk? Each token is stored as high-dimensional BF16 hidden-state
vectors from several layers, not a few bytes of text:
    seq_len * hidden_size * num_saved_layers * 2 bytes  per sample.
A few thousand 2048-token samples easily reaches ~140GB, so this script guards
free disk and aborts early instead of filling the volume.

Troubleshooting baked in:
    * clears stale /tmp/hidden_states/* before running (fixes "missing temp file")
    * refuses to start if free disk < --min-free-gb
    * exits non-zero with a clear message if the speculators CLI is missing
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

TMP_HIDDEN = Path("/tmp/hidden_states")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="Verifier model whose hidden states are captured.")
    p.add_argument("--data", default="data/sharegpt_qwen3.jsonl",
                   help="JSONL produced by prepare_data.py.")
    p.add_argument("--out", default="data/hidden_states",
                   help="Output directory for the generated hidden states.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=3000)
    p.add_argument("--min-free-gb", type=float, default=20.0,
                   help="Abort if free disk on --out falls below this.")
    p.add_argument("--tp-size", type=int, default=1,
                   help="Tensor-parallel size for the vLLM data generator.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command without running it.")
    return p.parse_args()


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def preflight(args: argparse.Namespace) -> None:
    data = Path(args.data)
    if not data.exists():
        sys.exit(f"ERROR: dataset {data} not found. Run prepare_data.py first.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Clear stale temp hidden states that cause "missing temporary file" errors.
    if TMP_HIDDEN.exists():
        print(f"==> Clearing stale {TMP_HIDDEN}")
        shutil.rmtree(TMP_HIDDEN, ignore_errors=True)

    have = free_gb(out)
    print(f"==> Free disk on {out}: {have:.1f} GB (min required {args.min_free_gb})")
    if have < args.min_free_gb:
        sys.exit("ERROR: not enough free disk. Reduce --max-samples or free space "
                 "before generating hidden states (a few thousand samples ~140GB).")


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the speculators offline data-generation command.

    speculators exposes hidden-state generation via its CLI. The exact
    subcommand/flag names are pinned by the v0.5.0 tutorial; adjust here if the
    installed tag differs (check: `speculators --help`).
    """
    return [
        "speculators", "generate-data",
        "--model", args.model,
        "--input", args.data,
        "--output-dir", args.out,
        "--max-seq-len", str(args.seq_len),
        "--max-samples", str(args.max_samples),
        "--tensor-parallel-size", str(args.tp_size),
    ]


def main() -> int:
    args = parse_args()
    preflight(args)

    if shutil.which("speculators") is None:
        sys.exit("ERROR: 'speculators' CLI not found. Activate speculators_venv "
                 "and confirm the editable install (see env_setup.sh).")

    cmd = build_command(args)
    print("==> Running:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return 0

    proc = subprocess.Popen(cmd)
    try:
        while proc.poll() is None:
            # Cheap periodic disk guard: kill generation before the disk fills.
            if free_gb(Path(args.out)) < args.min_free_gb:
                proc.terminate()
                sys.exit("ERROR: free disk dropped below threshold mid-run; "
                         "generation aborted. Reduce --max-samples and retry.")
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                continue
    finally:
        if proc.poll() is None:
            proc.terminate()

    rc = proc.returncode or 0
    if rc == 0:
        used = sum(f.stat().st_size for f in Path(args.out).rglob("*") if f.is_file())
        print(f"==> Done. Hidden states in {args.out} (~{used / 1e9:.1f} GB)")
        print("    Sanity check: hidden-state seq-len must match tokenized "
              "length; if not, verify the vLLM version first.")
    else:
        print(f"==> Generation exited with code {rc}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
