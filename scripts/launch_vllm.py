#!/usr/bin/env python3
"""Task 1 helper: launch a vLLM OpenAI-compatible server.

The offline EAGLE-3 tutorial ships its own ``scripts/launch_vllm.py``. This is a
thin, self-contained equivalent so the pipeline works even if the cloned repo's
copy moves. It simply shells out to ``vllm serve`` with the flags we need for
hidden-state generation (Task 1) and can be reused for benchmarking (Task 4).

Run inside vllm_venv:

    source vllm_venv/bin/activate
    python launch_vllm.py --model Qwen/Qwen3-8B --port 8000

Prefer serve.sh for the Task 4 benchmark configs; this script exists mainly for
the data-generation step, which needs a plain BF16 server.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-enable-prefix-caching", action="store_true",
                   help="Disable prefix caching (recommended for clean benchmarks).")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra args forwarded verbatim to 'vllm serve'.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        "vllm", "serve", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
        "--seed", str(args.seed),
    ]
    if args.no_enable_prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    # argparse.REMAINDER keeps a leading "--"; drop it if present.
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    cmd.extend(extra)

    print("==> Launching:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
