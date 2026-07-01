#!/usr/bin/env python3
"""Task 1 helper: launch the vLLM server for offline hidden-state extraction.

Hidden-state generation requires a vLLM instance configured to return hidden
states. That special configuration lives in the speculators repo's own
``scripts/launch_vllm.py`` (invocation: ``launch_vllm.py MODEL -- <vllm args>``).
This thin shim just execs that script so the extraction config is always
correct, regardless of vLLM version.

Run inside vllm_venv (the extraction server needs vLLM, which is NOT installed
in speculators_venv):

    source vllm_venv/bin/activate
    python launch_vllm.py Qwen/Qwen3-8B --port 8000
    # forward extra vLLM flags after --:
    python launch_vllm.py Qwen/Qwen3-8B --port 8000 -- --gpu-memory-utilization 0.9

generate_hidden_states.py can start this for you (it points at
vllm_venv/bin/python); run it manually only for debugging or when using
generate_hidden_states.py --no-manage-server.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parent / "speculators"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="Verifier model id/path, e.g. Qwen/Qwen3-8B.")
    p.add_argument("--repo", default=str(DEFAULT_REPO),
                   help="Path to the cloned speculators repo (has scripts/).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra vLLM args after -- (forwarded verbatim).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_script = Path(args.repo) / "scripts" / "launch_vllm.py"
    if not repo_script.exists():
        sys.exit(f"ERROR: {repo_script} not found. Set --repo to the cloned "
                 "speculators repo (env_setup.sh clones it next to this script).")

    # argparse.REMAINDER keeps the leading "--"; drop it if present.
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    # speculators expects: launch_vllm.py MODEL -- <vllm args...>
    cmd = [sys.executable, str(repo_script), args.model, "--",
           "--port", str(args.port), *extra]
    print("==> Launching:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.call(cmd, cwd=args.repo)


if __name__ == "__main__":
    sys.exit(main())
