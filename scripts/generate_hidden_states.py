#!/usr/bin/env python3
"""Task 1: offline hidden-state generation for EAGLE-3 training.

Orchestrates the offline workflow. It runs the speculators-side steps in THIS
env (speculators_venv) and launches the vLLM extraction server in the vLLM env
(vllm_venv) as a separate process -- the two talk over HTTP, which is exactly how
the tutorial splits the work (speculators has no vllm dependency; it uses an
openai client to hit the server).

  1. preprocess  -> speculators/scripts/prepare_data.py            [speculators_venv]
                    (ShareGPT .jsonl -> Arrow dataset + token_freq.pt in --work-dir)
  2. serve       -> speculators/scripts/launch_vllm.py             [vllm_venv]
                    (python -m vllm ... --speculative_config extract_hidden_states)
  3. generate    -> speculators/scripts/data_generation_offline.py [speculators_venv]
                    (openai client hits --endpoint, writes .safetensors)

Run inside speculators_venv:

    source speculators_venv/bin/activate
    python generate_hidden_states.py --model Qwen/Qwen3-8B \
        --data data/sharegpt_qwen3.jsonl --work-dir output \
        --seq-length 2048 --max-samples 3000 --min-free-gb 20

By default it auto-launches the server using ./vllm_venv/bin/python (override
with --vllm-python), or pass --no-manage-server to run against an already-running
server at --endpoint (start it yourself in vllm_venv via launch_vllm.py).

The input .jsonl is what prepare_data.py (our downloader) produced; speculators'
loader reads local .jsonl with a "conversations" key and from/value (or
role/content) turns, so the format matches.

Why so much disk? Each token is stored as high-dimensional BF16 hidden-state
vectors from several layers, not a few bytes of text. A few thousand 2048-token
samples easily reaches ~140GB, so this script guards free disk and aborts early.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = SCRIPT_DIR / "speculators"
# The extraction server needs vLLM, which lives in vllm_venv (not speculators_venv).
DEFAULT_VLLM_PY = SCRIPT_DIR / "vllm_venv" / "bin" / "python"
TMP_HIDDEN = Path("/tmp/hidden_states")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="Verifier model whose hidden states are captured.")
    p.add_argument("--data", default="data/sharegpt_qwen3.jsonl",
                   help="ShareGPT .jsonl produced by prepare_data.py.")
    p.add_argument("--work-dir", default="output",
                   help="Preprocessed dataset dir; hidden states go to "
                        "<work-dir>/hidden_states.")
    p.add_argument("--repo", default=str(DEFAULT_REPO),
                   help="Path to the cloned speculators repo (has scripts/).")
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=3000)
    p.add_argument("--concurrency", type=int, default=32,
                   help="Simultaneous vLLM requests during generation.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--endpoint", default=None,
                   help="vLLM OpenAI endpoint (default http://localhost:<port>/v1).")
    p.add_argument("--vllm-python", default=str(DEFAULT_VLLM_PY),
                   help="Python interpreter that has vLLM (vllm_venv) used to "
                        "auto-launch the extraction server.")
    p.add_argument("--no-manage-server", action="store_true",
                   help="Do not launch/stop the server; assume one is already "
                        "running at --endpoint (start it in vllm_venv).")
    p.add_argument("--min-free-gb", type=float, default=20.0,
                   help="Abort if free disk on --work-dir falls below this.")
    p.add_argument("--startup-timeout", type=int, default=900,
                   help="Seconds to wait for the vLLM /health endpoint.")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Reuse an existing preprocessed dataset in --work-dir.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without running them.")
    return p.parse_args()


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def script_path(repo: Path, name: str) -> Path:
    p = repo / "scripts" / name
    if not p.exists():
        sys.exit(f"ERROR: {p} not found. Set --repo to the cloned speculators "
                 "repo (env_setup.sh clones it next to this script).")
    return p


def run(cmd: list[str], dry: bool, **kw) -> None:
    print("==> Running:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return
    rc = subprocess.call(cmd, **kw)
    if rc != 0:
        sys.exit(f"ERROR: command failed (exit {rc}): {cmd[0]} {cmd[1] if len(cmd) > 1 else ''}")


def wait_for_health(port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def preflight(args: argparse.Namespace) -> Path:
    data = Path(args.data)
    if not data.exists():
        sys.exit(f"ERROR: dataset {data} not found. Run prepare_data.py first.")
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # Clear stale temp hidden states that cause "missing temporary file" errors.
    if TMP_HIDDEN.exists():
        print(f"==> Clearing stale {TMP_HIDDEN}")
        shutil.rmtree(TMP_HIDDEN, ignore_errors=True)
    have = free_gb(work)
    print(f"==> Free disk on {work}: {have:.1f} GB (min {args.min_free_gb})")
    if have < args.min_free_gb:
        sys.exit("ERROR: not enough free disk. Reduce --max-samples or free "
                 "space (a few thousand samples ~140GB).")
    return work


def preprocess(args: argparse.Namespace, work: Path) -> None:
    if args.skip_preprocess:
        print("==> Skipping preprocess (reusing existing dataset).")
        return
    cmd = [
        sys.executable, str(script_path(Path(args.repo), "prepare_data.py")),
        "--model", args.model,
        "--data", str(Path(args.data).resolve()),
        "--output", str(work.resolve()),
        "--seq-length", str(args.seq_length),
        "--max-samples", str(args.max_samples),
        # Keep token_freq.pt inside work-dir (default lands in cwd); train.py
        # looks for it there when --draft-vocab-size is used.
        "--token-freq-path", str((work / "token_freq.pt").resolve()),
    ]
    run(cmd, args.dry_run, cwd=args.repo)


def start_server(args: argparse.Namespace):
    """Launch the vLLM extraction server using the vllm_venv interpreter."""
    vllm_py = Path(args.vllm_python)
    if not vllm_py.exists():
        sys.exit(f"ERROR: vLLM interpreter {vllm_py} not found. The extraction "
                 "server needs vLLM (vllm_venv), not speculators_venv. Pass "
                 "--vllm-python <vllm_venv>/bin/python, or start the server "
                 "yourself and rerun with --no-manage-server.")
    repo_launch = script_path(Path(args.repo), "launch_vllm.py")
    # speculators launch_vllm.py: `launch_vllm.py MODEL -- <vllm args>`
    serve_cmd = [str(vllm_py), str(repo_launch), args.model, "--",
                 "--port", str(args.port)]
    print("==> Launching vLLM (hidden-state extraction):",
          " ".join(shlex.quote(c) for c in serve_cmd), flush=True)
    return subprocess.Popen(serve_cmd, cwd=args.repo, preexec_fn=os.setsid)


def generate(args: argparse.Namespace, work: Path) -> None:
    hs_out = work / "hidden_states"
    endpoint = args.endpoint or f"http://localhost:{args.port}/v1"
    gen_cmd = [
        sys.executable,
        str(script_path(Path(args.repo), "data_generation_offline.py")),
        "--preprocessed-data", str(work.resolve()),
        "--endpoint", endpoint,
        "--output", str(hs_out.resolve()),
        "--max-samples", str(args.max_samples),
        "--concurrency", str(args.concurrency),
        "--validate-outputs",
    ]

    if args.dry_run:
        if not args.no_manage_server:
            print("==> Would launch server with:", args.vllm_python)
        print("==> Would run:", " ".join(shlex.quote(c) for c in gen_cmd))
        return

    server = None if args.no_manage_server else start_server(args)
    try:
        if args.no_manage_server:
            print(f"==> Using already-running server at {endpoint}")
        if not wait_for_health(args.port, args.startup_timeout):
            raise RuntimeError(f"vLLM server not healthy at {endpoint}; if you "
                               "started it yourself, check the port/logs")
        print("==> Server healthy; starting hidden-state generation")
        print("==> Running:", " ".join(shlex.quote(c) for c in gen_cmd), flush=True)
        gen = subprocess.Popen(gen_cmd)
        # Poll: disk guard + server liveness while generation runs.
        while gen.poll() is None:
            if free_gb(work) < args.min_free_gb:
                gen.terminate()
                raise RuntimeError("free disk dropped below threshold mid-run; "
                                   "reduce --max-samples and retry")
            if server is not None and server.poll() is not None:
                gen.terminate()
                raise RuntimeError("vLLM server died during generation; check its logs")
            try:
                gen.wait(timeout=30)
            except subprocess.TimeoutExpired:
                continue
        if gen.returncode:
            raise RuntimeError(f"data_generation_offline.py exited {gen.returncode}")
    finally:
        if server is not None:
            print("==> Stopping vLLM server", flush=True)
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGINT)
                server.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(server.pid), signal.SIGKILL)
                except Exception:
                    pass


def main() -> int:
    args = parse_args()
    work = preflight(args)
    preprocess(args, work)
    generate(args, work)

    if not args.dry_run:
        hs_out = work / "hidden_states"
        used = sum(f.stat().st_size for f in hs_out.rglob("*") if f.is_file()) \
            if hs_out.exists() else 0
        print(f"\n==> Done. Hidden states in {hs_out} (~{used / 1e9:.1f} GB)")
        print("    --validate-outputs already checked that hidden-state seq-len "
              "matches token count; if it failed, verify the vLLM version first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
