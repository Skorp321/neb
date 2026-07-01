#!/usr/bin/env python3
"""Task 4: tune num_speculative_tokens for a speculative-decoding config.

For each candidate value it:
  1. starts a vLLM server via serve.sh with EAGLE-3 + that many draft tokens,
  2. waits for /health,
  3. runs run_benchmark.sh,
  4. parses Output token throughput, TPOT, and acceptance rate,
  5. tears the server down,
and finally prints a comparison table and the best value by output tok/s.

The optimal draft-token count differs per verifier (reference: BF16 -> 2,
FP8 -> 1) because more draft tokens only pay off while the extra accepted tokens
outrun the added draft+verify overhead. Tune the BF16 and FP8 configs
separately -- do NOT reuse one value for both.

Run inside vllm_venv:

    source vllm_venv/bin/activate
    python sweep_draft_tokens.py --model Qwen/Qwen3-8B \
        --draft-head output/checkpoints/best --values 1 2 3 4
    python sweep_draft_tokens.py --model Qwen3-8B-FP8-Dynamic \
        --draft-head output/checkpoints/best --values 1 2 3
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

FLOAT = r"([0-9]+\.?[0-9]*)"
PATTERNS = {
    "output_tok_s": re.compile(r"Output token throughput \(tok/s\):\s*" + FLOAT),
    "total_tok_s": re.compile(r"Total token throughput \(tok/s\):\s*" + FLOAT),
    "mean_tpot_ms": re.compile(r"Mean TPOT \(ms\):\s*" + FLOAT),
    "mean_ttft_ms": re.compile(r"Mean TTFT \(ms\):\s*" + FLOAT),
    # vLLM prints spec-decoding acceptance in the server log / bench summary.
    "acceptance_rate": re.compile(r"[Aa]cceptance rate:?\s*" + FLOAT),
    "acceptance_length": re.compile(r"[Aa]cceptance length:?\s*" + FLOAT),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="BF16 or FP8 model id/path to serve.")
    p.add_argument("--draft-head", required=True,
                   help="Path to the trained EAGLE-3 draft head (e.g. best).")
    p.add_argument("--values", type=int, nargs="+", default=[1, 2, 3],
                   help="Candidate num_speculative_tokens values to sweep.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=int, default=600,
                   help="Seconds to wait for the server /health endpoint.")
    p.add_argument("--label-prefix", default="spec",
                   help="Prefix for saved results/*.txt files.")
    return p.parse_args()


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


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pat in PATTERNS.items():
        m = pat.search(text)
        if m:
            out[key] = float(m.group(1))
    return out


def run_one(args: argparse.Namespace, n_spec: int) -> dict[str, float]:
    label = f"{args.label_prefix}_k{n_spec}"
    env = dict(os.environ, PORT=str(args.port))
    serve_cmd = [
        "bash", str(SCRIPT_DIR / "serve.sh"),
        "--model", args.model,
        "--draft-head", args.draft_head,
        "--num-spec-tokens", str(n_spec),
        "--port", str(args.port),
    ]
    print(f"\n===== num_speculative_tokens = {n_spec} =====")
    print("==> starting server:", " ".join(serve_cmd), flush=True)
    server = subprocess.Popen(serve_cmd, env=env, preexec_fn=os.setsid)
    try:
        if not wait_for_health(args.port, args.startup_timeout):
            raise RuntimeError("server did not become healthy in time")
        bench_cmd = [
            "bash", str(SCRIPT_DIR / "run_benchmark.sh"), args.model, label,
        ]
        print("==> benchmarking:", " ".join(bench_cmd), flush=True)
        proc = subprocess.run(bench_cmd, env=env, capture_output=True, text=True)
        combined = proc.stdout + proc.stderr
        sys.stdout.write(proc.stdout)
        metrics = parse_metrics(combined)
        metrics["num_spec_tokens"] = float(n_spec)
        return metrics
    finally:
        print("==> stopping server", flush=True)
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
    if not Path(args.draft_head).exists():
        print(f"WARNING: draft head {args.draft_head} not found on disk; "
              "make sure train_eagle3.py produced it.", file=sys.stderr)

    rows: list[dict[str, float]] = []
    for n in args.values:
        try:
            rows.append(run_one(args, n))
        except Exception as exc:  # keep sweeping remaining values
            print(f"==> k={n} failed: {exc}", file=sys.stderr)

    if not rows:
        sys.exit("ERROR: no successful benchmark runs.")

    cols = ["num_spec_tokens", "output_tok_s", "total_tok_s",
            "mean_tpot_ms", "mean_ttft_ms", "acceptance_rate", "acceptance_length"]
    print("\n================= SWEEP SUMMARY =================")
    print("\t".join(cols))
    for r in rows:
        print("\t".join(f"{r.get(c, float('nan')):.2f}" for c in cols))

    best = max(rows, key=lambda r: r.get("output_tok_s", 0.0))
    print(f"\n==> Best: num_speculative_tokens = {int(best['num_spec_tokens'])} "
          f"at {best.get('output_tok_s', 0):.2f} tok/s")
    print("    Justify with acceptance rate/length + TPOT: only increase draft "
          "tokens while accepted work grows faster than the added overhead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
