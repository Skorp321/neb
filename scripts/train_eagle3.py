#!/usr/bin/env python3
"""Task 2: train an EAGLE-3 draft head on precomputed hidden states.

Wraps ``torchrun ... speculators/scripts/train.py``. Training reads the
preprocessed dataset + cached hidden states produced in Task 1 and writes
checkpoints under --save-path. The speculators trainer saves the best checkpoint
itself (save_best in its TrainerConfig); enable a logger to inspect per-position
validation metrics.

Run inside speculators_venv:

    source speculators_venv/bin/activate
    python train_eagle3.py --verifier Qwen/Qwen3-8B \
        --data-path output --hidden-states output/hidden_states \
        --save-path output/checkpoints --epochs 5 --total-seq-len 2048 \
        --logger tensorboard

Metric glossary (answers for the write-up):
    full_acc_k  absolute acceptance: draft top-1 token == verifier token at
                position k given the full context.
    cond_acc_k  conditional acceptance: accuracy at position k GIVEN all earlier
                positions were already correct.
Accuracy drops with k because later positions condition on the draft's own
(possibly wrong) earlier predictions, so errors compound. If full_acc_0 is very
low, the data generation is almost always the culprit -- regenerate hidden
states (check tokenizer/template/vLLM version) before touching the recipe.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parent / "speculators"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verifier", default="Qwen/Qwen3-8B",
                   help="Verifier model the draft head is trained against "
                        "(train.py --verifier-name-or-path).")
    p.add_argument("--data-path", default="output",
                   help="Preprocessed dataset dir (has token_freq.pt etc).")
    p.add_argument("--hidden-states", default=None,
                   help="Cached hidden states (default: <data-path>/hidden_states).")
    p.add_argument("--save-path", default="output/checkpoints")
    p.add_argument("--repo", default=str(DEFAULT_REPO),
                   help="Path to the cloned speculators repo (has scripts/).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--total-seq-len", type=int, default=2048,
                   help="Must match the --seq-length used in Task 1.")
    p.add_argument("--on-missing", default="raise",
                   choices=["raise", "generate", "skip"],
                   help="Dataloader behaviour for samples without cached hidden "
                        "states. Offline training should use 'raise'.")
    p.add_argument("--draft-vocab-size", type=int, default=None,
                   help="Optional draft vocab size (needs token_freq.pt).")
    p.add_argument("--nproc", type=int, default=1,
                   help="torchrun --nproc_per_node (GPUs for training).")
    p.add_argument("--logger", default="tensorboard",
                   help="'' | trackio | wandb | tensorboard | comma-list.")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    repo = Path(args.repo)
    train_py = repo / "scripts" / "train.py"
    if not train_py.exists():
        sys.exit(f"ERROR: {train_py} not found. Set --repo to the cloned "
                 "speculators repo (env_setup.sh clones it).")
    hs = args.hidden_states or str(Path(args.data_path) / "hidden_states")
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={args.nproc}",
        str(train_py),
        "--verifier-name-or-path", args.verifier,
        "--data-path", args.data_path,
        "--hidden-states-path", hs,
        "--save-path", args.save_path,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--total-seq-len", str(args.total_seq_len),
        "--on-missing", args.on_missing,
    ]
    if args.draft_vocab_size is not None:
        cmd += ["--draft-vocab-size", str(args.draft_vocab_size)]
    if args.logger:
        cmd += ["--logger", args.logger, "--log-dir", args.log_dir]
    if args.run_name:
        cmd += ["--run-name", args.run_name]
    return cmd


def main() -> int:
    args = parse_args()
    if shutil.which("torchrun") is None:
        sys.exit("ERROR: 'torchrun' not found. Activate speculators_venv.")
    hs = Path(args.hidden_states or Path(args.data_path) / "hidden_states")
    if not hs.exists() and not args.dry_run:
        sys.exit(f"ERROR: hidden states {hs} not found. Run "
                 "generate_hidden_states.py first.")

    cmd = build_command(args)
    print("==> Running:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return 0

    rc = subprocess.call(cmd, cwd=args.repo)
    if rc != 0:
        print(f"==> Training exited with code {rc}", file=sys.stderr)
        return rc

    # Summarize checkpoints for Task 4. The trainer saves the best checkpoint
    # itself; point serve.sh at it (commonly a 'best' subdir under save-path).
    save = Path(args.save_path)
    ckpts = sorted(p for p in save.glob("*") if p.is_dir()) if save.exists() else []
    print(f"\n==> Checkpoints under {save}:")
    for c in ckpts:
        print(f"    {c}")
    best = save / "best"
    if best.exists():
        print(f"==> Use the best checkpoint for serving: {best}")
    else:
        print("==> Inspect the logger output (see --log-dir) for per-position "
              "val/full_acc_k and pick the best checkpoint for serve.sh.")
    print("    Reference ballpark: full_acc_0 ~ 0.46, decaying by position. "
          "If full_acc_0 is very low, fix data generation (Task 1) first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
