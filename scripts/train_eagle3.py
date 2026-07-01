#!/usr/bin/env python3
"""Task 2: train an EAGLE-3 draft head on precomputed hidden states.

Wraps the speculators EAGLE-3 trainer. It launches training, saves checkpoints
under output/checkpoints/, and — because the assignment is graded on per-position
acceptance, not just total loss — parses the trainer's metric log to pick and
print the best checkpoint for serving in Task 4.

Run inside speculators_venv:

    source speculators_venv/bin/activate
    python train_eagle3.py --verifier Qwen/Qwen3-8B \
        --hidden-states data/hidden_states \
        --out output/checkpoints --epochs 5 --log-positionwise

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
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verifier", default="Qwen/Qwen3-8B",
                   help="Verifier model the draft head is trained against.")
    p.add_argument("--hidden-states", default="data/hidden_states",
                   help="Directory produced by generate_hidden_states.py.")
    p.add_argument("--out", default="output/checkpoints")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-draft-positions", type=int, default=3,
                   help="Speculative positions to train/track (k = 0..N-1).")
    p.add_argument("--val-split", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-positionwise", action="store_true",
                   help="Emit per-position val metrics to metrics.jsonl.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_command(args: argparse.Namespace, metrics_path: Path) -> list[str]:
    """Build the speculators EAGLE-3 training command.

    Flag names follow the v0.5.0 offline tutorial; check `speculators train
    --help` if the installed tag differs.
    """
    return [
        "speculators", "train", "eagle3",
        "--verifier", args.verifier,
        "--hidden-states", args.hidden_states,
        "--output-dir", args.out,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.lr),
        "--num-draft-positions", str(args.num_draft_positions),
        "--val-split", str(args.val_split),
        "--seed", str(args.seed),
        "--metrics-file", str(metrics_path),
    ]


def pick_best_checkpoint(out_dir: Path, metrics_path: Path) -> None:
    """Report the best checkpoint by val loss for use in Task 4 serving."""
    if not metrics_path.exists():
        print(f"==> No metrics file at {metrics_path}; inspect trainer logs "
              "manually to select the best checkpoint.")
        return
    best = None
    with metrics_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            loss = rec.get("val/loss_epoch")
            if loss is None:
                continue
            if best is None or loss < best.get("val/loss_epoch", float("inf")):
                best = rec

    if best is None:
        print("==> Could not parse val/loss_epoch from metrics; check logs.")
        return

    print("\n==> Best checkpoint by val/loss_epoch:")
    for key in ("epoch", "val/loss_epoch",
                "val/loss_0_epoch", "val/full_acc_0_epoch", "val/cond_acc_0_epoch",
                "val/loss_1_epoch", "val/full_acc_1_epoch", "val/cond_acc_1_epoch",
                "val/loss_2_epoch", "val/full_acc_2_epoch", "val/cond_acc_2_epoch"):
        if key in best:
            print(f"    {key}: {best[key]}")

    # Persist the resolved path so serve.sh / sweep can reference it as "best".
    ckpt = best.get("checkpoint_path") or best.get("checkpoint")
    best_link = out_dir / "best"
    if ckpt and Path(ckpt).exists():
        if best_link.is_symlink() or best_link.exists():
            best_link.unlink() if best_link.is_symlink() else shutil.rmtree(best_link)
        try:
            best_link.symlink_to(Path(ckpt).resolve(), target_is_directory=True)
            print(f"==> Linked best checkpoint -> {best_link} ({ckpt})")
        except OSError:
            print(f"==> Best checkpoint: {ckpt} (could not create 'best' symlink)")
    else:
        print("==> Best checkpoint path not found in metrics; set it manually "
              f"under {out_dir}/best for serve.sh.")

    fa0 = best.get("val/full_acc_0_epoch")
    if isinstance(fa0, (int, float)) and fa0 < 0.30:
        print("    WARNING: full_acc_0 is low (<0.30). Inspect data generation "
              "before changing the training recipe; consider more samples.")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hs = Path(args.hidden_states)
    if not hs.exists():
        sys.exit(f"ERROR: hidden states {hs} not found. Run "
                 "generate_hidden_states.py first.")
    if shutil.which("speculators") is None:
        sys.exit("ERROR: 'speculators' CLI not found. Activate speculators_venv.")

    metrics_path = out_dir / "metrics.jsonl"
    cmd = build_command(args, metrics_path)
    print("==> Running:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return 0

    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"==> Training exited with code {rc}", file=sys.stderr)
        return rc
    pick_best_checkpoint(out_dir, metrics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
