"""Rebuild the current retrieval VQ stack from available base Stage1.

This wrapper exists because experiment checkpoints are easy to delete while
the sidecar retrieval JSONL is large and valuable. It can:

1. Train VQ-base if its VQ/stage1-vqbase checkpoints are missing.
2. Train retrieval-conditioned CodePrior from that VQ-base.

It does not train the original Stage1 autoencoder; pass --base_stage1 to an
existing Stage1 checkpoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train/rebuild retrieval VQ stack")
    parser.add_argument("--base_stage1", default="")
    parser.add_argument("--vq", default="vq_base_stage_rocstories_768_K2048_best.pt")
    parser.add_argument("--stage1_vqbase", default="stage1_vqbase_rocstories_768_K2048_best.pt")
    parser.add_argument("--retrieval_file", default="examples/rocstories_retrieval_contexts_full.jsonl")
    parser.add_argument("--code_prior_output", default="retrieval_full_code_prior_vqbase_best.pt")
    parser.add_argument("--code_prior_examples", default="retrieval_full_code_prior_examples.txt")
    parser.add_argument("--codebook_size", type=int, default=2048)
    parser.add_argument("--vq_epochs", type=int, default=8)
    parser.add_argument("--vq_batch_size", type=int, default=128)
    parser.add_argument("--code_epochs", type=int, default=10)
    parser.add_argument("--code_batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--retrieval_len", type=int, default=128)
    parser.add_argument("--retrieval_top_k", type=int, default=3)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--usage_entropy_weight", type=float, default=0.05)
    parser.add_argument("--adjacent_code_weight", type=float, default=0.05)
    parser.add_argument("--sample_codes", type=int, default=4)
    parser.add_argument("--sample_tau", type=float, default=0.8)
    parser.add_argument("--force_vq", action="store_true")
    parser.add_argument("--skip_code_prior", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def exists(path):
    return Path(path).exists()


def find_base_stage1():
    patterns = [
        "stage1_rocstories_768_rocstories_768_cosmos_best.pt",
        "stage1_rocstories_768_cosmos_best.pt",
        "stage1_rocstories_768_best.pt",
    ]
    for name in patterns:
        if exists(name):
            return name
    candidates = sorted(Path(".").glob("stage1*rocstories*768*best.pt"))
    if candidates:
        return str(candidates[0])
    return ""


def run(cmd, dry_run=False):
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    args = parse_args()
    base_stage1 = args.base_stage1 or find_base_stage1()
    if not base_stage1:
        raise SystemExit(
            "No base Stage1 checkpoint found. Pass --base_stage1, for example "
            "stage1_rocstories_768_rocstories_768_cosmos_best.pt"
        )
    if not exists(base_stage1):
        raise SystemExit(f"base Stage1 checkpoint not found: {base_stage1}")
    if not exists(args.retrieval_file):
        raise SystemExit(f"retrieval file not found: {args.retrieval_file}")

    need_vq = args.force_vq or not (exists(args.vq) and exists(args.stage1_vqbase))
    if need_vq:
        run(
            [
                args.python,
                "scripts/train_vq_base_stage.py",
                "--stage1",
                base_stage1,
                "--epochs",
                str(args.vq_epochs),
                "--batch_size",
                str(args.vq_batch_size),
                "--codebook_size",
                str(args.codebook_size),
                "--output",
                args.vq,
                "--stage1_output",
                args.stage1_vqbase,
            ],
            args.dry_run,
        )
    else:
        print(f"VQ-base checkpoints already exist: {args.vq}, {args.stage1_vqbase}", flush=True)

    if args.skip_code_prior:
        return
    run(
        [
            args.python,
            "scripts/train_tag_conditioned_code_prior.py",
            "--stage1",
            args.stage1_vqbase,
            "--vq",
            args.vq,
            "--condition_mode",
            "retrieval",
            "--retrieval_file",
            args.retrieval_file,
            "--retrieval_len",
            str(args.retrieval_len),
            "--retrieval_top_k",
            str(args.retrieval_top_k),
            "--train_size",
            str(args.train_size),
            "--epochs",
            str(args.code_epochs),
            "--batch_size",
            str(args.code_batch_size),
            "--layers",
            str(args.layers),
            "--hidden_dim",
            str(args.hidden_dim),
            "--usage_entropy_weight",
            str(args.usage_entropy_weight),
            "--adjacent_code_weight",
            str(args.adjacent_code_weight),
            "--sample_codes",
            str(args.sample_codes),
            "--sample_tau",
            str(args.sample_tau),
            "--output",
            args.code_prior_output,
            "--examples",
            args.code_prior_examples,
        ],
        args.dry_run,
    )


if __name__ == "__main__":
    main()
