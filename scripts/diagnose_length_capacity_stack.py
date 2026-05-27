"""Run length-capacity diagnostics, rebuilding VQ-base if needed."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild VQ-base if needed, then run length grid")
    parser.add_argument("--base_stage1", default="")
    parser.add_argument("--vq", default="vq_base_stage_rocstories_768_K2048_best.pt")
    parser.add_argument("--stage1_vqbase", default="stage1_vqbase_rocstories_768_K2048_best.pt")
    parser.add_argument("--retrieval_file", default="examples/rocstories_retrieval_contexts_full.jsonl")
    parser.add_argument("--prompt_lens", default="16,32,64")
    parser.add_argument("--suffix_lens", default="16,32,64,96,128")
    parser.add_argument("--train_size", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="results/length_capacity_retrieval")
    parser.add_argument("--vq_epochs", type=int, default=8)
    parser.add_argument("--vq_batch_size", type=int, default=128)
    parser.add_argument("--force_vq", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run(cmd, dry_run=False):
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    args = parse_args()
    need_vq = args.force_vq or not (Path(args.vq).exists() and Path(args.stage1_vqbase).exists())
    if need_vq:
        cmd = [
            args.python,
            "scripts/train_retrieval_vq_stack.py",
            "--vq",
            args.vq,
            "--stage1_vqbase",
            args.stage1_vqbase,
            "--retrieval_file",
            args.retrieval_file,
            "--vq_epochs",
            str(args.vq_epochs),
            "--vq_batch_size",
            str(args.vq_batch_size),
            "--skip_code_prior",
        ]
        if args.base_stage1:
            cmd.extend(["--base_stage1", args.base_stage1])
        if args.force_vq:
            cmd.append("--force_vq")
        run(cmd, args.dry_run)
    else:
        print(f"VQ-base checkpoints already exist: {args.vq}, {args.stage1_vqbase}", flush=True)

    cmd = [
        args.python,
        "scripts/run_length_capacity_grid.py",
        "--stage1",
        args.stage1_vqbase,
        "--vq",
        args.vq,
        "--condition_mode",
        "retrieval",
        "--retrieval_file",
        args.retrieval_file,
        "--prompt_lens",
        args.prompt_lens,
        "--suffix_lens",
        args.suffix_lens,
        "--train_size",
        str(args.train_size),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--output_dir",
        args.output_dir,
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
