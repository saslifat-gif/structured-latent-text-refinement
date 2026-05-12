"""Speed benchmark for GPT-2, reported Diffusion-LM, and local latent refinement."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from eval_text_metrics import ensure_output_dir, write_csv

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROCSTORIES = PROJECT_ROOT / "benchmark_rocstories.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROCStories speed benchmark")
    parser.add_argument("--dataset", default="rocstories", choices=("rocstories",))
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--prompt_len", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, choices=(128, 256, 768), default=256)
    parser.add_argument("--ode_steps", type=int, nargs="+", choices=(0, 1, 2, 4, 8, 16), default=[0, 4, 16])
    parser.add_argument("--batch_size", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--draft_source", choices=("synthetic", "manual", "none"), default="synthetic")
    parser.add_argument("--output_dir", default="results/rocstories")
    parser.add_argument("--stage1", default="stage1_best.pt")
    parser.add_argument("--stage2", default="stage2_conditional_flow_decoder_joint_best.pt")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.output_dir)
    rows = []

    for batch_size in args.batch_size:
        # Warmup uses a small local run and discards the file contents. The measured
        # pass below still includes draft encoding, DraftPrior, Riemannian flow,
        # fused readout, and decoder time for our model.
        warmup_samples = max(1, batch_size * args.warmup_batches)
        warmup_cmd = [
            sys.executable,
            str(BENCHMARK_ROCSTORIES),
            "--experiment",
            "fair",
            "--num_samples",
            str(warmup_samples),
            "--batch_size",
            str(batch_size),
            "--latent_dim",
            str(args.latent_dim),
            "--ode_steps",
            str(args.ode_steps[0]),
            "--draft_source",
            args.draft_source,
            "--output_dir",
            str(out_dir / "_warmup"),
            "--no_mauve",
            "--stage1",
            args.stage1,
            "--stage2",
            args.stage2,
        ]
        if args.local_files_only:
            warmup_cmd.append("--local_files_only")
        subprocess.run(warmup_cmd, check=False)

        for steps in args.ode_steps:
            cmd = [
            sys.executable,
                str(BENCHMARK_ROCSTORIES),
                "--experiment",
                "fair",
                "--num_samples",
                str(args.num_samples),
                "--batch_size",
                str(batch_size),
                "--latent_dim",
                str(args.latent_dim),
                "--ode_steps",
                str(steps),
                "--draft_source",
                args.draft_source,
                "--output_dir",
                str(out_dir / f"_speed_bs{batch_size}_steps{steps}"),
                "--no_mauve",
                "--stage1",
                args.stage1,
                "--stage2",
                args.stage2,
            ]
            if args.local_files_only:
                cmd.append("--local_files_only")
            subprocess.run(cmd, check=False)
            measured = read_csv(out_dir / f"_speed_bs{batch_size}_steps{steps}" / "fair_comparison.csv")
            for row in measured:
                row["batch_size"] = batch_size
                row["warmup_batches"] = args.warmup_batches
                rows.append(row)

    write_csv(out_dir / "speed_benchmark.csv", rows)
    print(f"wrote {out_dir / 'speed_benchmark.csv'}")


if __name__ == "__main__":
    main()
