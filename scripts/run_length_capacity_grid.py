"""Run prompt/target length capacity grid experiments.

This is an orchestration script around train_tag_conditioned_code_prior.py. It
trains one model per prompt_len/suffix_len pair, saves logs/checkpoints in a
grid directory, and summarizes final validation metrics into JSONL/CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integer list")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt/target length capacity grid")
    parser.add_argument("--stage1", default="stage1_vqbase_rocstories_768_K2048_best.pt")
    parser.add_argument("--vq", default="vq_base_stage_rocstories_768_K2048_best.pt")
    parser.add_argument("--condition_mode", choices=("raw", "tags", "expansion", "retrieval"), default="retrieval")
    parser.add_argument("--retrieval_file", default="examples/rocstories_retrieval_contexts_full.jsonl")
    parser.add_argument("--retrieval_top_k", type=int, default=3)
    parser.add_argument("--retrieval_len", type=int, default=128)
    parser.add_argument("--expansion_file", default="")
    parser.add_argument("--expansion_len", type=int, default=96)
    parser.add_argument("--expansion_choice", choices=("first", "random", "concat"), default="first")
    parser.add_argument("--prompt_lens", type=parse_int_list, default=parse_int_list("16,32,64"))
    parser.add_argument("--suffix_lens", type=parse_int_list, default=parse_int_list("16,32,64,96,128"))
    parser.add_argument("--train_size", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--usage_entropy_weight", type=float, default=0.05)
    parser.add_argument("--adjacent_code_weight", type=float, default=0.05)
    parser.add_argument("--sample_codes", type=int, default=4)
    parser.add_argument("--sample_tau", type=float, default=0.8)
    parser.add_argument("--output_dir", default="results/length_capacity_grid")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def metric_summary(metrics):
    if not metrics:
        return "no_metrics"
    keys = [
        "soft_ce",
        "hard_ce",
        "sample_ce",
        "sample_p",
        "sample_repeat_frac",
        "sample_unique_token_ratio",
        "hard_code_ppl",
        "hard_used_codes",
    ]
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.4g}")
            else:
                parts.append(f"{key}={value}")
    return " ".join(parts) if parts else "no_key_metrics"


def extract_json_block(text, start):
    brace_start = text.find("{", start)
    if brace_start < 0:
        return None
    depth = 0
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : idx + 1]
    return None


def parse_val_metrics(log_text):
    matches = list(re.finditer(r"val ep(\d+) \|", log_text))
    if not matches:
        return None, None
    last = matches[-1]
    block = extract_json_block(log_text, last.end())
    if not block:
        return int(last.group(1)), None
    return int(last.group(1)), json.loads(block)


def write_summary(rows, output_dir):
    jsonl_path = output_dir / "length_capacity_summary.jsonl"
    csv_path = output_dir / "length_capacity_summary.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    fieldnames = [
        "prompt_len",
        "suffix_len",
        "max_seq_len",
        "condition_mode",
        "status",
        "epoch",
        "soft_ce",
        "hard_ce",
        "sample_ce",
        "sample_p",
        "sample_repeat_frac",
        "sample_unique_token_ratio",
        "sample_code_ppl",
        "hard_code_ppl",
        "hard_used_codes",
        "target_code_ppl",
        "log",
        "checkpoint",
        "examples",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {jsonl_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)


def build_command(args, prompt_len, suffix_len, run_dir):
    max_seq_len = prompt_len + suffix_len
    name = f"{args.condition_mode}_p{prompt_len}_t{suffix_len}"
    cmd = [
        args.python,
        "scripts/train_tag_conditioned_code_prior.py",
        "--stage1",
        args.stage1,
        "--vq",
        args.vq,
        "--condition_mode",
        args.condition_mode,
        "--prompt_len",
        str(prompt_len),
        "--suffix_len",
        str(suffix_len),
        "--max_seq_len",
        str(max_seq_len),
        "--train_size",
        str(args.train_size),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
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
        str(run_dir / f"{name}_best.pt"),
        "--examples",
        str(run_dir / f"{name}_examples.txt"),
    ]
    if args.condition_mode == "retrieval":
        cmd.extend([
            "--retrieval_file",
            args.retrieval_file,
            "--retrieval_len",
            str(args.retrieval_len),
            "--retrieval_top_k",
            str(args.retrieval_top_k),
        ])
    elif args.condition_mode == "expansion":
        cmd.extend([
            "--expansion_file",
            args.expansion_file,
            "--expansion_len",
            str(args.expansion_len),
            "--expansion_choice",
            args.expansion_choice,
        ])
    if args.local_files_only:
        cmd.append("--local_files_only")
    return cmd


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for prompt_len in args.prompt_lens:
        for suffix_len in args.suffix_lens:
            run_name = f"{args.condition_mode}_p{prompt_len}_t{suffix_len}"
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            cmd = build_command(args, prompt_len, suffix_len, run_dir)
            row = {
                "prompt_len": prompt_len,
                "suffix_len": suffix_len,
                "max_seq_len": prompt_len + suffix_len,
                "condition_mode": args.condition_mode,
                "log": str(log_path),
                "checkpoint": str(run_dir / f"{run_name}_best.pt"),
                "examples": str(run_dir / f"{run_name}_examples.txt"),
                "status": "pending",
                "command": " ".join(cmd),
            }
            if args.skip_existing and log_path.exists():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                epoch, metrics = parse_val_metrics(text)
                row["status"] = "skipped_existing"
                row["epoch"] = epoch
                if metrics:
                    row.update(metrics)
                print(f"skip {run_name} | {metric_summary(metrics or {})}", flush=True)
                rows.append(row)
                write_summary(rows, output_dir)
                continue
            print(f"run {run_name}", flush=True)
            print(" ".join(cmd), flush=True)
            if args.dry_run:
                row["status"] = "dry_run"
                rows.append(row)
                continue
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return_code = process.wait()
            row["status"] = "ok" if return_code == 0 else f"failed:{return_code}"
            text = log_path.read_text(encoding="utf-8", errors="replace")
            epoch, metrics = parse_val_metrics(text)
            row["epoch"] = epoch
            if metrics:
                row.update(metrics)
            print(f"done {run_name} status={row['status']} | {metric_summary(metrics or {})}", flush=True)
            rows.append(row)
            write_summary(rows, output_dir)
            if return_code != 0:
                raise SystemExit(return_code)
    write_summary(rows, output_dir)


if __name__ == "__main__":
    main()
