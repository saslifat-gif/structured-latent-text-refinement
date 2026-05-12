"""ROCStories quality benchmarks for structured latent text refinement."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import DownloadConfig, load_dataset

from eval_text_metrics import append_jsonl, compute_text_metrics, ensure_output_dir, tokenize_text, write_csv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


DIFFUSION_LM_REPORTED_MAUVE = 0.043
SUPPORTED_LOCAL_LATENT_DIMS = {256}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROCStories quality/latency benchmark")
    parser.add_argument("--dataset", default="rocstories", choices=("rocstories",))
    parser.add_argument("--experiment", choices=("fair", "full", "latent_dim", "all"), default="all")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--prompt_len", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, choices=(128, 256, 768), default=256)
    parser.add_argument("--ode_steps", type=int, nargs="+", choices=(0, 1, 2, 4, 8, 16), default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--draft_source", choices=("synthetic", "manual", "none"), default="synthetic")
    parser.add_argument("--output_dir", default="results/rocstories")
    parser.add_argument("--stage1", default="stage1_best.pt")
    parser.add_argument("--stage2", default="stage2_conditional_flow_decoder_joint_best.pt")
    parser.add_argument("--gpt2_model", default="gpt2")
    parser.add_argument("--skip_gpt2", action="store_true")
    parser.add_argument("--skip_ours", action="store_true")
    parser.add_argument("--no_mauve", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--manual_draft_file", default=None, help="CSV/JSONL file with one draft field per ROCStories sample")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rocstories(num_samples: int, prompt_len: int, max_seq_len: int, local_files_only: bool) -> list[dict[str, str]]:
    errors = []
    download_config = DownloadConfig(local_files_only=local_files_only)
    candidates = [
        ("roc_stories", None),
        ("story_cloze", "2016"),
        ("story_cloze", "2018"),
    ]
    dataset = None
    for name, config in candidates:
        for split in ("validation", "test", "train"):
            try:
                if config is None:
                    dataset = load_dataset(name, split=split, download_config=download_config)
                else:
                    dataset = load_dataset(name, config, split=split, download_config=download_config)
                break
            except Exception as exc:  # pragma: no cover - depends on local datasets/cache.
                errors.append(f"{name}/{config}/{split}: {exc}")
        if dataset is not None:
            break
    if dataset is None:
        raise RuntimeError(
            "Could not load ROCStories. Install/cache a ROCStories-compatible dataset first. "
            "Tried: " + " | ".join(errors)
        )

    rows = []
    for row in dataset:
        text = row.get("story") or row.get("text")
        if not text:
            sentence_keys = [f"sentence{i}" for i in range(1, 6)]
            text = " ".join(str(row.get(key, "")).strip() for key in sentence_keys)
        tokens = tokenize_text(text)
        if len(tokens) < max_seq_len:
            continue
        prompt = " ".join(tokens[:prompt_len])
        suffix = " ".join(tokens[prompt_len:max_seq_len])
        rows.append({"prompt": prompt, "reference": suffix, "full_text": " ".join(tokens[:max_seq_len])})
        if len(rows) >= num_samples:
            break
    if not rows:
        raise RuntimeError("ROCStories loaded, but no examples were long enough for the requested 16/48 split.")
    return rows


def attach_manual_drafts(rows: list[dict[str, str]], manual_draft_file: str | None) -> None:
    if not manual_draft_file:
        return
    path = Path(manual_draft_file)
    if not path.exists():
        raise RuntimeError(f"manual draft file does not exist: {path}")
    drafts: list[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    drafts.append(str(item.get("draft") or item.get("manual_draft") or ""))
    else:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drafts.append(str(row.get("draft") or row.get("manual_draft") or ""))
    if len(drafts) < len(rows):
        raise RuntimeError(
            f"manual draft file has {len(drafts)} drafts, but benchmark needs {len(rows)} samples"
        )
    for row, draft in zip(rows, drafts):
        row["manual_draft"] = draft


def corrupt_synthetic_draft(reference: str, seed: int, drop_prob: float = 0.05) -> str:
    rng = random.Random(seed)
    kept = [tok for tok in tokenize_text(reference) if rng.random() >= drop_prob]
    return " ".join(kept) if kept else reference


def diffusion_lm_rows() -> list[dict[str, Any]]:
    rows = []
    for steps in (200, 2000):
        rows.append(
            {
                "model": "reported Diffusion-LM baseline",
                "dataset": "ROCStories",
                "latent_dim": 128,
                "steps": steps,
                "mauve": DIFFUSION_LM_REPORTED_MAUVE,
                "repetition_rate": None,
                "unique_token_ratio": None,
                "latency_per_sample": None,
                "tokens_per_second": None,
                "speedup_vs_gpt2": None,
                "status": "reported baseline, not reproduced",
            }
        )
    return rows


def generate_gpt2(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[str], float]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_model, local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.gpt2_model, local_files_only=args.local_files_only).to(device)
    model.eval()
    predictions = []
    start = time.perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        inputs = tokenizer([row["prompt"] for row in batch], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_seq_len - args.prompt_len,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        for item, prompt_ids in zip(output, inputs["input_ids"]):
            suffix_ids = item[prompt_ids.numel() :]
            predictions.append(tokenizer.decode(suffix_ids, skip_special_tokens=True))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return predictions, time.perf_counter() - start


def load_ours(args: argparse.Namespace):
    import inference_stage2_conditional as inf
    from parallel_decoder import cached_from_pretrained
    from transformers import BertTokenizer

    tokenizer = cached_from_pretrained(BertTokenizer)
    models = inf.load_models(args.stage1, args.stage2)
    return inf, tokenizer, models


def generate_ours(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    steps: int,
    latent_dim: int,
) -> tuple[list[str], float, str]:
    if args.draft_source == "none":
        raise RuntimeError("draft_source=none: no draft provided. Benchmarks do not use blank auto-drafts by default.")
    if args.draft_source == "manual" and any(not row.get("manual_draft") for row in rows):
        raise RuntimeError("manual draft mode requires --manual_draft_file with a non-empty draft column.")
    if latent_dim not in SUPPORTED_LOCAL_LATENT_DIMS:
        raise RuntimeError(
            f"latent_dim={latent_dim} requested, but the local inference checkpoint loader supports "
            f"{sorted(SUPPORTED_LOCAL_LATENT_DIMS)}. Provide a matching checkpoint/config before reproducing this row."
        )

    inf, tokenizer, models = load_ours(args)
    encoder, decoder, flow_net, metric_net, start_prior, aux_token_head, aux_logit_fusion_beta, mlm_model = models
    predictions = []
    start = time.perf_counter()
    for idx, row in enumerate(rows):
        if args.draft_source == "manual":
            draft = row["manual_draft"]
            draft_status = "external/manual draft; demo mode, not autonomous generation"
        else:
            draft = corrupt_synthetic_draft(row["reference"], args.seed + idx)
            draft_status = "synthetic corrupted target draft; controlled diagnostic"
        debug = inf.generate(
            row["prompt"],
            flow_net,
            metric_net,
            encoder,
            decoder,
            mlm_model,
            tokenizer,
            n_samples=1,
            seq_len=args.max_seq_len,
            latent_dim=latent_dim,
            steps=steps,
            start_prior=start_prior,
            aux_token_head=aux_token_head,
            aux_logit_fusion_beta=aux_logit_fusion_beta,
            draft_text=draft,
            allow_latent_fallback=False,
            return_debug=True,
        )
        output = debug["fused"][0] if debug["fused"] is not None else debug["flow"][0]
        predictions.append(output)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return predictions, time.perf_counter() - start, draft_status


def add_metric_row(
    *,
    model: str,
    rows: list[dict[str, str]],
    predictions: list[str],
    latency: float,
    latent_dim: int | str,
    steps: int | str,
    include_mauve: bool,
    status: str,
    gpt2_latency_per_sample: float | None = None,
) -> dict[str, Any]:
    references = [row["reference"] for row in rows]
    metrics = compute_text_metrics(
        predictions,
        references,
        include_mauve=include_mauve,
        latency_seconds=latency,
        generated_tokens=sum(len(tokenize_text(text)) for text in predictions),
    )
    speedup = None
    if gpt2_latency_per_sample and metrics["latency_per_sample"]:
        speedup = gpt2_latency_per_sample / float(metrics["latency_per_sample"])
    return {
        "model": model,
        "dataset": "ROCStories",
        "latent_dim": latent_dim,
        "steps": steps,
        **metrics,
        "speedup_vs_gpt2": speedup,
        "status": status,
    }


def run_table(args: argparse.Namespace, table_name: str, rows: list[dict[str, str]], latent_dims: list[int], steps_list: list[int]) -> None:
    out_dir = ensure_output_dir(args.output_dir)
    table_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    gpt2_latency_per_sample = None

    if not args.skip_gpt2:
        print("running GPT-2 autoregressive baseline: 48 autoregressive suffix steps", flush=True)
        try:
            predictions, latency = generate_gpt2(rows, args)
            gpt2_row = add_metric_row(
                model="GPT-2 autoregressive baseline",
                rows=rows,
                predictions=predictions,
                latency=latency,
                latent_dim="-",
                steps="48 AR",
                include_mauve=not args.no_mauve,
                status="reproduced locally",
            )
            gpt2_latency_per_sample = gpt2_row["latency_per_sample"]
            table_rows.append(gpt2_row)
            for source, pred in zip(rows, predictions):
                sample_rows.append({"table": table_name, "model": "GPT-2", **source, "prediction": pred})
        except Exception as exc:
            table_rows.append({"model": "GPT-2 autoregressive baseline", "status": f"unavailable: {exc}"})

    table_rows.extend(diffusion_lm_rows())

    if not args.skip_ours:
        for latent_dim in latent_dims:
            label = (
                "full-capacity 768-dimensional BERT latent"
                if latent_dim == 768
                else "capacity-matched 128-dimensional latent"
                if latent_dim == 128
                else "local 256-dimensional latent"
            )
            print(f"running ours: {label}, draft_source={args.draft_source}", flush=True)
            for steps in steps_list:
                try:
                    predictions, latency, status = generate_ours(rows, args, steps, latent_dim)
                    table_rows.append(
                        add_metric_row(
                            model=f"Ours DraftPrior + Flow ({label})",
                            rows=rows,
                            predictions=predictions,
                            latency=latency,
                            latent_dim=latent_dim,
                            steps=steps,
                            include_mauve=not args.no_mauve,
                            status=status,
                            gpt2_latency_per_sample=gpt2_latency_per_sample,
                        )
                    )
                    for source, pred in zip(rows, predictions):
                        sample_rows.append({"table": table_name, "model": f"Ours steps={steps}", **source, "prediction": pred})
                except Exception as exc:
                    table_rows.append(
                        {
                            "model": f"Ours DraftPrior + Flow ({label})",
                            "dataset": "ROCStories",
                            "latent_dim": latent_dim,
                            "steps": steps,
                            "status": f"unavailable: {exc}",
                        }
                    )

    write_csv(out_dir / f"{table_name}.csv", table_rows)
    append_jsonl(out_dir / "generated_samples.jsonl", sample_rows)
    print(f"wrote {out_dir / f'{table_name}.csv'}", flush=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = ensure_output_dir(args.output_dir)
    samples_path = out_dir / "generated_samples.jsonl"
    if samples_path.exists():
        samples_path.unlink()
    rows = load_rocstories(args.num_samples, args.prompt_len, args.max_seq_len, args.local_files_only)
    attach_manual_drafts(rows, args.manual_draft_file)

    experiments = [args.experiment] if args.experiment != "all" else ["fair", "full", "latent_dim"]
    for experiment in experiments:
        if experiment == "fair":
            latent_dims = [args.latent_dim]
            run_table(args, "fair_comparison", rows, latent_dims, [args.ode_steps[-1]])
        elif experiment == "full":
            run_table(args, "full_strength_768", rows, [768], args.ode_steps)
        elif experiment == "latent_dim":
            run_table(args, "latent_dim_ablation", rows, [128, 256, 768], [4, 16])


if __name__ == "__main__":
    main()
