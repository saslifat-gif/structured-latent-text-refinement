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
SUPPORTED_LOCAL_LATENT_DIMS = {256, 768}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROCStories quality/latency benchmark")
    parser.add_argument("--dataset", default="rocstories", choices=("rocstories",))
    parser.add_argument("--experiment", choices=("fair", "full", "latent_dim", "all"), default="all")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--prompt_len", type=int, default=16)
    parser.add_argument(
        "--split_strategy",
        choices=("sentence", "token"),
        default="sentence",
        help="ROCStories split: first 2 sentences -> last 3 sentences, or legacy fixed-token 16/48.",
    )
    parser.add_argument("--latent_dim", type=int, choices=(128, 256, 768), default=256)
    parser.add_argument("--ode_steps", type=int, nargs="+", choices=(0, 1, 2, 4, 8, 16), default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--draft_source", choices=("synthetic", "manual", "none"), default="synthetic")
    parser.add_argument("--output_dir", default="results/rocstories")
    parser.add_argument("--stage1", default="stage1_best.pt")
    parser.add_argument("--stage2", default="stage2_conditional_flow_decoder_joint_best.pt")
    parser.add_argument("--gpt2_model", default="gpt2")
    parser.add_argument(
        "--diffusion_lm_file",
        default=None,
        help="Optional local Diffusion-LM predictions, one per benchmark row. Supports JSONL, CSV, or TXT.",
    )
    parser.add_argument("--diffusion_lm_steps", default="local", help="Step label for --diffusion_lm_file rows.")
    parser.add_argument("--diffusion_lm_label", default="Diffusion-LM local reproduction")
    parser.add_argument(
        "--diffusion_lm_latency_seconds",
        type=float,
        default=None,
        help="Optional total generation latency for the local Diffusion-LM prediction file.",
    )
    parser.add_argument("--skip_reported_diffusion_lm", action="store_true")
    parser.add_argument(
        "--export_eval_file",
        default=None,
        help="Write the exact benchmark rows/prompts/references to JSONL for external generators such as Diffusion-LM.",
    )
    parser.add_argument("--export_only", action="store_true", help="Only export --export_eval_file, then exit.")
    parser.add_argument("--skip_gpt2", action="store_true")
    parser.add_argument("--skip_ours", action="store_true")
    parser.add_argument("--no_mauve", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--rocstories_file",
        default=None,
        help="Local ROCStories/Story Cloze CSV, TSV, JSONL, or TXT file. Preferred when HF dataset scripts are unavailable.",
    )
    parser.add_argument("--manual_draft_file", default=None, help="CSV/JSONL file with one draft field per ROCStories sample")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sentence_parts_from_row(row: dict[str, Any]) -> list[str]:
    sentence_key_sets = [
        [f"sentence{i}" for i in range(1, 6)],
        [f"Sentence{i}" for i in range(1, 6)],
        [f"InputSentence{i}" for i in range(1, 5)] + ["RandomFifthSentenceQuiz1"],
        [f"InputSentence{i}" for i in range(1, 5)] + ["RandomFifthSentenceQuiz2"],
    ]
    for keys in sentence_key_sets:
        parts = [str(row.get(key, "")).strip() for key in keys]
        if sum(bool(part) for part in parts) >= 5:
            return parts

    for prompt_key, continuation_key in (
        ("prompt", "continuation"),
        ("Prompt", "Continuation"),
    ):
        if row.get(prompt_key) and row.get(continuation_key):
            prompt = str(row[prompt_key]).strip()
            continuation = str(row[continuation_key]).strip()
            prompt_parts = split_story_sentences(prompt)
            continuation_parts = split_story_sentences(continuation)
            parts = prompt_parts + continuation_parts
            if len(parts) >= 5:
                return parts[:5]
    return []


def split_story_sentences(text: str) -> list[str]:
    import re

    text = " ".join(str(text).strip().split())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def story_text_from_row(row: dict[str, Any]) -> str:
    sentence_parts = sentence_parts_from_row(row)
    if sentence_parts:
        return " ".join(sentence_parts)

    for key in ("story", "text", "full_text", "Story", "Text"):
        if row.get(key):
            return str(row[key])
    return ""


def sentence_split_example(row: dict[str, Any], max_seq_len: int) -> dict[str, Any] | None:
    sentences = sentence_parts_from_row(row)
    if len(sentences) < 5:
        sentences = split_story_sentences(story_text_from_row(row))
    if len(sentences) < 5:
        return None

    prompt = " ".join(sentences[:2])
    target = " ".join(sentences[2:5])
    full_tokens = tokenize_text(f"{prompt} {target}")
    prompt_tokens = tokenize_text(prompt)
    target_tokens = tokenize_text(target)
    if not prompt_tokens or not target_tokens:
        return None
    if len(full_tokens) > max_seq_len:
        budget = max(max_seq_len - len(prompt_tokens), 1)
        target_tokens = target_tokens[:budget]
        full_tokens = prompt_tokens + target_tokens
    if len(full_tokens) < 8:
        return None
    return {
        "prompt": prompt,
        "reference": " ".join(target_tokens),
        "full_text": " ".join(full_tokens),
        "prompt_len": len(prompt_tokens),
        "target_len": len(target_tokens),
        "split_strategy": "sentence",
    }


def token_split_example(row: dict[str, Any], prompt_len: int, max_seq_len: int) -> dict[str, Any] | None:
    text = story_text_from_row(row)
    tokens = tokenize_text(text)
    if len(tokens) < max_seq_len:
        return None
    prompt = " ".join(tokens[:prompt_len])
    suffix = " ".join(tokens[prompt_len:max_seq_len])
    return {
        "prompt": prompt,
        "reference": suffix,
        "full_text": " ".join(tokens[:max_seq_len]),
        "prompt_len": prompt_len,
        "target_len": max_seq_len - prompt_len,
        "split_strategy": "token",
    }


def rows_to_examples(
    raw_rows: list[dict[str, Any]],
    num_samples: int,
    prompt_len: int,
    max_seq_len: int,
    split_strategy: str,
) -> list[dict[str, Any]]:
    rows = []
    for row in raw_rows:
        if split_strategy == "sentence":
            example = sentence_split_example(row, max_seq_len)
        else:
            example = token_split_example(row, prompt_len, max_seq_len)
        if example is None:
            continue
        rows.append(example)
        if len(rows) >= num_samples:
            break
    return rows


def load_rocstories_file(
    path: str | Path,
    num_samples: int,
    prompt_len: int,
    max_seq_len: int,
    split_strategy: str,
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"ROCStories file does not exist: {path}")

    suffix = path.suffix.lower()
    raw_rows: list[dict[str, Any]] = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            raw_rows = [json.loads(line) for line in f if line.strip()]
    elif suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f, delimiter=delimiter))
    else:
        with path.open("r", encoding="utf-8") as f:
            raw_rows = [{"text": line.strip()} for line in f if line.strip()]

    rows = rows_to_examples(raw_rows, num_samples, prompt_len, max_seq_len, split_strategy)
    if not rows:
        raise RuntimeError(
            f"Loaded {path}, but found no stories long enough for max_seq_len={max_seq_len}. "
            "For sentence mode, provide five-sentence ROCStories rows. For token mode, rows must have enough tokens."
        )
    return rows


def load_rocstories(
    num_samples: int,
    prompt_len: int,
    max_seq_len: int,
    local_files_only: bool,
    rocstories_file: str | None = None,
    split_strategy: str = "sentence",
) -> list[dict[str, Any]]:
    if rocstories_file:
        return load_rocstories_file(rocstories_file, num_samples, prompt_len, max_seq_len, split_strategy)

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
            "The old Hugging Face Story Cloze dataset script may fail on recent `datasets` releases; "
            "prefer passing a local file with `--rocstories_file path/to/rocstories.csv`. "
            "Tried: " + " | ".join(errors)
        )

    rows = rows_to_examples(list(dataset), num_samples, prompt_len, max_seq_len, split_strategy)
    if not rows:
        raise RuntimeError("ROCStories loaded, but no examples matched the requested split strategy.")
    return rows


def attach_manual_drafts(rows: list[dict[str, Any]], manual_draft_file: str | None) -> None:
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


def load_prediction_file(path: str | None, expected_count: int, label: str) -> list[str] | None:
    if not path:
        return None
    pred_path = Path(path)
    if not pred_path.exists():
        raise RuntimeError(f"{label} prediction file does not exist: {pred_path}")

    predictions: list[str] = []
    suffix = pred_path.suffix.lower()
    if suffix == ".jsonl":
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                pred = (
                    item.get("prediction")
                    or item.get("generated")
                    or item.get("generation")
                    or item.get("text")
                    or item.get("output")
                )
                predictions.append(str(pred or ""))
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with pred_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                pred = (
                    row.get("prediction")
                    or row.get("generated")
                    or row.get("generation")
                    or row.get("text")
                    or row.get("output")
                )
                predictions.append(str(pred or ""))
    else:
        with pred_path.open("r", encoding="utf-8") as f:
            predictions = [line.strip() for line in f if line.strip()]

    if len(predictions) < expected_count:
        raise RuntimeError(
            f"{label} prediction file has {len(predictions)} rows, but benchmark needs {expected_count}"
        )
    if len(predictions) > expected_count:
        predictions = predictions[:expected_count]
    return predictions


def export_eval_file(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    if not args.export_eval_file:
        return
    from transformers import BertTokenizer
    from parallel_decoder import cached_from_pretrained

    tokenizer = cached_from_pretrained(BertTokenizer)
    suffix_slots = checkpoint_suffix_slots(args.stage2, args.max_seq_len, args.prompt_len)
    path = Path(args.export_eval_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            item = {
                "id": idx,
                "prompt": row["prompt"],
                "reference": row["reference"],
                "metric_reference": fixed_suffix_reference(row, tokenizer, suffix_slots),
                "full_text": row.get("full_text", ""),
                "prompt_len": row.get("prompt_len"),
                "target_len": row.get("target_len"),
                "split_strategy": row.get("split_strategy"),
                "suffix_slots": suffix_slots,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} benchmark rows to {path}", flush=True)


def corrupt_synthetic_draft(reference: str, seed: int, drop_prob: float = 0.05) -> str:
    rng = random.Random(seed)
    kept = [tok for tok in tokenize_text(reference) if rng.random() >= drop_prob]
    return " ".join(kept) if kept else reference


def pack_prompt_suffix_inputs(
    tokenizer,
    prompts: list[str],
    suffixes: list[str],
    prompt_slots: int,
    suffix_slots: int,
) -> dict[str, torch.Tensor]:
    """Pack text into the same fixed prompt/suffix slots used during training."""
    prompt_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=prompt_slots,
        padding="max_length",
        truncation=True,
    )
    suffix_inputs = tokenizer(
        suffixes,
        return_tensors="pt",
        max_length=suffix_slots,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    return {
        "input_ids": torch.cat([prompt_inputs["input_ids"], suffix_inputs["input_ids"]], dim=1),
        "attention_mask": torch.cat([prompt_inputs["attention_mask"], suffix_inputs["attention_mask"]], dim=1),
    }


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


def generate_gpt2(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[str], float]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_model, local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.gpt2_model, local_files_only=args.local_files_only).to(device)
    model.eval()
    predictions = []
    start = time.perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        inputs = tokenizer([row["prompt"] for row in batch], return_tensors="pt", padding=True, truncation=True).to(device)
        max_new_tokens = max(int(row.get("target_len", args.max_seq_len - args.prompt_len)) for row in batch)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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
    inf.tokenizer = tokenizer
    models = inf.load_models(args.stage1, args.stage2)
    return inf, tokenizer, models


def generate_ours(
    rows: list[dict[str, Any]],
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
    original_prompt_len = inf.PROMPT_LEN
    original_max_seq_len = inf.MAX_SEQ_LEN
    original_mlm_draft_len = inf.MLM_DRAFT_LEN
    model_prompt_len = inf.PROMPT_LEN
    model_max_seq_len = inf.MAX_SEQ_LEN
    start = time.perf_counter()
    try:
        inf.PROMPT_LEN = model_prompt_len
        inf.MAX_SEQ_LEN = model_max_seq_len
        inf.MLM_DRAFT_LEN = model_max_seq_len - model_prompt_len
        suffix_len = model_max_seq_len - model_prompt_len
        batch_size = max(1, args.batch_size)
        draft_status = (
            "external/manual draft; demo mode, not autonomous generation"
            if args.draft_source == "manual"
            else "synthetic corrupted target draft; controlled diagnostic"
        )
        for start_idx in range(0, len(rows), batch_size):
            batch = rows[start_idx:start_idx + batch_size]
            prompts = [row["prompt"] for row in batch]
            suffix_refs = [fixed_suffix_reference(row, tokenizer, suffix_len) for row in batch]
            if args.draft_source == "manual":
                drafts = [row["manual_draft"] for row in batch]
            else:
                drafts = [
                    corrupt_synthetic_draft(suffix_refs[offset], args.seed + start_idx + offset)
                    for offset, row in enumerate(batch)
                ]

            prompt_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                max_length=model_prompt_len,
                padding="max_length",
                truncation=True,
            )
            draft_inputs = pack_prompt_suffix_inputs(
                tokenizer,
                prompts,
                drafts,
                model_prompt_len,
                suffix_len,
            )
            input_ids = prompt_inputs["input_ids"].to(inf.device)
            attention_mask = prompt_inputs["attention_mask"].to(inf.device)
            draft_ids = draft_inputs["input_ids"].to(inf.device)
            draft_mask = draft_inputs["attention_mask"].to(inf.device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=inf.device.type == "cuda"):
                    z_prompt = decoder.compress(encoder(input_ids, attention_mask))
                z_cond = inf.prompt_condition(z_prompt, attention_mask)
                z_draft = inf.encode_suffix_latents(encoder, decoder, draft_ids, draft_mask)
                z_draft = z_draft[:, :suffix_len, :]
                with torch.amp.autocast("cuda", enabled=inf.device.type == "cuda"):
                    z_prior = inf.denoise_draft_latents(start_prior, z_draft, z_cond, suffix_len, inf.device)
                if steps == 0:
                    z = z_prior
                else:
                    z, _metric_snapshot = inf.sample_suffix_latents(
                        flow_net,
                        metric_net,
                        z_cond,
                        len(batch),
                        suffix_len,
                        latent_dim,
                        inf.device,
                        steps=steps,
                        start_prior=start_prior,
                        z_start=z_prior,
                    )
                z_prompt_exp = z_prompt.expand(len(batch), model_prompt_len, latent_dim)
                fused = inf.decode_fused_suffix_texts(
                    decoder,
                    tokenizer,
                    flow_net,
                    aux_token_head,
                    aux_logit_fusion_beta,
                    z_prompt_exp,
                    z,
                )
                outputs = fused if fused is not None else inf.decode_suffix_texts(decoder, tokenizer, z_prompt_exp, z)
            predictions.extend(outputs)
    finally:
        inf.PROMPT_LEN = original_prompt_len
        inf.MAX_SEQ_LEN = original_max_seq_len
        inf.MLM_DRAFT_LEN = original_mlm_draft_len
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return predictions, time.perf_counter() - start, draft_status


def fixed_suffix_reference(row: dict[str, Any], tokenizer, suffix_slots: int) -> str:
    ids = tokenizer(
        row["reference"],
        add_special_tokens=False,
        truncation=True,
        max_length=suffix_slots,
    )["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def checkpoint_suffix_slots(stage2_path: str, fallback_max_seq_len: int, fallback_prompt_len: int) -> int:
    try:
        ckpt = torch.load(stage2_path, map_location="cpu", weights_only=False)
        max_seq_len = int(ckpt.get("max_seq_len", fallback_max_seq_len))
        prompt_len = int(ckpt.get("prompt_len", fallback_prompt_len))
    except Exception:
        max_seq_len = fallback_max_seq_len
        prompt_len = fallback_prompt_len
    return max(1, max_seq_len - prompt_len)


def add_metric_row(
    *,
    model: str,
    rows: list[dict[str, Any]],
    predictions: list[str],
    latency: float,
    latent_dim: int | str,
    steps: int | str,
    include_mauve: bool,
    status: str,
    gpt2_latency_per_sample: float | None = None,
    references: list[str] | None = None,
) -> dict[str, Any]:
    references = references if references is not None else [row["reference"] for row in rows]
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
                status=f"reproduced locally; split={args.split_strategy}",
            )
            gpt2_latency_per_sample = gpt2_row["latency_per_sample"]
            table_rows.append(gpt2_row)
            for source, pred in zip(rows, predictions):
                sample_rows.append({"table": table_name, "model": "GPT-2", **source, "prediction": pred})
        except Exception as exc:
            table_rows.append({"model": "GPT-2 autoregressive baseline", "status": f"unavailable: {exc}"})

    if args.diffusion_lm_file:
        print(f"scoring local Diffusion-LM predictions from {args.diffusion_lm_file}", flush=True)
        try:
            predictions = load_prediction_file(args.diffusion_lm_file, len(rows), "Diffusion-LM")
            from transformers import BertTokenizer
            from parallel_decoder import cached_from_pretrained

            tokenizer = cached_from_pretrained(BertTokenizer)
            suffix_slots = checkpoint_suffix_slots(args.stage2, args.max_seq_len, args.prompt_len)
            fixed_refs = [fixed_suffix_reference(row, tokenizer, suffix_slots) for row in rows]
            table_rows.append(
                add_metric_row(
                    model=args.diffusion_lm_label,
                    rows=rows,
                    predictions=predictions or [],
                    latency=args.diffusion_lm_latency_seconds or 0.0,
                    latent_dim=128,
                    steps=args.diffusion_lm_steps,
                    include_mauve=not args.no_mauve,
                    status=f"locally reproduced from {args.diffusion_lm_file}; references truncated to {suffix_slots} BERT suffix slots",
                    gpt2_latency_per_sample=gpt2_latency_per_sample,
                    references=fixed_refs,
                )
            )
            for source, ref, pred in zip(rows, fixed_refs, predictions or []):
                sample_rows.append({
                    "table": table_name,
                    "model": args.diffusion_lm_label,
                    **source,
                    "metric_reference": ref,
                    "reference_note": f"truncated to {suffix_slots} BERT suffix slots for local benchmark metrics",
                    "prediction": pred,
                })
        except Exception as exc:
            table_rows.append({"model": args.diffusion_lm_label, "status": f"unavailable: {exc}"})

    if not args.skip_reported_diffusion_lm:
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
                    from transformers import BertTokenizer
                    from parallel_decoder import cached_from_pretrained

                    tokenizer = cached_from_pretrained(BertTokenizer)
                    suffix_slots = checkpoint_suffix_slots(args.stage2, args.max_seq_len, args.prompt_len)
                    fixed_refs = [fixed_suffix_reference(row, tokenizer, suffix_slots) for row in rows]
                    table_rows.append(
                        add_metric_row(
                            model=f"Ours DraftPrior + Flow ({label})",
                            rows=rows,
                            predictions=predictions,
                            latency=latency,
                            latent_dim=latent_dim,
                            steps=steps,
                            include_mauve=not args.no_mauve,
                            status=f"{status}; references truncated to {suffix_slots} BERT suffix slots for fixed-slot latent evaluation",
                            gpt2_latency_per_sample=gpt2_latency_per_sample,
                            references=fixed_refs,
                        )
                    )
                    for source, ref, pred in zip(rows, fixed_refs, predictions):
                        sample_rows.append({
                            "table": table_name,
                            "model": f"Ours steps={steps}",
                            **source,
                            "metric_reference": ref,
                            "reference_note": f"truncated to {suffix_slots} BERT suffix slots for latent model metrics",
                            "prediction": pred,
                        })
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
    rows = load_rocstories(
        args.num_samples,
        args.prompt_len,
        args.max_seq_len,
        args.local_files_only,
        args.rocstories_file,
        args.split_strategy,
    )
    attach_manual_drafts(rows, args.manual_draft_file)
    export_eval_file(args, rows)
    if args.export_only:
        return

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
