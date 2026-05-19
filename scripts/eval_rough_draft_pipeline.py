"""Evaluate whether a rough local text draft supplies the missing suffix information.

This is a diagnostic script. Stage1 and optional DraftPrior stay frozen. The
script encodes a rough draft suffix into latent space, optionally repairs it with
DraftPrior, then compares decoder readability against the true ROCStories suffix.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmark_rocstories import corrupt_synthetic_draft, load_rocstories as benchmark_load_rocstories
from benchmark_rocstories import fixed_suffix_reference, pack_prompt_suffix_inputs
from eval_text_metrics import tokenize_text
from parallel_decoder import BertTokenizer, cached_from_pretrained
import stage2_data as s2data
from stage2_config import TARGET_LATENT_MEAN, TARGET_LATENT_STD
import stage2_riemannian as rfm
from stage2_riemannian import DenoisingPrior, suffix_positions
from train_basin_projector import decoder_ce_stats, encode_latents, load_stage1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a rough-draft-first latent pipeline diagnostic")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--draft_prior", default=None, help="Optional frozen DraftPrior checkpoint.")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--rocstories_file", default=None)
    parser.add_argument("--split_strategy", choices=("sentence", "token"), default="sentence")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--draft_source",
        choices=("synthetic", "copy_prompt", "hf"),
        default="synthetic",
        help="Rough text draft source. synthetic is reference corruption for a controlled upper-bound test.",
    )
    parser.add_argument("--synthetic_drop_prob", type=float, default=0.30)
    parser.add_argument("--hf_draft_model", default="distilgpt2")
    parser.add_argument("--hf_batch_size", type=int, default=8)
    parser.add_argument("--hf_temperature", type=float, default=0.8)
    parser.add_argument("--hf_top_p", type=float, default=0.95)
    parser.add_argument("--hf_top_k", type=int, default=50)
    parser.add_argument("--draft_alpha", type=float, default=None, help="Override DraftPrior alpha from checkpoint.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output_dir", default="results/rough_draft_pipeline")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rows_from_stage2_data(
    num_samples: int,
    prompt_len: int,
    max_seq_len: int,
    local_files_only: bool,
    rocstories_file: str | None,
    split_strategy: str,
) -> list[dict[str, Any]]:
    original_local_only = s2data.ROCSTORIES_LOCAL_FILES_ONLY
    original_file = s2data.ROCSTORIES_FILE
    original_source = s2data.ROCSTORIES_SOURCE
    try:
        s2data.ROCSTORIES_LOCAL_FILES_ONLY = local_files_only
        if rocstories_file:
            s2data.ROCSTORIES_FILE = rocstories_file
            s2data.ROCSTORIES_SOURCE = "file"
        raw_rows = s2data.load_rocstories_rows()
    finally:
        s2data.ROCSTORIES_LOCAL_FILES_ONLY = original_local_only
        s2data.ROCSTORIES_FILE = original_file
        s2data.ROCSTORIES_SOURCE = original_source

    prompt_sentences = int(getattr(s2data, "ROCSTORIES_PROMPT_SENTENCES", 2))
    target_sentences = int(getattr(s2data, "ROCSTORIES_TARGET_SENTENCES", 3))
    needed = prompt_sentences + target_sentences
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        parts = s2data.sentence_parts_from_row(raw)
        if len(parts) < needed:
            continue
        if split_strategy == "sentence":
            prompt = " ".join(parts[:prompt_sentences])
            reference = " ".join(parts[prompt_sentences:needed])
            full_tokens = tokenize_text(f"{prompt} {reference}")
            prompt_tokens = tokenize_text(prompt)
            reference_tokens = tokenize_text(reference)
            if not prompt_tokens or not reference_tokens:
                continue
            if len(full_tokens) > max_seq_len:
                budget = max(max_seq_len - len(prompt_tokens), 1)
                reference_tokens = reference_tokens[:budget]
                full_tokens = prompt_tokens + reference_tokens
            rows.append(
                {
                    "prompt": prompt,
                    "reference": " ".join(reference_tokens),
                    "full_text": " ".join(full_tokens),
                    "prompt_len": len(prompt_tokens),
                    "target_len": len(reference_tokens),
                    "split_strategy": "sentence",
                }
            )
        else:
            tokens = tokenize_text(" ".join(parts))
            if len(tokens) < max_seq_len:
                continue
            rows.append(
                {
                    "prompt": " ".join(tokens[:prompt_len]),
                    "reference": " ".join(tokens[prompt_len:max_seq_len]),
                    "full_text": " ".join(tokens[:max_seq_len]),
                    "prompt_len": prompt_len,
                    "target_len": max_seq_len - prompt_len,
                    "split_strategy": "token",
                }
            )
        if len(rows) >= num_samples:
            break
    if not rows:
        raise RuntimeError("stage2_data ROCStories fallback loaded rows, but none matched the requested split")
    return rows


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    try:
        rows = benchmark_load_rocstories(
            args.num_samples,
            args.prompt_len,
            args.max_seq_len,
            args.local_files_only,
            args.rocstories_file,
            args.split_strategy,
        )
        return rows, warnings
    except Exception as exc:
        warning = f"benchmark ROCStories loader failed; trying stage2_data fallback: {exc}"
        print(f"WARNING: {warning}", flush=True)
        warnings.append(warning)
        rows = rows_from_stage2_data(
            args.num_samples,
            args.prompt_len,
            args.max_seq_len,
            args.local_files_only,
            args.rocstories_file,
            args.split_strategy,
        )
        return rows, warnings


def load_draft_prior(path: str | None, latent_dim: int, device: torch.device):
    if not path:
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    prior = DenoisingPrior(
        latent_dim=int(ckpt.get("latent_dim", latent_dim)),
        hidden_dim=int(ckpt.get("denoising_hidden_dim", ckpt.get("start_transformer_hidden_dim", 512))),
        num_layers=int(ckpt.get("denoising_layers", ckpt.get("start_transformer_layers", 4))),
        num_heads=int(ckpt.get("denoising_heads", ckpt.get("start_transformer_heads", 8))),
    ).to(device)
    state = ckpt.get("denoising_prior", ckpt.get("draft_prior"))
    if state is None:
        raise RuntimeError(f"No denoising_prior/draft_prior state found in {path}")
    prior.load_state_dict(state)
    prior.eval()
    for param in prior.parameters():
        param.requires_grad_(False)
    alpha = float(ckpt.get("draft_alpha", ckpt.get("denoising_prior_alpha", 0.7)))
    print(f"loaded DraftPrior: {path} | alpha={alpha:.3f}", flush=True)
    return prior, alpha


def make_copy_prompt_drafts(rows: list[dict[str, Any]], tokenizer, suffix_len: int) -> list[str]:
    drafts = []
    for row in rows:
        ids = tokenizer(
            row["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=suffix_len,
        )["input_ids"]
        drafts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return drafts


def make_hf_drafts(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_draft_model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.hf_draft_model, local_files_only=args.local_files_only).to(device)
    model.eval()

    drafts: list[str] = []
    start = time.perf_counter()
    with torch.no_grad():
        for start_idx in range(0, len(rows), args.hf_batch_size):
            batch = rows[start_idx : start_idx + args.hf_batch_size]
            prompts = [row["prompt"] for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.suffix_len,
                do_sample=True,
                temperature=args.hf_temperature,
                top_p=args.hf_top_p,
                top_k=args.hf_top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
            drafts.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    print(f"generated HF rough drafts in {time.perf_counter() - start:.2f}s", flush=True)
    return drafts


def make_drafts(rows: list[dict[str, Any]], args: argparse.Namespace, tokenizer) -> list[str]:
    if args.draft_source == "synthetic":
        return [
            corrupt_synthetic_draft(fixed_suffix_reference(row, tokenizer, args.suffix_len), args.seed + idx, args.synthetic_drop_prob)
            for idx, row in enumerate(rows)
        ]
    if args.draft_source == "copy_prompt":
        return make_copy_prompt_drafts(rows, tokenizer, args.suffix_len)
    if args.draft_source == "hf":
        return make_hf_drafts(rows, args)
    raise ValueError(f"unsupported draft source: {args.draft_source}")


@torch.no_grad()
def repair_with_draft_prior(draft_prior, z_draft, z_prompt, suffix_mask, alpha: float):
    pos = suffix_positions(z_prompt.size(0), z_draft.size(1), z_prompt.device, z_prompt.dtype)
    alpha_t = z_prompt.new_full((z_prompt.size(0),), alpha)
    beta = max(0.0, 1.0 - alpha * alpha) ** 0.5
    noise = torch.randn_like(z_draft) * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    z_t = alpha * z_draft + beta * noise
    if suffix_mask is not None:
        z_t = z_t * suffix_mask.to(z_t.dtype).unsqueeze(-1)
    return draft_prior(z_t, z_prompt, alpha_t, pos, suffix_mask)


@torch.no_grad()
def decode_suffixes(decoder, tokenizer, z_prompt, z_suffix) -> list[str]:
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))[:, z_prompt.size(1) :]
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in ids]


def mean_metric(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_ce_svg(path: Path, direct: list[float], repaired: list[float], title: str) -> None:
    width, height = 900, 620
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    values = direct + repaired
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"200\"></svg>\n", encoding="utf-8")
        return
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    def x_of(v: float) -> float:
        return margin + (v - lo) / (hi - lo) * plot_w

    def y_of(v: float) -> float:
        return height - margin - (v - lo) / (hi - lo) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial" font-size="24">{_svg_escape(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222" stroke-width="1"/>',
        f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="16">direct rough-draft CE</text>',
    ]
    if repaired:
        parts.append(
            f'<text x="20" y="{height / 2:.1f}" transform="rotate(-90 20 {height / 2:.1f})" '
            'text-anchor="middle" font-family="Arial" font-size="16">DraftPrior-repaired CE</text>'
        )
        parts.append(
            f'<line x1="{margin}" y1="{y_of(lo):.2f}" x2="{width - margin}" y2="{y_of(hi):.2f}" '
            'stroke="#888" stroke-width="1" stroke-dasharray="6 6"/>'
        )
        for idx, (d_ce, r_ce) in enumerate(zip(direct, repaired)):
            color = "#1f9d55" if r_ce < d_ce else "#d9480f"
            parts.append(
                f'<circle cx="{x_of(d_ce):.2f}" cy="{y_of(r_ce):.2f}" r="4" fill="{color}" fill-opacity="0.72">'
                f'<title>#{idx} direct={d_ce:.3f} repaired={r_ce:.3f}</title></circle>'
            )
        better = sum(1 for d_ce, r_ce in zip(direct, repaired) if r_ce < d_ce)
        parts.append(
            f'<text x="{width - margin}" y="{margin - 18}" text-anchor="end" font-family="Arial" font-size="14">'
            f'repaired better: {better}/{min(len(direct), len(repaired))}</text>'
        )
    else:
        denom = max(len(direct) - 1, 1)
        for idx, d_ce in enumerate(direct):
            x = margin + idx / denom * plot_w
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y_of(d_ce):.2f}" r="4" fill="#2563eb" fill-opacity="0.72">'
                f'<title>#{idx} direct={d_ce:.3f}</title></circle>'
            )
        parts.append(
            f'<text x="20" y="{height / 2:.1f}" transform="rotate(-90 20 {height / 2:.1f})" '
            'text-anchor="middle" font-family="Arial" font-size="16">direct CE</text>'
        )

    for tick in range(5):
        v = lo + tick / 4 * (hi - lo)
        x = x_of(v)
        y = y_of(v)
        parts.append(f'<line x1="{x:.2f}" y1="{height - margin}" x2="{x:.2f}" y2="{height - margin + 6}" stroke="#222"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - margin + 22}" text-anchor="middle" font-family="Arial" font-size="12">{v:.2f}</text>')
        parts.append(f'<line x1="{margin - 6}" y1="{y:.2f}" x2="{margin}" y2="{y:.2f}" stroke="#222"/>')
        parts.append(f'<text x="{margin - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{v:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.max_seq_len is None:
        args.max_seq_len = args.prompt_len + args.suffix_len
    else:
        args.suffix_len = args.max_seq_len - args.prompt_len
    if args.suffix_len <= 0:
        raise ValueError("suffix_len must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)
    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim, stage1_ckpt = load_stage1(args.stage1, device)
    args.prompt_len = int(stage1_ckpt.get("prompt_len", args.prompt_len))
    args.max_seq_len = int(stage1_ckpt.get("max_seq_len", args.max_seq_len))
    args.suffix_len = args.max_seq_len - args.prompt_len
    rfm.PROMPT_LEN = args.prompt_len
    rfm.MAX_SEQ_LEN = args.max_seq_len
    print(
        f"slots: prompt_len={args.prompt_len} suffix_len={args.suffix_len} max_seq_len={args.max_seq_len}",
        flush=True,
    )

    draft_prior, ckpt_alpha = load_draft_prior(args.draft_prior, latent_dim, device)
    draft_alpha = args.draft_alpha if args.draft_alpha is not None else ckpt_alpha

    rows, warnings = load_rows(args)
    rows = rows[: args.num_samples]
    references = [fixed_suffix_reference(row, tokenizer, args.suffix_len) for row in rows]
    drafts = make_drafts(rows, args, tokenizer)
    print(f"loaded rows={len(rows)} draft_source={args.draft_source}", flush=True)

    example_rows: list[dict[str, Any]] = []
    direct_ce: list[float] = []
    direct_p: list[float] = []
    direct_top1: list[float] = []
    repaired_ce: list[float] = []
    repaired_p: list[float] = []
    repaired_top1: list[float] = []

    for start_idx in range(0, len(rows), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(rows))
        batch_rows = rows[start_idx:end_idx]
        prompts = [row["prompt"] for row in batch_rows]
        ref_batch = references[start_idx:end_idx]
        draft_batch = drafts[start_idx:end_idx]

        ref_pack = pack_prompt_suffix_inputs(tokenizer, prompts, ref_batch, args.prompt_len, args.suffix_len)
        draft_pack = pack_prompt_suffix_inputs(tokenizer, prompts, draft_batch, args.prompt_len, args.suffix_len)
        ref_ids = ref_pack["input_ids"].to(device)
        ref_mask = ref_pack["attention_mask"].to(device)
        draft_ids = draft_pack["input_ids"].to(device)
        draft_mask = draft_pack["attention_mask"].to(device)

        z_ref = encode_latents(encoder, decoder, ref_ids, ref_mask)
        z_draft_full = encode_latents(encoder, decoder, draft_ids, draft_mask)
        z_prompt = z_ref[:, : args.prompt_len]
        z_draft = z_draft_full[:, args.prompt_len :]
        suffix_ids = ref_ids[:, args.prompt_len :]
        suffix_mask = ref_mask[:, args.prompt_len :]

        ce, prob, top1 = decoder_ce_stats(decoder, z_prompt, z_draft, suffix_ids, suffix_mask)
        direct_ce.extend(ce.detach().cpu().tolist())
        direct_p.extend(prob.detach().cpu().tolist())
        direct_top1.extend(top1.detach().cpu().tolist())
        direct_text = decode_suffixes(decoder, tokenizer, z_prompt, z_draft)

        repaired_text: list[str | None] = [None] * len(batch_rows)
        if draft_prior is not None:
            z_repaired = repair_with_draft_prior(draft_prior, z_draft, z_prompt, suffix_mask, float(draft_alpha))
            ce_r, prob_r, top1_r = decoder_ce_stats(decoder, z_prompt, z_repaired, suffix_ids, suffix_mask)
            repaired_ce.extend(ce_r.detach().cpu().tolist())
            repaired_p.extend(prob_r.detach().cpu().tolist())
            repaired_top1.extend(top1_r.detach().cpu().tolist())
            repaired_text = decode_suffixes(decoder, tokenizer, z_prompt, z_repaired)

        for local_idx, row in enumerate(batch_rows):
            global_idx = start_idx + local_idx
            item = {
                "idx": global_idx,
                "prompt": row["prompt"],
                "reference": ref_batch[local_idx],
                "rough_draft": draft_batch[local_idx],
                "direct_decode": direct_text[local_idx],
                "direct_ce": direct_ce[global_idx],
                "direct_p": direct_p[global_idx],
                "direct_top1": direct_top1[global_idx],
            }
            if draft_prior is not None:
                item.update(
                    {
                        "repaired_decode": repaired_text[local_idx],
                        "repaired_ce": repaired_ce[global_idx],
                        "repaired_p": repaired_p[global_idx],
                        "repaired_top1": repaired_top1[global_idx],
                    }
                )
            example_rows.append(item)

    summary = {
        "stage1": args.stage1,
        "draft_prior": args.draft_prior,
        "draft_source": args.draft_source,
        "num_examples": len(rows),
        "prompt_len": args.prompt_len,
        "suffix_len": args.suffix_len,
        "max_seq_len": args.max_seq_len,
        "draft_alpha": draft_alpha,
        "warnings": warnings,
        "direct_ce": mean_metric(direct_ce),
        "direct_p": mean_metric(direct_p),
        "direct_top1": mean_metric(direct_top1),
        "repaired_ce": mean_metric(repaired_ce),
        "repaired_p": mean_metric(repaired_p),
        "repaired_top1": mean_metric(repaired_top1),
    }

    examples_path = output_dir / "rough_draft_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as f:
        for item in example_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    csv_path = output_dir / "rough_draft_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(example_rows[0].keys()) if example_rows else ["idx"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(example_rows)

    summary_path = output_dir / "rough_draft_summary.json"
    svg_path = output_dir / "rough_draft_ce_scatter.svg"
    write_ce_svg(svg_path, direct_ce, repaired_ce, f"Rough Draft CE: {args.draft_source}")
    summary["outputs"] = [str(examples_path), str(csv_path), str(svg_path), str(summary_path)]
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
