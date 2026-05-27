"""Visualize decoder repetition on PromptPrior-generated suffix latents."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import stage2_data as s2data
import stage2_losses as s2losses
import stage2_riemannian as rfm
from parallel_decoder import cached_from_pretrained
from stage2_config import SEED
from train_basin_projector import load_prompt_prior, prompt_prior_latents
from train_code_prior import encode_latents, load_stage1
from transformers import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect why PromptPrior decoded suffixes repeat")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--prompt_prior", required=True)
    parser.add_argument("--decoder_checkpoint", default="", help="Optional adapted Stage1-style decoder checkpoint to compare.")
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_size", type=int, default=512)
    parser.add_argument("--num_examples", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_dir", default="results/decoder_repetition")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint_path(path_value, label, required=True):
    if not path_value:
        if required:
            raise SystemExit(f"{label} checkpoint path is required")
        return ""
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if required:
        raise SystemExit(f"{label} checkpoint not found: {path_value}")
    return ""


def configure_data(args):
    if args.max_seq_len != args.prompt_len + args.suffix_len:
        args.max_seq_len = args.prompt_len + args.suffix_len
    s2data.PROMPT_LEN = args.prompt_len
    s2data.DATASET_NAME = args.dataset
    s2data.ROCSTORIES_LOCAL_FILES_ONLY = args.local_files_only
    rfm.PROMPT_LEN = args.prompt_len
    rfm.MAX_SEQ_LEN = args.max_seq_len
    s2losses.PROMPT_LEN = args.prompt_len


def freeze(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def load_decoder_only(path, latent_dim, device):
    _encoder, decoder, ckpt_latent_dim = load_stage1(path, device)
    if int(ckpt_latent_dim) != int(latent_dim):
        raise RuntimeError(f"decoder checkpoint latent_dim={ckpt_latent_dim}, expected {latent_dim}")
    freeze(decoder)
    return decoder


def token_text(tokenizer, token_id):
    return tokenizer.convert_ids_to_tokens([int(token_id)])[0]


def decoded_text(tokenizer, ids, mask):
    valid_ids = [int(tok) for tok, keep in zip(ids.tolist(), mask.tolist()) if int(keep)]
    return tokenizer.decode(valid_ids, skip_special_tokens=True).strip()


def longest_repeat_run(ids, mask):
    best_token = None
    best_len = 0
    current_token = None
    current_len = 0
    for token, keep in zip(ids.tolist(), mask.tolist()):
        if not int(keep):
            continue
        token = int(token)
        if token == current_token:
            current_len += 1
        else:
            current_token = token
            current_len = 1
        if current_len > best_len:
            best_len = current_len
            best_token = token
    return best_token, best_len


@torch.no_grad()
def inspect_decoder(args, tokenizer, decoder, name, z_prompt, z_suffix, suffix_ids, suffix_mask, example_offset):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))[:, args.prompt_len :]
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    pred_ids = probs.argmax(dim=-1)
    pred_prob = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)
    target_prob = probs.gather(-1, suffix_ids.unsqueeze(-1)).squeeze(-1)
    top_probs, top_ids = probs.topk(min(args.top_k, probs.size(-1)), dim=-1)

    example_rows = []
    token_rows = []
    for b in range(pred_ids.size(0)):
        mask = suffix_mask[b].bool()
        valid_pred = pred_ids[b][mask]
        valid_target = suffix_ids[b][mask]
        if valid_pred.numel() == 0:
            continue
        repeat_frac = valid_pred[1:].eq(valid_pred[:-1]).float().mean().item() if valid_pred.numel() > 1 else 0.0
        counts = Counter(int(tok) for tok in valid_pred.tolist())
        dominant_id, dominant_count = counts.most_common(1)[0]
        run_id, run_len = longest_repeat_run(pred_ids[b], suffix_mask[b])
        row = {
            "decoder": name,
            "example_id": example_offset + b,
            "ce": float(-(target_prob[b][mask].clamp_min(1e-12).log()).mean().item()),
            "target_prob": float(target_prob[b][mask].mean().item()),
            "top1": float(pred_ids[b][mask].eq(suffix_ids[b][mask]).float().mean().item()),
            "entropy": float(entropy[b][mask].mean().item()),
            "pred_prob": float(pred_prob[b][mask].mean().item()),
            "repeat_frac": repeat_frac,
            "unique_ratio": float(len(counts) / max(valid_pred.numel(), 1)),
            "dominant_token": token_text(tokenizer, dominant_id),
            "dominant_frac": float(dominant_count / max(valid_pred.numel(), 1)),
            "longest_run_token": token_text(tokenizer, run_id) if run_id is not None else "",
            "longest_run_len": int(run_len),
            "target_text": tokenizer.decode(valid_target.tolist(), skip_special_tokens=True).strip(),
            "pred_text": tokenizer.decode(valid_pred.tolist(), skip_special_tokens=True).strip(),
        }
        example_rows.append(row)

        previous_id = None
        for pos in range(pred_ids.size(1)):
            if not bool(mask[pos]):
                continue
            pred_id = int(pred_ids[b, pos].item())
            target_id = int(suffix_ids[b, pos].item())
            top = [
                f"{token_text(tokenizer, int(tok))}:{float(prob):.3f}"
                for tok, prob in zip(top_ids[b, pos].tolist(), top_probs[b, pos].tolist())
            ]
            token_rows.append(
                {
                    "decoder": name,
                    "example_id": example_offset + b,
                    "pos": pos,
                    "target_token": token_text(tokenizer, target_id),
                    "pred_token": token_text(tokenizer, pred_id),
                    "is_repeat": int(previous_id == pred_id) if previous_id is not None else 0,
                    "is_correct": int(pred_id == target_id),
                    "entropy": float(entropy[b, pos].item()),
                    "pred_prob": float(pred_prob[b, pos].item()),
                    "target_prob": float(target_prob[b, pos].item()),
                    "topk": " | ".join(top),
                }
            )
            previous_id = pred_id
    return example_rows, token_rows


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def style_for_token(row):
    entropy = float(row["entropy"])
    target_prob = float(row["target_prob"])
    if int(row["is_repeat"]):
        return "background:#fecaca;border-color:#ef4444"
    if target_prob < 0.02:
        return "background:#fed7aa;border-color:#f97316"
    if entropy < 1.5:
        return "background:#fde68a;border-color:#eab308"
    return "background:#dcfce7;border-color:#22c55e"


def write_html(path, example_rows, token_rows, max_examples=16):
    by_key = {}
    for row in token_rows:
        by_key.setdefault((row["decoder"], row["example_id"]), []).append(row)

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#111827}"
        ".ex{border-top:1px solid #d1d5db;padding:18px 0}.tok{display:inline-block;padding:3px 5px;margin:2px;"
        "border:1px solid #d1d5db;border-radius:4px;font-size:13px}.small{color:#4b5563;font-size:13px}"
        "pre{white-space:pre-wrap;background:#f9fafb;padding:10px;border-radius:6px}</style>",
        "<h1>Decoder Repetition Diagnostic</h1>",
        "<p class='small'>Red = adjacent repeat, orange = low target probability, yellow = low entropy, green = healthier token.</p>",
    ]
    shown = 0
    for row in example_rows:
        if shown >= max_examples:
            break
        shown += 1
        key = (row["decoder"], row["example_id"])
        parts.append("<div class='ex'>")
        parts.append(
            f"<h2>{html.escape(row['decoder'])} example {row['example_id']}</h2>"
            f"<p class='small'>CE={row['ce']:.3f} p={row['target_prob']:.3f} top1={row['top1']:.3f} "
            f"H={row['entropy']:.2f} repeat={row['repeat_frac']:.2f} unique={row['unique_ratio']:.2f} "
            f"dominant={html.escape(str(row['dominant_token']))} ({row['dominant_frac']:.2f}) "
            f"longest_run={html.escape(str(row['longest_run_token']))} x{row['longest_run_len']}</p>"
        )
        parts.append(f"<p><b>Target</b></p><pre>{html.escape(row['target_text'])}</pre>")
        parts.append(f"<p><b>Prediction</b></p><pre>{html.escape(row['pred_text'])}</pre>")
        parts.append("<p><b>Tokens</b></p><div>")
        for tok in by_key.get(key, []):
            title = (
                f"target={tok['target_token']} pred={tok['pred_token']} "
                f"target_prob={tok['target_prob']:.4f} pred_prob={tok['pred_prob']:.4f} "
                f"entropy={tok['entropy']:.2f} top={tok['topk']}"
            )
            parts.append(
                f"<span class='tok' style='{style_for_token(tok)}' title='{html.escape(title)}'>"
                f"{html.escape(tok['pred_token'])}</span>"
            )
        parts.append("</div></div>")
    path.write_text("\n".join(parts), encoding="utf-8")


def numeric_mean(rows):
    totals = {}
    counts = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in sorted(totals)}


def grouped_numeric_mean(rows, group_key):
    groups = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key, "")), []).append(row)
    return {key: numeric_mean(value) for key, value in sorted(groups.items())}


def dominant_token_summary(rows, limit=12):
    by_decoder = {}
    for row in rows:
        decoder = str(row.get("decoder", ""))
        token = str(row.get("dominant_token", ""))
        if not token:
            continue
        by_decoder.setdefault(decoder, Counter())[token] += 1
    return {
        decoder: [{"token": token, "examples": count} for token, count in counts.most_common(limit)]
        for decoder, counts in sorted(by_decoder.items())
    }


def latent_collapse_diag(z_real, z_pred, mask=None):
    z_real = z_real.float()
    z_pred = z_pred.float()
    if mask is not None:
        valid = mask.bool().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1).float()
        pred_pos_std = ((z_pred - (z_pred * valid).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)) * valid).std(dim=1).mean()
        real_pos_std = ((z_real - (z_real * valid).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)) * valid).std(dim=1).mean()
        pair_mask = mask[:, :-1].bool() & mask[:, 1:].bool()
    else:
        pred_pos_std = z_pred.std(dim=1).mean()
        real_pos_std = z_real.std(dim=1).mean()
        pair_mask = None
    pred_adj = torch.nn.functional.cosine_similarity(z_pred[:, :-1, :], z_pred[:, 1:, :], dim=-1)
    real_adj = torch.nn.functional.cosine_similarity(z_real[:, :-1, :], z_real[:, 1:, :], dim=-1)
    if pair_mask is not None and pair_mask.any():
        pred_adj_cos = pred_adj[pair_mask].mean()
        real_adj_cos = real_adj[pair_mask].mean()
    else:
        pred_adj_cos = pred_adj.mean()
        real_adj_cos = real_adj.mean()
    return {
        "pred_pos_std": float(pred_pos_std.item()),
        "real_pos_std": float(real_pos_std.item()),
        "pos_std_ratio": float((pred_pos_std / real_pos_std.clamp_min(1e-8)).item()),
        "pred_adj_cos": float(pred_adj_cos.item()),
        "real_adj_cos": float(real_adj_cos.item()),
        "adj_cos_gap": float((pred_adj_cos - real_adj_cos).item()),
        "pred_token_std": float(z_pred.std(dim=-1).mean().item()),
        "real_token_std": float(z_real.std(dim=-1).mean().item()),
    }


def main():
    args = parse_args()
    seed_everything(SEED)
    configure_data(args)
    args.stage1 = resolve_checkpoint_path(args.stage1, "Stage1")
    args.prompt_prior = resolve_checkpoint_path(args.prompt_prior, "PromptPrior")
    args.decoder_checkpoint = resolve_checkpoint_path(args.decoder_checkpoint, "Decoder", required=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, base_decoder, latent_dim = load_stage1(args.stage1, device)
    freeze(encoder)
    freeze(base_decoder)
    decoders = [("base", base_decoder)]
    if args.decoder_checkpoint:
        adapted_decoder = load_decoder_only(args.decoder_checkpoint, latent_dim, device)
        decoders.append(("adapted", adapted_decoder))
    prompt_prior, prompt_ckpt = load_prompt_prior(args.prompt_prior, latent_dim, device)

    train_loader, _val_loader = s2data.build_stage2_dataloaders(
        tokenizer,
        args.train_size,
        args.batch_size,
        args.max_seq_len,
    )

    example_rows = []
    token_rows = []
    latent_diags = []
    example_offset = 0
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            z = encode_latents(encoder, base_decoder, input_ids, attention_mask)
            z_prompt = z[:, : args.prompt_len]
            z_target = z[:, args.prompt_len :]
            suffix_ids = input_ids[:, args.prompt_len :]
            suffix_mask = attention_mask[:, args.prompt_len :]
            z_pred = prompt_prior_latents(prompt_prior, z_prompt, args.suffix_len, suffix_mask, z_target)
            z_pred = z_pred.to(device=z_target.device, dtype=z_target.dtype)
            take = min(input_ids.size(0), args.num_examples - example_offset)
            if take <= 0:
                break
            latent_diags.append(latent_collapse_diag(z_target[:take], z_pred[:take], suffix_mask[:take]))
            for name, decoder in decoders:
                erows, trows = inspect_decoder(
                    args,
                    tokenizer,
                    decoder,
                    name,
                    z_prompt[:take],
                    z_pred[:take],
                    suffix_ids[:take],
                    suffix_mask[:take],
                    example_offset,
                )
                example_rows.extend(erows)
                token_rows.extend(trows)
            example_offset += take
            if example_offset >= args.num_examples:
                break

    summary = {
        "stage1": args.stage1,
        "prompt_prior": args.prompt_prior,
        "decoder_checkpoint": args.decoder_checkpoint,
        "prompt_prior_type": prompt_ckpt.get("type"),
        "num_examples": example_offset,
        "latent_collapse": numeric_mean(latent_diags),
        "example_metrics": numeric_mean(example_rows),
        "example_metrics_by_decoder": grouped_numeric_mean(example_rows, "decoder"),
        "dominant_tokens_by_decoder": dominant_token_summary(example_rows),
        "outputs": [
            str(out_dir / "decoder_repetition_examples.jsonl"),
            str(out_dir / "decoder_repetition_examples.csv"),
            str(out_dir / "decoder_repetition_tokens.csv"),
            str(out_dir / "decoder_repetition.html"),
        ],
    }
    write_jsonl(out_dir / "decoder_repetition_examples.jsonl", example_rows)
    write_csv(out_dir / "decoder_repetition_examples.csv", example_rows)
    write_csv(out_dir / "decoder_repetition_tokens.csv", token_rows)
    write_html(out_dir / "decoder_repetition.html", example_rows, token_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
