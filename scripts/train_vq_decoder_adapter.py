"""Train a small decoder adapter for VQ/generated-like suffix latents."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import stage2_data as s2data
import stage2_losses as s2losses
import stage2_riemannian as rfm
from parallel_decoder import cached_from_pretrained
from stage2_config import SEED
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss
from stage2_riemannian import VQDecoderAdapter, suffix_positions
from train_code_prior import encode_latents, load_stage1, load_vq, mean, token_collapse_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ decoder adapter")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--vq", required=True)
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--delta_scale", type=float, default=0.5)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.05)
    parser.add_argument("--cos_weight", type=float, default=0.05)
    parser.add_argument("--delta_weight", type=float, default=0.02)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--example_max_tokens", type=int, default=48)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--examples", default="vq_decoder_adapter_examples.txt")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_data(args):
    s2data.PROMPT_LEN = args.prompt_len
    s2data.DATASET_NAME = args.dataset
    s2data.ROCSTORIES_LOCAL_FILES_ONLY = args.local_files_only
    rfm.PROMPT_LEN = args.prompt_len
    rfm.MAX_SEQ_LEN = args.max_seq_len
    s2losses.PROMPT_LEN = args.prompt_len


def add(total, stats):
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + float(value)
    total["n"] = total.get("n", 0) + 1


def score_latents(args, decoder, z_prompt, z_suffix, z_target, suffix_ids, suffix_mask, prefix):
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_suffix[: args.decode_batch]], dim=1))
    ce, p, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids[: args.decode_batch],
        suffix_mask[: args.decode_batch],
    )
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_suffix[valid], z_target.detach()[valid]) if valid.any() else F.mse_loss(z_suffix, z_target.detach())
    _cos_loss, cos_val = rollout_cosine_alignment_loss(z_suffix, z_target, suffix_mask)
    max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(
        decoder,
        z_prompt[: args.decode_batch],
        z_suffix[: args.decode_batch],
        suffix_mask[: args.decode_batch],
        args.prompt_len,
    )
    return {
        f"{prefix}_ce": ce.detach().item(),
        f"{prefix}_p": p,
        f"{prefix}_top1": top1,
        f"{prefix}_mse": mse.detach().item(),
        f"{prefix}_cos": cos_val,
        f"{prefix}_max_token_frac": max_frac,
        f"{prefix}_unique_token_ratio": unique_tok,
        f"{prefix}_repeat_frac": rep_frac,
        f"{prefix}_punct_frac": punct_frac,
    }


def compute_loss(args, decoder, adapter, z_prompt, z_q, z_target, suffix_ids, suffix_mask):
    pos = suffix_positions(z_q.size(0), z_q.size(1), z_q.device, z_q.dtype)
    z_adapt, delta = adapter(z_q, z_prompt, pos, suffix_mask, return_delta=True)
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_adapt[: args.decode_batch]], dim=1))
    ce, p, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids[: args.decode_batch],
        suffix_mask[: args.decode_batch],
    )
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_adapt[valid], z_target.detach()[valid]) if valid.any() else F.mse_loss(z_adapt, z_target.detach())
    cos_loss, cos_val = rollout_cosine_alignment_loss(z_adapt, z_target, suffix_mask)
    valid = suffix_mask.bool()
    delta_reg = delta[valid].pow(2).mean() if valid.any() else delta.pow(2).mean()
    loss = (
        args.ce_weight * ce
        + args.mse_weight * mse
        + args.cos_weight * cos_loss
        + args.delta_weight * delta_reg
    )
    max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(
        decoder,
        z_prompt[: args.decode_batch],
        z_adapt[: args.decode_batch],
        suffix_mask[: args.decode_batch],
        args.prompt_len,
    )
    stats = {
        "adapt_ce": ce.detach().item(),
        "adapt_p": p,
        "adapt_top1": top1,
        "adapt_mse": mse.detach().item(),
        "adapt_cos": cos_val,
        "adapt_max_token_frac": max_frac,
        "adapt_unique_token_ratio": unique_tok,
        "adapt_repeat_frac": rep_frac,
        "adapt_punct_frac": punct_frac,
        "delta_reg": delta_reg.detach().item(),
    }
    return loss, stats, z_adapt


@torch.no_grad()
def write_examples(path, tokenizer, decoder, z_prompt, z_q, z_adapt, input_ids, args):
    q_logits = decoder.decode_from_latent(torch.cat([z_prompt, z_q], dim=1))[:, args.prompt_len : args.prompt_len + args.example_max_tokens]
    a_logits = decoder.decode_from_latent(torch.cat([z_prompt, z_adapt], dim=1))[:, args.prompt_len : args.prompt_len + args.example_max_tokens]
    q_ids = q_logits.argmax(dim=-1)
    a_ids = a_logits.argmax(dim=-1)
    rows = []
    for i in range(min(12, input_ids.size(0))):
        prompt = tokenizer.decode(input_ids[i, : args.prompt_len], skip_special_tokens=True).strip()
        target = tokenizer.decode(input_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        q_out = tokenizer.decode(q_ids[i], skip_special_tokens=True).strip()
        a_out = tokenizer.decode(a_ids[i], skip_special_tokens=True).strip()
        rows.append(
            f"--- example {i + 1}\n"
            f"prompt: {prompt}\n"
            f"target: {target}\n"
            f"vq: {q_out}\n"
            f"adapter: {a_out}\n"
        )
    Path(path).write_text("\n".join(rows), encoding="utf-8")


def main():
    args = parse_args()
    if args.max_seq_len != args.prompt_len + args.suffix_len:
        args.max_seq_len = args.prompt_len + args.suffix_len
    seed_everything(SEED)
    configure_data(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    vq, _vq_ckpt = load_vq(args.vq, latent_dim, device)
    if args.output is None:
        args.output = f"vq_decoder_adapter_rocstories_{latent_dim}_K{vq.codebook_size}_best.pt"
    adapter = VQDecoderAdapter(
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        delta_scale=args.delta_scale,
    ).to(device)
    train_loader, val_loader = s2data.build_stage2_dataloaders(tokenizer, args.train_size, args.batch_size, args.max_seq_len)
    optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_ce = float("inf")
    print(f"VQDecoderAdapter params={sum(p.numel() for p in adapter.parameters()):,}", flush=True)

    for epoch in range(args.epochs):
        adapter.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"adapter ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad():
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                z_q, _code_ids, _vq_loss, _parts = vq(z_target, suffix_mask)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, stats, _z_adapt = compute_loss(args, decoder, adapter, z_prompt, z_q, z_target, suffix_ids, suffix_mask)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            add(total, stats)
            if step % 50 == 0:
                print(
                    f"ep{epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f} "
                    f"adapt_ce={stats['adapt_ce']:.3f} p={stats['adapt_p']:.3f} "
                    f"rep={stats['adapt_repeat_frac']:.3f} delta={stats['delta_reg']:.5f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(ce=f"{stats['adapt_ce']:.3f}", p=f"{stats['adapt_p']:.3f}")
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        adapter.eval()
        val_total = {}
        last = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"adapter ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                z_q, _code_ids, _vq_loss, _parts = vq(z_target, suffix_mask)
                pos = suffix_positions(z_q.size(0), z_q.size(1), device, z_q.dtype)
                z_adapt = adapter(z_q, z_prompt, pos, suffix_mask)
                raw = score_latents(args, decoder, z_prompt, z_q, z_target, suffix_ids, suffix_mask, "vq")
                adapted = score_latents(args, decoder, z_prompt, z_adapt, z_target, suffix_ids, suffix_mask, "adapt")
                add(val_total, raw | adapted)
                last = (z_prompt[:12], z_q[:12], z_adapt[:12], input_ids[:12])
        val_mean = mean(val_total)
        print(f"val ep{epoch + 1} | {json.dumps(val_mean, indent=2)}", flush=True)
        if val_mean["adapt_ce"] < best_ce:
            best_ce = val_mean["adapt_ce"]
            torch.save(
                {
                    "adapter": adapter.state_dict(),
                    "best_adapt_ce": best_ce,
                    "val": val_mean,
                    "latent_dim": latent_dim,
                    "codebook_size": vq.codebook_size,
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "delta_scale": args.delta_scale,
                    "prompt_len": args.prompt_len,
                    "max_seq_len": args.max_seq_len,
                    "vq_path": args.vq,
                    "type": "vq_decoder_adapter",
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | adapt_ce={best_ce:.4f}", flush=True)
            if last is not None:
                write_examples(args.examples, tokenizer, decoder, *last, args)


if __name__ == "__main__":
    main()
