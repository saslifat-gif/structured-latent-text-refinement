"""Train a base-stage decoder-readable VQ latent language.

This makes VQ codes part of the core latent construction path:

    target text -> frozen encoder/compress -> real suffix latents
    real suffix latents -> VQ code ids/code embeddings
    prompt latents + VQ embeddings -> trainable decoder readout -> target tokens

The encoder and decoder compress layer stay frozen. The VQ tokenizer and decoder
readout learn together so later prompt models can predict discrete codes instead
of unstable arbitrary 768D vectors.
"""

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

import stage2_data as s2data
import stage2_losses as s2losses
import stage2_riemannian as rfm
from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained
from stage2_config import SEED
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss
from stage2_riemannian import VQLatentTokenizer
from train_code_prior import code_usage, token_collapse_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train foundational VQ latent language + decoder readout")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=768)
    parser.add_argument("--codebook_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--decoder_lr", type=float, default=5e-5)
    parser.add_argument("--bert_lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--ce_weight", type=float, default=2.0)
    parser.add_argument("--mse_weight", type=float, default=0.05)
    parser.add_argument("--cos_weight", type=float, default=0.05)
    parser.add_argument("--commit_weight", type=float, default=1.0)
    parser.add_argument("--usage_entropy_weight", type=float, default=0.05)
    parser.add_argument("--adjacent_code_weight", type=float, default=0.05)
    parser.add_argument("--soft_temp", type=float, default=0.2)
    parser.add_argument("--bert_last_layers", type=int, default=0)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--stage1_output", default=None)
    parser.add_argument("--examples", default="vq_base_stage_examples.txt")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_stage1(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    latent_dim = int(ckpt.get("latent_dim", 256))
    encoder = BertEncoder().to(device)
    decoder = ParallelDecoder(latent_dim=latent_dim).to(device)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    freeze(encoder)
    return encoder, decoder, latent_dim


def configure_decoder(decoder, bert_last_layers):
    for param in decoder.parameters():
        param.requires_grad_(False)
    # Compress defines the frozen Stage1 latent space. Decoder readout adapts to VQ codes.
    for module in (decoder.project_up, decoder.to_logits):
        for param in module.parameters():
            param.requires_grad_(True)
    bert_params = []
    if bert_last_layers > 0:
        layers = list(decoder.bert.encoder.layer)[-int(bert_last_layers) :]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad_(True)
                bert_params.append(param)
    return [p for p in decoder.parameters() if p.requires_grad], bert_params


@torch.no_grad()
def encode_latents(encoder, decoder, input_ids, attention_mask):
    return decoder.compress(encoder(input_ids, attention_mask))


def soft_code_probs(vq, z, temp):
    z_e = vq.encoder(z)
    flat = z_e.reshape(-1, z_e.size(-1)).float()
    codebook = vq.codebook.weight.float()
    distances = (
        flat.pow(2).sum(dim=1, keepdim=True)
        - 2.0 * flat @ codebook.t()
        + codebook.pow(2).sum(dim=1).unsqueeze(0)
    )
    probs = torch.softmax(-distances / max(temp, 1e-5), dim=-1).reshape(*z.shape[:2], -1)
    return probs


def code_regularizers(vq, z_target, code_ids, mask, temp):
    probs = soft_code_probs(vq, z_target, temp)
    valid = mask.bool()
    if valid.any():
        marginal = probs[valid].mean(dim=0)
        usage_entropy = -(marginal.clamp_min(1e-8) * marginal.clamp_min(1e-8).log()).sum()
    else:
        usage_entropy = -(probs.reshape(-1, probs.size(-1)).mean(dim=0).clamp_min(1e-8).log()).sum()
    usage_loss = -usage_entropy

    if probs.size(1) > 1:
        adjacent_same = (probs[:, :-1] * probs[:, 1:]).sum(dim=-1)
        pair_mask = valid[:, :-1] & valid[:, 1:]
        adjacent_loss = adjacent_same[pair_mask].mean() if pair_mask.any() else adjacent_same.mean()
    else:
        adjacent_loss = probs.new_tensor(0.0)
    hard_adj = 0.0
    if code_ids.size(1) > 1:
        pair_mask = valid[:, :-1] & valid[:, 1:]
        same = code_ids[:, 1:].eq(code_ids[:, :-1]).float()
        hard_adj = same[pair_mask].mean().item() if pair_mask.any() else same.mean().item()
    return usage_loss, adjacent_loss, usage_entropy.detach().item(), hard_adj


def norm_gap(z_q, z_target, mask):
    valid = mask.bool()
    if valid.any():
        return F.smooth_l1_loss(z_q.norm(dim=-1)[valid], z_target.detach().norm(dim=-1)[valid])
    return F.smooth_l1_loss(z_q.norm(dim=-1), z_target.detach().norm(dim=-1))


def compute_loss(args, decoder, vq, z_prompt, z_target, suffix_ids, suffix_mask):
    z_q, code_ids, vq_loss, parts = vq(z_target, suffix_mask)
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_q[: args.decode_batch]], dim=1))
    ce, p, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids[: args.decode_batch],
        suffix_mask[: args.decode_batch],
    )
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_q[valid], z_target.detach()[valid]) if valid.any() else F.mse_loss(z_q, z_target.detach())
    cos_loss, cos_val = rollout_cosine_alignment_loss(z_q, z_target, suffix_mask)
    usage_loss, adj_code_loss, usage_entropy, hard_adj = code_regularizers(
        vq, z_target.detach(), code_ids.detach(), suffix_mask, args.soft_temp
    )
    max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(
        decoder,
        z_prompt[: args.decode_batch],
        z_q[: args.decode_batch],
        suffix_mask[: args.decode_batch],
        args.prompt_len,
    )
    loss = (
        args.ce_weight * ce
        + args.mse_weight * mse
        + args.cos_weight * cos_loss
        + args.commit_weight * vq_loss
        + args.usage_entropy_weight * usage_loss
        + args.adjacent_code_weight * adj_code_loss
    )
    return loss, z_q.detach(), code_ids.detach(), {
        "ce": ce.detach().item(),
        "p": p,
        "top1": top1,
        "mse": mse.detach().item(),
        "cos": cos_val,
        "vq": vq_loss.detach().item(),
        "codebook": parts["codebook_loss"].detach().item(),
        "commit": parts["commitment_loss"].detach().item(),
        "usage_entropy": usage_entropy,
        "usage_loss": usage_loss.detach().item(),
        "adj_code": adj_code_loss.detach().item(),
        "hard_adj_code": hard_adj,
        "max_token_frac": max_frac,
        "unique_token_ratio": unique_tok,
        "repeat_frac": rep_frac,
        "punct_frac": punct_frac,
        "norm": norm_gap(z_q, z_target, suffix_mask).detach().item(),
    }


def add(total, stats):
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + float(value)
    total["n"] = total.get("n", 0) + 1


def mean(total):
    n = max(total.get("n", 0), 1)
    return {key: value / n for key, value in total.items() if key != "n"}


@torch.no_grad()
def write_examples(path, tokenizer, decoder, z_prompt, z_q, input_ids, args):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_q], dim=1))
    pred_ids = logits.argmax(dim=-1)
    rows = []
    for i in range(min(16, input_ids.size(0))):
        prompt = tokenizer.decode(input_ids[i, : args.prompt_len], skip_special_tokens=True).strip()
        target = tokenizer.decode(input_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        pred = tokenizer.decode(pred_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        rows.append(f"--- example {i + 1}\nprompt: {prompt}\ntarget: {target}\nvq-base recon: {pred}\n")
    Path(path).write_text("\n".join(rows), encoding="utf-8")


def main():
    args = parse_args()
    seed_everything(SEED)
    configure_data(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)
    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    if latent_dim != args.latent_dim:
        print(f"latent_dim override: checkpoint={latent_dim}, args={args.latent_dim}", flush=True)
        args.latent_dim = latent_dim
    decoder_params, bert_params = configure_decoder(decoder, args.bert_last_layers)
    vq = VQLatentTokenizer(latent_dim, args.codebook_size, args.commitment).to(device)
    if args.output is None:
        args.output = f"vq_base_stage_rocstories_{latent_dim}_K{args.codebook_size}_best.pt"
    if args.stage1_output is None:
        args.stage1_output = f"stage1_vqbase_rocstories_{latent_dim}_K{args.codebook_size}_best.pt"

    train_loader, val_loader = s2data.build_stage2_dataloaders(
        tokenizer, args.train_size, args.batch_size, args.max_seq_len
    )
    param_groups = [
        {"params": vq.parameters(), "lr": args.lr},
        {"params": decoder_params, "lr": args.decoder_lr},
    ]
    if bert_params:
        param_groups.append({"params": bert_params, "lr": args.bert_lr})
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = float("inf")
    print(
        f"VQ base params vq={sum(p.numel() for p in vq.parameters()):,} "
        f"decoder_trainable={sum(p.numel() for p in decoder.parameters() if p.requires_grad):,} "
        f"K={args.codebook_size} bert_last_layers={args.bert_last_layers}",
        flush=True,
    )

    for epoch in range(args.epochs):
        vq.train()
        decoder.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"vq-base ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad():
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, _z_q, code_ids, stats = compute_loss(
                    args, decoder, vq, z_prompt, z_target, suffix_ids, suffix_mask
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(vq.parameters()) + [p for p in decoder.parameters() if p.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            add(total, stats)
            if step % 50 == 0:
                ppl, dead, used = vq.usage_stats(code_ids, suffix_mask)
                print(
                    f"ep{epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f} "
                    f"ce={stats['ce']:.3f} p={stats['p']:.3f} rep={stats['repeat_frac']:.3f} "
                    f"uent={stats['usage_entropy']:.2f} adjcode={stats['hard_adj_code']:.3f} "
                    f"ppl={ppl:.1f} used={used} dead={dead:.1f}%",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(
                    ce=f"{stats['ce']:.2f}",
                    p=f"{stats['p']:.3f}",
                    rep=f"{stats['repeat_frac']:.2f}",
                    adj=f"{stats['hard_adj_code']:.2f}",
                )
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        vq.eval()
        decoder.eval()
        val_total = {}
        usage_counts = torch.zeros(args.codebook_size, device=device)
        last = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"vq-base ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                _loss, z_q, code_ids, stats = compute_loss(
                    args, decoder, vq, z_prompt, z_target, suffix_ids, suffix_mask
                )
                add(val_total, stats)
                ids = code_ids[suffix_mask.bool()]
                if ids.numel():
                    usage_counts += torch.bincount(ids, minlength=args.codebook_size).to(usage_counts)
                last = (z_prompt[:16], z_q[:16], input_ids[:16])
        val_mean = mean(val_total)
        probs = usage_counts / usage_counts.sum().clamp_min(1.0)
        usage_ppl = float((-(probs[probs > 0] * probs[probs > 0].log()).sum()).exp().item())
        used = int((usage_counts > 0).sum().item())
        dead = float(100.0 * (args.codebook_size - used) / args.codebook_size)
        val_mean.update({"code_ppl": usage_ppl, "used_codes": used, "dead_code_percent": dead})
        print(f"val ep{epoch + 1} | {json.dumps(val_mean, indent=2)}", flush=True)
        score = val_mean["ce"] + val_mean["repeat_frac"] + 0.01 * val_mean["hard_adj_code"]
        if score < best_score:
            best_score = score
            ckpt = {
                "vq": vq.state_dict(),
                "decoder": decoder.state_dict(),
                "best_score": best_score,
                "val": val_mean,
                "latent_dim": latent_dim,
                "codebook_size": args.codebook_size,
                "prompt_len": args.prompt_len,
                "max_seq_len": args.max_seq_len,
                "commitment": args.commitment,
                "ce_weight": args.ce_weight,
                "mse_weight": args.mse_weight,
                "cos_weight": args.cos_weight,
                "commit_weight": args.commit_weight,
                "usage_entropy_weight": args.usage_entropy_weight,
                "adjacent_code_weight": args.adjacent_code_weight,
                "soft_temp": args.soft_temp,
                "bert_last_layers": args.bert_last_layers,
                "stage1": args.stage1,
                "type": "vq_base_stage",
                "epoch": epoch,
            }
            torch.save(ckpt, args.output)
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "latent_dim": latent_dim,
                    "source_stage1": args.stage1,
                    "vq_base_stage": args.output,
                    "type": "stage1_vqbase_decoder",
                    "epoch": epoch,
                },
                args.stage1_output,
            )
            print(f"saved {args.output} and {args.stage1_output} | score={best_score:.4f}", flush=True)
            if last is not None:
                write_examples(args.examples, tokenizer, decoder, *last, args)


if __name__ == "__main__":
    main()
