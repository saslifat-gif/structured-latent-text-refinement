"""Train a route-aware prompt -> VQ token-code prior."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
from stage2_riemannian import RouteCodePrior, suffix_positions
from train_code_prior import (
    compute_losses,
    encode_latents,
    hard_stats,
    load_stage1,
    load_vq,
    mean,
    sampled_stats,
    write_examples,
)
from train_hier_code_prior import length_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train route-aware prompt-to-VQ-code prior")
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
    parser.add_argument("--plan_slots", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mixer_layers", type=int, default=2)
    parser.add_argument("--mixer_scale", type=float, default=0.5)
    parser.add_argument("--route_scale", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.2)
    parser.add_argument("--code_ce_weight", type=float, default=1.0)
    parser.add_argument("--decoder_ce_weight", type=float, default=1.0)
    parser.add_argument("--token_ce_weight", type=float, default=0.25)
    parser.add_argument("--event_plan_weight", type=float, default=0.0)
    parser.add_argument("--mse_weight", type=float, default=0.03)
    parser.add_argument("--cos_weight", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.02)
    parser.add_argument("--usage_entropy_weight", type=float, default=0.05)
    parser.add_argument("--valid_weight", type=float, default=0.25)
    parser.add_argument("--end_weight", type=float, default=0.50)
    parser.add_argument("--route_smooth_weight", type=float, default=0.05)
    parser.add_argument("--route_entropy_weight", type=float, default=0.02)
    parser.add_argument("--route_usage_weight", type=float, default=0.02)
    parser.add_argument("--sample_codes", type=int, default=4)
    parser.add_argument("--sample_tau", type=float, default=0.9)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--examples", default="route_code_prior_examples.txt")
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


def route_losses(route_probs, suffix_mask):
    valid = suffix_mask.bool()
    if route_probs.size(1) > 1:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        route_delta = (route_probs[:, 1:] - route_probs[:, :-1]).pow(2).sum(dim=-1)
        smooth_loss = route_delta[pair_valid].mean() if pair_valid.any() else route_delta.mean()
    else:
        smooth_loss = route_probs.new_tensor(0.0)
    token_entropy = -(route_probs.clamp_min(1e-8) * route_probs.clamp_min(1e-8).log()).sum(dim=-1)
    entropy_loss = token_entropy[valid].mean() if valid.any() else token_entropy.mean()
    if valid.any():
        marginal = route_probs[valid].mean(dim=0)
    else:
        marginal = route_probs.reshape(-1, route_probs.size(-1)).mean(dim=0)
    usage_entropy = -(marginal.clamp_min(1e-8) * marginal.clamp_min(1e-8).log()).sum()
    usage_loss = -usage_entropy
    hard_routes = route_probs.argmax(dim=-1)
    used_routes = torch.unique(hard_routes[valid]).numel() if valid.any() else torch.unique(hard_routes).numel()
    return smooth_loss, entropy_loss, usage_loss, {
        "route_smooth": smooth_loss.detach().item(),
        "route_entropy": entropy_loss.detach().item(),
        "route_usage_entropy": usage_entropy.detach().item(),
        "route_used": float(used_routes),
    }


def token_ce_loss(token_logits, suffix_ids, suffix_mask):
    valid = suffix_mask.bool()
    if not valid.any():
        return token_logits.new_tensor(0.0), {
            "token_ce": 0.0,
            "token_p": 0.0,
            "token_top1": 0.0,
        }
    logits = token_logits[valid]
    targets = suffix_ids[valid]
    loss = F.cross_entropy(logits, targets, reduction="mean")
    probs = torch.softmax(logits.detach(), dim=-1)
    target_p = probs.gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    top1 = probs.argmax(dim=-1).eq(targets).float().mean().item()
    return loss, {
        "token_ce": loss.detach().item(),
        "token_p": target_p,
        "token_top1": top1,
    }


def event_plan_loss(plans, z_target, target_sentence_mask, suffix_mask):
    if target_sentence_mask is None:
        zero = plans.new_tensor(0.0)
        return zero, {"event_plan_loss": 0.0, "event_plan_cos": 0.0}
    sent_mask = target_sentence_mask.to(device=z_target.device, dtype=torch.bool)
    if sent_mask.dim() != 3:
        zero = plans.new_tensor(0.0)
        return zero, {"event_plan_loss": 0.0, "event_plan_cos": 0.0}
    sent_mask = sent_mask & suffix_mask.bool().unsqueeze(1)
    event_count = min(plans.size(1), sent_mask.size(1))
    if event_count <= 0:
        zero = plans.new_tensor(0.0)
        return zero, {"event_plan_loss": 0.0, "event_plan_cos": 0.0}
    plans = plans[:, :event_count]
    sent_mask = sent_mask[:, :event_count]
    denom = sent_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(z_target.dtype)
    targets = (sent_mask.unsqueeze(-1).to(z_target.dtype) * z_target.unsqueeze(1)).sum(dim=2) / denom
    valid = sent_mask.any(dim=-1)
    if not valid.any():
        zero = plans.new_tensor(0.0)
        return zero, {"event_plan_loss": 0.0, "event_plan_cos": 0.0}
    pred = plans[valid]
    target = targets[valid].detach()
    mse = F.mse_loss(pred, target)
    cos = F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()
    loss = mse + (1.0 - cos.to(mse.dtype))
    return loss, {
        "event_plan_loss": loss.detach().item(),
        "event_plan_cos": cos.detach().item(),
    }


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
        args.output = f"route_code_prior_rocstories_{latent_dim}_K{vq.codebook_size}_routes{args.plan_slots}_best.pt"

    model = RouteCodePrior(
        latent_dim=latent_dim,
        codebook_size=vq.codebook_size,
        plan_slots=args.plan_slots,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.hidden_dim,
        mixer_layers=args.mixer_layers,
        mixer_scale=args.mixer_scale,
        route_scale=args.route_scale,
    ).to(device)
    valid_head = nn.Linear(latent_dim, 1).to(device)
    end_head = nn.Linear(latent_dim, 1).to(device)
    token_head = nn.Linear(latent_dim, tokenizer.vocab_size).to(device)
    train_loader, val_loader = s2data.build_stage2_dataloaders(
        tokenizer,
        args.train_size,
        args.batch_size,
        args.max_seq_len,
    )
    optimizer = AdamW(
        list(model.parameters())
        + list(valid_head.parameters())
        + list(end_head.parameters())
        + list(token_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = float("inf")
    print(
        f"RouteCodePrior params={sum(p.numel() for p in model.parameters()):,} "
        f"K={vq.codebook_size} routes={args.plan_slots}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        valid_head.train()
        end_head.train()
        token_head.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"route-code ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            target_sentence_mask = batch.get("target_sentence_mask")
            if target_sentence_mask is not None:
                target_sentence_mask = target_sentence_mask.to(device, non_blocking=True)
            with torch.no_grad():
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                code_targets = vq.encode(z_target, suffix_mask)
            suffix_ids = input_ids[:, args.prompt_len :]
            pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                code_logits, aux, prior_hidden = model(z_prompt, pos, suffix_mask, return_aux=True, return_hidden=True)
                loss, stats = compute_losses(
                    args,
                    decoder,
                    vq,
                    z_prompt,
                    z_target,
                    code_logits,
                    code_targets,
                    suffix_ids,
                    suffix_mask,
                )
                valid_logits = valid_head(prior_hidden).squeeze(-1)
                end_logits = end_head(prior_hidden).squeeze(-1)
                token_logits = token_head(prior_hidden)
                valid_loss, end_loss, length_stats = length_losses(valid_logits, end_logits, suffix_mask)
                token_loss, token_stats = token_ce_loss(token_logits, suffix_ids, suffix_mask)
                event_loss, event_stats = event_plan_loss(
                    aux["plans"],
                    z_target,
                    target_sentence_mask,
                    suffix_mask,
                )
                smooth_loss, entropy_loss, usage_loss, route_stats = route_losses(aux["route_probs"], suffix_mask)
                loss = (
                    loss
                    + args.valid_weight * valid_loss
                    + args.end_weight * end_loss
                    + args.token_ce_weight * token_loss
                    + args.event_plan_weight * event_loss
                    + args.route_smooth_weight * smooth_loss
                    + args.route_entropy_weight * entropy_loss
                    + args.route_usage_weight * usage_loss
                )
                stats.update(length_stats)
                stats.update(token_stats)
                stats.update(event_stats)
                stats.update(route_stats)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(valid_head.parameters())
                + list(end_head.parameters())
                + list(token_head.parameters()),
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            add(total, stats)
            if step % 50 == 0:
                print(
                    f"ep{epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f} "
                    f"code_ce={stats['code_ce']:.3f} acc={stats['code_acc']:.3f} "
                    f"soft_ce={stats['soft_ce']:.3f} p={stats['soft_p']:.3f} "
                    f"tok_ce={stats['token_ce']:.3f} tok_p={stats['token_p']:.3f} "
                    f"evt_cos={stats['event_plan_cos']:.3f} "
                    f"end_mae={stats['end_mae']:.1f} route_used={stats['route_used']:.0f} "
                    f"route_ent={stats['route_entropy']:.2f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(
                    code=f"{stats['code_ce']:.2f}",
                    ce=f"{stats['soft_ce']:.2f}",
                    route=f"{stats['route_used']:.0f}",
                )
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        model.eval()
        valid_head.eval()
        end_head.eval()
        token_head.eval()
        val_total = {}
        last = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"route-code ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                target_sentence_mask = batch.get("target_sentence_mask")
                if target_sentence_mask is not None:
                    target_sentence_mask = target_sentence_mask.to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                code_targets = vq.encode(z_target, suffix_mask)
                pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
                code_logits, aux, prior_hidden = model(z_prompt, pos, suffix_mask, return_aux=True, return_hidden=True)
                _loss, soft = compute_losses(
                    args,
                    decoder,
                    vq,
                    z_prompt,
                    z_target,
                    code_logits,
                    code_targets,
                    suffix_ids,
                    suffix_mask,
                )
                valid_logits = valid_head(prior_hidden).squeeze(-1)
                end_logits = end_head(prior_hidden).squeeze(-1)
                token_logits = token_head(prior_hidden)
                _valid_loss, _end_loss, length_stats = length_losses(valid_logits, end_logits, suffix_mask)
                _token_loss, token_stats = token_ce_loss(token_logits, suffix_ids, suffix_mask)
                _event_loss, event_stats = event_plan_loss(
                    aux["plans"],
                    z_target,
                    target_sentence_mask,
                    suffix_mask,
                )
                _smooth_loss, _entropy_loss, _usage_loss, route_stats = route_losses(aux["route_probs"], suffix_mask)
                hard, z_hard, _ids = hard_stats(
                    args,
                    decoder,
                    vq,
                    z_prompt,
                    z_target,
                    code_logits,
                    code_targets,
                    suffix_ids,
                    suffix_mask,
                )
                sample, z_sample = sampled_stats(args, decoder, vq, z_prompt, z_target, code_logits, suffix_ids, suffix_mask)
                add(val_total, soft | length_stats | token_stats | event_stats | route_stats | hard | sample)
                last = (z_prompt[:16], (z_sample if z_sample is not None else z_hard)[:16], input_ids[:16])
        val_mean = mean(val_total)
        print(f"val ep{epoch + 1} | {json.dumps(val_mean, indent=2)}", flush=True)
        score = val_mean.get("sample_ce", val_mean["hard_ce"]) + val_mean.get("sample_repeat_frac", 0.0)
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "route_code_prior": model.state_dict(),
                    "valid_head": valid_head.state_dict(),
                    "end_head": end_head.state_dict(),
                    "token_head": token_head.state_dict(),
                    "best_score": best_score,
                    "val": val_mean,
                    "latent_dim": latent_dim,
                    "codebook_size": vq.codebook_size,
                    "prompt_len": args.prompt_len,
                    "max_seq_len": args.max_seq_len,
                    "plan_slots": args.plan_slots,
                    "layers": args.layers,
                    "heads": args.heads,
                    "hidden_dim": args.hidden_dim,
                    "mixer_layers": args.mixer_layers,
                    "mixer_scale": args.mixer_scale,
                    "route_scale": args.route_scale,
                    "tau": args.tau,
                    "entropy_weight": args.entropy_weight,
                    "usage_entropy_weight": args.usage_entropy_weight,
                    "valid_weight": args.valid_weight,
                    "end_weight": args.end_weight,
                    "token_ce_weight": args.token_ce_weight,
                    "event_plan_weight": args.event_plan_weight,
                    "route_smooth_weight": args.route_smooth_weight,
                    "route_entropy_weight": args.route_entropy_weight,
                    "route_usage_weight": args.route_usage_weight,
                    "sample_codes": args.sample_codes,
                    "sample_tau": args.sample_tau,
                    "vq_path": args.vq,
                    "type": "route_code_prior",
                    "has_length_heads": True,
                    "has_token_head": True,
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | score={best_score:.4f}", flush=True)
            if last is not None:
                write_examples(args.examples, tokenizer, decoder, *last, args)


if __name__ == "__main__":
    main()
