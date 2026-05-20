"""Train a Riemannian refiner on sampled CodePrior/VQ suffix latents."""

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
from stage2_config import FLOW_REFINE_SCALE, SEED
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss
from stage2_riemannian import CodePrior, FlowNet, HierCodePrior, MetricNet, RouteCodePrior, suffix_positions
from train_code_prior import (
    code_usage,
    encode_latents,
    load_stage1,
    load_vq,
    mean,
    token_collapse_stats,
)


class ValidMaskHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train sampled CodePrior + Riemannian MetricRefiner")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--vq", required=True)
    parser.add_argument("--code_prior", required=True)
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--sample_tau", type=float, default=0.9)
    parser.add_argument("--sample_codes", type=int, default=4)
    parser.add_argument("--rollout_steps", type=int, default=4)
    parser.add_argument("--decode_batch", type=int, default=32)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--hidden_weight", type=float, default=0.05)
    parser.add_argument("--mse_weight", type=float, default=0.05)
    parser.add_argument("--cos_weight", type=float, default=0.05)
    parser.add_argument("--delta_weight", type=float, default=0.01)
    parser.add_argument("--mask_weight", type=float, default=0.1)
    parser.add_argument("--repeat_weight", type=float, default=0.02)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
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


def freeze(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def resolve_checkpoint_path(path_value, label):
    placeholders = {
        "STAGE1_CHECKPOINT.pt",
        "VQ_CHECKPOINT.pt",
        "CODE_PRIOR_CHECKPOINT.pt",
        "HIER_CODE_PRIOR_CHECKPOINT.pt",
    }
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    hint = ""
    if path.name in placeholders:
        hint = f" The value `{path_value}` is a placeholder; replace it with the real {label} checkpoint path."
    raise SystemExit(
        f"{label} checkpoint not found: {path_value}.{hint} "
        "Run `find /workspace/structured-latent-text-refinement -name '*.pt'` "
        "or pass the full checkpoint path."
    )


def load_code_prior(path, latent_dim, codebook_size, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    prior_type = ckpt.get("type", "code_prior")
    common = {
        "latent_dim": latent_dim,
        "codebook_size": codebook_size,
        "num_layers": int(ckpt.get("layers", 2)),
        "num_heads": int(ckpt.get("heads", 8)),
        "ffn_dim": int(ckpt.get("hidden_dim", 512)),
        "mixer_layers": int(ckpt.get("mixer_layers", 2)),
        "mixer_scale": float(ckpt.get("mixer_scale", 0.5)),
    }
    if prior_type == "route_code_prior" or "route_code_prior" in ckpt:
        prior = RouteCodePrior(
            plan_slots=int(ckpt.get("plan_slots", 4)),
            route_scale=float(ckpt.get("route_scale", 1.0)),
            **common,
        ).to(device)
        state = ckpt.get("route_code_prior")
    elif prior_type == "hier_code_prior" or "hier_code_prior" in ckpt:
        prior = HierCodePrior(plan_slots=int(ckpt.get("plan_slots", 8)), **common).to(device)
        state = ckpt.get("hier_code_prior")
    else:
        prior = CodePrior(**common).to(device)
        state = ckpt.get("code_prior")
    if state is None:
        raise RuntimeError(f"No CodePrior state found in {path}")
    prior.load_state_dict(state)
    freeze(prior)
    return prior, ckpt


@torch.no_grad()
def sample_vq_latents(prior, vq, z_prompt, suffix_mask, sample_tau):
    pos = suffix_positions(z_prompt.size(0), suffix_mask.size(1), z_prompt.device, z_prompt.dtype)
    logits = prior(z_prompt, pos, suffix_mask)
    probs = torch.softmax(logits / max(sample_tau, 1e-5), dim=-1)
    ids = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(logits.shape[:2])
    z0 = vq.decode_codes(ids, suffix_mask)
    return z0, ids, logits


def masked_natural_velocity(flow_net, metric_net, z, t, z_prompt, pos, mask):
    force = flow_net(z, t, z_prompt, pos, mask)
    pooled = z_prompt.mean(dim=1).unsqueeze(1).expand_as(z)
    g = metric_net(
        z.reshape(-1, z.size(-1)),
        t.reshape(-1),
        pooled.reshape(-1, z.size(-1)),
        pos.reshape(-1),
    ).reshape_as(z)
    velocity = rfm.clamp_velocity(force / g.clamp_min(1e-3))
    if mask is not None:
        velocity = velocity * mask.to(velocity.dtype).unsqueeze(-1)
    return velocity, g


def refine_latents(flow_net, metric_net, z0, z_prompt, suffix_mask, steps):
    z = z0
    pos = suffix_positions(z.size(0), z.size(1), z.device, z.dtype)
    steps = max(1, int(steps))
    for idx in range(steps):
        t = torch.full((z.size(0), z.size(1)), idx / steps, device=z.device, dtype=z.dtype)
        velocity, _g = masked_natural_velocity(flow_net, metric_net, z, t, z_prompt, pos, suffix_mask)
        z = z + FLOW_REFINE_SCALE * velocity / steps
        if suffix_mask is not None:
            z = z * suffix_mask.to(z.dtype).unsqueeze(-1)
    return z


def adjacent_repeat_penalty(logits, suffix_ids, mask):
    probs = logits.float().softmax(dim=-1)
    suffix_probs = probs[:, rfm.PROMPT_LEN :, :]
    if suffix_probs.size(1) < 2:
        return logits.new_tensor(0.0), 0.0
    same_prob = (suffix_probs[:, 1:] * suffix_probs[:, :-1]).sum(dim=-1)
    valid = (suffix_ids[:, 1:] != 0) & (suffix_ids[:, :-1] != 0)
    if mask is not None:
        valid = valid & mask[:, 1:].bool() & mask[:, :-1].bool()
    if not valid.any():
        return logits.new_tensor(0.0), 0.0
    value = same_prob[valid].mean()
    return value, value.detach().item()


def refiner_losses(args, decoder, flow_net, metric_net, mask_head, z_prompt, z0, z_target, suffix_ids, suffix_mask):
    z_refined = refine_latents(flow_net, metric_net, z0, z_prompt, suffix_mask, args.rollout_steps)
    n_decode = min(args.decode_batch, z_refined.size(0))
    pred_seq = torch.cat([z_prompt[:n_decode], z_refined[:n_decode]], dim=1)
    target_seq = torch.cat([z_prompt[:n_decode], z_target[:n_decode]], dim=1)
    logits, pred_hidden = decoder.decode_from_latent(pred_seq, return_hidden=True)
    with torch.no_grad():
        _target_logits, target_hidden = decoder.decode_from_latent(target_seq, return_hidden=True)
    ce, p, top1 = rollout_flow_token_ce_loss(logits, suffix_ids[:n_decode], suffix_mask[:n_decode])
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_refined[valid], z_target.detach()[valid]) if valid.any() else F.mse_loss(z_refined, z_target.detach())
    cos_loss, cos_val = rollout_cosine_alignment_loss(z_refined, z_target, suffix_mask)
    delta = z_refined - z0.detach()
    delta_reg = delta[valid].pow(2).mean() if valid.any() else delta.pow(2).mean()
    hidden_valid = suffix_mask[:n_decode].bool()
    pred_suffix_hidden = pred_hidden[:, args.prompt_len :, :]
    target_suffix_hidden = target_hidden[:, args.prompt_len :, :]
    hidden_loss = (
        F.smooth_l1_loss(pred_suffix_hidden[hidden_valid].float(), target_suffix_hidden.detach()[hidden_valid].float())
        if hidden_valid.any()
        else pred_hidden.new_tensor(0.0)
    )
    mask_logits = mask_head(z_refined)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, suffix_mask.float())
    repeat_loss, repeat_prob = adjacent_repeat_penalty(logits, suffix_ids[:n_decode], suffix_mask[:n_decode])
    loss = (
        args.ce_weight * ce
        + args.hidden_weight * hidden_loss
        + args.mse_weight * mse
        + args.cos_weight * cos_loss
        + args.delta_weight * delta_reg
        + args.mask_weight * mask_loss
        + args.repeat_weight * repeat_loss
    )
    return loss, z_refined.detach(), {
        "refined_ce": ce.detach().item(),
        "refined_p": p,
        "refined_top1": top1,
        "hidden_loss": hidden_loss.detach().item(),
        "mse": mse.detach().item(),
        "cos": cos_val,
        "delta_reg": delta_reg.detach().item(),
        "mask_loss": mask_loss.detach().item(),
        "repeat_prob": repeat_prob,
    }


@torch.no_grad()
def eval_decode_stats(args, decoder, vq, z_prompt, z_target, z_suffix, ids, suffix_ids, suffix_mask, prefix):
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_suffix[: args.decode_batch]], dim=1))
    ce, p, top1 = rollout_flow_token_ce_loss(logits, suffix_ids[: args.decode_batch], suffix_mask[: args.decode_batch])
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_suffix[valid], z_target[valid]) if valid.any() else F.mse_loss(z_suffix, z_target)
    _cos_loss, cos_val = rollout_cosine_alignment_loss(z_suffix, z_target, suffix_mask)
    max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(
        decoder,
        z_prompt[: args.decode_batch],
        z_suffix[: args.decode_batch],
        suffix_mask[: args.decode_batch],
        args.prompt_len,
    )
    ppl, unique_code_ratio, top1_code_frac, used_codes = code_usage(ids, suffix_mask, vq.codebook_size)
    return {
        f"{prefix}_ce": ce.detach().item(),
        f"{prefix}_p": p,
        f"{prefix}_top1": top1,
        f"{prefix}_mse": mse.detach().item(),
        f"{prefix}_cos": cos_val,
        f"{prefix}_code_ppl": ppl,
        f"{prefix}_unique_code_ratio": unique_code_ratio,
        f"{prefix}_top1_code_frac": top1_code_frac,
        f"{prefix}_used_codes": used_codes,
        f"{prefix}_max_token_frac": max_frac,
        f"{prefix}_unique_token_ratio": unique_tok,
        f"{prefix}_repeat_frac": rep_frac,
        f"{prefix}_punct_frac": punct_frac,
    }


def add(total, stats):
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + float(value)
    total["n"] = total.get("n", 0) + 1


def main():
    args = parse_args()
    if args.max_seq_len != args.prompt_len + args.suffix_len:
        args.max_seq_len = args.prompt_len + args.suffix_len
    args.stage1 = resolve_checkpoint_path(args.stage1, "Stage1")
    args.vq = resolve_checkpoint_path(args.vq, "VQ")
    args.code_prior = resolve_checkpoint_path(args.code_prior, "CodePrior")
    seed_everything(SEED)
    configure_data(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    vq, _vq_ckpt = load_vq(args.vq, latent_dim, device)
    prior, prior_ckpt = load_code_prior(args.code_prior, latent_dim, vq.codebook_size, device)
    flow_net = FlowNet(latent_dim=latent_dim).to(device)
    metric_net = MetricNet(latent_dim=latent_dim).to(device)
    mask_head = ValidMaskHead(latent_dim).to(device)
    if args.output is None:
        stem = Path(args.code_prior).stem
        args.output = f"{stem}_metric_refiner_best.pt"

    train_loader, val_loader = s2data.build_stage2_dataloaders(
        tokenizer,
        args.train_size,
        args.batch_size,
        args.max_seq_len,
    )
    optimizer = AdamW(
        list(flow_net.parameters()) + list(metric_net.parameters()) + list(mask_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = float("inf")
    print(
        f"MetricRefiner params={sum(p.numel() for p in flow_net.parameters()) + sum(p.numel() for p in metric_net.parameters()) + sum(p.numel() for p in mask_head.parameters()):,} "
        f"prior_type={prior_ckpt.get('type', 'code_prior')} K={vq.codebook_size}",
        flush=True,
    )

    for epoch in range(args.epochs):
        flow_net.train()
        metric_net.train()
        mask_head.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"metric-refiner ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad():
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                z0, _ids, _logits = sample_vq_latents(prior, vq, z_prompt, suffix_mask, args.sample_tau)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, _z_refined, stats = refiner_losses(
                    args,
                    decoder,
                    flow_net,
                    metric_net,
                    mask_head,
                    z_prompt,
                    z0,
                    z_target,
                    suffix_ids,
                    suffix_mask,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(flow_net.parameters()) + list(metric_net.parameters()) + list(mask_head.parameters()),
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            add(total, stats)
            if step % 50 == 0:
                print(
                    f"ep{epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f} "
                    f"ce={stats['refined_ce']:.3f} p={stats['refined_p']:.3f} "
                    f"cos={stats['cos']:.3f} rep={stats['repeat_prob']:.3f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(ce=f"{stats['refined_ce']:.2f}", p=f"{stats['refined_p']:.3f}", cos=f"{stats['cos']:.2f}")
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        flow_net.eval()
        metric_net.eval()
        mask_head.eval()
        val_total = {}
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"metric-refiner ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                sample_total = {}
                for _sample_idx in range(max(1, args.sample_codes)):
                    z0, ids, _logits = sample_vq_latents(prior, vq, z_prompt, suffix_mask, args.sample_tau)
                    z_refined = refine_latents(flow_net, metric_net, z0, z_prompt, suffix_mask, args.rollout_steps)
                    base = eval_decode_stats(args, decoder, vq, z_prompt, z_target, z0, ids, suffix_ids, suffix_mask, "sample")
                    refined = eval_decode_stats(
                        args,
                        decoder,
                        vq,
                        z_prompt,
                        z_target,
                        z_refined,
                        ids,
                        suffix_ids,
                        suffix_mask,
                        "refined",
                    )
                    mask_logits = mask_head(z_refined)
                    mask_acc = (mask_logits.sigmoid().ge(0.5) == suffix_mask.bool()).float().mean().item()
                    add(sample_total, base | refined | {"mask_acc": mask_acc})
                add(val_total, mean(sample_total))
        val_mean = mean(val_total)
        print(f"val ep{epoch + 1} | {json.dumps(val_mean, indent=2)}", flush=True)
        score = val_mean["refined_ce"] + val_mean["refined_repeat_frac"]
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "flow_net": flow_net.state_dict(),
                    "metric_net": metric_net.state_dict(),
                    "mask_head": mask_head.state_dict(),
                    "best_score": best_score,
                    "val": val_mean,
                    "latent_dim": latent_dim,
                    "codebook_size": vq.codebook_size,
                    "prompt_len": args.prompt_len,
                    "max_seq_len": args.max_seq_len,
                    "suffix_len": args.suffix_len,
                    "sample_tau": args.sample_tau,
                    "rollout_steps": args.rollout_steps,
                    "vq_path": args.vq,
                    "code_prior_path": args.code_prior,
                    "type": "codeprior_metric_refiner",
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | score={best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
