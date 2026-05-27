"""Train a VQ code prior conditioned on prompt latents plus a rough future plan."""

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
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss
from stage2_riemannian import VQLatentTokenizer, suffix_positions
from train_basin_projector import load_prompt_prior, prompt_prior_latents
from train_code_prior import (
    code_ce_loss,
    code_usage,
    encode_latents,
    load_stage1,
    mean,
    token_collapse_stats,
)


class PlanConditionedCodePrior(nn.Module):
    """Parallel VQ-code predictor conditioned on prompt plus rough suffix plan."""

    def __init__(
        self,
        latent_dim=768,
        codebook_size=2048,
        num_layers=3,
        num_heads=8,
        ffn_dim=768,
        mixer_layers=2,
        mixer_kernel=5,
        mixer_scale=0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.mixer_scale = mixer_scale
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.prompt_type = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.plan_type = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.query_plan_gate = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )
        self.layers = nn.ModuleList([
            rfm._StartTransformerLayer(latent_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.mixers = nn.ModuleList([
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(latent_dim),
                    "conv": nn.Conv1d(
                        latent_dim,
                        latent_dim,
                        kernel_size=mixer_kernel,
                        padding=mixer_kernel // 2,
                        groups=latent_dim,
                    ),
                    "gate": nn.Linear(latent_dim, latent_dim),
                    "out": nn.Linear(latent_dim, latent_dim),
                }
            )
            for _ in range(mixer_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, codebook_size)

    def forward(self, z_prompt, z_plan, pos, mask=None):
        context = torch.cat(
            [
                z_prompt + self.prompt_type.to(z_prompt.dtype),
                z_plan + self.plan_type.to(z_plan.dtype),
            ],
            dim=1,
        )
        x = self.pos_proj(pos.unsqueeze(-1))
        if z_plan.size(1) == x.size(1):
            gate = self.query_plan_gate(torch.cat([x, z_plan], dim=-1))
            x = x + gate * z_plan
        for layer in self.layers:
            x = layer(x, context)
        for mixer in self.mixers:
            h = mixer["norm"](x)
            conv = mixer["conv"](h.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(mixer["gate"](h))
            x = x + self.mixer_scale * mixer["out"](conv * gate)
            if mask is not None:
                x = x * mask.to(x.dtype).unsqueeze(-1)
        logits = self.out_proj(self.out_norm(x))
        if mask is not None:
            logits = logits.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description="Train prompt+rough-plan conditioned VQ code prior")
    parser.add_argument("--stage1", default="stage1_vqbase_rocstories_768_K2048_best.pt")
    parser.add_argument("--vq", default="vq_base_stage_rocstories_768_K2048_best.pt")
    parser.add_argument("--prompt_prior", required=True)
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mixer_layers", type=int, default=2)
    parser.add_argument("--mixer_scale", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--code_ce_weight", type=float, default=1.0)
    parser.add_argument("--decoder_ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.03)
    parser.add_argument("--cos_weight", type=float, default=0.05)
    parser.add_argument("--usage_entropy_weight", type=float, default=0.05)
    parser.add_argument("--adjacent_code_weight", type=float, default=0.05)
    parser.add_argument("--sample_codes", type=int, default=4)
    parser.add_argument("--sample_tau", type=float, default=0.8)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--examples", default="plan_conditioned_code_prior_examples.txt")
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


def resolve_checkpoint_path(path_value, label):
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit(f"{label} checkpoint not found: {path_value}")


def freeze(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def load_vq(path, latent_dim, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    codebook_size = int(ckpt.get("codebook_size", 512))
    vq = VQLatentTokenizer(latent_dim, codebook_size).to(device)
    state = ckpt.get("vq")
    if state is None:
        raise RuntimeError(f"No vq state found in {path}")
    vq.load_state_dict(state)
    freeze(vq)
    return vq, ckpt


def soft_code_embeddings(vq, code_logits, tau):
    probs = torch.softmax(code_logits / max(tau, 1e-5), dim=-1)
    z = probs @ vq.codebook.weight
    return vq.decoder(z), probs


def adjacent_code_loss(probs, mask):
    if probs.size(1) < 2:
        return probs.new_tensor(0.0), 0.0
    adjacent_same = (probs[:, :-1] * probs[:, 1:]).sum(dim=-1)
    pair_mask = mask[:, :-1].bool() & mask[:, 1:].bool()
    soft = adjacent_same[pair_mask].mean() if pair_mask.any() else adjacent_same.mean()
    return soft, soft.detach().item()


def compute_losses(args, decoder, vq, z_prompt, z_plan, z_target, code_logits, code_targets, suffix_ids, suffix_mask):
    code_loss, code_acc = code_ce_loss(code_logits, code_targets, suffix_mask)
    z_soft, probs = soft_code_embeddings(vq, code_logits, args.tau)
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_soft[: args.decode_batch]], dim=1))
    dec_ce, p, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids[: args.decode_batch],
        suffix_mask[: args.decode_batch],
    )
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_soft[valid], z_target.detach()[valid]) if valid.any() else F.mse_loss(z_soft, z_target.detach())
    cos_loss, cos_val = rollout_cosine_alignment_loss(z_soft, z_target, suffix_mask)
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    marginal = probs[valid].mean(dim=0) if valid.any() else probs.reshape(-1, probs.size(-1)).mean(dim=0)
    usage_entropy = -(marginal.clamp_min(1e-8) * marginal.clamp_min(1e-8).log()).sum()
    usage_loss = -usage_entropy
    adj_loss, adj_value = adjacent_code_loss(probs, suffix_mask)
    max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(
        decoder,
        z_prompt[: args.decode_batch],
        z_soft[: args.decode_batch],
        suffix_mask[: args.decode_batch],
        args.prompt_len,
    )
    loss = (
        args.code_ce_weight * code_loss
        + args.decoder_ce_weight * dec_ce
        + args.mse_weight * mse
        + args.cos_weight * cos_loss
        + args.usage_entropy_weight * usage_loss
        + args.adjacent_code_weight * adj_loss
    )
    plan_cos = F.cosine_similarity(z_plan.float(), z_target.float(), dim=-1)
    plan_cos = plan_cos[valid].mean().item() if valid.any() else plan_cos.mean().item()
    return loss, {
        "code_ce": code_loss.detach().item(),
        "code_acc": code_acc,
        "soft_ce": dec_ce.detach().item(),
        "soft_p": p,
        "soft_top1": top1,
        "mse": mse.detach().item(),
        "cos": cos_val,
        "code_entropy": entropy[valid].mean().detach().item() if valid.any() else entropy.mean().detach().item(),
        "usage_entropy": usage_entropy.detach().item(),
        "adj_code": adj_value,
        "plan_cos": plan_cos,
        "max_token_frac": max_frac,
        "unique_token_ratio": unique_tok,
        "repeat_frac": rep_frac,
        "punct_frac": punct_frac,
    }


@torch.no_grad()
def decode_code_stats(args, decoder, vq, z_prompt, z_target, ids, suffix_ids, suffix_mask, prefix):
    z_hard = vq.decode_codes(ids, suffix_mask)
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_hard[: args.decode_batch]], dim=1))
    ce, p, top1 = rollout_flow_token_ce_loss(logits, suffix_ids[: args.decode_batch], suffix_mask[: args.decode_batch])
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_hard[valid], z_target[valid]) if valid.any() else F.mse_loss(z_hard, z_target)
    _cos_loss, cos_val = rollout_cosine_alignment_loss(z_hard, z_target, suffix_mask)
    ppl, unique_code_ratio, top1_code_frac, used_codes = code_usage(ids, suffix_mask, vq.codebook_size)
    max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(
        decoder, z_prompt[: args.decode_batch], z_hard[: args.decode_batch], suffix_mask[: args.decode_batch], args.prompt_len
    )
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
    }, z_hard, ids


@torch.no_grad()
def hard_stats(args, decoder, vq, z_prompt, z_target, code_logits, code_targets, suffix_ids, suffix_mask):
    hard_ids = code_logits.argmax(dim=-1)
    hard, z_hard, ids = decode_code_stats(
        args, decoder, vq, z_prompt, z_target, hard_ids, suffix_ids, suffix_mask, "hard"
    )
    target_usage = code_usage(code_targets, suffix_mask, vq.codebook_size)
    hard.update(
        {
            "target_code_ppl": target_usage[0],
            "target_unique_code_ratio": target_usage[1],
            "target_top1_code_frac": target_usage[2],
            "target_used_codes": target_usage[3],
        }
    )
    return hard, z_hard, ids


@torch.no_grad()
def sampled_stats(args, decoder, vq, z_prompt, z_target, code_logits, suffix_ids, suffix_mask):
    if args.sample_codes <= 0:
        return {}, None
    probs = torch.softmax(code_logits / max(args.sample_tau, 1e-5), dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    stats_total = {}
    last_z = None
    for _idx in range(args.sample_codes):
        sampled = torch.multinomial(flat, 1).reshape(code_logits.shape[:2])
        stats, z_sample, _ids = decode_code_stats(
            args, decoder, vq, z_prompt, z_target, sampled, suffix_ids, suffix_mask, "sample"
        )
        last_z = z_sample
        for key, value in stats.items():
            stats_total[key] = stats_total.get(key, 0.0) + float(value)
    return {key: value / args.sample_codes for key, value in stats_total.items()}, last_z


def add(total, stats):
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + float(value)
    total["n"] = total.get("n", 0) + 1


@torch.no_grad()
def write_examples(path, tokenizer, decoder, z_prompt, z_hard, input_ids, args):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_hard], dim=1))
    pred = logits.argmax(dim=-1)
    rows = []
    for i in range(min(16, input_ids.size(0))):
        prompt = tokenizer.decode(input_ids[i, : args.prompt_len], skip_special_tokens=True).strip()
        target = tokenizer.decode(input_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        out = tokenizer.decode(pred[i, args.prompt_len :], skip_special_tokens=True).strip()
        rows.append(f"--- example {i + 1}\nprompt: {prompt}\ntarget: {target}\nplan-code prior: {out}\n")
    Path(path).write_text("\n".join(rows), encoding="utf-8")


def main():
    args = parse_args()
    seed_everything(SEED)
    configure_data(args)
    args.stage1 = resolve_checkpoint_path(args.stage1, "Stage1")
    args.vq = resolve_checkpoint_path(args.vq, "VQ")
    args.prompt_prior = resolve_checkpoint_path(args.prompt_prior, "PromptPrior")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)
    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    vq, _vq_ckpt = load_vq(args.vq, latent_dim, device)
    prompt_prior, prompt_ckpt = load_prompt_prior(args.prompt_prior, latent_dim, device)
    prior = PlanConditionedCodePrior(
        latent_dim=latent_dim,
        codebook_size=vq.codebook_size,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.hidden_dim,
        mixer_layers=args.mixer_layers,
        mixer_scale=args.mixer_scale,
    ).to(device)
    if args.output is None:
        args.output = f"plan_code_prior_vqbase_rocstories_{latent_dim}_K{vq.codebook_size}_best.pt"
    train_loader, val_loader = s2data.build_stage2_dataloaders(tokenizer, args.train_size, args.batch_size, args.max_seq_len)
    optimizer = AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_ce = float("inf")
    print(
        f"PlanCodePrior params={sum(p.numel() for p in prior.parameters()):,} K={vq.codebook_size} "
        f"prompt_prior_type={prompt_ckpt.get('type')}",
        flush=True,
    )

    for epoch in range(args.epochs):
        prior.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"plan-code ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad():
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                z_plan = prompt_prior_latents(prompt_prior, z_prompt, args.suffix_len, suffix_mask, z_target)
                z_plan = z_plan.to(device=z_target.device, dtype=z_target.dtype)
                code_targets = vq.encode(z_target, suffix_mask)
                pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                code_logits = prior(z_prompt, z_plan, pos, suffix_mask)
                loss, stats = compute_losses(
                    args, decoder, vq, z_prompt, z_plan, z_target, code_logits, code_targets, suffix_ids, suffix_mask
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            add(total, stats)
            if step % 50 == 0:
                print(
                    f"ep{epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f} "
                    f"code_ce={stats['code_ce']:.3f} acc={stats['code_acc']:.3f} "
                    f"soft_ce={stats['soft_ce']:.3f} p={stats['soft_p']:.3f} "
                    f"rep={stats['repeat_frac']:.3f} plan_cos={stats['plan_cos']:.3f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(
                    code=f"{stats['code_ce']:.2f}",
                    ce=f"{stats['soft_ce']:.2f}",
                    p=f"{stats['soft_p']:.3f}",
                    rep=f"{stats['repeat_frac']:.2f}",
                )
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        prior.eval()
        val_total = {}
        last = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"plan-code ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                z_plan = prompt_prior_latents(prompt_prior, z_prompt, args.suffix_len, suffix_mask, z_target)
                z_plan = z_plan.to(device=z_target.device, dtype=z_target.dtype)
                code_targets = vq.encode(z_target, suffix_mask)
                pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
                code_logits = prior(z_prompt, z_plan, pos, suffix_mask)
                _loss, soft = compute_losses(
                    args, decoder, vq, z_prompt, z_plan, z_target, code_logits, code_targets, suffix_ids, suffix_mask
                )
                hard, z_hard, _ids = hard_stats(
                    args, decoder, vq, z_prompt, z_target, code_logits, code_targets, suffix_ids, suffix_mask
                )
                sample, z_sample = sampled_stats(args, decoder, vq, z_prompt, z_target, code_logits, suffix_ids, suffix_mask)
                add(val_total, soft | hard | sample)
                last = (z_prompt[:16], (z_sample if z_sample is not None else z_hard)[:16], input_ids[:16])
        val_mean = mean(val_total)
        print(f"val ep{epoch + 1} | {json.dumps(val_mean, indent=2)}", flush=True)
        if val_mean["hard_ce"] < best_ce:
            best_ce = val_mean["hard_ce"]
            torch.save(
                {
                    "plan_code_prior": prior.state_dict(),
                    "best_hard_ce": best_ce,
                    "val": val_mean,
                    "latent_dim": latent_dim,
                    "codebook_size": vq.codebook_size,
                    "prompt_len": args.prompt_len,
                    "max_seq_len": args.max_seq_len,
                    "layers": args.layers,
                    "heads": args.heads,
                    "hidden_dim": args.hidden_dim,
                    "mixer_layers": args.mixer_layers,
                    "mixer_scale": args.mixer_scale,
                    "tau": args.tau,
                    "usage_entropy_weight": args.usage_entropy_weight,
                    "adjacent_code_weight": args.adjacent_code_weight,
                    "sample_codes": args.sample_codes,
                    "sample_tau": args.sample_tau,
                    "vq_path": args.vq,
                    "stage1_path": args.stage1,
                    "prompt_prior_path": args.prompt_prior,
                    "prompt_prior_type": prompt_ckpt.get("type"),
                    "type": "plan_conditioned_code_prior",
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | hard_ce={best_ce:.4f}", flush=True)
            if last is not None:
                write_examples(args.examples, tokenizer, decoder, *last, args)


if __name__ == "__main__":
    main()
