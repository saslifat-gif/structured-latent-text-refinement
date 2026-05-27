"""Train Stage1 decoder robustness on real, noisy, and PromptPrior latents."""

from __future__ import annotations

import argparse
import copy
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
from stage2_config import SEED, TARGET_LATENT_STD
from stage2_losses import rollout_flow_token_ce_loss
from train_basin_projector import load_prompt_prior, prompt_prior_latents
from train_code_prior import encode_latents, load_stage1, mean, token_collapse_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Adapt decoder basin to PromptPrior-generated latents")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--prompt_prior", required=True)
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bert_lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--real_frac", type=float, default=0.50)
    parser.add_argument("--noise_frac", type=float, default=0.25)
    parser.add_argument("--pred_frac", type=float, default=0.25)
    parser.add_argument("--loss_mode", choices=("all", "mix"), default="all")
    parser.add_argument("--real_ce_weight", type=float, default=0.25)
    parser.add_argument("--noise_ce_weight", type=float, default=0.25)
    parser.add_argument("--pred_ce_weight", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0.05, help="Noise scale multiplied by TARGET_LATENT_STD.")
    parser.add_argument("--preserve_kl_weight", type=float, default=0.05)
    parser.add_argument("--kl_temp", type=float, default=1.0)
    parser.add_argument("--bert_last_layers", type=int, default=0)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint_path(path_value, label):
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit(f"{label} checkpoint not found: {path_value}")


def configure_data(args):
    if args.max_seq_len != args.prompt_len + args.suffix_len:
        args.max_seq_len = args.prompt_len + args.suffix_len
    s2data.PROMPT_LEN = args.prompt_len
    s2data.DATASET_NAME = args.dataset
    s2data.ROCSTORIES_LOCAL_FILES_ONLY = args.local_files_only
    rfm.PROMPT_LEN = args.prompt_len
    rfm.MAX_SEQ_LEN = args.max_seq_len
    s2losses.PROMPT_LEN = args.prompt_len


def configure_decoder_trainable(decoder, bert_last_layers):
    for param in decoder.parameters():
        param.requires_grad_(False)
    train_groups = []
    for module in (decoder.project_up, decoder.to_logits):
        for param in module.parameters():
            param.requires_grad_(True)
        train_groups.append({"params": list(module.parameters())})
    bert_params = []
    if bert_last_layers > 0:
        layers = list(decoder.bert.encoder.layer)[-int(bert_last_layers) :]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad_(True)
                bert_params.append(param)
    return train_groups, bert_params


def mixture_latents(args, z_target, z_pred, suffix_mask):
    bsz = z_target.size(0)
    probs = torch.tensor(
        [args.real_frac, args.noise_frac, args.pred_frac],
        device=z_target.device,
        dtype=torch.float32,
    )
    probs = probs / probs.sum().clamp_min(1e-8)
    choice = torch.multinomial(probs, bsz, replacement=True)
    z_pred = z_pred.detach().to(device=z_target.device, dtype=z_target.dtype)
    z_noisy = z_target + torch.randn_like(z_target) * (args.noise_std * TARGET_LATENT_STD)
    z_noisy = z_noisy.to(dtype=z_target.dtype)
    z_mix = z_target.clone()
    z_mix[choice == 1] = z_noisy[choice == 1]
    z_mix[choice == 2] = z_pred[choice == 2]
    if suffix_mask is not None:
        z_mix = z_mix * suffix_mask.to(z_mix.dtype).unsqueeze(-1)
    counts = {
        "real_frac_seen": choice.eq(0).float().mean().item(),
        "noise_frac_seen": choice.eq(1).float().mean().item(),
        "pred_frac_seen": choice.eq(2).float().mean().item(),
    }
    return z_mix, counts


def noisy_latents(args, z_target, suffix_mask):
    z_noisy = z_target + torch.randn_like(z_target) * (args.noise_std * TARGET_LATENT_STD)
    z_noisy = z_noisy.to(dtype=z_target.dtype)
    if suffix_mask is not None:
        z_noisy = z_noisy * suffix_mask.to(z_noisy.dtype).unsqueeze(-1)
    return z_noisy


def suffix_ce(logits, suffix_ids, suffix_mask, prompt_len):
    suffix_logits = logits[:, prompt_len:]
    valid = suffix_mask.bool()
    if not valid.any():
        return logits.new_tensor(0.0)
    return F.cross_entropy(suffix_logits[valid], suffix_ids[valid], reduction="mean")


def preserve_kl(student_logits, teacher_logits, suffix_mask, prompt_len, temp):
    if temp <= 0:
        temp = 1.0
    s = student_logits[:, prompt_len:].float() / temp
    t = teacher_logits[:, prompt_len:].float() / temp
    token_kl = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="none").sum(dim=-1)
    valid = suffix_mask.bool()
    if valid.any():
        return token_kl[valid].mean() * (temp * temp)
    return token_kl.mean() * (temp * temp)


def compute_batch(args, decoder, teacher_decoder, prompt_prior, batch, device, train):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    with torch.no_grad():
        z = encode_latents(teacher_decoder._robust_encoder, teacher_decoder, input_ids, attention_mask)
        z_prompt = z[:, : args.prompt_len]
        z_target = z[:, args.prompt_len :]
        suffix_ids = input_ids[:, args.prompt_len :]
        suffix_mask = attention_mask[:, args.prompt_len :]
        z_pred = prompt_prior_latents(prompt_prior, z_prompt, args.suffix_len, suffix_mask, z_target)
        z_pred = z_pred.detach().to(device=z_target.device, dtype=z_target.dtype)
        z_noisy = noisy_latents(args, z_target, suffix_mask)
        if train and args.loss_mode == "mix":
            z_mix, mix_stats = mixture_latents(args, z_target, z_pred, suffix_mask)
        else:
            mix_stats = {
                "real_frac_seen": 0.0,
                "noise_frac_seen": 0.0,
                "pred_frac_seen": 1.0,
            }
    if train and args.loss_mode == "mix":
        z_suffix = z_mix
        z_seq = torch.cat([z_prompt, z_suffix], dim=1)
        logits = decoder.decode_from_latent(z_seq)
        ce = suffix_ce(logits, suffix_ids, suffix_mask, args.prompt_len)
        real_ce = logits.new_tensor(0.0)
        noise_ce = logits.new_tensor(0.0)
        pred_ce = ce
    elif train:
        real_logits = decoder.decode_from_latent(torch.cat([z_prompt, z_target], dim=1))
        noise_logits = decoder.decode_from_latent(torch.cat([z_prompt, z_noisy], dim=1))
        pred_logits = decoder.decode_from_latent(torch.cat([z_prompt, z_pred], dim=1))
        real_ce = suffix_ce(real_logits, suffix_ids, suffix_mask, args.prompt_len)
        noise_ce = suffix_ce(noise_logits, suffix_ids, suffix_mask, args.prompt_len)
        pred_ce = suffix_ce(pred_logits, suffix_ids, suffix_mask, args.prompt_len)
        ce = pred_ce
        logits = pred_logits
        z_suffix = z_pred
    else:
        z_suffix = z_pred
        logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))
        ce = suffix_ce(logits, suffix_ids, suffix_mask, args.prompt_len)
        real_ce = logits.new_tensor(0.0)
        noise_ce = logits.new_tensor(0.0)
        pred_ce = ce
    kl = logits.new_tensor(0.0)
    if train and args.preserve_kl_weight > 0:
        with torch.no_grad():
            teacher_logits = teacher_decoder.decode_from_latent(torch.cat([z_prompt, z_target], dim=1))
        if args.loss_mode == "all":
            kl = preserve_kl(real_logits, teacher_logits, suffix_mask, args.prompt_len, args.kl_temp)
        else:
            kl = preserve_kl(logits, teacher_logits, suffix_mask, args.prompt_len, args.kl_temp)
    if train and args.loss_mode == "all":
        loss = (
            args.real_ce_weight * real_ce
            + args.noise_ce_weight * noise_ce
            + args.pred_ce_weight * pred_ce
            + args.preserve_kl_weight * kl
        )
    else:
        loss = ce + args.preserve_kl_weight * kl
    with torch.no_grad():
        eval_logits = logits[: args.decode_batch]
        eval_ids = suffix_ids[: args.decode_batch]
        eval_mask = suffix_mask[: args.decode_batch]
        _flow_ce, p, top1 = rollout_flow_token_ce_loss(eval_logits, eval_ids, eval_mask)
        max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(
            decoder,
            z_prompt[: args.decode_batch],
            z_suffix[: args.decode_batch],
            suffix_mask[: args.decode_batch],
            args.prompt_len,
        )
    stats = {
        "loss": loss.detach().item(),
        "ce": ce.detach().item(),
        "real_ce": real_ce.detach().item(),
        "noise_ce": noise_ce.detach().item(),
        "pred_ce": pred_ce.detach().item(),
        "preserve_kl": kl.detach().item(),
        "target_prob": p,
        "top1": top1,
        "max_token_frac": max_frac,
        "unique_token_ratio": unique_tok,
        "repeat_frac": rep_frac,
        "punct_frac": punct_frac,
        **mix_stats,
    }
    return loss, stats


def add(total, stats):
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + float(value)
    total["n"] = total.get("n", 0) + 1


def evaluate(args, decoder, teacher_decoder, prompt_prior, val_loader, device):
    decoder.eval()
    total = {}
    with torch.no_grad():
        iterator = val_loader
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(val_loader), desc="decoder-robust val")
        for batch in iterator:
            _loss, stats = compute_batch(args, decoder, teacher_decoder, prompt_prior, batch, device, train=False)
            add(total, {f"pred_{key}": value for key, value in stats.items()})
    return mean(total)


def main():
    args = parse_args()
    seed_everything(SEED)
    configure_data(args)
    args.stage1 = resolve_checkpoint_path(args.stage1, "Stage1")
    args.prompt_prior = resolve_checkpoint_path(args.prompt_prior, "PromptPrior")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    teacher_decoder = copy.deepcopy(decoder).to(device)
    teacher_decoder._robust_encoder = encoder
    for param in teacher_decoder.parameters():
        param.requires_grad_(False)
    prompt_prior, prompt_ckpt = load_prompt_prior(args.prompt_prior, latent_dim, device)
    train_groups, bert_params = configure_decoder_trainable(decoder, args.bert_last_layers)
    param_groups = [
        {"params": [p for group in train_groups for p in group["params"]], "lr": args.lr},
    ]
    if bert_params:
        param_groups.append({"params": bert_params, "lr": args.bert_lr})
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if args.output is None:
        stem = Path(args.prompt_prior).stem
        args.output = f"{stem}_decoder_robust_best.pt"

    train_loader, val_loader = s2data.build_stage2_dataloaders(
        tokenizer,
        args.train_size,
        args.batch_size,
        args.max_seq_len,
    )
    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(
        f"decoder robustness trainable={trainable:,} prompt_prior_type={prompt_ckpt.get('type')} "
        f"loss_mode={args.loss_mode} mix=real{args.real_frac:g}/noise{args.noise_frac:g}/pred{args.pred_frac:g} "
        f"ce_weights=real{args.real_ce_weight:g}/noise{args.noise_ce_weight:g}/pred{args.pred_ce_weight:g}",
        flush=True,
    )

    best_score = float("inf")
    for epoch in range(args.epochs):
        decoder.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"decoder-robust ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, stats = compute_batch(args, decoder, teacher_decoder, prompt_prior, batch, device, train=True)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in decoder.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            add(total, stats)
            if step % 50 == 0:
                print(
                    f"ep{epoch + 1} step {step}/{len(train_loader)} loss={stats['loss']:.4f} "
                    f"pred_ce={stats['pred_ce']:.3f} p={stats['target_prob']:.3f} rep={stats['repeat_frac']:.3f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(ce=f"{stats['pred_ce']:.2f}", p=f"{stats['target_prob']:.3f}", rep=f"{stats['repeat_frac']:.2f}")
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)
        val_stats = evaluate(args, decoder, teacher_decoder, prompt_prior, val_loader, device)
        print(f"val ep{epoch + 1} | {json.dumps(val_stats, indent=2)}", flush=True)
        score = val_stats["pred_ce"] + val_stats["pred_repeat_frac"]
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "best_score": best_score,
                    "val": val_stats,
                    "latent_dim": latent_dim,
                    "prompt_len": args.prompt_len,
                    "suffix_len": args.suffix_len,
                    "max_seq_len": args.max_seq_len,
                    "stage1_path": args.stage1,
                    "prompt_prior_path": args.prompt_prior,
                    "prompt_prior_type": prompt_ckpt.get("type"),
                    "real_frac": args.real_frac,
                    "noise_frac": args.noise_frac,
                    "pred_frac": args.pred_frac,
                    "loss_mode": args.loss_mode,
                    "real_ce_weight": args.real_ce_weight,
                    "noise_ce_weight": args.noise_ce_weight,
                    "pred_ce_weight": args.pred_ce_weight,
                    "noise_std": args.noise_std,
                    "preserve_kl_weight": args.preserve_kl_weight,
                    "bert_last_layers": args.bert_last_layers,
                    "type": "stage1_decoder_robustness",
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | score={best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
