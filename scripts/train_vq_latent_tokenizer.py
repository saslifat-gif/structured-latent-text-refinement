"""Train a decoder-readable VQ tokenizer over frozen Stage1 suffix latents."""

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ latent tokenizer on frozen Stage1 suffix latents")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.05)
    parser.add_argument("--cos_weight", type=float, default=0.10)
    parser.add_argument("--commit_weight", type=float, default=1.0)
    parser.add_argument("--norm_weight", type=float, default=0.01)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--examples", default="vq_latent_tokenizer_examples.txt")
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


def load_stage1(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    latent_dim = int(ckpt.get("latent_dim", 256))
    encoder = BertEncoder().to(device)
    decoder = ParallelDecoder(latent_dim=latent_dim).to(device)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    freeze(encoder)
    freeze(decoder)
    return encoder, decoder, latent_dim


@torch.no_grad()
def encode_latents(encoder, decoder, input_ids, attention_mask):
    return decoder.compress(encoder(input_ids, attention_mask))


def norm_gap(z_q, z_target, mask):
    valid = mask.bool()
    if valid.any():
        return F.smooth_l1_loss(z_q.norm(dim=-1)[valid], z_target.detach().norm(dim=-1)[valid])
    return F.smooth_l1_loss(z_q.norm(dim=-1), z_target.detach().norm(dim=-1))


def compute_stats(args, decoder, z_prompt, z_q, z_target, suffix_ids, suffix_mask, vq_loss):
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_q[: args.decode_batch]], dim=1))
    ce, p, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids[: args.decode_batch],
        suffix_mask[: args.decode_batch],
    )
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_q[valid], z_target.detach()[valid]) if valid.any() else F.mse_loss(z_q, z_target.detach())
    cos_loss, cos_val = rollout_cosine_alignment_loss(z_q, z_target, suffix_mask)
    nloss = norm_gap(z_q, z_target, suffix_mask)
    loss = (
        args.ce_weight * ce
        + args.mse_weight * mse
        + args.cos_weight * cos_loss
        + args.commit_weight * vq_loss
        + args.norm_weight * nloss
    )
    return loss, {
        "ce": ce.detach().item(),
        "p": p,
        "top1": top1,
        "mse": mse.detach().item(),
        "cos": cos_val,
        "vq": vq_loss.detach().item(),
        "norm": nloss.detach().item(),
    }


def add(total, stats):
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + value
    total["n"] = total.get("n", 0) + 1


def mean(total):
    n = max(total.get("n", 0), 1)
    return {key: value / n for key, value in total.items() if key != "n"}


@torch.no_grad()
def write_examples(path, tokenizer, decoder, z_prompt, z_q, input_ids, args):
    texts = []
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_q], dim=1))
    pred_ids = logits.argmax(dim=-1)
    n = min(16, input_ids.size(0))
    for i in range(n):
        prompt = tokenizer.decode(input_ids[i, : args.prompt_len], skip_special_tokens=True).strip()
        target = tokenizer.decode(input_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        pred = tokenizer.decode(pred_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        texts.append(f"--- example {i + 1}\nprompt: {prompt}\ntarget: {target}\nvq recon: {pred}\n")
    Path(path).write_text("\n".join(texts), encoding="utf-8")


def main():
    args = parse_args()
    if args.max_seq_len != args.prompt_len + args.suffix_len:
        args.max_seq_len = args.prompt_len + args.suffix_len
    if args.output is None:
        args.output = f"vq_latent_tokenizer_rocstories_{args.latent_dim}_K{args.codebook_size}_best.pt"
    seed_everything(SEED)
    configure_data(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    vq = VQLatentTokenizer(latent_dim, args.codebook_size, args.commitment).to(device)
    train_loader, val_loader = s2data.build_stage2_dataloaders(tokenizer, args.train_size, args.batch_size, args.max_seq_len)
    optimizer = AdamW(vq.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_ce = float("inf")
    print(f"VQ params={sum(p.numel() for p in vq.parameters()):,} K={args.codebook_size}", flush=True)

    for epoch in range(args.epochs):
        vq.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"vq ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                z_q, code_ids, vq_loss, _parts = vq(z_target, suffix_mask)
                loss, stats = compute_stats(args, decoder, z_prompt, z_q, z_target, suffix_ids, suffix_mask, vq_loss)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            add(total, stats)
            if step % 50 == 0:
                ppl, dead, used = vq.usage_stats(code_ids.detach(), suffix_mask)
                print(
                    f"ep{epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f} "
                    f"ce={stats['ce']:.3f} p={stats['p']:.3f} top1={stats['top1']:.3f} "
                    f"mse={stats['mse']:.4f} cos={stats['cos']:.3f} vq={stats['vq']:.4f} "
                    f"ppl={ppl:.1f} used={used} dead={dead:.1f}%",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(ce=f"{stats['ce']:.3f}", p=f"{stats['p']:.3f}", vq=f"{stats['vq']:.3f}")
        train_mean = mean(total)
        print(f"ep{epoch + 1} train | {json.dumps(train_mean, indent=2)}", flush=True)

        vq.eval()
        val_total = {}
        usage_counts = torch.zeros(args.codebook_size, device=device)
        last_batch = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"vq ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                z_q, code_ids, vq_loss, _parts = vq(z_target, suffix_mask)
                _loss, stats = compute_stats(args, decoder, z_prompt, z_q, z_target, suffix_ids, suffix_mask, vq_loss)
                add(val_total, stats)
                ids = code_ids[suffix_mask.bool()]
                if ids.numel():
                    usage_counts += torch.bincount(ids, minlength=args.codebook_size).to(usage_counts)
                last_batch = (z_prompt[:16], z_q[:16], input_ids[:16])
        val_mean = mean(val_total)
        probs = usage_counts / usage_counts.sum().clamp_min(1.0)
        usage_ppl = float((-(probs[probs > 0] * probs[probs > 0].log()).sum()).exp().item())
        used = int((usage_counts > 0).sum().item())
        dead = float(100.0 * (args.codebook_size - used) / args.codebook_size)
        print(
            f"val ep{epoch + 1} | ce={val_mean['ce']:.3f} p={val_mean['p']:.3f} top1={val_mean['top1']:.3f} "
            f"mse={val_mean['mse']:.4f} cos={val_mean['cos']:.3f} vq={val_mean['vq']:.4f} "
            f"code_ppl={usage_ppl:.1f} used={used} dead={dead:.1f}%",
            flush=True,
        )
        if val_mean["ce"] < best_ce:
            best_ce = val_mean["ce"]
            torch.save(
                {
                    "vq": vq.state_dict(),
                    "best_val_ce": best_ce,
                    "val": val_mean,
                    "code_perplexity": usage_ppl,
                    "used_codes": used,
                    "dead_code_percent": dead,
                    "latent_dim": latent_dim,
                    "codebook_size": args.codebook_size,
                    "prompt_len": args.prompt_len,
                    "max_seq_len": args.max_seq_len,
                    "type": "vq_latent_tokenizer",
                    "stage1": args.stage1,
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | val_ce={best_ce:.4f}", flush=True)
            if last_batch is not None:
                write_examples(args.examples, tokenizer, decoder, *last_batch, args)


if __name__ == "__main__":
    main()
