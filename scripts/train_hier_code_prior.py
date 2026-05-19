"""Train a prompt -> plan slots -> VQ token-code prior."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
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
from stage2_riemannian import HierCodePrior, suffix_positions
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train hierarchical prompt-to-VQ-code prior")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--vq", required=True)
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--plan_slots", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mixer_layers", type=int, default=2)
    parser.add_argument("--mixer_scale", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.2)
    parser.add_argument("--code_ce_weight", type=float, default=1.0)
    parser.add_argument("--decoder_ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.03)
    parser.add_argument("--cos_weight", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.02)
    parser.add_argument("--usage_entropy_weight", type=float, default=0.05)
    parser.add_argument("--sample_codes", type=int, default=4)
    parser.add_argument("--sample_tau", type=float, default=0.9)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--examples", default="hier_code_prior_examples.txt")
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
        args.output = f"hier_code_prior_rocstories_{latent_dim}_K{vq.codebook_size}_plan{args.plan_slots}_best.pt"

    model = HierCodePrior(
        latent_dim=latent_dim,
        codebook_size=vq.codebook_size,
        plan_slots=args.plan_slots,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.hidden_dim,
        mixer_layers=args.mixer_layers,
        mixer_scale=args.mixer_scale,
    ).to(device)
    train_loader, val_loader = s2data.build_stage2_dataloaders(
        tokenizer,
        args.train_size,
        args.batch_size,
        args.max_seq_len,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_ce = float("inf")
    print(
        f"HierCodePrior params={sum(p.numel() for p in model.parameters()):,} "
        f"K={vq.codebook_size} plan_slots={args.plan_slots}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"hier-code ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad():
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                code_targets = vq.encode(z_target, suffix_mask)
            suffix_ids = input_ids[:, args.prompt_len :]
            pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                code_logits = model(z_prompt, pos, suffix_mask)
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
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            add(total, stats)
            if step % 50 == 0:
                print(
                    f"ep{epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f} "
                    f"code_ce={stats['code_ce']:.3f} acc={stats['code_acc']:.3f} "
                    f"soft_ce={stats['soft_ce']:.3f} p={stats['soft_p']:.3f} "
                    f"ent={stats['code_entropy']:.2f} usage_ent={stats['usage_entropy']:.2f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(
                    code=f"{stats['code_ce']:.2f}",
                    ce=f"{stats['soft_ce']:.2f}",
                    acc=f"{stats['code_acc']:.2f}",
                )
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        model.eval()
        val_total = {}
        last = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"hier-code ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                code_targets = vq.encode(z_target, suffix_mask)
                pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
                code_logits = model(z_prompt, pos, suffix_mask)
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
                add(val_total, soft | hard | sample)
                last = (z_prompt[:16], (z_sample if z_sample is not None else z_hard)[:16], input_ids[:16])
        val_mean = mean(val_total)
        print(f"val ep{epoch + 1} | {json.dumps(val_mean, indent=2)}", flush=True)
        if val_mean["hard_ce"] < best_ce:
            best_ce = val_mean["hard_ce"]
            torch.save(
                {
                    "hier_code_prior": model.state_dict(),
                    "best_hard_ce": best_ce,
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
                    "tau": args.tau,
                    "entropy_weight": args.entropy_weight,
                    "usage_entropy_weight": args.usage_entropy_weight,
                    "sample_codes": args.sample_codes,
                    "sample_tau": args.sample_tau,
                    "vq_path": args.vq,
                    "type": "hier_code_prior",
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | hard_ce={best_ce:.4f}", flush=True)
            if last is not None:
                write_examples(args.examples, tokenizer, decoder, *last, args)


if __name__ == "__main__":
    main()
