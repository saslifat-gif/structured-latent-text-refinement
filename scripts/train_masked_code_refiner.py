"""Train a masked iterative VQ-code refiner for parallel code generation."""

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
from stage2_riemannian import MaskedCodeRefiner, suffix_positions
from train_code_prior import (
    code_ce_loss,
    compute_losses,
    decode_code_stats,
    encode_latents,
    load_stage1,
    load_vq,
    mean,
    sampled_stats,
    write_examples,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train masked iterative VQ-code refiner")
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
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mixer_layers", type=int, default=2)
    parser.add_argument("--mixer_scale", type=float, default=0.5)
    parser.add_argument("--min_mask_prob", type=float, default=0.25)
    parser.add_argument("--mask_prob", type=float, default=0.35)
    parser.add_argument("--random_replace_prob", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--code_ce_weight", type=float, default=1.0)
    parser.add_argument("--decoder_ce_weight", type=float, default=0.5)
    parser.add_argument("--mse_weight", type=float, default=0.02)
    parser.add_argument("--cos_weight", type=float, default=0.03)
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument("--usage_entropy_weight", type=float, default=0.02)
    parser.add_argument("--sample_codes", type=int, default=4)
    parser.add_argument("--sample_tau", type=float, default=0.9)
    parser.add_argument("--iter_steps", type=int, default=4)
    parser.add_argument("--remask_frac", type=float, default=0.35)
    parser.add_argument("--example_max_tokens", type=int, default=48)
    parser.add_argument("--example_token_temp", type=float, default=0.8)
    parser.add_argument("--example_top_k", type=int, default=30)
    parser.add_argument("--example_top_p", type=float, default=0.9)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--examples", default="masked_code_refiner_examples.txt")
    parser.add_argument("--latest_examples", default="masked_code_refiner_latest_examples.txt")
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


def top_p_filter(probs, top_p):
    if top_p <= 0 or top_p >= 1:
        return probs
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(dim=-1, index=sorted_idx, src=sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def sample_token_ids(logits, temp, top_k, top_p):
    probs = torch.softmax(logits.float() / max(temp, 1e-5), dim=-1)
    if top_k > 0 and top_k < probs.size(-1):
        values, idx = probs.topk(top_k, dim=-1)
        kept = torch.zeros_like(probs).scatter(dim=-1, index=idx, src=values)
        probs = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    probs = top_p_filter(probs, top_p)
    return torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(logits.shape[:-1])


@torch.no_grad()
def write_masked_examples(path, tokenizer, decoder, z_prompt, z_iter, input_ids, args):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_iter], dim=1))[:, args.prompt_len :]
    cutoff = min(args.example_max_tokens, logits.size(1))
    logits = logits[:, :cutoff]
    pred = sample_token_ids(logits, args.example_token_temp, args.example_top_k, args.example_top_p)
    rows = []
    for i in range(min(16, input_ids.size(0))):
        prompt = tokenizer.decode(input_ids[i, : args.prompt_len], skip_special_tokens=True).strip()
        target = tokenizer.decode(input_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        out = tokenizer.decode(pred[i], skip_special_tokens=True).strip()
        rows.append(
            f"--- example {i + 1}\n"
            f"prompt: {prompt}\n"
            f"target: {target}\n"
            f"masked iter sample: {out}\n"
        )
    Path(path).write_text("\n".join(rows), encoding="utf-8")


def corrupt_codes(code_targets, suffix_mask, codebook_size, mask_prob, random_replace_prob):
    valid = suffix_mask.bool()
    noise = torch.rand(code_targets.shape, device=code_targets.device)
    masked = (noise < mask_prob) & valid
    no_mask_rows = valid.any(dim=1) & ~masked.any(dim=1)
    if no_mask_rows.any():
        first_valid = valid[no_mask_rows].float().argmax(dim=1)
        masked[no_mask_rows.nonzero(as_tuple=False).squeeze(1), first_valid] = True
    known_mask = valid & ~masked
    corrupted = code_targets.clone()
    replace = (torch.rand(code_targets.shape, device=code_targets.device) < random_replace_prob) & known_mask
    if replace.any():
        corrupted[replace] = torch.randint(0, codebook_size, (int(replace.sum().item()),), device=code_targets.device)
    corrupted = corrupted.masked_fill(~valid, 0)
    return corrupted, known_mask, masked


def masked_code_loss(logits, targets, masked, suffix_mask):
    valid = masked.bool() & suffix_mask.bool()
    if not valid.any():
        valid = suffix_mask.bool()
    return code_ce_loss(logits, targets, valid)


@torch.no_grad()
def iterative_ids(args, model, vq, z_prompt, suffix_mask, sample_tau):
    batch, length = suffix_mask.shape
    device = z_prompt.device
    ids = torch.zeros((batch, length), dtype=torch.long, device=device)
    known = torch.zeros((batch, length), dtype=torch.bool, device=device)
    pos = suffix_positions(batch, length, device, z_prompt.dtype)
    logits = None
    steps = max(1, int(args.iter_steps))
    for step in range(steps):
        code_emb = vq.decode_codes(ids, suffix_mask)
        logits = model(z_prompt, code_emb, known, pos, suffix_mask)
        probs = torch.softmax(logits / max(sample_tau, 1e-5), dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(batch, length)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        ids = torch.where(suffix_mask.bool(), sampled, ids)
        if step < steps - 1:
            keep = suffix_mask.bool()
            valid_counts = keep.sum(dim=1)
            next_known = torch.zeros_like(known)
            for row in range(batch):
                count = int(valid_counts[row].item())
                if count <= 0:
                    continue
                keep_count = max(1, int(round(count * (1.0 - args.remask_frac))))
                keep_count = min(count, keep_count)
                row_conf = conf[row].masked_fill(~keep[row], -1.0)
                idx = row_conf.topk(keep_count).indices
                next_known[row, idx] = True
            known = next_known & suffix_mask.bool()
        else:
            known = suffix_mask.bool()
    return ids, logits


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
        args.output = f"masked_code_refiner_rocstories_{latent_dim}_K{vq.codebook_size}_best.pt"
    model = MaskedCodeRefiner(
        latent_dim=latent_dim,
        codebook_size=vq.codebook_size,
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
    best_score = float("inf")
    print(
        f"MaskedCodeRefiner params={sum(p.numel() for p in model.parameters()):,} "
        f"K={vq.codebook_size} mask_prob={args.mask_prob}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"masked-code ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad():
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                code_targets = vq.encode(z_target, suffix_mask)
                effective_mask_prob = random.uniform(
                    min(args.min_mask_prob, args.mask_prob),
                    max(args.min_mask_prob, args.mask_prob),
                )
                corrupted, known_mask, masked = corrupt_codes(
                    code_targets,
                    suffix_mask,
                    vq.codebook_size,
                    effective_mask_prob,
                    args.random_replace_prob,
                )
                code_emb = vq.decode_codes(corrupted, suffix_mask)
            pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                code_logits = model(z_prompt, code_emb, known_mask, pos, suffix_mask)
                recon_loss, recon_acc = masked_code_loss(code_logits, code_targets, masked, suffix_mask)
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
                loss = loss + args.code_ce_weight * recon_loss
                stats["masked_code_ce"] = recon_loss.detach().item()
                stats["masked_code_acc"] = recon_acc
                stats["mask_frac"] = (masked & suffix_mask.bool()).float().sum().item() / suffix_mask.bool().float().sum().clamp_min(1).item()
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
                    f"mask_ce={stats['masked_code_ce']:.3f} mask_acc={stats['masked_code_acc']:.3f} "
                    f"soft_ce={stats['soft_ce']:.3f} p={stats['soft_p']:.3f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(
                    mask=f"{stats['masked_code_ce']:.2f}",
                    ce=f"{stats['soft_ce']:.2f}",
                    acc=f"{stats['masked_code_acc']:.2f}",
                )
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        model.eval()
        val_total = {}
        last = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"masked-code ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                code_targets = vq.encode(z_target, suffix_mask)
                ids, code_logits = iterative_ids(args, model, vq, z_prompt, suffix_mask, args.sample_tau)
                iter_stats, z_iter, _ids = decode_code_stats(
                    args,
                    decoder,
                    vq,
                    z_prompt,
                    z_target,
                    ids,
                    suffix_ids,
                    suffix_mask,
                    "iter",
                )
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
                sample, _z_sample = sampled_stats(args, decoder, vq, z_prompt, z_target, code_logits, suffix_ids, suffix_mask)
                add(val_total, soft | iter_stats | sample)
                last = (z_prompt[:16], z_iter[:16], input_ids[:16])
        val_mean = mean(val_total)
        print(f"val ep{epoch + 1} | {json.dumps(val_mean, indent=2)}", flush=True)
        if last is not None and args.latest_examples:
            write_masked_examples(args.latest_examples, tokenizer, decoder, *last, args)
        score = val_mean.get("iter_ce", val_mean.get("sample_ce", 999.0)) + val_mean.get("iter_repeat_frac", 0.0)
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "masked_code_refiner": model.state_dict(),
                    "best_score": best_score,
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
                    "min_mask_prob": args.min_mask_prob,
                    "mask_prob": args.mask_prob,
                    "random_replace_prob": args.random_replace_prob,
                    "iter_steps": args.iter_steps,
                    "remask_frac": args.remask_frac,
                    "example_max_tokens": args.example_max_tokens,
                    "example_token_temp": args.example_token_temp,
                    "example_top_k": args.example_top_k,
                    "example_top_p": args.example_top_p,
                    "sample_codes": args.sample_codes,
                    "sample_tau": args.sample_tau,
                    "vq_path": args.vq,
                    "type": "masked_code_refiner",
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | score={best_score:.4f}", flush=True)
            if last is not None:
                write_masked_examples(args.examples, tokenizer, decoder, *last, args)


if __name__ == "__main__":
    main()
