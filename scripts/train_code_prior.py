"""Train a prompt-conditioned parallel discrete latent code prior."""

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
from stage2_riemannian import CodePrior, VQLatentTokenizer, suffix_positions


def parse_args():
    parser = argparse.ArgumentParser(description="Train prompt-to-VQ-code prior")
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
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mixer_layers", type=int, default=2)
    parser.add_argument("--mixer_scale", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--code_ce_weight", type=float, default=1.0)
    parser.add_argument("--decoder_ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.03)
    parser.add_argument("--cos_weight", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument("--usage_entropy_weight", type=float, default=0.0)
    parser.add_argument("--sample_codes", type=int, default=1)
    parser.add_argument("--sample_tau", type=float, default=1.0)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--examples", default="code_prior_examples.txt")
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


@torch.no_grad()
def encode_latents(encoder, decoder, input_ids, attention_mask):
    return decoder.compress(encoder(input_ids, attention_mask))


def code_ce_loss(logits, target_ids, mask):
    valid = mask.bool()
    if not valid.any():
        return logits.new_tensor(0.0), 0.0
    loss = F.cross_entropy(logits[valid], target_ids[valid], reduction="mean")
    acc = logits[valid].argmax(dim=-1).eq(target_ids[valid]).float().mean().item()
    return loss, acc


def soft_code_embeddings(vq, code_logits, tau):
    probs = torch.softmax(code_logits / max(tau, 1e-5), dim=-1)
    z = probs @ vq.codebook.weight
    return vq.decoder(z), probs


def code_usage(ids, mask, k):
    valid = ids[mask.bool()]
    if valid.numel() == 0:
        return 0.0, 0.0, 0.0, 0
    counts = torch.bincount(valid.reshape(-1), minlength=k).float()
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    ppl = float(entropy.exp().item())
    used = int((counts > 0).sum().item())
    unique_ratio = float(used / max(k, 1))
    top1_frac = float(counts.max().item() / counts.sum().clamp_min(1.0).item())
    return ppl, unique_ratio, top1_frac, used


def token_collapse_stats(decoder, z_prompt, z_suffix, mask, prompt_len):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))[:, prompt_len:]
    pred = logits.argmax(dim=-1)
    valid = mask.bool()
    if not valid.any():
        return 0.0, 0.0, 0.0
    max_fracs = []
    unique_fracs = []
    repeat_fracs = []
    punct_ids = {999, 1012, 1029, 999, 1025, 1024, 1010}
    punct = 0
    total = 0
    for row, row_mask in zip(pred, valid):
        ids = row[row_mask]
        if ids.numel() == 0:
            continue
        total += ids.numel()
        punct += sum(1 for tok in ids.tolist() if int(tok) in punct_ids)
        counts = torch.bincount(ids)
        max_fracs.append(float(counts.max().item() / ids.numel()))
        unique_fracs.append(float((counts > 0).sum().item() / ids.numel()))
        if ids.numel() > 1:
            repeat_fracs.append(float(ids[1:].eq(ids[:-1]).float().mean().item()))
    denom = max(len(max_fracs), 1)
    return (
        sum(max_fracs) / denom,
        sum(unique_fracs) / denom,
        sum(repeat_fracs) / max(len(repeat_fracs), 1),
        punct / max(total, 1),
    )


def compute_losses(args, decoder, vq, z_prompt, z_target, code_logits, code_targets, suffix_ids, suffix_mask):
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
    ent_reg = -entropy[valid].mean() if valid.any() else -entropy.mean()
    if valid.any():
        marginal = probs[valid].mean(dim=0)
    else:
        marginal = probs.reshape(-1, probs.size(-1)).mean(dim=0)
    usage_entropy = -(marginal.clamp_min(1e-8) * marginal.clamp_min(1e-8).log()).sum()
    usage_ent_reg = -usage_entropy
    loss = (
        args.code_ce_weight * code_loss
        + args.decoder_ce_weight * dec_ce
        + args.mse_weight * mse
        + args.cos_weight * cos_loss
        + args.entropy_weight * ent_reg
        + args.usage_entropy_weight * usage_ent_reg
    )
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
    }


@torch.no_grad()
def decode_code_stats(args, decoder, vq, z_prompt, z_target, ids, suffix_ids, suffix_mask, prefix):
    z_hard = vq.decode_codes(ids, suffix_mask)
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_hard[: args.decode_batch]], dim=1))
    ce, p, top1 = rollout_flow_token_ce_loss(logits, suffix_ids[: args.decode_batch], suffix_mask[: args.decode_batch])
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_hard[valid], z_target[valid]) if valid.any() else F.mse_loss(z_hard, z_target)
    cos_loss, cos_val = rollout_cosine_alignment_loss(z_hard, z_target, suffix_mask)
    ppl, unique_code_ratio, top1_code_frac, used_codes = code_usage(ids, suffix_mask, vq.codebook_size)
    max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(decoder, z_prompt[: args.decode_batch], z_hard[: args.decode_batch], suffix_mask[: args.decode_batch], args.prompt_len)
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


def mean(total):
    n = max(total.get("n", 0), 1)
    return {key: value / n for key, value in total.items() if key != "n"}


@torch.no_grad()
def write_examples(path, tokenizer, decoder, z_prompt, z_hard, input_ids, args):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_hard], dim=1))
    pred = logits.argmax(dim=-1)
    rows = []
    for i in range(min(16, input_ids.size(0))):
        prompt = tokenizer.decode(input_ids[i, : args.prompt_len], skip_special_tokens=True).strip()
        target = tokenizer.decode(input_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        out = tokenizer.decode(pred[i, args.prompt_len :], skip_special_tokens=True).strip()
        rows.append(f"--- example {i + 1}\nprompt: {prompt}\ntarget: {target}\ncode prior: {out}\n")
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
    vq, vq_ckpt = load_vq(args.vq, latent_dim, device)
    if args.output is None:
        args.output = f"code_prior_rocstories_{latent_dim}_K{vq.codebook_size}_best.pt"
    prior = CodePrior(
        latent_dim=latent_dim,
        codebook_size=vq.codebook_size,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.hidden_dim,
        mixer_layers=args.mixer_layers,
        mixer_scale=args.mixer_scale,
    ).to(device)
    train_loader, val_loader = s2data.build_stage2_dataloaders(tokenizer, args.train_size, args.batch_size, args.max_seq_len)
    optimizer = AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_ce = float("inf")
    print(f"CodePrior params={sum(p.numel() for p in prior.parameters()):,} K={vq.codebook_size}", flush=True)

    for epoch in range(args.epochs):
        prior.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"code ep{epoch + 1}/{args.epochs} train")
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
                code_logits = prior(z_prompt, pos, suffix_mask)
                loss, stats = compute_losses(args, decoder, vq, z_prompt, z_target, code_logits, code_targets, suffix_ids, suffix_mask)
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
                    f"soft_ce={stats['soft_ce']:.3f} p={stats['soft_p']:.3f} ent={stats['code_entropy']:.2f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(code=f"{stats['code_ce']:.2f}", ce=f"{stats['soft_ce']:.2f}", acc=f"{stats['code_acc']:.2f}")
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        prior.eval()
        val_total = {}
        last = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"code ep{epoch + 1}/{args.epochs} val")
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
                code_logits = prior(z_prompt, pos, suffix_mask)
                _loss, soft = compute_losses(args, decoder, vq, z_prompt, z_target, code_logits, code_targets, suffix_ids, suffix_mask)
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
                    "code_prior": prior.state_dict(),
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
                    "entropy_weight": args.entropy_weight,
                    "usage_entropy_weight": args.usage_entropy_weight,
                    "sample_codes": args.sample_codes,
                    "sample_tau": args.sample_tau,
                    "vq_path": args.vq,
                    "type": "code_prior",
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | hard_ce={best_ce:.4f}", flush=True)
            if last is not None:
                write_examples(args.examples, tokenizer, decoder, *last, args)


if __name__ == "__main__":
    main()
