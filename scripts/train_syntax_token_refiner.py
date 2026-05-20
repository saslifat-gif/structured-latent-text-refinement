"""Train a parallel token-space syntax refiner from rough drafts."""

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
from stage2_riemannian import SyntaxTokenRefiner, VQDecoderAdapter, suffix_positions
from train_code_prior import encode_latents, load_stage1, load_vq, mean
from train_codeprior_metric_refiner import load_code_prior, resolve_checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train rough-draft -> syntax-refined token logits")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--vq", required=True)
    parser.add_argument("--code_prior", default="")
    parser.add_argument("--decoder_adapter", default="")
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mixer_layers", type=int, default=2)
    parser.add_argument("--mixer_scale", type=float, default=0.5)
    parser.add_argument("--draft_mode", choices=("route", "vq", "corrupt"), default="route")
    parser.add_argument("--sample_tau", type=float, default=0.8)
    parser.add_argument("--corrupt_prob", type=float, default=0.35)
    parser.add_argument("--random_replace_prob", type=float, default=0.10)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--draft_keep_weight", type=float, default=0.05)
    parser.add_argument("--repeat_weight", type=float, default=0.05)
    parser.add_argument("--decode", choices=("argmax", "sample"), default="sample")
    parser.add_argument("--token_temp", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--example_max_tokens", type=int, default=48)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--examples", default="syntax_token_refiner_examples.txt")
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


def load_decoder_adapter(path, latent_dim, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    adapter = VQDecoderAdapter(
        latent_dim=latent_dim,
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
        layers=int(ckpt.get("layers", 2)),
        delta_scale=float(ckpt.get("delta_scale", 0.5)),
    ).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)
    return adapter, ckpt


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


def sample_tokens(logits, temp=0.8, top_k=30, top_p=0.9):
    probs = torch.softmax(logits.float() / max(temp, 1e-5), dim=-1)
    if top_k > 0 and top_k < probs.size(-1):
        values, idx = probs.topk(top_k, dim=-1)
        kept = torch.zeros_like(probs).scatter(dim=-1, index=idx, src=values)
        probs = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    probs = top_p_filter(probs, top_p)
    return torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(logits.shape[:-1])


def token_stats(logits, targets, mask, prefix):
    valid = mask.bool()
    if not valid.any():
        return logits.new_tensor(0.0), {f"{prefix}_ce": 0.0, f"{prefix}_p": 0.0, f"{prefix}_top1": 0.0}
    flat_logits = logits[valid]
    flat_targets = targets[valid]
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="mean")
    probs = torch.softmax(flat_logits.detach(), dim=-1)
    target_p = probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1).mean().item()
    top1 = probs.argmax(dim=-1).eq(flat_targets).float().mean().item()
    return ce, {f"{prefix}_ce": ce.detach().item(), f"{prefix}_p": target_p, f"{prefix}_top1": top1}


def repeat_loss(logits, mask):
    if logits.size(1) < 2:
        return logits.new_tensor(0.0), 0.0
    probs = torch.softmax(logits.float(), dim=-1)
    pair_valid = mask[:, 1:].bool() & mask[:, :-1].bool()
    same = (probs[:, 1:] * probs[:, :-1]).sum(dim=-1)
    loss = same[pair_valid].mean() if pair_valid.any() else same.mean()
    ids = logits.argmax(dim=-1)
    rep = ids[:, 1:].eq(ids[:, :-1]).float()
    rep_val = rep[pair_valid].mean().item() if pair_valid.any() else 0.0
    return loss, rep_val


def token_collapse_from_ids(ids, mask):
    valid = mask.bool()
    unique_fracs = []
    max_fracs = []
    repeat_fracs = []
    for row, row_mask in zip(ids, valid):
        toks = row[row_mask]
        if toks.numel() == 0:
            continue
        counts = torch.bincount(toks)
        max_fracs.append(float(counts.max().item() / toks.numel()))
        unique_fracs.append(float((counts > 0).sum().item() / toks.numel()))
        if toks.numel() > 1:
            repeat_fracs.append(float(toks[1:].eq(toks[:-1]).float().mean().item()))
    return {
        "max_token_frac": sum(max_fracs) / max(len(max_fracs), 1),
        "unique_token_ratio": sum(unique_fracs) / max(len(unique_fracs), 1),
        "repeat_frac": sum(repeat_fracs) / max(len(repeat_fracs), 1),
    }


@torch.no_grad()
def decode_suffix_logits_in_chunks(args, decoder, z_prompt, z_suffix):
    chunks = []
    for start in range(0, z_suffix.size(0), args.decode_batch):
        end = min(start + args.decode_batch, z_suffix.size(0))
        logits = decoder.decode_from_latent(torch.cat([z_prompt[start:end], z_suffix[start:end]], dim=1))
        chunks.append(logits[:, args.prompt_len :, :])
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def make_draft(args, tokenizer, decoder, vq, prior, adapter, z_prompt, z_target, suffix_ids, suffix_mask):
    if args.draft_mode == "route":
        if prior is None:
            raise RuntimeError("--draft_mode route requires --code_prior")
        pos = suffix_positions(z_prompt.size(0), suffix_mask.size(1), z_prompt.device, z_prompt.dtype)
        code_logits = prior(z_prompt, pos, suffix_mask)
        probs = torch.softmax(code_logits / max(args.sample_tau, 1e-5), dim=-1)
        code_ids = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(code_logits.shape[:2])
        z_draft = vq.decode_codes(code_ids, suffix_mask)
    elif args.draft_mode == "vq":
        z_draft, _code_ids, _vq_loss, _parts = vq(z_target, suffix_mask)
    else:
        keep = torch.rand_like(suffix_ids.float()) > args.corrupt_prob
        rand = torch.randint(0, tokenizer.vocab_size, suffix_ids.shape, device=suffix_ids.device)
        replace = torch.rand_like(suffix_ids.float()) < args.random_replace_prob
        draft_ids = torch.where(keep, suffix_ids, torch.full_like(suffix_ids, tokenizer.mask_token_id))
        draft_ids = torch.where(replace & suffix_mask.bool(), rand, draft_ids)
        # The corrupted-token mode is only a lightweight fallback; use zero latents for unknown draft tokens.
        z_draft = torch.zeros_like(z_target)
        conf = keep.to(z_target.dtype) * suffix_mask.to(z_target.dtype)
        return draft_ids, z_draft, conf
    if adapter is not None:
        pos = suffix_positions(z_draft.size(0), z_draft.size(1), z_draft.device, z_draft.dtype)
        z_draft = adapter(z_draft, z_prompt, pos, suffix_mask)
    suffix_logits = decode_suffix_logits_in_chunks(args, decoder, z_prompt, z_draft)
    if args.decode == "sample":
        draft_ids = sample_tokens(suffix_logits, args.token_temp, args.top_k, args.top_p)
    else:
        draft_ids = suffix_logits.argmax(dim=-1)
    probs = torch.softmax(suffix_logits.float(), dim=-1).max(dim=-1).values
    return draft_ids, z_draft, probs.to(z_target.dtype)


def compute_loss(args, model, z_prompt, draft_ids, z_draft, draft_conf, suffix_ids, suffix_mask):
    pos = suffix_positions(z_draft.size(0), z_draft.size(1), z_draft.device, z_draft.dtype)
    logits = model(z_prompt, draft_ids, z_draft, pos, suffix_mask, draft_conf)
    ce, stats = token_stats(logits, suffix_ids, suffix_mask, "refined")
    keep_loss = F.cross_entropy(logits[suffix_mask.bool()], draft_ids[suffix_mask.bool()], reduction="mean")
    rep_loss, rep = repeat_loss(logits, suffix_mask)
    loss = args.ce_weight * ce + args.draft_keep_weight * keep_loss + args.repeat_weight * rep_loss
    pred = logits.argmax(dim=-1)
    valid = suffix_mask.bool()
    draft_acc = draft_ids[valid].eq(suffix_ids[valid]).float().mean().item() if valid.any() else 0.0
    draft_rep = (
        draft_ids[:, 1:].eq(draft_ids[:, :-1]).float()[suffix_mask[:, 1:].bool() & suffix_mask[:, :-1].bool()].mean().item()
        if logits.size(1) > 1 and (suffix_mask[:, 1:].bool() & suffix_mask[:, :-1].bool()).any()
        else 0.0
    )
    stats.update({
        "draft_acc": draft_acc,
        "draft_repeat_frac": draft_rep,
        "draft_keep_ce": keep_loss.detach().item(),
        "repeat_loss": rep_loss.detach().item(),
        "refined_repeat_frac": rep,
    })
    stats.update({f"refined_{k}": v for k, v in token_collapse_from_ids(pred, suffix_mask).items()})
    return loss, stats, logits


@torch.no_grad()
def write_examples(path, tokenizer, input_ids, draft_ids, refined_logits, args):
    if args.decode == "sample":
        refined_ids = sample_tokens(
            refined_logits[:, : args.example_max_tokens],
            args.token_temp,
            args.top_k,
            args.top_p,
        )
    else:
        refined_ids = refined_logits[:, : args.example_max_tokens].argmax(dim=-1)
    rows = []
    for i in range(min(12, input_ids.size(0))):
        prompt = tokenizer.decode(input_ids[i, : args.prompt_len], skip_special_tokens=True).strip()
        target = tokenizer.decode(input_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        draft = tokenizer.decode(draft_ids[i, : args.example_max_tokens], skip_special_tokens=True).strip()
        refined = tokenizer.decode(refined_ids[i], skip_special_tokens=True).strip()
        rows.append(
            f"--- example {i + 1}\n"
            f"prompt: {prompt}\n"
            f"target: {target}\n"
            f"draft: {draft}\n"
            f"syntax refined: {refined}\n"
        )
    Path(path).write_text("\n".join(rows), encoding="utf-8")


def main():
    args = parse_args()
    if args.max_seq_len != args.prompt_len + args.suffix_len:
        args.max_seq_len = args.prompt_len + args.suffix_len
    args.stage1 = resolve_checkpoint_path(args.stage1, "Stage1")
    args.vq = resolve_checkpoint_path(args.vq, "VQ")
    if args.code_prior:
        args.code_prior = resolve_checkpoint_path(args.code_prior, "CodePrior")
    if args.decoder_adapter:
        args.decoder_adapter = resolve_checkpoint_path(args.decoder_adapter, "DecoderAdapter")
    seed_everything(SEED)
    configure_data(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    vq, _vq_ckpt = load_vq(args.vq, latent_dim, device)
    prior = None
    if args.code_prior:
        prior, _prior_ckpt = load_code_prior(args.code_prior, latent_dim, vq.codebook_size, device)
    adapter = None
    if args.decoder_adapter:
        adapter, _adapter_ckpt = load_decoder_adapter(args.decoder_adapter, latent_dim, device)
    if args.output is None:
        suffix = "route" if args.code_prior else args.draft_mode
        args.output = f"syntax_token_refiner_rocstories_{latent_dim}_K{vq.codebook_size}_{suffix}_best.pt"

    model = SyntaxTokenRefiner(
        vocab_size=tokenizer.vocab_size,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        mixer_layers=args.mixer_layers,
        mixer_scale=args.mixer_scale,
        pad_token_id=tokenizer.pad_token_id or 0,
    ).to(device)
    train_loader, val_loader = s2data.build_stage2_dataloaders(tokenizer, args.train_size, args.batch_size, args.max_seq_len)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_ce = float("inf")
    print(f"SyntaxTokenRefiner params={sum(p.numel() for p in model.parameters()):,} draft_mode={args.draft_mode}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"syntax-refiner ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad():
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                draft_ids, z_draft, draft_conf = make_draft(
                    args, tokenizer, decoder, vq, prior, adapter, z_prompt, z_target, suffix_ids, suffix_mask
                )
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, stats, _logits = compute_loss(
                    args, model, z_prompt, draft_ids, z_draft, draft_conf, suffix_ids, suffix_mask
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
                    f"ref_ce={stats['refined_ce']:.3f} p={stats['refined_p']:.3f} "
                    f"top1={stats['refined_top1']:.3f} rep={stats['refined_repeat_frac']:.3f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(ce=f"{stats['refined_ce']:.3f}", p=f"{stats['refined_p']:.3f}")
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        model.eval()
        val_total = {}
        last = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"syntax-refiner ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                draft_ids, z_draft, draft_conf = make_draft(
                    args, tokenizer, decoder, vq, prior, adapter, z_prompt, z_target, suffix_ids, suffix_mask
                )
                _loss, stats, logits = compute_loss(
                    args, model, z_prompt, draft_ids, z_draft, draft_conf, suffix_ids, suffix_mask
                )
                add(val_total, stats)
                last = (input_ids[:12], draft_ids[:12], logits[:12])
        val_mean = mean(val_total)
        print(f"val ep{epoch + 1} | {json.dumps(val_mean, indent=2)}", flush=True)
        if val_mean["refined_ce"] < best_ce:
            best_ce = val_mean["refined_ce"]
            torch.save(
                {
                    "syntax_refiner": model.state_dict(),
                    "best_refined_ce": best_ce,
                    "val": val_mean,
                    "latent_dim": latent_dim,
                    "vocab_size": tokenizer.vocab_size,
                    "codebook_size": vq.codebook_size,
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "heads": args.heads,
                    "mixer_layers": args.mixer_layers,
                    "mixer_scale": args.mixer_scale,
                    "prompt_len": args.prompt_len,
                    "max_seq_len": args.max_seq_len,
                    "draft_mode": args.draft_mode,
                    "vq_path": args.vq,
                    "code_prior_path": args.code_prior,
                    "decoder_adapter_path": args.decoder_adapter,
                    "type": "syntax_token_refiner",
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | refined_ce={best_ce:.4f}", flush=True)
            if last is not None:
                write_examples(args.examples, tokenizer, *last, args)


if __name__ == "__main__":
    main()
