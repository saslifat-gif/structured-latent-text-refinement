"""Train a prompt-only sentence-plan latent prior.

This is a separate experiment from PromptPrior. It keeps Stage1 frozen, predicts
a small set of internal plan/event vectors from the prompt, expands those plan
vectors into suffix-token latents, and trains the expanded latents through the
frozen parallel decoder.
"""

from __future__ import annotations

import argparse
import os
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

from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained
from stage2_config import (
    DATASET_NAME,
    EPOCHS,
    LATENT_DIM,
    LOG_EVERY,
    MAX_SEQ_LEN,
    PROMPT_LEN,
    ROCSTORIES_SPLIT,
    SEED,
    TARGET_LATENT_MEAN,
    TARGET_LATENT_STD,
    TRAIN_BATCH_SIZE,
    TRAIN_SIZE,
)
from stage2_data import build_stage2_dataloaders
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss


def parse_args() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an env-configured prompt-only sentence-plan latent prior. "
            "Configuration is read from PLAN_PRIOR_* and shared Stage2 env vars."
        )
    )
    parser.parse_args()


PLAN_PRIOR_EPOCHS = int(os.environ.get("PLAN_PRIOR_EPOCHS", str(min(EPOCHS, 8))))
PLAN_PRIOR_LR = float(os.environ.get("PLAN_PRIOR_LR", "1e-4"))
PLAN_PRIOR_WEIGHT_DECAY = float(os.environ.get("PLAN_PRIOR_WEIGHT_DECAY", "0.01"))
PLAN_PRIOR_CODES = int(os.environ.get("PLAN_PRIOR_CODES", "3"))
PLAN_PRIOR_HIDDEN = int(os.environ.get("PLAN_PRIOR_HIDDEN", "768"))
PLAN_PRIOR_LAYERS = int(os.environ.get("PLAN_PRIOR_LAYERS", "3"))
PLAN_PRIOR_HEADS = int(os.environ.get("PLAN_PRIOR_HEADS", "8"))
PLAN_PRIOR_DROPOUT = float(os.environ.get("PLAN_PRIOR_DROPOUT", "0.05"))
PLAN_PRIOR_OUTPUT_SCALE = float(os.environ.get("PLAN_PRIOR_OUTPUT_SCALE", "1.0"))
PLAN_PRIOR_OUTPUT_BOUND = os.environ.get("PLAN_PRIOR_OUTPUT_BOUND", "true").lower() in ("1", "true", "yes", "on")
PLAN_PRIOR_CE_WEIGHT = float(os.environ.get("PLAN_PRIOR_CE_WEIGHT", "1.0"))
PLAN_PRIOR_MSE_WEIGHT = float(os.environ.get("PLAN_PRIOR_MSE_WEIGHT", "0.05"))
PLAN_PRIOR_COS_WEIGHT = float(os.environ.get("PLAN_PRIOR_COS_WEIGHT", "0.10"))
PLAN_PRIOR_PLAN_WEIGHT = float(os.environ.get("PLAN_PRIOR_PLAN_WEIGHT", "0.20"))
PLAN_PRIOR_VAR_WEIGHT = float(os.environ.get("PLAN_PRIOR_VAR_WEIGHT", "0.02"))
PLAN_PRIOR_ADJ_REP_WEIGHT = float(os.environ.get("PLAN_PRIOR_ADJ_REP_WEIGHT", "1.0"))
PLAN_PRIOR_ENTROPY_WEIGHT = float(os.environ.get("PLAN_PRIOR_ENTROPY_WEIGHT", "0.02"))
PLAN_PRIOR_ENTROPY_FLOOR = float(os.environ.get("PLAN_PRIOR_ENTROPY_FLOOR", "6.5"))
PLAN_PRIOR_DECODE_BATCH = int(os.environ.get("PLAN_PRIOR_DECODE_BATCH", "64"))
PLAN_PRIOR_EXAMPLES = int(os.environ.get("PLAN_PRIOR_EXAMPLES", "16"))
PLAN_PRIOR_EXAMPLE_DECODE = os.environ.get("PLAN_PRIOR_EXAMPLE_DECODE", "sample").lower()
PLAN_PRIOR_EXAMPLE_TEMP = float(os.environ.get("PLAN_PRIOR_EXAMPLE_TEMP", "0.8"))
PLAN_PRIOR_EXAMPLE_TOPK = int(os.environ.get("PLAN_PRIOR_EXAMPLE_TOPK", "30"))
PLAN_PRIOR_EXAMPLE_TOPP = float(os.environ.get("PLAN_PRIOR_EXAMPLE_TOPP", "0.85"))
PLAN_PRIOR_CHECKPOINT = os.environ.get(
    "PLAN_PRIOR_CHECKPOINT",
    f"sentence_plan_prior_{DATASET_NAME}_{LATENT_DIM}_{PROMPT_LEN}x{MAX_SEQ_LEN - PROMPT_LEN}_best.pt",
)
PLAN_PRIOR_EXAMPLES_PATH = os.environ.get("PLAN_PRIOR_EXAMPLES_PATH", "sentence_plan_prior_examples.txt")
STAGE1_VARIANT = os.environ.get("STAGE1_VARIANT", "").strip()


def variant_suffix(name: str) -> str:
    clean = name.strip().strip("_")
    return f"_{clean}" if clean else ""


def default_stage1_checkpoint() -> str:
    if DATASET_NAME == "rocstories":
        return f"stage1_rocstories_{LATENT_DIM}{variant_suffix(STAGE1_VARIANT)}_best.pt"
    return f"stage1{variant_suffix(STAGE1_VARIANT)}_best.pt"


STAGE1_CHECKPOINT = os.environ.get("STAGE1_CHECKPOINT", default_stage1_checkpoint())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


class SentencePlanPrior(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_codes: int = 3,
        hidden_dim: int = 768,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.plan_queries = nn.Parameter(torch.randn(num_codes, latent_dim) * 0.02)
        self.prompt_norm = nn.LayerNorm(latent_dim)
        self.plan_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True),
                        "norm1": nn.LayerNorm(latent_dim),
                        "ff": nn.Sequential(
                            nn.Linear(latent_dim, hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, latent_dim),
                        ),
                        "norm2": nn.LayerNorm(latent_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.plan_id = nn.Embedding(num_codes, latent_dim)
        self.expand = nn.Sequential(
            nn.LayerNorm(latent_dim * 3),
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_prompt: torch.Tensor, suffix_len: int, suffix_mask: torch.Tensor | None = None):
        batch = z_prompt.size(0)
        prompt = self.prompt_norm(z_prompt)
        prompt_pool = prompt.mean(dim=1)
        plans = self.plan_queries.unsqueeze(0).expand(batch, -1, -1)
        for block in self.plan_blocks:
            attn_out, _ = block["attn"](plans, prompt, prompt, need_weights=False)
            plans = block["norm1"](plans + attn_out)
            plans = block["norm2"](plans + block["ff"](plans))

        pos = torch.linspace(0.0, 1.0, suffix_len, device=z_prompt.device, dtype=z_prompt.dtype)
        raw_idx = torch.floor(pos * self.num_codes).long().clamp_max(self.num_codes - 1)
        plan_per_pos = plans[:, raw_idx, :]
        pos_emb = self.pos_proj(pos.view(1, suffix_len, 1).expand(batch, -1, -1))
        plan_id = self.plan_id(raw_idx).unsqueeze(0).expand(batch, -1, -1)
        prompt_exp = prompt_pool.unsqueeze(1).expand(-1, suffix_len, -1)
        x = torch.cat([plan_per_pos + plan_id, pos_emb, prompt_exp], dim=-1)
        raw = self.out_proj(self.out_norm(self.expand(x)))
        if PLAN_PRIOR_OUTPUT_BOUND:
            z = TARGET_LATENT_MEAN + PLAN_PRIOR_OUTPUT_SCALE * TARGET_LATENT_STD * torch.tanh(raw)
        else:
            z = raw
        if suffix_mask is not None:
            z = z * suffix_mask.to(z.dtype).unsqueeze(-1)
        return plans, z


def load_stage1(device: torch.device):
    ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device, weights_only=False)
    latent_dim = int(ckpt.get("latent_dim", LATENT_DIM))
    encoder = BertEncoder().to(device)
    decoder = ParallelDecoder(latent_dim=latent_dim).to(device)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    freeze(encoder)
    freeze(decoder)
    print(f"stage1 loaded from {STAGE1_CHECKPOINT} | encoder + decoder frozen", flush=True)
    return encoder, decoder, latent_dim


@torch.no_grad()
def encode_latents(encoder, decoder, input_ids, attention_mask):
    return decoder.compress(encoder(input_ids, attention_mask))


def target_plan_codes(z_target: torch.Tensor, mask: torch.Tensor | None, num_codes: int):
    batch, suffix_len, dim = z_target.shape
    out = []
    global_mask = mask.bool() if mask is not None else torch.ones(batch, suffix_len, device=z_target.device, dtype=torch.bool)
    for code_idx in range(num_codes):
        start = int(round(code_idx * suffix_len / num_codes))
        end = int(round((code_idx + 1) * suffix_len / num_codes))
        seg = z_target[:, start:end]
        seg_mask = global_mask[:, start:end]
        denom = seg_mask.sum(dim=1, keepdim=True).clamp_min(1).to(z_target.dtype)
        mean = (seg * seg_mask.to(z_target.dtype).unsqueeze(-1)).sum(dim=1) / denom
        empty = seg_mask.sum(dim=1) == 0
        if empty.any():
            full_denom = global_mask.sum(dim=1, keepdim=True).clamp_min(1).to(z_target.dtype)
            full_mean = (z_target * global_mask.to(z_target.dtype).unsqueeze(-1)).sum(dim=1) / full_denom
            mean = torch.where(empty.unsqueeze(-1), full_mean, mean)
        out.append(mean)
    return torch.stack(out, dim=1)


def variance_floor_loss(pred, target, mask):
    if mask is not None and mask.bool().any():
        pred_valid = pred[mask.bool()]
        target_valid = target.detach()[mask.bool()]
    else:
        pred_valid = pred.reshape(-1, pred.size(-1))
        target_valid = target.detach().reshape(-1, target.size(-1))
    pred_std = pred_valid.float().std().clamp_min(1e-6)
    target_std = target_valid.float().std().clamp_min(1e-6)
    return F.relu(target_std - pred_std), pred_std.detach().item(), target_std.detach().item()


def decoded_anti_loop_losses(decoder, z_prompt, z_suffix, target_mask):
    adj_rep = z_suffix.new_tensor(0.0)
    entropy_floor = z_suffix.new_tensor(0.0)
    entropy_mean = 0.0
    if PLAN_PRIOR_ADJ_REP_WEIGHT <= 0 and PLAN_PRIOR_ENTROPY_WEIGHT <= 0:
        return adj_rep, entropy_floor, entropy_mean
    n = min(PLAN_PRIOR_DECODE_BATCH, z_suffix.size(0))
    logits = decoder.decode_from_latent(torch.cat([z_prompt[:n], z_suffix[:n]], dim=1))[:, PROMPT_LEN:]
    probs = logits.softmax(dim=-1)
    mask = target_mask[:n].bool() if target_mask is not None else torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
    if PLAN_PRIOR_ADJ_REP_WEIGHT > 0 and logits.size(1) > 1:
        adjacent_same = (probs[:, :-1] * probs[:, 1:]).sum(dim=-1)
        pair_mask = mask[:, :-1] & mask[:, 1:]
        adj_rep = adjacent_same[pair_mask].mean() if pair_mask.any() else adjacent_same.mean()
    if PLAN_PRIOR_ENTROPY_WEIGHT > 0:
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
        entropy_valid = entropy[mask] if mask.any() else entropy.reshape(-1)
        entropy_mean = entropy_valid.mean().detach().item()
        entropy_floor = F.relu(PLAN_PRIOR_ENTROPY_FLOOR - entropy_valid).mean()
    return adj_rep, entropy_floor, entropy_mean


def compute_loss(decoder, z_prompt, plans, z_suffix, z_target, target_plans, suffix_ids, target_mask):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))
    ce, target_prob, top1 = rollout_flow_token_ce_loss(logits, suffix_ids, target_mask)
    mse = F.mse_loss(z_suffix[target_mask.bool()], z_target.detach()[target_mask.bool()]) if target_mask.bool().any() else F.mse_loss(z_suffix, z_target.detach())
    cos_loss, cos_val = rollout_cosine_alignment_loss(z_suffix, z_target, target_mask)
    plan_loss = F.smooth_l1_loss(plans, target_plans.detach())
    vloss, pred_std, target_std = variance_floor_loss(z_suffix, z_target, target_mask)
    adj_rep, entropy_floor, entropy = decoded_anti_loop_losses(decoder, z_prompt, z_suffix, target_mask)
    loss = (
        PLAN_PRIOR_CE_WEIGHT * ce
        + PLAN_PRIOR_MSE_WEIGHT * mse
        + PLAN_PRIOR_COS_WEIGHT * cos_loss
        + PLAN_PRIOR_PLAN_WEIGHT * plan_loss
        + PLAN_PRIOR_VAR_WEIGHT * vloss
        + PLAN_PRIOR_ADJ_REP_WEIGHT * adj_rep
        + PLAN_PRIOR_ENTROPY_WEIGHT * entropy_floor
    )
    return loss, {
        "ce": ce.detach().item(),
        "p": target_prob,
        "top1": top1,
        "mse": mse.detach().item(),
        "cos": cos_val,
        "plan": plan_loss.detach().item(),
        "var": vloss.detach().item(),
        "adj": adj_rep.detach().item(),
        "ent_floor": entropy_floor.detach().item(),
        "ent": entropy,
        "pred_std": pred_std,
        "target_std": target_std,
    }


def new_total():
    return {key: 0.0 for key in ("ce", "p", "top1", "mse", "cos", "plan", "var", "adj", "ent_floor", "ent", "pred_std", "target_std")} | {"n": 0}


def add_stats(total, stats):
    for key in total:
        if key != "n" and key in stats:
            total[key] += stats[key]
    total["n"] += 1


def mean_stats(total):
    n = max(total["n"], 1)
    return {key: value / n for key, value in total.items() if key != "n"}


def sample_token_ids(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature <= 0:
        return logits.argmax(dim=-1)
    scores = logits.float() / max(temperature, 1e-5)
    if top_k > 0 and top_k < scores.size(-1):
        kth = scores.topk(top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_idx = scores.sort(dim=-1, descending=True)
        probs = sorted_scores.softmax(dim=-1)
        remove = probs.cumsum(dim=-1) > top_p
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, sorted_idx, sorted_scores)
    probs = scores.softmax(dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(logits.shape[:-1])


def decode_suffix(tokenizer, decoder, z_prompt, z_suffix):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))
    if PLAN_PRIOR_EXAMPLE_DECODE in ("sample", "sampling", "topk", "topp"):
        ids = sample_token_ids(
            logits,
            PLAN_PRIOR_EXAMPLE_TEMP,
            PLAN_PRIOR_EXAMPLE_TOPK,
            PLAN_PRIOR_EXAMPLE_TOPP,
        )
    else:
        ids = logits.argmax(dim=-1)
    return [tokenizer.decode(ids[i, PROMPT_LEN:], skip_special_tokens=True).strip() for i in range(ids.size(0))]


@torch.no_grad()
def write_examples(tokenizer, encoder, decoder, model, val_loader, device, epoch):
    model.eval()
    rows = []
    made = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        z = encode_latents(encoder, decoder, input_ids, attention_mask)
        z_prompt = z[:, :PROMPT_LEN]
        target_mask = attention_mask[:, PROMPT_LEN:]
        n = min(PLAN_PRIOR_EXAMPLES - made, input_ids.size(0))
        plans, z_suffix = model(z_prompt, MAX_SEQ_LEN - PROMPT_LEN, target_mask)
        pred_texts = decode_suffix(tokenizer, decoder, z_prompt[:n], z_suffix[:n])
        target_texts = [tokenizer.decode(input_ids[i, PROMPT_LEN:], skip_special_tokens=True).strip() for i in range(n)]
        prompt_texts = [tokenizer.decode(input_ids[i, :PROMPT_LEN], skip_special_tokens=True).strip() for i in range(n)]
        for i in range(n):
            rows.append(
                f"--- example {made + 1} epoch {epoch + 1}\n"
                f"prompt: {prompt_texts[i]}\n"
                f"target: {target_texts[i]}\n"
                f"plan prior sample: {pred_texts[i]}\n"
            )
            made += 1
            if made >= PLAN_PRIOR_EXAMPLES:
                Path(PLAN_PRIOR_EXAMPLES_PATH).write_text("\n".join(rows), encoding="utf-8")
                print(f"saved {made} examples to {PLAN_PRIOR_EXAMPLES_PATH}", flush=True)
                return
    Path(PLAN_PRIOR_EXAMPLES_PATH).write_text("\n".join(rows), encoding="utf-8")
    print(f"saved {made} examples to {PLAN_PRIOR_EXAMPLES_PATH}", flush=True)


def main():
    parse_args()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)
    print(
        f"sentence-plan config | dataset={DATASET_NAME} split={ROCSTORIES_SPLIT if DATASET_NAME == 'rocstories' else 'legacy'} "
        f"prompt_slots={PROMPT_LEN} suffix_slots={MAX_SEQ_LEN - PROMPT_LEN} latent_dim={LATENT_DIM} "
        f"codes={PLAN_PRIOR_CODES} train_size={TRAIN_SIZE} batch={TRAIN_BATCH_SIZE} epochs={PLAN_PRIOR_EPOCHS} "
        f"output_bound={PLAN_PRIOR_OUTPUT_BOUND} output_scale={PLAN_PRIOR_OUTPUT_SCALE} "
        f"loss ce={PLAN_PRIOR_CE_WEIGHT} mse={PLAN_PRIOR_MSE_WEIGHT} cos={PLAN_PRIOR_COS_WEIGHT} "
        f"plan={PLAN_PRIOR_PLAN_WEIGHT} var={PLAN_PRIOR_VAR_WEIGHT} adj={PLAN_PRIOR_ADJ_REP_WEIGHT} "
        f"ent={PLAN_PRIOR_ENTROPY_WEIGHT}@{PLAN_PRIOR_ENTROPY_FLOOR} checkpoint={PLAN_PRIOR_CHECKPOINT}",
        flush=True,
    )

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(device)
    model = SentencePlanPrior(
        latent_dim=latent_dim,
        num_codes=PLAN_PRIOR_CODES,
        hidden_dim=PLAN_PRIOR_HIDDEN,
        num_layers=PLAN_PRIOR_LAYERS,
        num_heads=PLAN_PRIOR_HEADS,
        dropout=PLAN_PRIOR_DROPOUT,
    ).to(device)
    train_loader, val_loader = build_stage2_dataloaders(
        tokenizer,
        train_size=TRAIN_SIZE,
        batch_size=TRAIN_BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
    )
    optimizer = AdamW(model.parameters(), lr=PLAN_PRIOR_LR, weight_decay=PLAN_PRIOR_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_val_ce = float("inf")
    print(f"SentencePlanPrior params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    for epoch in range(PLAN_PRIOR_EPOCHS):
        model.train()
        total = new_total()
        total_loss = 0.0
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"plan ep{epoch + 1}/{PLAN_PRIOR_EPOCHS} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, :PROMPT_LEN]
                z_target = z[:, PROMPT_LEN:]
                target_mask = attention_mask[:, PROMPT_LEN:]
                suffix_ids = input_ids[:, PROMPT_LEN:]
                target_plans = target_plan_codes(z_target, target_mask, PLAN_PRIOR_CODES)
                plans, z_suffix = model(z_prompt, z_target.size(1), target_mask)
                loss, stats = compute_loss(decoder, z_prompt, plans, z_suffix, z_target, target_plans, suffix_ids, target_mask)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            add_stats(total, stats)
            if tqdm is not None:
                iterator.set_postfix(loss=f"{loss.item():.3f}", ce=f"{stats['ce']:.3f}", p=f"{stats['p']:.3f}", ent=f"{stats['ent']:.2f}")
            if step % LOG_EVERY == 0:
                print(
                    f"ep{epoch + 1} step {step}/{len(train_loader)} | loss {loss.item():.4f} "
                    f"| ce {stats['ce']:.4f} p={stats['p']:.3f} top1={stats['top1']:.3f} "
                    f"| mse {stats['mse']:.4f} cos {stats['cos']:.3f} plan {stats['plan']:.4f} "
                    f"| adj {stats['adj']:.4f} ent {stats['ent']:.2f} std {stats['pred_std']:.3f}/{stats['target_std']:.3f}",
                    flush=True,
                )
        train_mean = mean_stats(total)
        print(
            f"ep{epoch + 1} train | avg_loss={total_loss / max(total['n'], 1):.4f} "
            f"ce={train_mean['ce']:.3f} p={train_mean['p']:.3f} top1={train_mean['top1']:.3f} "
            f"plan={train_mean['plan']:.4f} adj={train_mean['adj']:.4f} ent={train_mean['ent']:.2f}",
            flush=True,
        )

        model.eval()
        val_total = new_total()
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"plan ep{epoch + 1}/{PLAN_PRIOR_EPOCHS} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    z = encode_latents(encoder, decoder, input_ids, attention_mask)
                    z_prompt = z[:, :PROMPT_LEN]
                    z_target = z[:, PROMPT_LEN:]
                    target_mask = attention_mask[:, PROMPT_LEN:]
                    suffix_ids = input_ids[:, PROMPT_LEN:]
                    target_plans = target_plan_codes(z_target, target_mask, PLAN_PRIOR_CODES)
                    plans, z_suffix = model(z_prompt, z_target.size(1), target_mask)
                    _loss, stats = compute_loss(decoder, z_prompt, plans, z_suffix, z_target, target_plans, suffix_ids, target_mask)
                add_stats(val_total, stats)
        val_mean = mean_stats(val_total)
        print(
            f"val ep{epoch + 1} | ce={val_mean['ce']:.3f} p={val_mean['p']:.3f} top1={val_mean['top1']:.3f} "
            f"mse={val_mean['mse']:.4f} cos={val_mean['cos']:.3f} plan={val_mean['plan']:.4f} "
            f"adj={val_mean['adj']:.4f} ent={val_mean['ent']:.2f} std={val_mean['pred_std']:.3f}/{val_mean['target_std']:.3f}",
            flush=True,
        )
        if val_mean["ce"] < best_val_ce:
            best_val_ce = val_mean["ce"]
            torch.save(
                {
                    "sentence_plan_prior": model.state_dict(),
                    "best_val_ce": best_val_ce,
                    "val": val_mean,
                    "latent_dim": latent_dim,
                    "plan_codes": PLAN_PRIOR_CODES,
                    "plan_output_bound": PLAN_PRIOR_OUTPUT_BOUND,
                    "plan_output_scale": PLAN_PRIOR_OUTPUT_SCALE,
                    "prompt_len": PROMPT_LEN,
                    "max_seq_len": MAX_SEQ_LEN,
                    "dataset_name": DATASET_NAME,
                    "dataset_split": ROCSTORIES_SPLIT if DATASET_NAME == "rocstories" else "legacy_fixed_token",
                    "stage1_variant": STAGE1_VARIANT,
                    "type": "sentence_plan_prior",
                    "epoch": epoch,
                },
                PLAN_PRIOR_CHECKPOINT,
            )
            print(f"saved {PLAN_PRIOR_CHECKPOINT} | val_ce={best_val_ce:.4f}", flush=True)
        write_examples(tokenizer, encoder, decoder, model, val_loader, device, epoch)


if __name__ == "__main__":
    main()
