"""Train a bounded projector that moves PromptPrior latents toward a readable basin.

This is a diagnostic post-prompt-prior experiment: Stage1 and PromptPrior stay
frozen, and only the small BasinProjector is optimized.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained
from stage2_config import (
    MAX_SEQ_LEN,
    PROMPT_LEN,
    SEED,
    TARGET_LATENT_MEAN,
    TARGET_LATENT_STD,
    TRAIN_BATCH_SIZE,
    TRAIN_SIZE,
)
import stage2_data as s2data
from stage2_data import build_stage2_dataloaders
import stage2_riemannian as rfm
from stage2_riemannian import StartTransformer, suffix_positions
from transformers import BertTokenizer


SPECIAL_IDS = {0, 101, 102, 103}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PromptPrior basin projector diagnostic")
    parser.add_argument("--stage1", default="stage1_rocstories_768_best.pt")
    parser.add_argument("--prompt_prior", default="prompt_prior_rocstories_768_best.pt")
    parser.add_argument("--output", default="basin_projector_rocstories_768_best.pt")
    parser.add_argument("--last_output", default=None, help="Optional per-epoch checkpoint written before validation.")
    parser.add_argument("--metrics_csv", default="results/basin_projector_metrics.csv")
    parser.add_argument("--dataset", choices=("rocstories", "wikitext"), default="rocstories")
    parser.add_argument("--rocstories_file", default=None)
    parser.add_argument(
        "--rocstories_source",
        choices=("auto", "file", "hub"),
        default="auto",
        help="Only used when --dataset=rocstories.",
    )
    parser.add_argument("--rocstories_local_files_only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--prompt_len", type=int, default=None)
    parser.add_argument(
        "--suffix_len",
        type=int,
        default=None,
        help="Optional target/suffix slot count. If set with --prompt_len, max_seq_len=prompt_len+suffix_len.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--residual_scale", type=float, default=0.2)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.10)
    parser.add_argument("--cos_weight", type=float, default=0.10)
    parser.add_argument("--hidden_weight", type=float, default=0.10)
    parser.add_argument("--smooth_weight", type=float, default=0.01)
    parser.add_argument("--delta_weight", type=float, default=0.01)
    parser.add_argument(
        "--source",
        choices=("prompt_prior", "moa", "prompt_prior_moa"),
        default="prompt_prior",
        help="Latent source before BasinProjector.",
    )
    parser.add_argument("--moa_k", type=int, default=32)
    parser.add_argument("--moa_temp", type=float, default=0.7)
    parser.add_argument("--moa_topk", type=int, default=4)
    parser.add_argument("--moa_noise", type=float, default=0.02)
    parser.add_argument("--moa_mix", type=float, default=0.5, help="MoA weight for --source=prompt_prior_moa.")
    parser.add_argument("--moa_entropy_weight", type=float, default=0.001)
    parser.add_argument("--moa_init_batches", type=int, default=8)
    parser.add_argument("--target", choices=("real", "synthetic_draft"), default="synthetic_draft")
    parser.add_argument("--draft_drop_prob", type=float, default=0.03)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--val_batches", type=int, default=200, help="Cap validation batches; <=0 runs full validation.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def load_stage1(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    latent_dim = int(ckpt.get("latent_dim", 256))
    encoder = BertEncoder().to(device)
    decoder = ParallelDecoder(latent_dim=latent_dim).to(device)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    freeze(encoder)
    freeze(decoder)
    return encoder, decoder, latent_dim, ckpt


def load_prompt_prior(path: str, latent_dim: int, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("prompt_prior")
    if state is None:
        raise RuntimeError(f"No prompt_prior state found in {path}")
    ckpt_latent_dim = int(ckpt.get("latent_dim", latent_dim))
    mixer = str(ckpt.get("prompt_prior_mixer", "none")).lower()
    stochastic = bool(ckpt.get("prompt_prior_stochastic", False))
    memory = bool(ckpt.get("prompt_prior_memory", False))
    use_wrapper = (
        any(key.startswith("base.") for key in state)
        or mixer not in ("none", "off", "false", "0")
        or stochastic
        or memory
    )
    if use_wrapper:
        mixer_layers = 0 if mixer in ("none", "off", "false", "0") else int(ckpt.get("prompt_prior_mixer_layers", 2))
        memory_slots = int(ckpt.get("max_seq_len", MAX_SEQ_LEN)) - int(ckpt.get("prompt_len", PROMPT_LEN))
        if "memory_values" in state:
            memory_slots = int(state["memory_values"].shape[1])
        model = PromptPriorWithMixer(
            latent_dim=ckpt_latent_dim,
            num_layers=int(ckpt.get("layers", ckpt.get("start_transformer_layers", 4))),
            num_heads=int(ckpt.get("heads", ckpt.get("start_transformer_heads", 8))),
            ffn_dim=int(ckpt.get("hidden_dim", ckpt.get("start_transformer_hidden_dim", 512))),
            mixer_layers=mixer_layers,
            mixer_kernel=int(ckpt.get("prompt_prior_mixer_kernel", 5)),
            mixer_scale=float(ckpt.get("prompt_prior_mixer_scale", 0.5)),
            use_noise=stochastic,
            noise_input_scale=float(ckpt.get("prompt_prior_noise_input_scale", 0.2)),
            use_memory=memory,
            memory_size=int(ckpt.get("prompt_prior_memory_size", 512)),
            memory_temp=float(ckpt.get("prompt_prior_memory_temp", 0.2)),
            memory_scale=float(ckpt.get("prompt_prior_memory_scale", 1.0)),
            memory_topk=int(ckpt.get("prompt_prior_memory_topk", 0)),
            memory_slots=memory_slots,
        ).to(device)
    else:
        model = StartTransformer(
            latent_dim=ckpt_latent_dim,
            num_layers=int(ckpt.get("layers", ckpt.get("start_transformer_layers", 4))),
            num_heads=int(ckpt.get("heads", ckpt.get("start_transformer_heads", 8))),
            ffn_dim=int(ckpt.get("hidden_dim", ckpt.get("start_transformer_hidden_dim", 512))),
        ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    norm_on = bool(ckpt.get("prompt_prior_normalize_output", False))
    norm_mode = str(ckpt.get("prompt_prior_normalize_mode", "sequence"))
    norm_source = str(ckpt.get("prompt_prior_normalize_source", "target"))
    norm_scale = float(ckpt.get("prompt_prior_normalize_scale", 1.0))
    model._basin_noise_std_scale = float(ckpt.get("prompt_prior_noise_std_scale", 1.0))
    model._basin_normalize_output = norm_on
    model._basin_normalize_mode = norm_mode
    model._basin_normalize_source = norm_source
    model._basin_normalize_scale = norm_scale
    model._basin_output_mean = float(ckpt.get("prompt_prior_output_mean", TARGET_LATENT_MEAN))
    model._basin_output_std = float(ckpt.get("prompt_prior_output_std", TARGET_LATENT_STD))
    arch = (
        f"mixer={mixer} layers={int(ckpt.get('prompt_prior_mixer_layers', 0)) if use_wrapper else 0} "
        f"stoch={stochastic} memory={memory} "
        f"norm={'on' if norm_on else 'off'}:{norm_mode}x{norm_scale:g}"
    )
    print(
        f"loaded PromptPrior architecture: {arch} | missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    freeze(model)
    return model, ckpt


class ParallelStateMixerBlock(nn.Module):
    def __init__(self, latent_dim, kernel_size=5, residual_scale=0.5):
        super().__init__()
        padding = kernel_size // 2
        self.residual_scale = residual_scale
        self.norm = nn.LayerNorm(latent_dim)
        self.in_proj = nn.Linear(latent_dim, latent_dim * 2)
        self.dwconv = nn.Conv1d(
            latent_dim,
            latent_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=latent_dim,
        )
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, mask=None):
        h = self.norm(x)
        u, gate = self.in_proj(h).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)
        gate = torch.sigmoid(gate)
        if mask is not None:
            m = mask.to(u.dtype).unsqueeze(-1)
            u = u * m
            gate = gate * m
        weighted = u * gate
        denom_f = gate.cumsum(dim=1).clamp_min(1e-5)
        state_f = weighted.cumsum(dim=1) / denom_f
        denom_b = gate.flip(1).cumsum(dim=1).flip(1).clamp_min(1e-5)
        state_b = weighted.flip(1).cumsum(dim=1).flip(1) / denom_b
        mixed = u + 0.5 * (state_f + state_b)
        x = x + self.residual_scale * self.out_proj(mixed)
        if mask is not None:
            x = x * mask.to(x.dtype).unsqueeze(-1)
        return x


class PromptPriorWithMixer(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers,
        num_heads,
        ffn_dim,
        mixer_layers,
        mixer_kernel,
        mixer_scale,
        use_noise=False,
        noise_input_scale=0.2,
        use_memory=False,
        memory_size=512,
        memory_temp=0.2,
        memory_scale=1.0,
        memory_topk=0,
        memory_slots=64,
    ):
        super().__init__()
        self.use_noise = use_noise
        self.noise_input_scale = noise_input_scale
        self.use_memory = use_memory
        self.memory_temp = memory_temp
        self.memory_scale = memory_scale
        self.memory_topk = memory_topk
        self.base = StartTransformer(latent_dim=latent_dim, num_layers=num_layers, num_heads=num_heads, ffn_dim=ffn_dim)
        self.mixer = nn.ModuleList(
            [
                ParallelStateMixerBlock(latent_dim=latent_dim, kernel_size=mixer_kernel, residual_scale=mixer_scale)
                for _ in range(mixer_layers)
            ]
        )
        self.noise_proj = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim))
        nn.init.eye_(self.noise_proj[-1].weight)
        nn.init.zeros_(self.noise_proj[-1].bias)
        self.prompt_query = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim))
        nn.init.eye_(self.prompt_query[-1].weight)
        nn.init.zeros_(self.prompt_query[-1].bias)
        self.memory_norm = nn.LayerNorm(latent_dim)
        self.memory_keys = nn.Parameter(torch.randn(memory_size, latent_dim) * 0.02)
        self.memory_values = nn.Parameter(
            torch.randn(memory_size, memory_slots, latent_dim) * (0.2 * TARGET_LATENT_STD) + TARGET_LATENT_MEAN
        )
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, z_prompt, pos, mask=None, z_init=None):
        x = self.base(z_prompt, pos, mask)
        if self.use_memory:
            prompt_mask = z_prompt.abs().sum(dim=-1) > 0
            denom = prompt_mask.sum(dim=1, keepdim=True).clamp_min(1).to(z_prompt.dtype)
            pooled = (z_prompt * prompt_mask.to(z_prompt.dtype).unsqueeze(-1)).sum(dim=1) / denom
            q = F.normalize(self.prompt_query(pooled), dim=-1)
            keys = F.normalize(self.memory_keys, dim=-1)
            scores = (q @ keys.t()) / max(self.memory_temp, 1e-4)
            if self.memory_topk > 0 and self.memory_topk < scores.size(-1):
                top_vals, top_idx = scores.topk(self.memory_topk, dim=-1)
                masked_scores = scores.new_full(scores.shape, -float("inf"))
                scores = masked_scores.scatter(dim=-1, index=top_idx, src=top_vals)
            weights = torch.softmax(scores, dim=-1)
            z_mem = torch.einsum("bn,ntd->btd", weights, self.memory_values)[:, : x.size(1), :]
            x = x + self.memory_scale * self.memory_norm(z_mem)
            if mask is not None:
                x = x * mask.to(x.dtype).unsqueeze(-1)
        if self.use_noise:
            if z_init is None:
                raise ValueError("PromptPriorWithMixer requires z_init when use_noise=True")
            x = x + self.noise_input_scale * self.noise_proj(z_init)
            if mask is not None:
                x = x * mask.to(x.dtype).unsqueeze(-1)
        for block in self.mixer:
            x = block(x, mask)
        x = self.out_norm(x)
        if mask is not None:
            x = x * mask.to(x.dtype).unsqueeze(-1)
        return x


def resolve_slots(args, stage1_ckpt, prompt_ckpt):
    prompt_len = args.prompt_len
    if prompt_len is None:
        prompt_len = prompt_ckpt.get("prompt_len", stage1_ckpt.get("prompt_len", PROMPT_LEN))
    prompt_len = int(prompt_len)

    if args.suffix_len is not None:
        max_seq_len = prompt_len + int(args.suffix_len)
    elif args.max_seq_len is not None:
        max_seq_len = int(args.max_seq_len)
    else:
        max_seq_len = prompt_ckpt.get("max_seq_len", stage1_ckpt.get("max_seq_len", MAX_SEQ_LEN))
        max_seq_len = int(max_seq_len)

    if max_seq_len <= prompt_len:
        raise ValueError(
            f"max_seq_len must be larger than prompt_len; got prompt_len={prompt_len} "
            f"max_seq_len={max_seq_len}"
        )
    args.prompt_len = prompt_len
    args.max_seq_len = max_seq_len
    args.suffix_len = max_seq_len - prompt_len


class BasinProjector(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, depth: int, residual_scale: float):
        super().__init__()
        self.residual_scale = residual_scale
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
        )
        self.in_proj = nn.Linear(latent_dim * 2 + 1, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, z_prompt, pos, mask=None):
        prompt = z_prompt.mean(dim=1).unsqueeze(1).expand_as(z)
        h = self.in_proj(torch.cat([z, prompt, pos.unsqueeze(-1)], dim=-1))
        h = h + self.prompt_proj(prompt)
        for block in self.blocks:
            h = h + block(h)
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)
        delta = self.residual_scale * torch.tanh(self.out_proj(self.out_norm(h)))
        if mask is not None:
            delta = delta * mask.to(delta.dtype).unsqueeze(-1)
        return z + delta, delta


class PromptMoASource(nn.Module):
    def __init__(self, latent_dim: int, suffix_len: int, num_archetypes: int, hidden_dim: int):
        super().__init__()
        self.num_archetypes = num_archetypes
        self.suffix_len = suffix_len
        self.prototypes = nn.Parameter(
            torch.randn(num_archetypes, suffix_len, latent_dim) * (0.2 * TARGET_LATENT_STD) + TARGET_LATENT_MEAN
        )
        self.router = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_archetypes),
        )

    def forward(self, z_prompt, mask=None, temperature=0.7, topk=0, noise_scale=0.0):
        prompt_mask = z_prompt.abs().sum(dim=-1) > 0
        denom = prompt_mask.sum(dim=1, keepdim=True).clamp_min(1).to(z_prompt.dtype)
        pooled = (z_prompt * prompt_mask.to(z_prompt.dtype).unsqueeze(-1)).sum(dim=1) / denom
        logits = self.router(pooled)
        scores = logits / max(temperature, 1e-4)
        if topk > 0 and topk < scores.size(-1):
            top_vals, top_idx = scores.topk(topk, dim=-1)
            masked = scores.new_full(scores.shape, -float("inf"))
            scores = masked.scatter(dim=-1, index=top_idx, src=top_vals)
        weights = torch.softmax(scores, dim=-1)
        z = torch.einsum("bk,ktd->btd", weights, self.prototypes)
        if noise_scale > 0:
            z = z + noise_scale * torch.randn_like(z)
        if mask is not None:
            z = z * mask.to(z.dtype).unsqueeze(-1)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        max_prob = weights.max(dim=-1).values.mean()
        return z, {"entropy": entropy, "max_prob": max_prob}


@torch.no_grad()
def encode_latents(encoder, decoder, input_ids, attention_mask):
    return decoder.compress(encoder(input_ids, attention_mask))


@torch.no_grad()
def prompt_prior_latents(prompt_prior, z_prompt, suffix_len, mask, target):
    pos = suffix_positions(z_prompt.size(0), suffix_len, z_prompt.device, z_prompt.dtype)
    kwargs = {}
    if getattr(prompt_prior, "use_noise", False):
        z_init = (
            torch.randn_like(target) * (prompt_prior._basin_noise_std_scale * TARGET_LATENT_STD)
            + TARGET_LATENT_MEAN
        )
        if mask is not None:
            z_init = z_init * mask.to(z_init.dtype).unsqueeze(-1)
        kwargs["z_init"] = z_init
    pred = prompt_prior(z_prompt, pos, mask, **kwargs)
    return normalize_prompt_output(prompt_prior, pred, target, mask)


def masked_sequence_stats(z, mask):
    if mask is None:
        zf = z.float()
        return zf.mean(dim=(1, 2), keepdim=True).to(z.dtype), zf.std(dim=(1, 2), keepdim=True).clamp_min(1e-5).to(z.dtype)
    valid = mask.to(z.dtype).unsqueeze(-1)
    denom = (valid.sum(dim=(1, 2), keepdim=True) * z.size(-1)).clamp_min(1.0)
    zf = z.float()
    valid_f = valid.float()
    mean = (zf * valid_f).sum(dim=(1, 2), keepdim=True) / denom.float()
    var = (((zf - mean) * valid_f).pow(2).sum(dim=(1, 2), keepdim=True) / denom.float()).clamp_min(1e-10)
    return mean.to(z.dtype), var.sqrt().to(z.dtype)


def masked_feature_stats(z, mask):
    if mask is None:
        zf = z.float()
        return zf.mean(dim=1, keepdim=True).to(z.dtype), zf.std(dim=1, keepdim=True).clamp_min(1e-5).to(z.dtype)
    valid = mask.to(z.dtype).unsqueeze(-1)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    zf = z.float()
    valid_f = valid.float()
    mean = (zf * valid_f).sum(dim=1, keepdim=True) / denom.float()
    var = (((zf - mean) * valid_f).pow(2).sum(dim=1, keepdim=True) / denom.float()).clamp_min(1e-10)
    return mean.to(z.dtype), var.sqrt().to(z.dtype)


def normalize_prompt_output(prompt_prior, pred, target, mask):
    if not getattr(prompt_prior, "_basin_normalize_output", False):
        return pred
    mode = getattr(prompt_prior, "_basin_normalize_mode", "sequence")
    if mode == "sequence":
        stats_fn = masked_sequence_stats
    elif mode == "feature":
        stats_fn = masked_feature_stats
    else:
        raise ValueError(f"unsupported PromptPrior normalize mode: {mode}")
    pred_mean, pred_std = stats_fn(pred, mask)
    pred_norm = (pred - pred_mean) / pred_std
    source = getattr(prompt_prior, "_basin_normalize_source", "target")
    if source == "target":
        target_mean, target_std = stats_fn(target.detach(), mask)
    elif source == "global":
        shape = (pred.size(0), 1, 1)
        target_mean = pred.new_full(shape, getattr(prompt_prior, "_basin_output_mean", TARGET_LATENT_MEAN))
        target_std = pred.new_full(shape, getattr(prompt_prior, "_basin_output_std", TARGET_LATENT_STD))
    else:
        raise ValueError(f"unsupported PromptPrior normalize source: {source}")
    out = pred_norm * (getattr(prompt_prior, "_basin_normalize_scale", 1.0) * target_std.clamp_min(1e-5)) + target_mean
    if mask is not None:
        out = out * mask.to(out.dtype).unsqueeze(-1)
    return out


def make_synthetic_draft_ids(input_ids, attention_mask, prompt_len, drop_prob):
    draft = input_ids.clone()
    draft_mask = attention_mask.clone()
    suffix_len = input_ids.size(1) - prompt_len
    for batch_idx in range(input_ids.size(0)):
        kept = []
        suffix_ids = input_ids[batch_idx, prompt_len:].tolist()
        suffix_mask = attention_mask[batch_idx, prompt_len:].tolist()
        for token_id, token_mask in zip(suffix_ids, suffix_mask):
            if not token_mask or int(token_id) in SPECIAL_IDS:
                continue
            if random.random() >= drop_prob:
                kept.append(int(token_id))
        if not kept:
            kept = [int(token_id) for token_id, token_mask in zip(suffix_ids, suffix_mask) if token_mask]
        kept = kept[:suffix_len]
        padded = kept + [0] * (suffix_len - len(kept))
        draft[batch_idx, prompt_len:] = torch.tensor(padded, device=input_ids.device, dtype=input_ids.dtype)
        draft_mask[batch_idx, prompt_len:] = (draft[batch_idx, prompt_len:] != 0).to(draft_mask.dtype)
    return draft, draft_mask


def decoder_ce_stats(decoder, z_prompt, z_suffix, suffix_ids, suffix_mask):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))[:, z_prompt.size(1):]
    vocab = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        suffix_ids.reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape_as(suffix_ids)
    mask = suffix_mask.bool()
    denom = mask.sum(dim=1).clamp_min(1)
    ce = (loss * mask.to(loss.dtype)).sum(dim=1) / denom
    probs = logits.softmax(dim=-1)
    target_prob = probs.gather(-1, suffix_ids.unsqueeze(-1)).squeeze(-1)
    target_prob = (target_prob * mask.to(target_prob.dtype)).sum(dim=1) / denom
    top1 = logits.argmax(dim=-1).eq(suffix_ids)
    top1_acc = (top1 * mask).sum(dim=1).to(torch.float32) / denom
    return ce, target_prob, top1_acc


def decoder_stats_and_hidden_loss(decoder, z_prompt, z_suffix, z_target, suffix_ids, suffix_mask):
    logits, hidden = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1), return_hidden=True)
    logits = logits[:, z_prompt.size(1):]
    hidden = hidden[:, z_prompt.size(1):]
    vocab = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        suffix_ids.reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape_as(suffix_ids)
    mask = suffix_mask.bool()
    denom = mask.sum(dim=1).clamp_min(1)
    ce = (loss * mask.to(loss.dtype)).sum(dim=1) / denom
    probs = logits.softmax(dim=-1)
    target_prob = probs.gather(-1, suffix_ids.unsqueeze(-1)).squeeze(-1)
    target_prob = (target_prob * mask.to(target_prob.dtype)).sum(dim=1) / denom
    top1 = logits.argmax(dim=-1).eq(suffix_ids)
    top1_acc = (top1 * mask).sum(dim=1).to(torch.float32) / denom
    with torch.no_grad():
        _target_logits, target_hidden = decoder.decode_from_latent(
            torch.cat([z_prompt, z_target], dim=1),
            return_hidden=True,
        )
        target_hidden = target_hidden[:, z_prompt.size(1):]
    if mask.any():
        hidden_loss = F.mse_loss(hidden[mask], target_hidden.detach()[mask])
    else:
        hidden_loss = F.mse_loss(hidden, target_hidden.detach())
    return ce, target_prob, top1_acc, hidden_loss


def masked_losses(pred, target, mask):
    valid = mask.bool()
    if valid.any():
        mse = F.mse_loss(pred[valid], target.detach()[valid])
        cos = F.cosine_similarity(pred[valid], target.detach()[valid], dim=-1).mean()
    else:
        mse = F.mse_loss(pred, target.detach())
        cos = F.cosine_similarity(pred, target.detach(), dim=-1).mean()
    return mse, cos


def local_smoothness_loss(pred, target, mask):
    if pred.size(1) < 2:
        return pred.new_tensor(0.0)
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target.detach()[:, 1:] - target.detach()[:, :-1]
    pair_mask = mask[:, 1:].bool() & mask[:, :-1].bool()
    if pair_mask.any():
        return F.smooth_l1_loss(pred_diff[pair_mask], target_diff[pair_mask])
    return F.smooth_l1_loss(pred_diff, target_diff)


def mean_dict(rows):
    out = {}
    if not rows:
        return out
    for key in rows[0]:
        out[key] = sum(float(row[key]) for row in rows) / len(rows)
    return out


def progress(iterable, total, desc, disabled):
    if tqdm is None or disabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def choose_source_latents(args, prompt_prior, moa_source, z_prompt, suffix_mask, target):
    prior = None
    moa = None
    moa_stats = {}
    if args.source in ("prompt_prior", "prompt_prior_moa"):
        prior = prompt_prior_latents(prompt_prior, z_prompt, target.size(1), suffix_mask, target)
    if args.source in ("moa", "prompt_prior_moa"):
        moa, moa_stats = moa_source(
            z_prompt,
            suffix_mask,
            temperature=args.moa_temp,
            topk=args.moa_topk,
            noise_scale=args.moa_noise,
        )
    if args.source == "prompt_prior":
        return prior, moa_stats
    if args.source == "moa":
        return moa, moa_stats
    mix = min(max(args.moa_mix, 0.0), 1.0)
    return (1.0 - mix) * prior + mix * moa, moa_stats


def prepare_batch(args, encoder, decoder, prompt_prior, moa_source, batch, device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    z_data = encode_latents(encoder, decoder, input_ids, attention_mask)
    z_prompt = z_data[:, : args.prompt_len]
    z_real = z_data[:, args.prompt_len:]
    suffix_ids = input_ids[:, args.prompt_len:]
    suffix_mask = attention_mask[:, args.prompt_len:]

    draft_ids, draft_mask = make_synthetic_draft_ids(
        input_ids,
        attention_mask,
        args.prompt_len,
        args.draft_drop_prob,
    )
    z_draft = encode_latents(encoder, decoder, draft_ids, draft_mask)[:, args.prompt_len:]
    target = z_draft if args.target == "synthetic_draft" else z_real
    z_prior, source_stats = choose_source_latents(args, prompt_prior, moa_source, z_prompt, suffix_mask, target)
    return z_prompt, z_prior, target, z_real, suffix_ids, suffix_mask, source_stats


def evaluate(args, encoder, decoder, prompt_prior, moa_source, projector, val_loader, device):
    projector.eval()
    if moa_source is not None:
        moa_source.eval()
    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
            if args.val_batches > 0 and batch_idx > args.val_batches:
                break
            z_prompt, z_prior, target, z_real, suffix_ids, suffix_mask, source_stats = prepare_batch(
                args,
                encoder,
                decoder,
                prompt_prior,
                moa_source,
                batch,
                device,
            )
            pos = suffix_positions(z_prior.size(0), z_prior.size(1), device, z_prior.dtype)
            z_proj, delta = projector(z_prior, z_prompt, pos, suffix_mask)
            direct_ce, direct_p, direct_top1 = decoder_ce_stats(decoder, z_prompt, z_prior, suffix_ids, suffix_mask)
            proj_ce, proj_p, proj_top1 = decoder_ce_stats(decoder, z_prompt, z_proj, suffix_ids, suffix_mask)
            direct_mse, direct_cos = masked_losses(z_prior, target, suffix_mask)
            proj_mse, proj_cos = masked_losses(z_proj, target, suffix_mask)
            n = min(args.decode_batch, z_proj.size(0))
            _ce, _p, _top1, hidden_loss = decoder_stats_and_hidden_loss(
                decoder,
                z_prompt[:n],
                z_proj[:n],
                target[:n],
                suffix_ids[:n],
                suffix_mask[:n],
            )
            smooth = local_smoothness_loss(z_proj, target, suffix_mask)
            rows.append(
                {
                    "direct_ce": direct_ce.mean().item(),
                    "projected_ce": proj_ce.mean().item(),
                    "direct_p": direct_p.mean().item(),
                    "projected_p": proj_p.mean().item(),
                    "direct_top1": direct_top1.mean().item(),
                    "projected_top1": proj_top1.mean().item(),
                    "direct_cos": direct_cos.item(),
                    "projected_cos": proj_cos.item(),
                    "direct_mse": direct_mse.item(),
                    "projected_mse": proj_mse.item(),
                    "projected_hidden": hidden_loss.item(),
                    "projected_smooth": smooth.item(),
                    "delta_norm": delta.norm(dim=-1)[suffix_mask.bool()].mean().item(),
                    "source_entropy": float(source_stats.get("entropy", z_proj.new_tensor(0.0)).detach().cpu()),
                    "source_max_prob": float(source_stats.get("max_prob", z_proj.new_tensor(0.0)).detach().cpu()),
                }
            )
    projector.train()
    if moa_source is not None:
        moa_source.train()
    return mean_dict(rows)


def projector_checkpoint(args, projector, moa_source, prompt_ckpt, latent_dim, epoch, best_ce=None):
    payload = {
        "basin_projector": projector.state_dict(),
        "stage1": args.stage1,
        "prompt_prior": args.prompt_prior,
        "prompt_prior_type": prompt_ckpt.get("type"),
        "latent_dim": latent_dim,
        "dataset": args.dataset,
        "rocstories_file": args.rocstories_file,
        "rocstories_source": args.rocstories_source,
        "rocstories_local_files_only": args.rocstories_local_files_only,
        "prompt_len": args.prompt_len,
        "suffix_len": args.suffix_len,
        "max_seq_len": args.max_seq_len,
        "target": args.target,
        "draft_drop_prob": args.draft_drop_prob,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "residual_scale": args.residual_scale,
        "ce_weight": args.ce_weight,
        "mse_weight": args.mse_weight,
        "cos_weight": args.cos_weight,
        "hidden_weight": args.hidden_weight,
        "smooth_weight": args.smooth_weight,
        "delta_weight": args.delta_weight,
        "source": args.source,
        "moa_k": args.moa_k,
        "moa_temp": args.moa_temp,
        "moa_topk": args.moa_topk,
        "moa_noise": args.moa_noise,
        "moa_mix": args.moa_mix,
        "moa_entropy_weight": args.moa_entropy_weight,
        "val_batches": args.val_batches,
        "epoch": epoch,
    }
    if moa_source is not None:
        payload["moa_source"] = moa_source.state_dict()
    if best_ce is not None:
        payload["best_projected_ce"] = best_ce
    return payload


def append_metrics(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def initialize_moa_from_targets(args, moa_source, train_loader, encoder, decoder, device):
    if moa_source is None:
        return
    targets = []
    seen = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        z_data = encode_latents(encoder, decoder, input_ids, attention_mask)
        if args.target == "real":
            target = z_data[:, args.prompt_len:]
        else:
            draft_ids, draft_mask = make_synthetic_draft_ids(
                input_ids,
                attention_mask,
                args.prompt_len,
                args.draft_drop_prob,
            )
            target = encode_latents(encoder, decoder, draft_ids, draft_mask)[:, args.prompt_len:]
        targets.append(target.detach().float().cpu())
        seen += 1
        if sum(chunk.size(0) for chunk in targets) >= args.moa_k or seen >= args.moa_init_batches:
            break
    if not targets:
        print("WARNING: MoA prototype init skipped; no target latents collected", flush=True)
        return
    values = torch.cat(targets, dim=0)
    if values.size(0) < args.moa_k:
        repeat = (args.moa_k + values.size(0) - 1) // values.size(0)
        values = values.repeat((repeat, 1, 1))
    values = values[: args.moa_k].to(device=device, dtype=moa_source.prototypes.dtype)
    slots = min(values.size(1), moa_source.prototypes.size(1))
    moa_source.prototypes.data.zero_()
    moa_source.prototypes.data[:, :slots].copy_(values[:, :slots])
    print(
        f"initialized MoA prototypes from target latents | k={args.moa_k} "
        f"batches={seen} std={moa_source.prototypes.detach().float().std().item():.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim, stage1_ckpt = load_stage1(args.stage1, device)
    prompt_prior, prompt_ckpt = load_prompt_prior(args.prompt_prior, latent_dim, device)
    resolve_slots(args, stage1_ckpt, prompt_ckpt)
    rfm.PROMPT_LEN = args.prompt_len
    rfm.MAX_SEQ_LEN = args.max_seq_len
    s2data.DATASET_NAME = args.dataset
    s2data.PROMPT_LEN = args.prompt_len
    if args.dataset == "rocstories":
        s2data.ROCSTORIES_SOURCE = args.rocstories_source
        s2data.ROCSTORIES_LOCAL_FILES_ONLY = args.rocstories_local_files_only
        if args.rocstories_file:
            s2data.ROCSTORIES_FILE = args.rocstories_file
            s2data.ROCSTORIES_SOURCE = "file"
    print(
        f"dataset={args.dataset} prompt_len={args.prompt_len} "
        f"suffix_len={args.suffix_len} max_seq_len={args.max_seq_len}",
        flush=True,
    )

    train_loader, val_loader = build_stage2_dataloaders(
        tokenizer,
        args.train_size,
        args.batch_size,
        args.max_seq_len,
    )
    projector = BasinProjector(
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        residual_scale=args.residual_scale,
    ).to(device)
    moa_source = None
    if args.source in ("moa", "prompt_prior_moa"):
        moa_source = PromptMoASource(
            latent_dim=latent_dim,
            suffix_len=args.suffix_len,
            num_archetypes=args.moa_k,
            hidden_dim=args.hidden_dim,
        ).to(device)
        initialize_moa_from_targets(args, moa_source, train_loader, encoder, decoder, device)
    params = list(projector.parameters())
    if moa_source is not None:
        params.extend(moa_source.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_ce = float("inf")
    metrics_path = Path(args.metrics_csv)
    for epoch in range(1, args.epochs + 1):
        projector.train()
        if moa_source is not None:
            moa_source.train()
        running = []
        for step, batch in enumerate(
            progress(train_loader, len(train_loader), f"basin epoch {epoch}", args.no_progress),
            start=1,
        ):
            z_prompt, z_prior, target, _z_real, suffix_ids, suffix_mask, source_stats = prepare_batch(
                args,
                encoder,
                decoder,
                prompt_prior,
                moa_source,
                batch,
                device,
            )
            pos = suffix_positions(z_prior.size(0), z_prior.size(1), device, z_prior.dtype)
            z_proj, delta = projector(z_prior, z_prompt, pos, suffix_mask)
            n = min(args.decode_batch, z_proj.size(0))
            ce, target_prob, top1, hidden_loss = decoder_stats_and_hidden_loss(
                decoder,
                z_prompt[:n],
                z_proj[:n],
                target[:n],
                suffix_ids[:n],
                suffix_mask[:n],
            )
            mse, cos = masked_losses(z_proj, target, suffix_mask)
            smooth = local_smoothness_loss(z_proj, target, suffix_mask)
            valid_delta = delta.norm(dim=-1)[suffix_mask.bool()]
            delta_loss = valid_delta.mean() if valid_delta.numel() else delta.norm(dim=-1).mean()
            loss = (
                args.ce_weight * ce.mean()
                + args.mse_weight * mse
                + args.cos_weight * (1.0 - cos)
                + args.hidden_weight * hidden_loss
                + args.smooth_weight * smooth
                + args.delta_weight * delta_loss
            )
            source_entropy = source_stats.get("entropy", z_proj.new_tensor(0.0))
            source_max_prob = source_stats.get("max_prob", z_proj.new_tensor(0.0))
            if moa_source is not None and args.moa_entropy_weight > 0:
                loss = loss - args.moa_entropy_weight * source_entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            running.append(
                {
                    "loss": loss.detach().item(),
                    "ce": ce.detach().mean().item(),
                    "p": target_prob.detach().mean().item(),
                    "top1": top1.detach().mean().item(),
                    "mse": mse.detach().item(),
                    "cos": cos.detach().item(),
                    "hidden": hidden_loss.detach().item(),
                    "smooth": smooth.detach().item(),
                    "delta": delta_loss.detach().item(),
                    "source_entropy": source_entropy.detach().item(),
                    "source_max_prob": source_max_prob.detach().item(),
                }
            )
            if args.log_every > 0 and step % args.log_every == 0:
                stats = mean_dict(running[-args.log_every:])
                print(
                    f"epoch={epoch} step={step} loss={stats['loss']:.4f} "
                    f"ce={stats['ce']:.4f} p={stats['p']:.4f} top1={stats['top1']:.4f} "
                    f"cos={stats['cos']:.4f} hid={stats['hidden']:.4f} "
                    f"sm={stats['smooth']:.4f} delta={stats['delta']:.4f} "
                    f"srcH={stats['source_entropy']:.3f} srcMax={stats['source_max_prob']:.3f}",
                    flush=True,
                )

        train_stats = mean_dict(running)
        last_output = args.last_output or args.output.replace("_best.pt", "_last.pt")
        torch.save(
            projector_checkpoint(args, projector, moa_source, prompt_ckpt, latent_dim, epoch),
            last_output,
        )
        print(f"saved {last_output} before validation", flush=True)
        val_stats = evaluate(args, encoder, decoder, prompt_prior, moa_source, projector, val_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
        }
        append_metrics(metrics_path, row)
        print(json.dumps(row, indent=2), flush=True)
        if val_stats.get("projected_ce", float("inf")) < best_ce:
            best_ce = val_stats["projected_ce"]
            torch.save(
                projector_checkpoint(args, projector, moa_source, prompt_ckpt, latent_dim, epoch, best_ce=best_ce),
                args.output,
            )
            print(f"saved {args.output} | projected_ce={best_ce:.4f}", flush=True)


if __name__ == "__main__":
    main()
