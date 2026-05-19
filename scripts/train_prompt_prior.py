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
    DENOISING_PRIOR_ALPHA,
    DENOISING_PRIOR_PATH,
    EPOCHS,
    FLOW_DEPTH,
    FLOW_HIDDEN_DIM,
    FLOW_REFINE_SCALE,
    LATENT_DIM,
    LOG_EVERY,
    MAX_SEQ_LEN,
    METRIC_HIDDEN_DIM,
    METRIC_LOG_BOUND,
    ODE_STEPS,
    PROMPT_LEN,
    ROCSTORIES_SPLIT,
    SEED,
    START_TRANSFORMER_HEADS,
    START_TRANSFORMER_HIDDEN_DIM,
    START_TRANSFORMER_LAYERS,
    TARGET_LATENT_MEAN,
    TARGET_LATENT_STD,
    TRAIN_BATCH_SIZE,
    TRAIN_SIZE,
)
from stage2_data import build_stage2_dataloaders
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss
from stage2_riemannian import DenoisingPrior, FlowNet, MetricNet, StartTransformer, natural_velocity, suffix_positions


PROMPT_PRIOR_EPOCHS = int(os.environ.get("PROMPT_PRIOR_EPOCHS", str(min(EPOCHS, 10))))
PROMPT_PRIOR_LR = float(os.environ.get("PROMPT_PRIOR_LR", "1e-4"))
PROMPT_PRIOR_CE_WEIGHT = float(os.environ.get("PROMPT_PRIOR_CE_WEIGHT", "1.0"))
PROMPT_PRIOR_MSE_WEIGHT = float(os.environ.get("PROMPT_PRIOR_MSE_WEIGHT", "0.05"))
PROMPT_PRIOR_COS_WEIGHT = float(os.environ.get("PROMPT_PRIOR_COS_WEIGHT", "0.10"))
PROMPT_PRIOR_NORM_WEIGHT = float(os.environ.get("PROMPT_PRIOR_NORM_WEIGHT", "0.01"))
PROMPT_PRIOR_DECODE_BATCH = int(os.environ.get("PROMPT_PRIOR_DECODE_BATCH", "64"))
PROMPT_PRIOR_EXAMPLES = int(os.environ.get("PROMPT_PRIOR_EXAMPLES", "16"))
PROMPT_PRIOR_EXAMPLE_DECODE = os.environ.get("PROMPT_PRIOR_EXAMPLE_DECODE", "argmax").lower()
PROMPT_PRIOR_EXAMPLE_TEMP = float(os.environ.get("PROMPT_PRIOR_EXAMPLE_TEMP", "0.9"))
PROMPT_PRIOR_EXAMPLE_TOPK = int(os.environ.get("PROMPT_PRIOR_EXAMPLE_TOPK", "50"))
PROMPT_PRIOR_EXAMPLE_TOPP = float(os.environ.get("PROMPT_PRIOR_EXAMPLE_TOPP", "0.95"))
PROMPT_PRIOR_MODE = os.environ.get("PROMPT_PRIOR_MODE", "direct").lower()
PROMPT_PRIOR_DRAFT_DROP_PROB = float(os.environ.get("PROMPT_PRIOR_DRAFT_DROP_PROB", "0.10"))
PROMPT_PRIOR_DRAFT_REPLACE_PROB = float(os.environ.get("PROMPT_PRIOR_DRAFT_REPLACE_PROB", "0.00"))
PROMPT_PRIOR_VAR_WEIGHT = float(
    os.environ.get("PROMPT_PRIOR_VAR_WEIGHT", "0.02" if PROMPT_PRIOR_MODE == "draft" else "0.0")
)
PROMPT_PRIOR_NORMALIZE_OUTPUT = os.environ.get(
    "PROMPT_PRIOR_NORMALIZE_OUTPUT",
    "false",
).lower() in ("1", "true", "yes", "on")
PROMPT_PRIOR_NORMALIZE_SOURCE = os.environ.get("PROMPT_PRIOR_NORMALIZE_SOURCE", "target").lower()
PROMPT_PRIOR_NORMALIZE_MODE = os.environ.get("PROMPT_PRIOR_NORMALIZE_MODE", "sequence").lower()
PROMPT_PRIOR_NORMALIZE_SCALE = float(os.environ.get("PROMPT_PRIOR_NORMALIZE_SCALE", "1.0"))
PROMPT_PRIOR_OUTPUT_MEAN = float(os.environ.get("PROMPT_PRIOR_OUTPUT_MEAN", str(TARGET_LATENT_MEAN)))
PROMPT_PRIOR_OUTPUT_STD = float(os.environ.get("PROMPT_PRIOR_OUTPUT_STD", str(TARGET_LATENT_STD)))
PROMPT_PRIOR_PROGRESS = os.environ.get("PROMPT_PRIOR_PROGRESS", "true").lower() in ("1", "true", "yes", "on")
PROMPT_PRIOR_MIXER = os.environ.get("PROMPT_PRIOR_MIXER", "none").lower()
PROMPT_PRIOR_MIXER_LAYERS = int(os.environ.get("PROMPT_PRIOR_MIXER_LAYERS", "2"))
PROMPT_PRIOR_MIXER_KERNEL = int(os.environ.get("PROMPT_PRIOR_MIXER_KERNEL", "5"))
PROMPT_PRIOR_MIXER_SCALE = float(os.environ.get("PROMPT_PRIOR_MIXER_SCALE", "0.5"))
PROMPT_PRIOR_STOCHASTIC = os.environ.get("PROMPT_PRIOR_STOCHASTIC", "false").lower() in ("1", "true", "yes", "on")
PROMPT_PRIOR_NOISE_STD_SCALE = float(os.environ.get("PROMPT_PRIOR_NOISE_STD_SCALE", "1.0"))
PROMPT_PRIOR_NOISE_INPUT_SCALE = float(os.environ.get("PROMPT_PRIOR_NOISE_INPUT_SCALE", "0.2"))
PROMPT_PRIOR_VAL_SAMPLES = int(os.environ.get("PROMPT_PRIOR_VAL_SAMPLES", "1"))
PROMPT_PRIOR_MEMORY = os.environ.get("PROMPT_PRIOR_MEMORY", "false").lower() in ("1", "true", "yes", "on")
PROMPT_PRIOR_MEMORY_SIZE = int(os.environ.get("PROMPT_PRIOR_MEMORY_SIZE", "512"))
PROMPT_PRIOR_MEMORY_TEMP = float(os.environ.get("PROMPT_PRIOR_MEMORY_TEMP", "0.2"))
PROMPT_PRIOR_MEMORY_SCALE = float(os.environ.get("PROMPT_PRIOR_MEMORY_SCALE", "1.0"))
PROMPT_PRIOR_MEMORY_TOPK = int(os.environ.get("PROMPT_PRIOR_MEMORY_TOPK", "0"))
PROMPT_PRIOR_MEMORY_INIT = os.environ.get(
    "PROMPT_PRIOR_MEMORY_INIT",
    "real" if PROMPT_PRIOR_MEMORY else "random",
).lower()
PROMPT_PRIOR_MEMORY_INIT_MAX_BATCHES = int(os.environ.get("PROMPT_PRIOR_MEMORY_INIT_MAX_BATCHES", "32"))
PROMPT_PRIOR_GROUP_SAMPLES = int(os.environ.get("PROMPT_PRIOR_GROUP_SAMPLES", "1"))
PROMPT_PRIOR_RANKING_WEIGHT = float(os.environ.get("PROMPT_PRIOR_RANKING_WEIGHT", "0.0"))
PROMPT_PRIOR_RANKING_TEMP = float(os.environ.get("PROMPT_PRIOR_RANKING_TEMP", "0.25"))
PROMPT_PRIOR_ADJ_REP_WEIGHT = float(os.environ.get("PROMPT_PRIOR_ADJ_REP_WEIGHT", "0.0"))
PROMPT_PRIOR_ENTROPY_WEIGHT = float(os.environ.get("PROMPT_PRIOR_ENTROPY_WEIGHT", "0.0"))
PROMPT_PRIOR_ENTROPY_FLOOR = float(os.environ.get("PROMPT_PRIOR_ENTROPY_FLOOR", "3.5"))
PROMPT_PRIOR_STAGE2 = os.environ.get("PROMPT_PRIOR_STAGE2", "")
PROMPT_PRIOR_USE_FLOW = os.environ.get("PROMPT_PRIOR_USE_FLOW", "true").lower() in ("1", "true", "yes", "on")
STAGE1_VARIANT = os.environ.get("STAGE1_VARIANT", "").strip()
PROMPT_PRIOR_VARIANT = os.environ.get("PROMPT_PRIOR_VARIANT", STAGE1_VARIANT).strip()


def variant_suffix(variant):
    return f"_{variant}" if variant else ""


def prompt_prior_default_name():
    if PROMPT_PRIOR_MODE == "pipeline":
        mode_suffix = "_pipeline"
    elif PROMPT_PRIOR_MODE == "draft":
        mode_suffix = f"_draftdrop{int(round(PROMPT_PRIOR_DRAFT_DROP_PROB * 100)):02d}"
    else:
        mode_suffix = ""
    if PROMPT_PRIOR_STOCHASTIC:
        mode_suffix = f"{mode_suffix}_stoch"
    if PROMPT_PRIOR_MEMORY:
        mode_suffix = f"{mode_suffix}_mem{PROMPT_PRIOR_MEMORY_SIZE}"
    if DATASET_NAME == "rocstories":
        return f"prompt_prior_rocstories_{LATENT_DIM}{variant_suffix(PROMPT_PRIOR_VARIANT)}{mode_suffix}_best.pt"
    return f"prompt_prior{variant_suffix(PROMPT_PRIOR_VARIANT)}{mode_suffix}_best.pt"


CHECKPOINT_PATH = os.environ.get(
    "PROMPT_PRIOR_CHECKPOINT",
    prompt_prior_default_name(),
)
EXAMPLES_PATH = os.environ.get("PROMPT_PRIOR_EXAMPLES_PATH", "prompt_prior_examples.txt")


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def progress_bar(iterable, total, desc):
    if tqdm is None or not PROMPT_PRIOR_PROGRESS:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)


def log_line(message):
    if tqdm is not None and PROMPT_PRIOR_PROGRESS:
        tqdm.write(message)
    else:
        print(message, flush=True)


def active_feature_label():
    norm = "normoff"
    if PROMPT_PRIOR_NORMALIZE_OUTPUT:
        norm = f"n{PROMPT_PRIOR_NORMALIZE_MODE[:4]}x{PROMPT_PRIOR_NORMALIZE_SCALE:g}"
    flags = [
        PROMPT_PRIOR_MODE,
        "mem" if PROMPT_PRIOR_MEMORY else "nomem",
        PROMPT_PRIOR_MIXER,
        "stoch" if PROMPT_PRIOR_STOCHASTIC else "det",
        norm,
    ]
    return " ".join(flags)


def default_stage1_checkpoint():
    if DATASET_NAME == "rocstories":
        return f"stage1_rocstories_{LATENT_DIM}{variant_suffix(STAGE1_VARIANT)}_best.pt"
    return f"stage1{variant_suffix(STAGE1_VARIANT)}_best.pt"


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
        out = self.out_proj(mixed)
        x = x + self.residual_scale * out
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
        memory_slots=32,
    ):
        super().__init__()
        self.use_noise = use_noise
        self.noise_input_scale = noise_input_scale
        self.use_memory = use_memory
        self.memory_temp = memory_temp
        self.memory_scale = memory_scale
        self.memory_topk = memory_topk
        self.base = StartTransformer(
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
        )
        self.mixer = nn.ModuleList([
            ParallelStateMixerBlock(
                latent_dim=latent_dim,
                kernel_size=mixer_kernel,
                residual_scale=mixer_scale,
            )
            for _ in range(mixer_layers)
        ])
        self.noise_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.eye_(self.noise_proj[-1].weight)
        nn.init.zeros_(self.noise_proj[-1].bias)
        self.prompt_query = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.eye_(self.prompt_query[-1].weight)
        nn.init.zeros_(self.prompt_query[-1].bias)
        self.memory_norm = nn.LayerNorm(latent_dim)
        self.memory_keys = nn.Parameter(torch.randn(memory_size, latent_dim) * 0.02)
        self.memory_values = nn.Parameter(
            torch.randn(memory_size, memory_slots, latent_dim) * (0.2 * TARGET_LATENT_STD)
            + TARGET_LATENT_MEAN
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
            z_mem = torch.einsum("bn,ntd->btd", weights, self.memory_values)
            z_mem = z_mem[:, : x.size(1), :]
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


@torch.no_grad()
def initialize_memory_from_real_latents(model, train_loader, encoder, decoder, device):
    if not PROMPT_PRIOR_MEMORY or PROMPT_PRIOR_MEMORY_INIT in ("random", "none", "off", "false", "0"):
        return
    if not isinstance(model, PromptPriorWithMixer) or not model.use_memory:
        return
    if PROMPT_PRIOR_MEMORY_INIT not in ("real", "dataset", "latents"):
        raise ValueError("PROMPT_PRIOR_MEMORY_INIT must be 'real' or 'random'")

    model.eval()
    keys = []
    values = []
    seen_batches = 0
    target_n = model.memory_values.size(0)
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            z_prompt, z_target, target_mask, _suffix_ids, _target_aux = build_prompt_prior_targets(
                encoder,
                decoder,
                input_ids,
                attention_mask,
            )
        prompt_mask = z_prompt.abs().sum(dim=-1) > 0
        denom = prompt_mask.sum(dim=1, keepdim=True).clamp_min(1).to(z_prompt.dtype)
        pooled = (z_prompt * prompt_mask.to(z_prompt.dtype).unsqueeze(-1)).sum(dim=1) / denom
        if target_mask is not None:
            z_target = z_target * target_mask.to(z_target.dtype).unsqueeze(-1)
        keys.append(pooled.detach().float().cpu())
        values.append(z_target.detach().float().cpu())
        seen_batches += 1
        if sum(chunk.size(0) for chunk in values) >= target_n or seen_batches >= PROMPT_PRIOR_MEMORY_INIT_MAX_BATCHES:
            break

    if not values:
        print("memory init skipped | no batches collected", flush=True)
        return

    key_tensor = torch.cat(keys, dim=0)
    value_tensor = torch.cat(values, dim=0)
    if value_tensor.size(0) < target_n:
        repeat = (target_n + value_tensor.size(0) - 1) // value_tensor.size(0)
        key_tensor = key_tensor.repeat((repeat, 1))
        value_tensor = value_tensor.repeat((repeat, 1, 1))
    key_tensor = key_tensor[:target_n].to(device=device, dtype=model.memory_keys.dtype)
    value_tensor = value_tensor[:target_n].to(device=device, dtype=model.memory_values.dtype)
    slots = min(value_tensor.size(1), model.memory_values.size(1))
    model.memory_keys.data.copy_(key_tensor)
    model.memory_values.data.zero_()
    model.memory_values.data[:, :slots, :].copy_(value_tensor[:, :slots, :])
    print(
        f"memory initialized from real latents | entries={target_n} batches={seen_batches} "
        f"value_std={model.memory_values.detach().float().std().item():.4f}",
        flush=True,
    )
    model.train()


def build_prompt_prior_model():
    if (
        PROMPT_PRIOR_MIXER in ("none", "off", "false", "0")
        and not PROMPT_PRIOR_STOCHASTIC
        and not PROMPT_PRIOR_MEMORY
    ):
        return StartTransformer(
            latent_dim=LATENT_DIM,
            num_layers=START_TRANSFORMER_LAYERS,
            num_heads=START_TRANSFORMER_HEADS,
            ffn_dim=START_TRANSFORMER_HIDDEN_DIM,
        )
    if PROMPT_PRIOR_MIXER not in ("none", "off", "false", "0", "mamba", "ssm", "state"):
        raise ValueError("PROMPT_PRIOR_MIXER must be 'none', 'mamba', 'ssm', or 'state'")
    mixer_layers = 0 if PROMPT_PRIOR_MIXER in ("none", "off", "false", "0") else PROMPT_PRIOR_MIXER_LAYERS
    return PromptPriorWithMixer(
        latent_dim=LATENT_DIM,
        num_layers=START_TRANSFORMER_LAYERS,
        num_heads=START_TRANSFORMER_HEADS,
        ffn_dim=START_TRANSFORMER_HIDDEN_DIM,
        mixer_layers=mixer_layers,
        mixer_kernel=PROMPT_PRIOR_MIXER_KERNEL,
        mixer_scale=PROMPT_PRIOR_MIXER_SCALE,
        use_noise=PROMPT_PRIOR_STOCHASTIC,
        noise_input_scale=PROMPT_PRIOR_NOISE_INPUT_SCALE,
        use_memory=PROMPT_PRIOR_MEMORY,
        memory_size=PROMPT_PRIOR_MEMORY_SIZE,
        memory_temp=PROMPT_PRIOR_MEMORY_TEMP,
        memory_scale=PROMPT_PRIOR_MEMORY_SCALE,
        memory_topk=PROMPT_PRIOR_MEMORY_TOPK,
        memory_slots=MAX_SEQ_LEN - PROMPT_LEN,
    )


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def encode_latents(encoder, decoder, input_ids, attention_mask):
    with torch.no_grad():
        return decoder.compress(encoder(input_ids, attention_mask))


def valid_mse(pred, target, mask):
    if mask is not None and mask.bool().any():
        valid = mask.bool()
        return F.mse_loss(pred[valid], target[valid].detach())
    return F.mse_loss(pred, target.detach())


def norm_gap_loss(pred, target, mask):
    pred_norm = pred.norm(dim=-1)
    target_norm = target.detach().norm(dim=-1)
    if mask is not None and mask.bool().any():
        valid = mask.bool()
        return F.smooth_l1_loss(pred_norm[valid], target_norm[valid])
    return F.smooth_l1_loss(pred_norm, target_norm)


def variance_floor_loss(pred, target, mask):
    if mask is not None and mask.bool().any():
        valid = mask.bool()
        pred_valid = pred[valid]
        target_valid = target.detach()[valid]
    else:
        pred_valid = pred.reshape(-1, pred.size(-1))
        target_valid = target.detach().reshape(-1, target.size(-1))
    pred_std = pred_valid.float().std(dim=0).mean()
    target_std = target_valid.float().std(dim=0).mean().detach()
    floor = 0.75 * target_std
    return F.relu(floor - pred_std).pow(2), pred_std.detach().item(), target_std.detach().item()


def masked_sequence_stats(z, mask):
    if mask is None:
        zf = z.float()
        mean = zf.mean(dim=(1, 2), keepdim=True)
        std = zf.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return mean.to(z.dtype), std.to(z.dtype)

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
        mean = zf.mean(dim=1, keepdim=True)
        std = zf.std(dim=1, keepdim=True).clamp_min(1e-5)
        return mean.to(z.dtype), std.to(z.dtype)

    valid = mask.to(z.dtype).unsqueeze(-1)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    zf = z.float()
    valid_f = valid.float()
    mean = (zf * valid_f).sum(dim=1, keepdim=True) / denom.float()
    var = (((zf - mean) * valid_f).pow(2).sum(dim=1, keepdim=True) / denom.float()).clamp_min(1e-10)
    return mean.to(z.dtype), var.sqrt().to(z.dtype)


def normalize_prompt_output(pred, target, mask):
    if not PROMPT_PRIOR_NORMALIZE_OUTPUT:
        return pred

    if PROMPT_PRIOR_NORMALIZE_MODE == "sequence":
        stats_fn = masked_sequence_stats
    elif PROMPT_PRIOR_NORMALIZE_MODE == "feature":
        stats_fn = masked_feature_stats
    else:
        raise ValueError("PROMPT_PRIOR_NORMALIZE_MODE must be 'sequence' or 'feature'")

    pred_mean, pred_std = stats_fn(pred, mask)
    pred_norm = (pred - pred_mean) / pred_std

    if PROMPT_PRIOR_NORMALIZE_SOURCE == "target":
        target_mean, target_std = stats_fn(target.detach(), mask)
    elif PROMPT_PRIOR_NORMALIZE_SOURCE == "global":
        shape = (pred.size(0), 1, 1)
        target_mean = pred.new_full(shape, PROMPT_PRIOR_OUTPUT_MEAN)
        target_std = pred.new_full(shape, PROMPT_PRIOR_OUTPUT_STD)
    else:
        raise ValueError("PROMPT_PRIOR_NORMALIZE_SOURCE must be 'target' or 'global'")

    out = pred_norm * (PROMPT_PRIOR_NORMALIZE_SCALE * target_std.clamp_min(1e-5)) + target_mean
    if mask is not None:
        out = out * mask.to(out.dtype).unsqueeze(-1)
    return out


def decode_loss(decoder, z_prompt, z_suffix, suffix_ids, target_mask):
    n = min(PROMPT_PRIOR_DECODE_BATCH, z_suffix.size(0))
    logits = decoder.decode_from_latent(torch.cat([z_prompt[:n], z_suffix[:n]], dim=1))
    ce, target_prob, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids[:n],
        target_mask[:n] if target_mask is not None else None,
    )
    return ce, target_prob, top1


def load_frozen_draft_prior(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    prior = DenoisingPrior(
        latent_dim=ckpt.get("latent_dim", LATENT_DIM),
        hidden_dim=ckpt.get("denoising_hidden_dim", ckpt.get("start_transformer_hidden_dim", START_TRANSFORMER_HIDDEN_DIM)),
        num_layers=ckpt.get("denoising_layers", ckpt.get("start_transformer_layers", START_TRANSFORMER_LAYERS)),
        num_heads=ckpt.get("denoising_heads", ckpt.get("start_transformer_heads", START_TRANSFORMER_HEADS)),
    ).to(device)
    state = ckpt.get("denoising_prior", ckpt.get("draft_prior"))
    if state is None:
        raise RuntimeError(f"No denoising_prior/draft_prior state found in {path}")
    prior.load_state_dict(state)
    freeze(prior)
    alpha = float(ckpt.get("draft_alpha", ckpt.get("denoising_prior_alpha", DENOISING_PRIOR_ALPHA)))
    print(f"frozen DraftPrior loaded from {path} | alpha={alpha:.3f}", flush=True)
    return prior, alpha


def load_frozen_stage2_flow(path, device):
    if not path:
        return None, None, FLOW_REFINE_SCALE
    ckpt = torch.load(path, map_location=device, weights_only=False)
    flow_net = FlowNet(
        latent_dim=ckpt.get("latent_dim", LATENT_DIM),
        hidden_dim=ckpt.get("flow_hidden_dim", FLOW_HIDDEN_DIM),
        depth=ckpt.get("flow_depth", FLOW_DEPTH),
    ).to(device)
    metric_net = MetricNet(
        latent_dim=ckpt.get("latent_dim", LATENT_DIM),
        hidden_dim=ckpt.get("metric_hidden_dim", METRIC_HIDDEN_DIM),
        log_bound=ckpt.get("metric_log_bound", METRIC_LOG_BOUND),
    ).to(device)
    flow_state = {key.replace("_orig_mod.", ""): value for key, value in ckpt["flow_net"].items()}
    metric_state = {key.replace("_orig_mod.", ""): value for key, value in ckpt["metric_net"].items()}
    flow_net.load_state_dict(flow_state, strict=False)
    metric_net.load_state_dict(metric_state, strict=False)
    freeze(flow_net)
    freeze(metric_net)
    refine_scale = float(ckpt.get("flow_refine_scale", FLOW_REFINE_SCALE))
    print(f"frozen Stage2 flow loaded from {path} | refine_scale={refine_scale:.4f}", flush=True)
    return flow_net, metric_net, refine_scale


def apply_frozen_draft_prior(prior, z_draft, z_prompt, pos, target_mask, alpha):
    beta = max(0.0, 1.0 - alpha * alpha) ** 0.5
    noise = torch.randn_like(z_draft) * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    z_t = alpha * z_draft + beta * noise
    if target_mask is not None:
        z_t = z_t * target_mask.to(z_t.dtype).unsqueeze(-1)
    alpha_t = z_prompt.new_full((z_prompt.size(0),), alpha)
    return prior(z_t, z_prompt, alpha_t, pos, target_mask)


def apply_frozen_flow(flow_net, metric_net, z_start, z_prompt, target_mask, refine_scale):
    if flow_net is None or metric_net is None:
        return z_start
    z = z_start
    B, T, _D = z.shape
    pos = suffix_positions(B, T, z.device, z.dtype)
    dt = 1.0 / ODE_STEPS
    for step in range(ODE_STEPS):
        t = torch.full((B, T), step / ODE_STEPS, device=z.device, dtype=z.dtype)
        v, _g = natural_velocity(flow_net, metric_net, z, t, z_prompt, pos)
        z = z + refine_scale * v * dt
        if target_mask is not None:
            z = z * target_mask.to(z.dtype).unsqueeze(-1)
    return z


def apply_prompt_pipeline(z_direct, z_prompt, target_mask, pos, draft_prior, draft_alpha, flow_net, metric_net, refine_scale):
    if draft_prior is None:
        return z_direct, z_direct
    z_prior = apply_frozen_draft_prior(draft_prior, z_direct, z_prompt, pos, target_mask, draft_alpha)
    z_final = apply_frozen_flow(flow_net, metric_net, z_prior, z_prompt, target_mask, refine_scale)
    return z_prior, z_final


def compute_stats_and_loss(decoder, z_prompt, pred, z_target, suffix_ids, target_mask):
    ce, target_prob, top1 = decode_loss(decoder, z_prompt, pred, suffix_ids, target_mask)
    mse = valid_mse(pred, z_target, target_mask)
    cos_loss, cos_val = rollout_cosine_alignment_loss(pred, z_target, target_mask)
    nloss = norm_gap_loss(pred, z_target, target_mask)
    vloss, pred_std, target_std = variance_floor_loss(pred, z_target, target_mask)
    adj_rep = pred.new_tensor(0.0)
    entropy_floor = pred.new_tensor(0.0)
    entropy_mean = 0.0
    if PROMPT_PRIOR_ADJ_REP_WEIGHT > 0 or PROMPT_PRIOR_ENTROPY_WEIGHT > 0:
        n = min(PROMPT_PRIOR_DECODE_BATCH, pred.size(0))
        logits = decoder.decode_from_latent(torch.cat([z_prompt[:n], pred[:n]], dim=1))[:, PROMPT_LEN:]
        probs = logits.softmax(dim=-1)
        if target_mask is not None:
            mask = target_mask[:n].bool()
        else:
            mask = torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
        if PROMPT_PRIOR_ADJ_REP_WEIGHT > 0 and logits.size(1) > 1:
            adjacent_same = (probs[:, :-1] * probs[:, 1:]).sum(dim=-1)
            pair_mask = mask[:, :-1] & mask[:, 1:]
            if pair_mask.any():
                adj_rep = adjacent_same[pair_mask].mean()
            else:
                adj_rep = adjacent_same.mean()
        if PROMPT_PRIOR_ENTROPY_WEIGHT > 0:
            entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
            if mask.any():
                entropy_valid = entropy[mask]
            else:
                entropy_valid = entropy.reshape(-1)
            entropy_mean_tensor = entropy_valid.mean()
            entropy_mean = entropy_mean_tensor.detach().item()
            entropy_floor = F.relu(PROMPT_PRIOR_ENTROPY_FLOOR - entropy_valid).mean()
    loss = (
        PROMPT_PRIOR_CE_WEIGHT * ce
        + PROMPT_PRIOR_MSE_WEIGHT * mse
        + PROMPT_PRIOR_COS_WEIGHT * cos_loss
        + PROMPT_PRIOR_NORM_WEIGHT * nloss
        + PROMPT_PRIOR_VAR_WEIGHT * vloss
        + PROMPT_PRIOR_ADJ_REP_WEIGHT * adj_rep
        + PROMPT_PRIOR_ENTROPY_WEIGHT * entropy_floor
    )
    return loss, {
        "ce": ce.detach().item(),
        "p": target_prob,
        "top1": top1,
        "mse": mse.detach().item(),
        "cos": cos_val,
        "norm": nloss.detach().item(),
        "var": vloss.detach().item(),
        "adj_rep": adj_rep.detach().item(),
        "entropy_floor": entropy_floor.detach().item(),
        "entropy": entropy_mean,
        "pred_std": pred_std,
        "target_std": target_std,
    }


@torch.no_grad()
def local_output_reward(decoder, z_prompt, z_suffix, target_mask):
    n = min(PROMPT_PRIOR_DECODE_BATCH, z_suffix.size(0))
    logits = decoder.decode_from_latent(torch.cat([z_prompt[:n], z_suffix[:n]], dim=1))
    pred_ids = logits.argmax(dim=-1)[:, PROMPT_LEN:]
    masks = target_mask[:n].bool() if target_mask is not None else torch.ones_like(pred_ids, dtype=torch.bool)
    rewards = []
    for row, mask in zip(pred_ids, masks):
        ids = row[mask].tolist()
        ids = [int(tok) for tok in ids if int(tok) not in SPECIAL_IDS]
        if not ids:
            rewards.append(logits.new_tensor(-3.0))
            continue
        counts = {}
        punct = 0
        for tok in ids:
            counts[tok] = counts.get(tok, 0) + 1
            if tok in PUNCT_IDS:
                punct += 1
        length = len(ids)
        unique_ratio = len(counts) / max(length, 1)
        max_frac = max(counts.values()) / max(length, 1)
        punct_frac = punct / max(length, 1)
        length_ok = min(length / max(int(mask.sum().item()), 1), 1.0)
        reward = unique_ratio + 0.25 * length_ok - 1.5 * max_frac - 2.0 * punct_frac
        rewards.append(logits.new_tensor(reward))
    return torch.stack(rewards).mean()


def merge_sample_stats(stats_list, weights=None):
    keys = (
        "ce", "p", "top1", "mse", "cos", "norm", "var",
        "adj_rep", "entropy_floor", "entropy", "pred_std", "target_std",
    )
    if weights is None:
        inv = 1.0 / max(len(stats_list), 1)
        return {
            key: sum(stats[key] for stats in stats_list) * inv
            for key in keys
            if key in stats_list[0]
        }
    return {
        key: sum(float(w) * stats[key] for w, stats in zip(weights, stats_list))
        for key in keys
        if key in stats_list[0]
    }


def random_suffix_like(z_target, target_mask):
    z = torch.randn_like(z_target) * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    if target_mask is not None:
        z = z * target_mask.to(z.dtype).unsqueeze(-1)
    return z


def sample_prompt_noise(z_target, target_mask):
    z = (
        torch.randn_like(z_target)
        * (PROMPT_PRIOR_NOISE_STD_SCALE * TARGET_LATENT_STD)
        + TARGET_LATENT_MEAN
    )
    if target_mask is not None:
        z = z * target_mask.to(z.dtype).unsqueeze(-1)
    return z


def run_prompt_prior(model, z_prompt, pos, target_mask, z_target):
    if PROMPT_PRIOR_STOCHASTIC:
        z_init = sample_prompt_noise(z_target, target_mask)
        return model(z_prompt, pos, target_mask, z_init=z_init)
    return model(z_prompt, pos, target_mask)


def _is_important_token(token):
    clean = token[2:] if token.startswith("##") else token
    if clean in PUNCT_TOKENS:
        return True
    if any(ch.isdigit() for ch in clean):
        return True
    if len(clean) >= 6 and clean.isalpha():
        return True
    return False


def make_rough_draft_target_ids(input_ids, attention_mask, drop_prob, replace_prob):
    draft = input_ids.clone()
    B = input_ids.size(0)
    suffix_len = input_ids.size(1) - PROMPT_LEN

    for b in range(B):
        suffix_ids = input_ids[b, PROMPT_LEN:].tolist()
        suffix_mask = attention_mask[b, PROMPT_LEN:].tolist()
        kept = []
        for tok_id, tok_mask in zip(suffix_ids, suffix_mask):
            if tok_mask == 0 or tok_id in SPECIAL_IDS:
                continue
            token = tokenizer.convert_ids_to_tokens(int(tok_id))
            important = _is_important_token(token)
            if (not important) and random.random() < drop_prob:
                continue
            if (
                (not important)
                and COMMON_TOKEN_IDS
                and replace_prob > 0
                and token not in PUNCT_TOKENS
                and random.random() < replace_prob
            ):
                kept.append(random.choice(COMMON_TOKEN_IDS))
            else:
                kept.append(int(tok_id))

        kept = kept[:suffix_len]
        noisy_suffix = kept + [PAD_ID] * (suffix_len - len(kept))
        draft[b, PROMPT_LEN:] = torch.tensor(noisy_suffix, device=input_ids.device, dtype=input_ids.dtype)

    draft_mask = (draft != PAD_ID).to(attention_mask.dtype)
    draft[:, 0] = CLS_ID
    draft_mask[:, :PROMPT_LEN] = attention_mask[:, :PROMPT_LEN]
    return draft, draft_mask


def build_prompt_prior_targets(encoder, decoder, input_ids, attention_mask):
    z_data = encode_latents(encoder, decoder, input_ids, attention_mask)
    z_prompt = z_data[:, :PROMPT_LEN, :]
    aux = {}
    if PROMPT_PRIOR_MODE == "draft":
        draft_ids, draft_mask = make_rough_draft_target_ids(
            input_ids,
            attention_mask,
            PROMPT_PRIOR_DRAFT_DROP_PROB,
            PROMPT_PRIOR_DRAFT_REPLACE_PROB,
        )
        z_draft = encode_latents(encoder, decoder, draft_ids, draft_mask)
        aux["draft_ids"] = draft_ids
        aux["draft_mask"] = draft_mask
        return (
            z_prompt,
            z_draft[:, PROMPT_LEN:, :],
            draft_mask[:, PROMPT_LEN:],
            draft_ids[:, PROMPT_LEN:],
            aux,
        )
    return (
        z_prompt,
        z_data[:, PROMPT_LEN:, :],
        attention_mask[:, PROMPT_LEN:],
        input_ids[:, PROMPT_LEN:],
        aux,
    )


def add_stats(total, stats):
    for key, value in stats.items():
        if key in total:
            total[key] += value
    total["n"] += 1


def new_stats_total():
    return {
        "ce": 0.0,
        "p": 0.0,
        "top1": 0.0,
        "mse": 0.0,
        "cos": 0.0,
        "norm": 0.0,
        "var": 0.0,
        "adj_rep": 0.0,
        "entropy_floor": 0.0,
        "entropy": 0.0,
        "pred_std": 0.0,
        "target_std": 0.0,
        "n": 0,
    }


def mean_stats(total):
    n = max(total["n"], 1)
    return {
        key: total[key] / n
        for key in (
            "ce", "p", "top1", "mse", "cos", "norm", "var",
            "adj_rep", "entropy_floor", "entropy", "pred_std", "target_std",
        )
    }


def sample_latent_diversity(samples, mask):
    if len(samples) < 2:
        return 0.0
    vals = []
    valid = mask.bool() if mask is not None else None
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a = samples[i].float()
            b = samples[j].float()
            dist = (a - b).pow(2).sum(dim=-1).sqrt()
            if valid is not None and valid.any():
                vals.append(dist[valid].mean())
            else:
                vals.append(dist.mean())
    if not vals:
        return 0.0
    return torch.stack(vals).mean().detach().item()


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
        cum = probs.cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, sorted_idx, sorted_scores)
    probs = scores.softmax(dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    return torch.multinomial(flat, 1).reshape(logits.shape[:-1])


def decode_suffix(tokenizer, decoder, z_prompt, z_suffix):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))
    if PROMPT_PRIOR_EXAMPLE_DECODE in ("sample", "sampling", "topk", "topp"):
        pred_ids = sample_token_ids(
            logits,
            temperature=PROMPT_PRIOR_EXAMPLE_TEMP,
            top_k=PROMPT_PRIOR_EXAMPLE_TOPK,
            top_p=PROMPT_PRIOR_EXAMPLE_TOPP,
        )
    elif PROMPT_PRIOR_EXAMPLE_DECODE in ("argmax", "greedy"):
        pred_ids = logits.argmax(dim=-1)
    else:
        raise ValueError("PROMPT_PRIOR_EXAMPLE_DECODE must be 'argmax' or 'sample'")
    return [
        tokenizer.decode(pred_ids[i, PROMPT_LEN:], skip_special_tokens=True).strip()
        for i in range(pred_ids.size(0))
    ]


@torch.no_grad()
def write_examples(
    tokenizer,
    encoder,
    decoder,
    model,
    val_loader,
    device,
    epoch,
    draft_prior=None,
    draft_alpha=DENOISING_PRIOR_ALPHA,
    flow_net=None,
    metric_net=None,
    refine_scale=FLOW_REFINE_SCALE,
):
    model.eval()
    rows = []
    made = 0
    example_samples = max(1, min(PROMPT_PRIOR_VAL_SAMPLES, 4)) if PROMPT_PRIOR_STOCHASTIC else 1
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        z_prompt, z_target, target_mask, _suffix_ids, target_aux = build_prompt_prior_targets(
            encoder,
            decoder,
            input_ids,
            attention_mask,
        )
        pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
        n = min(PROMPT_PRIOR_EXAMPLES - made, input_ids.size(0))
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            prior_texts_by_sample = []
            final_texts_by_sample = []
            for _sample_idx in range(example_samples):
                pred = run_prompt_prior(model, z_prompt, pos, target_mask, z_target)
                pred = normalize_prompt_output(pred, z_target, target_mask)
                _z_prior, z_final = apply_prompt_pipeline(
                    pred,
                    z_prompt,
                    target_mask,
                    pos,
                    draft_prior,
                    draft_alpha,
                    flow_net,
                    metric_net,
                    refine_scale,
                )
                prior_texts_by_sample.append(decode_suffix(tokenizer, decoder, z_prompt[:n], pred[:n]))
                final_texts_by_sample.append(decode_suffix(tokenizer, decoder, z_prompt[:n], z_final[:n]))
            gauss = random_suffix_like(z_target, target_mask)
            _gauss_prior, gauss_final = apply_prompt_pipeline(
                gauss,
                z_prompt,
                target_mask,
                pos,
                draft_prior,
                draft_alpha,
                flow_net,
                metric_net,
                refine_scale,
            )
            gauss_texts = decode_suffix(tokenizer, decoder, z_prompt[:n], gauss_final[:n])
            target_texts = [
                tokenizer.decode(input_ids[i, PROMPT_LEN:], skip_special_tokens=True).strip()
                for i in range(n)
            ]
            draft_target_ids = target_aux.get("draft_ids", input_ids)
            draft_target_texts = [
                tokenizer.decode(draft_target_ids[i, PROMPT_LEN:], skip_special_tokens=True).strip()
                for i in range(n)
            ]
            prompt_texts = [
                tokenizer.decode(input_ids[i, :PROMPT_LEN], skip_special_tokens=True).strip()
                for i in range(n)
            ]

        for i in range(n):
            sample_lines = []
            for sample_idx in range(example_samples):
                sample_lines.append(
                    f"prompt prior sample {sample_idx + 1}: {prior_texts_by_sample[sample_idx][i]}\n"
                    f"pipeline sample {sample_idx + 1}: {final_texts_by_sample[sample_idx][i]}"
                )
            rows.append(
                f"--- example {made + 1} epoch {epoch + 1}\n"
                f"prompt: {prompt_texts[i]}\n"
                f"target: {target_texts[i]}\n"
                f"draft target: {draft_target_texts[i]}\n"
                f"{chr(10).join(sample_lines)}\n"
                f"gaussian: {gauss_texts[i]}\n"
            )
            made += 1
            if made >= PROMPT_PRIOR_EXAMPLES:
                Path(EXAMPLES_PATH).write_text("\n".join(rows), encoding="utf-8")
                print(f"saved {made} examples to {EXAMPLES_PATH}", flush=True)
                return


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "prompt-prior config | "
    f"dataset={DATASET_NAME} split={ROCSTORIES_SPLIT if DATASET_NAME == 'rocstories' else 'legacy'} "
    f"prompt_slots={PROMPT_LEN} max_seq={MAX_SEQ_LEN} latent_dim={LATENT_DIM} "
    f"train_size={TRAIN_SIZE} batch={TRAIN_BATCH_SIZE} epochs={PROMPT_PRIOR_EPOCHS} seed={SEED} "
    f"layers={START_TRANSFORMER_LAYERS} heads={START_TRANSFORMER_HEADS} "
    f"hidden={START_TRANSFORMER_HIDDEN_DIM} lr={PROMPT_PRIOR_LR} "
    f"loss ce={PROMPT_PRIOR_CE_WEIGHT} mse={PROMPT_PRIOR_MSE_WEIGHT} "
    f"cos={PROMPT_PRIOR_COS_WEIGHT} norm={PROMPT_PRIOR_NORM_WEIGHT} "
    f"var={PROMPT_PRIOR_VAR_WEIGHT} adj_rep={PROMPT_PRIOR_ADJ_REP_WEIGHT} "
    f"entropy={PROMPT_PRIOR_ENTROPY_WEIGHT}@{PROMPT_PRIOR_ENTROPY_FLOOR:.2f} "
    f"mode={PROMPT_PRIOR_MODE} use_flow={PROMPT_PRIOR_USE_FLOW} "
    f"draft_drop={PROMPT_PRIOR_DRAFT_DROP_PROB:.2f} draft_repl={PROMPT_PRIOR_DRAFT_REPLACE_PROB:.2f} "
    f"normalize={PROMPT_PRIOR_NORMALIZE_OUTPUT} norm_source={PROMPT_PRIOR_NORMALIZE_SOURCE} "
    f"norm_mode={PROMPT_PRIOR_NORMALIZE_MODE} norm_scale={PROMPT_PRIOR_NORMALIZE_SCALE:.3f} "
    f"global_stats=({PROMPT_PRIOR_OUTPUT_MEAN:.4f},{PROMPT_PRIOR_OUTPUT_STD:.4f}) "
    f"progress={PROMPT_PRIOR_PROGRESS and tqdm is not None} "
    f"mixer={PROMPT_PRIOR_MIXER} mixer_layers={PROMPT_PRIOR_MIXER_LAYERS} "
    f"mixer_kernel={PROMPT_PRIOR_MIXER_KERNEL} mixer_scale={PROMPT_PRIOR_MIXER_SCALE:.3f} "
    f"stochastic={PROMPT_PRIOR_STOCHASTIC} noise_std_scale={PROMPT_PRIOR_NOISE_STD_SCALE:.3f} "
    f"noise_input_scale={PROMPT_PRIOR_NOISE_INPUT_SCALE:.3f} val_samples={PROMPT_PRIOR_VAL_SAMPLES} "
    f"example_decode={PROMPT_PRIOR_EXAMPLE_DECODE} temp={PROMPT_PRIOR_EXAMPLE_TEMP:.2f} "
    f"topk={PROMPT_PRIOR_EXAMPLE_TOPK} topp={PROMPT_PRIOR_EXAMPLE_TOPP:.2f} "
    f"memory={PROMPT_PRIOR_MEMORY} memory_size={PROMPT_PRIOR_MEMORY_SIZE} "
    f"memory_temp={PROMPT_PRIOR_MEMORY_TEMP:.3f} memory_scale={PROMPT_PRIOR_MEMORY_SCALE:.3f} "
    f"memory_topk={PROMPT_PRIOR_MEMORY_TOPK} memory_init={PROMPT_PRIOR_MEMORY_INIT}",
    flush=True,
)

encoder = BertEncoder().to(device)
decoder = ParallelDecoder(latent_dim=LATENT_DIM).to(device)
stage1_checkpoint = os.environ.get("STAGE1_CHECKPOINT", default_stage1_checkpoint())
ckpt1 = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
decoder.load_state_dict(ckpt1["decoder"])
if "encoder" in ckpt1:
    encoder.load_state_dict(ckpt1["encoder"])
freeze(encoder)
freeze(decoder)
print(f"stage1 loaded from {stage1_checkpoint} | encoder + decoder frozen", flush=True)

tokenizer = cached_from_pretrained(BertTokenizer)
train_loader, val_loader = build_stage2_dataloaders(
    tokenizer,
    train_size=TRAIN_SIZE,
    batch_size=TRAIN_BATCH_SIZE,
    max_length=MAX_SEQ_LEN,
)
common_words = [
    "the", "a", "of", "and", "in", "to", "is", "was", "for", "with",
    "on", "as", "by", "from", "that", "this", "it", "are", "were", "be",
]
COMMON_TOKEN_IDS = [
    tokenizer.convert_tokens_to_ids(tok)
    for tok in common_words
    if tokenizer.convert_tokens_to_ids(tok) != tokenizer.unk_token_id
]
PAD_ID = tokenizer.pad_token_id
CLS_ID = tokenizer.cls_token_id
SPECIAL_IDS = set(tokenizer.all_special_ids)
PUNCT_TOKENS = {".", ",", ";", ":", "!", "?", "-", "(", ")", "'", '"'}
PUNCT_IDS = {
    tokenizer.convert_tokens_to_ids(tok)
    for tok in PUNCT_TOKENS
    if tokenizer.convert_tokens_to_ids(tok) != tokenizer.unk_token_id
}

model = build_prompt_prior_model().to(device)
if PROMPT_PRIOR_MODE not in ("direct", "pipeline", "draft"):
    raise ValueError("PROMPT_PRIOR_MODE must be 'direct', 'pipeline', or 'draft'")
initialize_memory_from_real_latents(model, train_loader, encoder, decoder, device)
draft_prior = None
draft_alpha = DENOISING_PRIOR_ALPHA
flow_net = None
metric_net = None
flow_refine_scale = FLOW_REFINE_SCALE
if PROMPT_PRIOR_MODE == "pipeline":
    draft_prior, draft_alpha = load_frozen_draft_prior(DENOISING_PRIOR_PATH, device)
    if PROMPT_PRIOR_USE_FLOW:
        if not PROMPT_PRIOR_STAGE2:
            raise RuntimeError("PROMPT_PRIOR_MODE=pipeline with PROMPT_PRIOR_USE_FLOW=true requires PROMPT_PRIOR_STAGE2")
        flow_net, metric_net, flow_refine_scale = load_frozen_stage2_flow(PROMPT_PRIOR_STAGE2, device)
optimizer = AdamW(model.parameters(), lr=PROMPT_PRIOR_LR)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
best_val_ce = float("inf")
print(f"PromptPrior params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
print(
    "ACTIVE PROMPT PRIOR | "
    f"{active_feature_label()} | "
    f"memory_size={PROMPT_PRIOR_MEMORY_SIZE} memory_temp={PROMPT_PRIOR_MEMORY_TEMP:.3f} "
    f"memory_scale={PROMPT_PRIOR_MEMORY_SCALE:.3f} memory_topk={PROMPT_PRIOR_MEMORY_TOPK} "
    f"memory_init={PROMPT_PRIOR_MEMORY_INIT} | "
    f"noise_input={PROMPT_PRIOR_NOISE_INPUT_SCALE:.3f} val_samples={PROMPT_PRIOR_VAL_SAMPLES} | "
    f"group_samples={PROMPT_PRIOR_GROUP_SAMPLES} ranking_weight={PROMPT_PRIOR_RANKING_WEIGHT:.3f} "
    f"ranking_temp={PROMPT_PRIOR_RANKING_TEMP:.3f} | "
    f"adj_rep_weight={PROMPT_PRIOR_ADJ_REP_WEIGHT:.3f} "
    f"entropy_weight={PROMPT_PRIOR_ENTROPY_WEIGHT:.3f} entropy_floor={PROMPT_PRIOR_ENTROPY_FLOOR:.2f} | "
    f"seed={SEED} checkpoint={CHECKPOINT_PATH}",
    flush=True,
)

for epoch in range(PROMPT_PRIOR_EPOCHS):
    model.train()
    train_total = {
        "ce": 0.0,
        "p": 0.0,
        "top1": 0.0,
        "mse": 0.0,
        "cos": 0.0,
        "norm": 0.0,
        "var": 0.0,
        "adj_rep": 0.0,
        "entropy_floor": 0.0,
        "entropy": 0.0,
        "pred_std": 0.0,
        "target_std": 0.0,
        "n": 0,
    }
    train_loss = 0.0

    train_iter = progress_bar(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"ep{epoch + 1}/{PROMPT_PRIOR_EPOCHS} train {active_feature_label()}",
    )
    for step, batch in train_iter:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            z_prompt, z_target, target_mask, suffix_ids, _target_aux = build_prompt_prior_targets(
                encoder,
                decoder,
                input_ids,
                attention_mask,
            )
            pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
            group_k = max(1, PROMPT_PRIOR_GROUP_SAMPLES if PROMPT_PRIOR_RANKING_WEIGHT > 0 else 1)
            sample_losses = []
            sample_stats = []
            sample_rewards = []
            for _sample_idx in range(group_k):
                pred = run_prompt_prior(model, z_prompt, pos, target_mask, z_target)
                pred = normalize_prompt_output(pred, z_target, target_mask)
                _z_prior, z_for_loss = apply_prompt_pipeline(
                    pred,
                    z_prompt,
                    target_mask,
                    pos,
                    draft_prior,
                    draft_alpha,
                    flow_net,
                    metric_net,
                    flow_refine_scale,
                )
                sample_loss, one_stats = compute_stats_and_loss(
                    decoder, z_prompt, z_for_loss, z_target, suffix_ids, target_mask
                )
                sample_losses.append(sample_loss)
                sample_stats.append(one_stats)
                sample_rewards.append(local_output_reward(decoder, z_prompt, z_for_loss, target_mask))
            if group_k > 1:
                reward_tensor = torch.stack(sample_rewards)
                weights = torch.softmax(reward_tensor / max(PROMPT_PRIOR_RANKING_TEMP, 1e-4), dim=0).detach()
                weighted_loss = sum(weight * sample_loss for weight, sample_loss in zip(weights, sample_losses))
                supervised_mean = torch.stack(sample_losses).mean()
                loss = (1.0 - PROMPT_PRIOR_RANKING_WEIGHT) * supervised_mean + PROMPT_PRIOR_RANKING_WEIGHT * weighted_loss
                stats = merge_sample_stats(sample_stats, weights.detach().float().cpu().tolist())
                stats["rank_reward"] = reward_tensor.mean().detach().item()
                stats["rank_best_reward"] = reward_tensor.max().detach().item()
                stats["rank_weight_max"] = weights.max().detach().item()
            else:
                loss = sample_losses[0]
                stats = sample_stats[0]
            if PROMPT_PRIOR_MODE == "pipeline":
                _direct_loss, direct_stats = compute_stats_and_loss(
                    decoder, z_prompt, pred, z_target, suffix_ids, target_mask
                )
                stats["direct_ce"] = direct_stats["ce"]
                stats["direct_p"] = direct_stats["p"]
                stats["direct_top1"] = direct_stats["top1"]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        add_stats(train_total, stats)
        if tqdm is not None and PROMPT_PRIOR_PROGRESS:
            train_iter.set_postfix(
                loss=f"{loss.item():.3f}",
                ce=f"{stats['ce']:.3f}",
                p=f"{stats['p']:.3f}",
                rew=f"{stats.get('rank_reward', 0.0):.2f}",
                std=f"{stats['pred_std']:.3f}/{stats['target_std']:.3f}",
            )
        if step % LOG_EVERY == 0:
            log_line(
                f"ep{epoch + 1} step {step}/{len(train_loader)} | loss {loss.item():.4f} "
                f"| ce {stats['ce']:.4f} p={stats['p']:.3f} top1={stats['top1']:.3f} "
                f"| mse {stats['mse']:.4f} cos {stats['cos']:.3f} norm {stats['norm']:.4f} "
                f"| var {stats['var']:.5f} std {stats['pred_std']:.3f}/{stats['target_std']:.3f} "
                f"| adj {stats.get('adj_rep', 0.0):.4f} ent {stats.get('entropy', 0.0):.2f} "
                f"| reward {stats.get('rank_reward', 0.0):.3f} best {stats.get('rank_best_reward', 0.0):.3f}"
            )
            if PROMPT_PRIOR_MODE == "pipeline":
                log_line(
                    f"  direct probe | ce {stats['direct_ce']:.4f} "
                    f"p={stats['direct_p']:.3f} top1={stats['direct_top1']:.3f}"
                )

    train_mean = mean_stats(train_total)
    print(
        f"ep{epoch + 1} train | avg_loss={train_loss / max(train_total['n'], 1):.4f} "
        f"ce={train_mean['ce']:.3f} p={train_mean['p']:.3f} top1={train_mean['top1']:.3f} "
        f"mse={train_mean['mse']:.4f} cos={train_mean['cos']:.3f} "
        f"adj={train_mean['adj_rep']:.4f} ent={train_mean['entropy']:.2f} "
        f"std={train_mean['pred_std']:.3f}/{train_mean['target_std']:.3f}",
        flush=True,
    )

    model.eval()
    val_total = new_stats_total()
    best_total = new_stats_total()
    gauss_total = new_stats_total()
    oracle_total = new_stats_total()
    diversity_total = 0.0
    diversity_n = 0

    with torch.no_grad():
        val_iter = progress_bar(
            val_loader,
            total=len(val_loader),
            desc=f"ep{epoch + 1}/{PROMPT_PRIOR_EPOCHS} val {active_feature_label()}",
        )
        for batch in val_iter:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z_prompt, z_target, target_mask, suffix_ids, _target_aux = build_prompt_prior_targets(
                    encoder,
                    decoder,
                    input_ids,
                    attention_mask,
                )
                pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)

                sample_stats = []
                sample_latents = []
                for _sample_idx in range(max(1, PROMPT_PRIOR_VAL_SAMPLES)):
                    pred = run_prompt_prior(model, z_prompt, pos, target_mask, z_target)
                    pred = normalize_prompt_output(pred, z_target, target_mask)
                    _z_prior, z_for_loss = apply_prompt_pipeline(
                        pred,
                        z_prompt,
                        target_mask,
                        pos,
                        draft_prior,
                        draft_alpha,
                        flow_net,
                        metric_net,
                        flow_refine_scale,
                    )
                    _loss, stats = compute_stats_and_loss(
                        decoder, z_prompt, z_for_loss, z_target, suffix_ids, target_mask
                    )
                    add_stats(val_total, stats)
                    sample_stats.append(stats)
                    sample_latents.append(z_for_loss.detach())
                best_stats = min(sample_stats, key=lambda item: item["ce"])
                add_stats(best_total, best_stats)
                diversity_total += sample_latent_diversity(sample_latents, target_mask)
                diversity_n += 1

                gauss = random_suffix_like(z_target, target_mask)
                _gauss_prior, gauss_for_loss = apply_prompt_pipeline(
                    gauss,
                    z_prompt,
                    target_mask,
                    pos,
                    draft_prior,
                    draft_alpha,
                    flow_net,
                    metric_net,
                    flow_refine_scale,
                )
                _gloss, gstats = compute_stats_and_loss(
                    decoder, z_prompt, gauss_for_loss, z_target, suffix_ids, target_mask
                )
                add_stats(gauss_total, gstats)

                _oloss, ostats = compute_stats_and_loss(
                    decoder, z_prompt, z_target, z_target, suffix_ids, target_mask
                )
                add_stats(oracle_total, ostats)
                if tqdm is not None and PROMPT_PRIOR_PROGRESS:
                    last_stats = sample_stats[-1]
                    val_iter.set_postfix(
                        ce=f"{last_stats['ce']:.3f}",
                        p=f"{last_stats['p']:.3f}",
                        std=f"{last_stats['pred_std']:.3f}/{last_stats['target_std']:.3f}",
                    )

    val_mean = mean_stats(val_total)
    best_mean = mean_stats(best_total)
    gauss_mean = mean_stats(gauss_total)
    oracle_mean = mean_stats(oracle_total)
    sample_diversity = diversity_total / max(diversity_n, 1)
    beats_gaussian = val_mean["ce"] < gauss_mean["ce"]
    val_label = (
        "pipeline"
        if PROMPT_PRIOR_MODE == "pipeline"
        else "draft"
        if PROMPT_PRIOR_MODE == "draft"
        else "prompt"
    )
    print(
        f"val ep{epoch + 1} | {val_label} ce={val_mean['ce']:.3f} p={val_mean['p']:.3f} "
        f"top1={val_mean['top1']:.3f} mse={val_mean['mse']:.4f} cos={val_mean['cos']:.3f} "
        f"adj={val_mean['adj_rep']:.4f} ent={val_mean['entropy']:.2f} "
        f"std={val_mean['pred_std']:.3f}/{val_mean['target_std']:.3f} "
        f"| gaussian ce={gauss_mean['ce']:.3f} p={gauss_mean['p']:.3f} top1={gauss_mean['top1']:.3f} "
        f"| oracle ce={oracle_mean['ce']:.3f} p={oracle_mean['p']:.3f} top1={oracle_mean['top1']:.3f} "
        f"| best-of-{max(1, PROMPT_PRIOR_VAL_SAMPLES)} ce={best_mean['ce']:.3f} "
        f"p={best_mean['p']:.3f} top1={best_mean['top1']:.3f} "
        f"| sample_div={sample_diversity:.4f} "
        f"| beats_gaussian={'yes' if beats_gaussian else 'no'}",
        flush=True,
    )

    if val_mean["ce"] < best_val_ce:
        best_val_ce = val_mean["ce"]
        torch.save(
            {
                "prompt_prior": model.state_dict(),
                "best_val_ce": best_val_ce,
                "val_prompt": val_mean,
                "val_prompt_best_of_k": best_mean,
                "val_gaussian": gauss_mean,
                "val_oracle": oracle_mean,
                "val_sample_diversity": sample_diversity,
                "beats_gaussian": beats_gaussian,
                "latent_dim": LATENT_DIM,
                "stage1_variant": STAGE1_VARIANT,
                "prompt_prior_variant": PROMPT_PRIOR_VARIANT,
                "prompt_len": PROMPT_LEN,
                "max_seq_len": MAX_SEQ_LEN,
                "dataset_name": DATASET_NAME,
                "dataset_split": ROCSTORIES_SPLIT if DATASET_NAME == "rocstories" else "legacy_fixed_token",
                "layers": START_TRANSFORMER_LAYERS,
                "heads": START_TRANSFORMER_HEADS,
                "hidden_dim": START_TRANSFORMER_HIDDEN_DIM,
                "type": (
                    "prompt_prior_pipeline"
                    if PROMPT_PRIOR_MODE == "pipeline"
                    else "prompt_prior_draft"
                    if PROMPT_PRIOR_MODE == "draft"
                    else "prompt_prior_diagnostic"
                ),
                "prompt_prior_mode": PROMPT_PRIOR_MODE,
                "prompt_prior_draft_drop_prob": PROMPT_PRIOR_DRAFT_DROP_PROB,
                "prompt_prior_draft_replace_prob": PROMPT_PRIOR_DRAFT_REPLACE_PROB,
                "prompt_prior_var_weight": PROMPT_PRIOR_VAR_WEIGHT,
                "prompt_prior_adj_rep_weight": PROMPT_PRIOR_ADJ_REP_WEIGHT,
                "prompt_prior_entropy_weight": PROMPT_PRIOR_ENTROPY_WEIGHT,
                "prompt_prior_entropy_floor": PROMPT_PRIOR_ENTROPY_FLOOR,
                "prompt_prior_normalize_output": PROMPT_PRIOR_NORMALIZE_OUTPUT,
                "prompt_prior_normalize_source": PROMPT_PRIOR_NORMALIZE_SOURCE,
                "prompt_prior_normalize_mode": PROMPT_PRIOR_NORMALIZE_MODE,
                "prompt_prior_normalize_scale": PROMPT_PRIOR_NORMALIZE_SCALE,
                "prompt_prior_output_mean": PROMPT_PRIOR_OUTPUT_MEAN,
                "prompt_prior_output_std": PROMPT_PRIOR_OUTPUT_STD,
                "prompt_prior_mixer": PROMPT_PRIOR_MIXER,
                "prompt_prior_mixer_layers": PROMPT_PRIOR_MIXER_LAYERS,
                "prompt_prior_mixer_kernel": PROMPT_PRIOR_MIXER_KERNEL,
                "prompt_prior_mixer_scale": PROMPT_PRIOR_MIXER_SCALE,
                "prompt_prior_stochastic": PROMPT_PRIOR_STOCHASTIC,
                "prompt_prior_noise_std_scale": PROMPT_PRIOR_NOISE_STD_SCALE,
                "prompt_prior_noise_input_scale": PROMPT_PRIOR_NOISE_INPUT_SCALE,
                "prompt_prior_val_samples": PROMPT_PRIOR_VAL_SAMPLES,
                "prompt_prior_memory": PROMPT_PRIOR_MEMORY,
                "prompt_prior_memory_size": PROMPT_PRIOR_MEMORY_SIZE,
                "prompt_prior_memory_temp": PROMPT_PRIOR_MEMORY_TEMP,
                "prompt_prior_memory_scale": PROMPT_PRIOR_MEMORY_SCALE,
                "prompt_prior_memory_topk": PROMPT_PRIOR_MEMORY_TOPK,
                "prompt_prior_memory_init": PROMPT_PRIOR_MEMORY_INIT,
                "prompt_prior_group_samples": PROMPT_PRIOR_GROUP_SAMPLES,
                "prompt_prior_ranking_weight": PROMPT_PRIOR_RANKING_WEIGHT,
                "prompt_prior_ranking_temp": PROMPT_PRIOR_RANKING_TEMP,
                "draft_prior_path": DENOISING_PRIOR_PATH if draft_prior is not None else None,
                "draft_prior_alpha": draft_alpha,
                "stage2_path": PROMPT_PRIOR_STAGE2 if flow_net is not None else None,
                "flow_refine_scale": flow_refine_scale,
                "ode_steps": ODE_STEPS,
                "epoch": epoch,
            },
            CHECKPOINT_PATH,
        )
        print(
            f"saved {CHECKPOINT_PATH} | val_ce={best_val_ce:.4f} "
            f"| beats_gaussian={'yes' if beats_gaussian else 'no'}",
            flush=True,
        )

    write_examples(
        tokenizer,
        encoder,
        decoder,
        model,
        val_loader,
        device,
        epoch,
        draft_prior=draft_prior,
        draft_alpha=draft_alpha,
        flow_net=flow_net,
        metric_net=metric_net,
        refine_scale=flow_refine_scale,
    )
