import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer

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
from stage2_riemannian import StartTransformer, suffix_positions


PROMPT_PRIOR_EPOCHS = int(os.environ.get("PROMPT_PRIOR_EPOCHS", str(min(EPOCHS, 10))))
PROMPT_PRIOR_LR = float(os.environ.get("PROMPT_PRIOR_LR", "1e-4"))
PROMPT_PRIOR_CE_WEIGHT = float(os.environ.get("PROMPT_PRIOR_CE_WEIGHT", "1.0"))
PROMPT_PRIOR_MSE_WEIGHT = float(os.environ.get("PROMPT_PRIOR_MSE_WEIGHT", "0.05"))
PROMPT_PRIOR_COS_WEIGHT = float(os.environ.get("PROMPT_PRIOR_COS_WEIGHT", "0.10"))
PROMPT_PRIOR_NORM_WEIGHT = float(os.environ.get("PROMPT_PRIOR_NORM_WEIGHT", "0.01"))
PROMPT_PRIOR_DECODE_BATCH = int(os.environ.get("PROMPT_PRIOR_DECODE_BATCH", "64"))
PROMPT_PRIOR_EXAMPLES = int(os.environ.get("PROMPT_PRIOR_EXAMPLES", "16"))
CHECKPOINT_PATH = os.environ.get(
    "PROMPT_PRIOR_CHECKPOINT",
    f"prompt_prior_rocstories_{LATENT_DIM}_best.pt" if DATASET_NAME == "rocstories" else "prompt_prior_best.pt",
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


def default_stage1_checkpoint():
    if DATASET_NAME == "rocstories":
        return f"stage1_rocstories_{LATENT_DIM}_best.pt"
    return "stage1_best.pt"


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


def decode_loss(decoder, z_prompt, z_suffix, suffix_ids, target_mask):
    n = min(PROMPT_PRIOR_DECODE_BATCH, z_suffix.size(0))
    logits = decoder.decode_from_latent(torch.cat([z_prompt[:n], z_suffix[:n]], dim=1))
    ce, target_prob, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids[:n],
        target_mask[:n] if target_mask is not None else None,
    )
    return ce, target_prob, top1


def compute_stats_and_loss(decoder, z_prompt, pred, z_target, suffix_ids, target_mask):
    ce, target_prob, top1 = decode_loss(decoder, z_prompt, pred, suffix_ids, target_mask)
    mse = valid_mse(pred, z_target, target_mask)
    cos_loss, cos_val = rollout_cosine_alignment_loss(pred, z_target, target_mask)
    nloss = norm_gap_loss(pred, z_target, target_mask)
    loss = (
        PROMPT_PRIOR_CE_WEIGHT * ce
        + PROMPT_PRIOR_MSE_WEIGHT * mse
        + PROMPT_PRIOR_COS_WEIGHT * cos_loss
        + PROMPT_PRIOR_NORM_WEIGHT * nloss
    )
    return loss, {
        "ce": ce.detach().item(),
        "p": target_prob,
        "top1": top1,
        "mse": mse.detach().item(),
        "cos": cos_val,
        "norm": nloss.detach().item(),
    }


def random_suffix_like(z_target, target_mask):
    z = torch.randn_like(z_target) * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    if target_mask is not None:
        z = z * target_mask.to(z.dtype).unsqueeze(-1)
    return z


def add_stats(total, stats):
    for key, value in stats.items():
        total[key] += value
    total["n"] += 1


def mean_stats(total):
    n = max(total["n"], 1)
    return {key: total[key] / n for key in ("ce", "p", "top1", "mse", "cos", "norm")}


def decode_suffix(tokenizer, decoder, z_prompt, z_suffix):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))
    pred_ids = logits.argmax(dim=-1)
    return [
        tokenizer.decode(pred_ids[i, PROMPT_LEN:], skip_special_tokens=True).strip()
        for i in range(pred_ids.size(0))
    ]


@torch.no_grad()
def write_examples(tokenizer, encoder, decoder, model, val_loader, device, epoch):
    model.eval()
    rows = []
    made = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        z_data = encode_latents(encoder, decoder, input_ids, attention_mask)
        z_prompt = z_data[:, :PROMPT_LEN, :]
        z_target = z_data[:, PROMPT_LEN:, :]
        target_mask = attention_mask[:, PROMPT_LEN:]
        pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
        n = min(PROMPT_PRIOR_EXAMPLES - made, input_ids.size(0))
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(z_prompt, pos, target_mask)
            gauss = random_suffix_like(z_target, target_mask)
            prior_texts = decode_suffix(tokenizer, decoder, z_prompt[:n], pred[:n])
            gauss_texts = decode_suffix(tokenizer, decoder, z_prompt[:n], gauss[:n])
            target_texts = [
                tokenizer.decode(input_ids[i, PROMPT_LEN:], skip_special_tokens=True).strip()
                for i in range(n)
            ]
            prompt_texts = [
                tokenizer.decode(input_ids[i, :PROMPT_LEN], skip_special_tokens=True).strip()
                for i in range(n)
            ]

        for i in range(n):
            rows.append(
                f"--- example {made + 1} epoch {epoch + 1}\n"
                f"prompt: {prompt_texts[i]}\n"
                f"target: {target_texts[i]}\n"
                f"prompt prior: {prior_texts[i]}\n"
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
    f"train_size={TRAIN_SIZE} batch={TRAIN_BATCH_SIZE} epochs={PROMPT_PRIOR_EPOCHS} "
    f"layers={START_TRANSFORMER_LAYERS} heads={START_TRANSFORMER_HEADS} "
    f"hidden={START_TRANSFORMER_HIDDEN_DIM} lr={PROMPT_PRIOR_LR} "
    f"loss ce={PROMPT_PRIOR_CE_WEIGHT} mse={PROMPT_PRIOR_MSE_WEIGHT} "
    f"cos={PROMPT_PRIOR_COS_WEIGHT} norm={PROMPT_PRIOR_NORM_WEIGHT}",
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

model = StartTransformer(
    latent_dim=LATENT_DIM,
    num_layers=START_TRANSFORMER_LAYERS,
    num_heads=START_TRANSFORMER_HEADS,
    ffn_dim=START_TRANSFORMER_HIDDEN_DIM,
).to(device)
optimizer = AdamW(model.parameters(), lr=PROMPT_PRIOR_LR)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
best_val_ce = float("inf")
print(f"PromptPrior params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

for epoch in range(PROMPT_PRIOR_EPOCHS):
    model.train()
    train_total = {"ce": 0.0, "p": 0.0, "top1": 0.0, "mse": 0.0, "cos": 0.0, "norm": 0.0, "n": 0}
    train_loss = 0.0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            z_data = encode_latents(encoder, decoder, input_ids, attention_mask)
            z_prompt = z_data[:, :PROMPT_LEN, :]
            z_target = z_data[:, PROMPT_LEN:, :]
            target_mask = attention_mask[:, PROMPT_LEN:]
            suffix_ids = input_ids[:, PROMPT_LEN:]
            pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
            pred = model(z_prompt, pos, target_mask)
            loss, stats = compute_stats_and_loss(
                decoder, z_prompt, pred, z_target, suffix_ids, target_mask
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        add_stats(train_total, stats)
        if step % LOG_EVERY == 0:
            print(
                f"ep{epoch + 1} step {step}/{len(train_loader)} | loss {loss.item():.4f} "
                f"| ce {stats['ce']:.4f} p={stats['p']:.3f} top1={stats['top1']:.3f} "
                f"| mse {stats['mse']:.4f} cos {stats['cos']:.3f} norm {stats['norm']:.4f}",
                flush=True,
            )

    train_mean = mean_stats(train_total)
    print(
        f"ep{epoch + 1} train | avg_loss={train_loss / max(train_total['n'], 1):.4f} "
        f"ce={train_mean['ce']:.3f} p={train_mean['p']:.3f} top1={train_mean['top1']:.3f} "
        f"mse={train_mean['mse']:.4f} cos={train_mean['cos']:.3f}",
        flush=True,
    )

    model.eval()
    val_total = {"ce": 0.0, "p": 0.0, "top1": 0.0, "mse": 0.0, "cos": 0.0, "norm": 0.0, "n": 0}
    gauss_total = {"ce": 0.0, "p": 0.0, "top1": 0.0, "mse": 0.0, "cos": 0.0, "norm": 0.0, "n": 0}
    oracle_total = {"ce": 0.0, "p": 0.0, "top1": 0.0, "mse": 0.0, "cos": 0.0, "norm": 0.0, "n": 0}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z_data = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z_data[:, :PROMPT_LEN, :]
                z_target = z_data[:, PROMPT_LEN:, :]
                target_mask = attention_mask[:, PROMPT_LEN:]
                suffix_ids = input_ids[:, PROMPT_LEN:]
                pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)

                pred = model(z_prompt, pos, target_mask)
                _loss, stats = compute_stats_and_loss(
                    decoder, z_prompt, pred, z_target, suffix_ids, target_mask
                )
                add_stats(val_total, stats)

                gauss = random_suffix_like(z_target, target_mask)
                _gloss, gstats = compute_stats_and_loss(
                    decoder, z_prompt, gauss, z_target, suffix_ids, target_mask
                )
                add_stats(gauss_total, gstats)

                _oloss, ostats = compute_stats_and_loss(
                    decoder, z_prompt, z_target, z_target, suffix_ids, target_mask
                )
                add_stats(oracle_total, ostats)

    val_mean = mean_stats(val_total)
    gauss_mean = mean_stats(gauss_total)
    oracle_mean = mean_stats(oracle_total)
    beats_gaussian = val_mean["ce"] < gauss_mean["ce"]
    print(
        f"val ep{epoch + 1} | prompt ce={val_mean['ce']:.3f} p={val_mean['p']:.3f} "
        f"top1={val_mean['top1']:.3f} mse={val_mean['mse']:.4f} cos={val_mean['cos']:.3f} "
        f"| gaussian ce={gauss_mean['ce']:.3f} p={gauss_mean['p']:.3f} top1={gauss_mean['top1']:.3f} "
        f"| oracle ce={oracle_mean['ce']:.3f} p={oracle_mean['p']:.3f} top1={oracle_mean['top1']:.3f} "
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
                "val_gaussian": gauss_mean,
                "val_oracle": oracle_mean,
                "beats_gaussian": beats_gaussian,
                "latent_dim": LATENT_DIM,
                "prompt_len": PROMPT_LEN,
                "max_seq_len": MAX_SEQ_LEN,
                "dataset_name": DATASET_NAME,
                "dataset_split": ROCSTORIES_SPLIT if DATASET_NAME == "rocstories" else "legacy_fixed_token",
                "layers": START_TRANSFORMER_LAYERS,
                "heads": START_TRANSFORMER_HEADS,
                "hidden_dim": START_TRANSFORMER_HIDDEN_DIM,
                "type": "prompt_prior_diagnostic",
                "epoch": epoch,
            },
            CHECKPOINT_PATH,
        )
        print(
            f"saved {CHECKPOINT_PATH} | val_ce={best_val_ce:.4f} "
            f"| beats_gaussian={'yes' if beats_gaussian else 'no'}",
            flush=True,
        )

    write_examples(tokenizer, encoder, decoder, model, val_loader, device, epoch)
