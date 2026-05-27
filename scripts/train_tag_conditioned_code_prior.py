"""Train VQ CodePrior with optional explicit prompt tag conditioning.

This is the clean prompt-information ablation:

    A: z_condition = encode(raw prompt)
    B: z_condition = encode(raw prompt) + encode(prompt-only tags)
    C: z_condition = encode(raw prompt + DeepSeek prompt-only expansion)
    D: z_condition = encode(raw prompt + retrieved ROCStories suffixes)

The target suffix is never used to build tags or expansions.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
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
from stage2_losses import rollout_cosine_alignment_loss, rollout_flow_token_ce_loss
from stage2_riemannian import VQLatentTokenizer, suffix_positions
from train_code_prior import code_ce_loss, code_usage, encode_latents, load_stage1, mean, token_collapse_stats


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "my", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "them", "then", "they",
    "to", "was", "we", "were", "with", "you",
}
ACTION_WORDS = {
    "ask", "asked", "buy", "bought", "call", "called", "decide", "decided", "find",
    "found", "get", "got", "give", "gave", "go", "goes", "went", "help", "helped",
    "learn", "learned", "like", "liked", "look", "looked", "love", "loved", "make",
    "made", "need", "needed", "play", "played", "run", "ran", "say", "said", "see",
    "saw", "take", "took", "tell", "told", "try", "tried", "want", "wanted", "work",
    "worked",
}


class TagConditionedCodePrior(nn.Module):
    def __init__(
        self,
        latent_dim=768,
        codebook_size=2048,
        num_layers=3,
        num_heads=8,
        ffn_dim=768,
        mixer_layers=2,
        mixer_kernel=5,
        mixer_scale=0.5,
    ):
        super().__init__()
        self.mixer_scale = mixer_scale
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.condition_norm = nn.LayerNorm(latent_dim)
        self.layers = nn.ModuleList([
            rfm._StartTransformerLayer(latent_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.mixers = nn.ModuleList([
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(latent_dim),
                    "conv": nn.Conv1d(latent_dim, latent_dim, kernel_size=mixer_kernel, padding=mixer_kernel // 2, groups=latent_dim),
                    "gate": nn.Linear(latent_dim, latent_dim),
                    "out": nn.Linear(latent_dim, latent_dim),
                }
            )
            for _ in range(mixer_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, codebook_size)

    def forward(self, z_condition, pos, mask=None):
        context = self.condition_norm(z_condition)
        x = self.pos_proj(pos.unsqueeze(-1))
        for layer in self.layers:
            x = layer(x, context)
        for mixer in self.mixers:
            h = mixer["norm"](x)
            conv = mixer["conv"](h.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(mixer["gate"](h))
            x = x + self.mixer_scale * mixer["out"](conv * gate)
            if mask is not None:
                x = x * mask.to(x.dtype).unsqueeze(-1)
        logits = self.out_proj(self.out_norm(x))
        if mask is not None:
            logits = logits.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description="Train raw/tag-conditioned VQ CodePrior")
    parser.add_argument("--stage1", default="stage1_vqbase_rocstories_768_K2048_best.pt")
    parser.add_argument("--vq", default="vq_base_stage_rocstories_768_K2048_best.pt")
    parser.add_argument("--condition_mode", choices=("raw", "tags", "expansion", "retrieval"), default="tags")
    parser.add_argument("--tag_len", type=int, default=32)
    parser.add_argument("--expansion_file", default="")
    parser.add_argument("--expansion_len", type=int, default=96)
    parser.add_argument("--expansion_choice", choices=("first", "random", "concat"), default="first")
    parser.add_argument("--filter_to_expansions", action="store_true")
    parser.add_argument("--retrieval_file", default="")
    parser.add_argument("--retrieval_len", type=int, default=128)
    parser.add_argument("--retrieval_top_k", type=int, default=3)
    parser.add_argument("--filter_to_retrieval", action="store_true")
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mixer_layers", type=int, default=2)
    parser.add_argument("--mixer_scale", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--code_ce_weight", type=float, default=1.0)
    parser.add_argument("--decoder_ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.03)
    parser.add_argument("--cos_weight", type=float, default=0.05)
    parser.add_argument("--usage_entropy_weight", type=float, default=0.05)
    parser.add_argument("--adjacent_code_weight", type=float, default=0.05)
    parser.add_argument("--sample_codes", type=int, default=4)
    parser.add_argument("--sample_tau", type=float, default=0.8)
    parser.add_argument("--decode_batch", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--examples", default="tag_conditioned_code_prior_examples.txt")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_data(args):
    if args.max_seq_len != args.prompt_len + args.suffix_len:
        args.max_seq_len = args.prompt_len + args.suffix_len
    s2data.PROMPT_LEN = args.prompt_len
    s2data.DATASET_NAME = args.dataset
    s2data.ROCSTORIES_LOCAL_FILES_ONLY = args.local_files_only
    rfm.PROMPT_LEN = args.prompt_len
    rfm.MAX_SEQ_LEN = args.max_seq_len
    s2losses.PROMPT_LEN = args.prompt_len


def resolve_checkpoint_path(path_value, label):
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit(f"{label} checkpoint not found: {path_value}")


def freeze(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


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


def clean_words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def build_tag_text(prompt_text):
    words = clean_words(prompt_text)
    content = [w for w in words if w not in STOPWORDS and len(w) > 2]
    actions = [w for w in content if w in ACTION_WORDS or w.endswith(("ed", "ing"))]
    entities = [w for w in content if w not in set(actions)]
    # Preserve order and keep the tag short.
    entities = list(dict.fromkeys(entities))[:10]
    actions = list(dict.fromkeys(actions))[:8]
    if not entities:
        entities = content[:6]
    return "entities: " + " ".join(entities) + " actions: " + " ".join(actions)


def normalize_prompt(text):
    return " ".join(str(text).strip().split()).lower()


def load_jsonl_record(line, path, line_no):
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        preview = line[:160].replace("\t", "\\t")
        raise SystemExit(
            f"Invalid JSONL in {path} at line {line_no}: {exc.msg}. "
            f"First chars: {preview!r}. Rebuild the file with --overwrite if it is partial/corrupt."
        ) from exc


def load_expansion_map(path):
    if not path:
        return {}
    expansion_path = Path(path)
    if not expansion_path.is_absolute():
        expansion_path = PROJECT_ROOT / expansion_path
    if not expansion_path.exists():
        raise SystemExit(f"expansion file not found: {path}")
    mapping = {}
    with expansion_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = load_jsonl_record(line, expansion_path, line_no)
            key = record.get("prompt_key") or normalize_prompt(record.get("prompt", ""))
            expansions = record.get("expansions") or []
            expansions = [str(item).strip() for item in expansions if str(item).strip()]
            if key and expansions:
                mapping[key] = expansions
    print(f"loaded prompt expansions: {len(mapping)} from {expansion_path}", flush=True)
    return mapping


def load_retrieval_map(path):
    if not path:
        return {}
    retrieval_path = Path(path)
    if not retrieval_path.is_absolute():
        retrieval_path = PROJECT_ROOT / retrieval_path
    if not retrieval_path.exists():
        raise SystemExit(f"retrieval file not found: {path}")
    mapping = {}
    with retrieval_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = load_jsonl_record(line, retrieval_path, line_no)
            key = record.get("prompt_key") or normalize_prompt(record.get("prompt", ""))
            retrieved = record.get("retrieved") or []
            suffixes = [
                str(item.get("suffix", "")).strip()
                for item in retrieved
                if isinstance(item, dict) and str(item.get("suffix", "")).strip()
            ]
            if key and suffixes:
                mapping[key] = suffixes
    print(f"loaded retrieval contexts: {len(mapping)} from {retrieval_path}", flush=True)
    return mapping


def build_filtered_dataloaders(tokenizer, args, condition_map, label):
    examples = s2data.build_rocstories_dataset(tokenizer, args.max_seq_len)
    filtered = []
    for example in examples:
        prompt = tokenizer.decode(example["input_ids"][: args.prompt_len], skip_special_tokens=True).strip()
        if normalize_prompt(prompt) in condition_map:
            filtered.append(example)
    if not filtered:
        raise RuntimeError(
            f"No ROCStories examples matched the {label} file. "
            "Check that prompt_key uses normalized first-two-sentence prompts."
        )
    random.Random(SEED).shuffle(filtered)
    train_cap = max(1, int(0.9 * len(filtered)))
    train_size = min(args.train_size, train_cap)
    train_rows = filtered[:train_size]
    val_rows = filtered[train_size:] or filtered[-min(len(filtered), 100):]
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=s2data.DATALOADER_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=s2data.seed_worker,
        generator=generator,
        persistent_workers=s2data.DATALOADER_NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_rows,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=s2data.DATALOADER_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=s2data.seed_worker,
        persistent_workers=s2data.DATALOADER_NUM_WORKERS > 0,
    )
    print(
        f"filtered {label} ROCStories batches: train={len(train_loader)} val={len(val_loader)} "
        f"matched={len(filtered)} total={len(examples)} train_examples={len(train_rows)} "
        f"val_examples={len(val_rows)} prompt_slots={args.prompt_len} target_slots={args.suffix_len}",
        flush=True,
    )
    return train_loader, val_loader


def make_tag_batch(tokenizer, input_ids, args):
    prompts = [
        tokenizer.decode(row[: args.prompt_len], skip_special_tokens=True).strip()
        for row in input_ids.detach().cpu()
    ]
    tag_texts = [build_tag_text(prompt) for prompt in prompts]
    encoded = tokenizer(
        tag_texts,
        padding="max_length",
        truncation=True,
        max_length=args.tag_len,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(input_ids.device), encoded["attention_mask"].to(input_ids.device), tag_texts


def choose_expansion(expansions, args, train):
    if not expansions:
        return ""
    if args.expansion_choice == "concat":
        return " ".join(expansions)
    if train and args.expansion_choice == "random":
        return random.choice(expansions)
    return expansions[0]


def make_expansion_batch(tokenizer, input_ids, expansion_map, args, train):
    prompts = [
        tokenizer.decode(row[: args.prompt_len], skip_special_tokens=True).strip()
        for row in input_ids.detach().cpu()
    ]
    condition_texts = []
    missing = 0
    for prompt in prompts:
        expansions = expansion_map.get(normalize_prompt(prompt), [])
        expansion = choose_expansion(expansions, args, train)
        if not expansion:
            missing += 1
            condition_texts.append(prompt)
        else:
            condition_texts.append((prompt + " " + expansion).strip())
    encoded = tokenizer(
        condition_texts,
        padding="max_length",
        truncation=True,
        max_length=args.expansion_len,
        return_tensors="pt",
    )
    return (
        encoded["input_ids"].to(input_ids.device),
        encoded["attention_mask"].to(input_ids.device),
        condition_texts,
        missing,
    )


def make_retrieval_batch(tokenizer, input_ids, retrieval_map, args):
    prompts = [
        tokenizer.decode(row[: args.prompt_len], skip_special_tokens=True).strip()
        for row in input_ids.detach().cpu()
    ]
    condition_texts = []
    missing = 0
    for prompt in prompts:
        suffixes = retrieval_map.get(normalize_prompt(prompt), [])[: args.retrieval_top_k]
        suffixes = [suffix for suffix in suffixes if suffix]
        if not suffixes:
            missing += 1
            condition_texts.append(prompt)
        else:
            retrieved_text = " ".join(f"retrieved story: {suffix}" for suffix in suffixes)
            condition_texts.append((prompt + " " + retrieved_text).strip())
    encoded = tokenizer(
        condition_texts,
        padding="max_length",
        truncation=True,
        max_length=args.retrieval_len,
        return_tensors="pt",
    )
    return (
        encoded["input_ids"].to(input_ids.device),
        encoded["attention_mask"].to(input_ids.device),
        condition_texts,
        missing,
    )


def soft_code_embeddings(vq, code_logits, tau):
    probs = torch.softmax(code_logits / max(tau, 1e-5), dim=-1)
    z = probs @ vq.codebook.weight
    return vq.decoder(z), probs


def adjacent_code_loss(probs, mask):
    if probs.size(1) < 2:
        return probs.new_tensor(0.0), 0.0
    adjacent_same = (probs[:, :-1] * probs[:, 1:]).sum(dim=-1)
    pair_mask = mask[:, :-1].bool() & mask[:, 1:].bool()
    soft = adjacent_same[pair_mask].mean() if pair_mask.any() else adjacent_same.mean()
    return soft, soft.detach().item()


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
    marginal = probs[valid].mean(dim=0) if valid.any() else probs.reshape(-1, probs.size(-1)).mean(dim=0)
    usage_entropy = -(marginal.clamp_min(1e-8) * marginal.clamp_min(1e-8).log()).sum()
    usage_loss = -usage_entropy
    adj_loss, adj_value = adjacent_code_loss(probs, suffix_mask)
    max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(
        decoder,
        z_prompt[: args.decode_batch],
        z_soft[: args.decode_batch],
        suffix_mask[: args.decode_batch],
        args.prompt_len,
    )
    loss = (
        args.code_ce_weight * code_loss
        + args.decoder_ce_weight * dec_ce
        + args.mse_weight * mse
        + args.cos_weight * cos_loss
        + args.usage_entropy_weight * usage_loss
        + args.adjacent_code_weight * adj_loss
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
        "adj_code": adj_value,
        "max_token_frac": max_frac,
        "unique_token_ratio": unique_tok,
        "repeat_frac": rep_frac,
        "punct_frac": punct_frac,
    }


@torch.no_grad()
def decode_code_stats(args, decoder, vq, z_prompt, z_target, ids, suffix_ids, suffix_mask, prefix):
    z_hard = vq.decode_codes(ids, suffix_mask)
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_hard[: args.decode_batch]], dim=1))
    ce, p, top1 = rollout_flow_token_ce_loss(logits, suffix_ids[: args.decode_batch], suffix_mask[: args.decode_batch])
    valid = suffix_mask.bool()
    mse = F.mse_loss(z_hard[valid], z_target[valid]) if valid.any() else F.mse_loss(z_hard, z_target)
    _cos_loss, cos_val = rollout_cosine_alignment_loss(z_hard, z_target, suffix_mask)
    ppl, unique_code_ratio, top1_code_frac, used_codes = code_usage(ids, suffix_mask, vq.codebook_size)
    max_frac, unique_tok, rep_frac, punct_frac = token_collapse_stats(
        decoder, z_prompt[: args.decode_batch], z_hard[: args.decode_batch], suffix_mask[: args.decode_batch], args.prompt_len
    )
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


@torch.no_grad()
def write_examples(path, tokenizer, decoder, z_prompt, z_hard, input_ids, condition_texts, args):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_hard], dim=1))
    pred = logits.argmax(dim=-1)
    rows = []
    for i in range(min(16, input_ids.size(0))):
        prompt = tokenizer.decode(input_ids[i, : args.prompt_len], skip_special_tokens=True).strip()
        target = tokenizer.decode(input_ids[i, args.prompt_len :], skip_special_tokens=True).strip()
        out = tokenizer.decode(pred[i, args.prompt_len :], skip_special_tokens=True).strip()
        condition = condition_texts[i] if i < len(condition_texts) else ""
        rows.append(
            f"--- example {i + 1}\n"
            f"prompt: {prompt}\n"
            f"condition: {condition}\n"
            f"target: {target}\n"
            f"code prior: {out}\n"
        )
    Path(path).write_text("\n".join(rows), encoding="utf-8")


def main():
    args = parse_args()
    seed_everything(SEED)
    configure_data(args)
    args.stage1 = resolve_checkpoint_path(args.stage1, "Stage1")
    args.vq = resolve_checkpoint_path(args.vq, "VQ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)
    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    vq, _vq_ckpt = load_vq(args.vq, latent_dim, device)
    expansion_map = load_expansion_map(args.expansion_file)
    retrieval_map = load_retrieval_map(args.retrieval_file)
    if args.condition_mode == "expansion" and not expansion_map:
        raise SystemExit("--condition_mode expansion requires --expansion_file with generated JSONL records")
    if args.condition_mode == "retrieval" and not retrieval_map:
        raise SystemExit("--condition_mode retrieval requires --retrieval_file with generated JSONL records")
    prior = TagConditionedCodePrior(
        latent_dim=latent_dim,
        codebook_size=vq.codebook_size,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.hidden_dim,
        mixer_layers=args.mixer_layers,
        mixer_scale=args.mixer_scale,
    ).to(device)
    if args.output is None:
        args.output = f"{args.condition_mode}_tag_code_prior_vqbase_rocstories_{latent_dim}_K{vq.codebook_size}_best.pt"
    if args.condition_mode == "expansion" or (args.filter_to_expansions and expansion_map):
        train_loader, val_loader = build_filtered_dataloaders(tokenizer, args, expansion_map, "expansion")
    elif args.condition_mode == "retrieval" or (args.filter_to_retrieval and retrieval_map):
        train_loader, val_loader = build_filtered_dataloaders(tokenizer, args, retrieval_map, "retrieval")
    else:
        train_loader, val_loader = s2data.build_stage2_dataloaders(
            tokenizer, args.train_size, args.batch_size, args.max_seq_len
        )
    optimizer = AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_ce = float("inf")
    print(
        f"TagCodePrior params={sum(p.numel() for p in prior.parameters()):,} K={vq.codebook_size} "
        f"condition_mode={args.condition_mode} tag_len={args.tag_len} "
        f"expansion_len={args.expansion_len} expansion_choice={args.expansion_choice} "
        f"retrieval_len={args.retrieval_len} retrieval_top_k={args.retrieval_top_k}",
        flush=True,
    )

    for epoch in range(args.epochs):
        prior.train()
        total = {}
        iterator = enumerate(train_loader)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"tag-code ep{epoch + 1}/{args.epochs} train")
        for step, batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad():
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                exp_missing_frac = 0.0
                retrieval_missing_frac = 0.0
                z_condition = z_prompt
                if args.condition_mode == "tags":
                    tag_ids, tag_mask, _tag_texts = make_tag_batch(tokenizer, input_ids, args)
                    z_tags = encode_latents(encoder, decoder, tag_ids, tag_mask)
                    z_condition = torch.cat([z_prompt, z_tags], dim=1)
                elif args.condition_mode == "expansion":
                    exp_ids, exp_mask, _condition_texts, missing = make_expansion_batch(
                        tokenizer, input_ids, expansion_map, args, train=True
                    )
                    exp_missing_frac = missing / max(1, input_ids.size(0))
                    z_condition = encode_latents(encoder, decoder, exp_ids, exp_mask)
                elif args.condition_mode == "retrieval":
                    ret_ids, ret_mask, _condition_texts, missing = make_retrieval_batch(
                        tokenizer, input_ids, retrieval_map, args
                    )
                    retrieval_missing_frac = missing / max(1, input_ids.size(0))
                    z_condition = encode_latents(encoder, decoder, ret_ids, ret_mask)
                code_targets = vq.encode(z_target, suffix_mask)
                pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                code_logits = prior(z_condition, pos, suffix_mask)
                loss, stats = compute_losses(
                    args, decoder, vq, z_prompt, z_target, code_logits, code_targets, suffix_ids, suffix_mask
                )
            if args.condition_mode == "expansion":
                stats["exp_missing_frac"] = exp_missing_frac
            elif args.condition_mode == "retrieval":
                stats["retrieval_missing_frac"] = retrieval_missing_frac
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
                    f"soft_ce={stats['soft_ce']:.3f} p={stats['soft_p']:.3f} rep={stats['repeat_frac']:.3f}",
                    flush=True,
                )
            if tqdm is not None:
                iterator.set_postfix(
                    code=f"{stats['code_ce']:.2f}",
                    ce=f"{stats['soft_ce']:.2f}",
                    p=f"{stats['soft_p']:.3f}",
                    rep=f"{stats['repeat_frac']:.2f}",
                )
        print(f"ep{epoch + 1} train | {json.dumps(mean(total), indent=2)}", flush=True)

        prior.eval()
        val_total = {}
        last = None
        with torch.no_grad():
            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_iter, total=len(val_loader), desc=f"tag-code ep{epoch + 1}/{args.epochs} val")
            for batch in val_iter:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                z = encode_latents(encoder, decoder, input_ids, attention_mask)
                z_prompt = z[:, : args.prompt_len]
                z_target = z[:, args.prompt_len :]
                suffix_mask = attention_mask[:, args.prompt_len :]
                suffix_ids = input_ids[:, args.prompt_len :]
                condition_texts = [""] * input_ids.size(0)
                exp_missing_frac = 0.0
                retrieval_missing_frac = 0.0
                z_condition = z_prompt
                if args.condition_mode == "tags":
                    tag_ids, tag_mask, condition_texts = make_tag_batch(tokenizer, input_ids, args)
                    z_tags = encode_latents(encoder, decoder, tag_ids, tag_mask)
                    z_condition = torch.cat([z_prompt, z_tags], dim=1)
                elif args.condition_mode == "expansion":
                    exp_ids, exp_mask, condition_texts, missing = make_expansion_batch(
                        tokenizer, input_ids, expansion_map, args, train=False
                    )
                    exp_missing_frac = missing / max(1, input_ids.size(0))
                    z_condition = encode_latents(encoder, decoder, exp_ids, exp_mask)
                elif args.condition_mode == "retrieval":
                    ret_ids, ret_mask, condition_texts, missing = make_retrieval_batch(
                        tokenizer, input_ids, retrieval_map, args
                    )
                    retrieval_missing_frac = missing / max(1, input_ids.size(0))
                    z_condition = encode_latents(encoder, decoder, ret_ids, ret_mask)
                code_targets = vq.encode(z_target, suffix_mask)
                pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
                code_logits = prior(z_condition, pos, suffix_mask)
                _loss, soft = compute_losses(
                    args, decoder, vq, z_prompt, z_target, code_logits, code_targets, suffix_ids, suffix_mask
                )
                hard, z_hard, _ids = hard_stats(
                    args, decoder, vq, z_prompt, z_target, code_logits, code_targets, suffix_ids, suffix_mask
                )
                sample, z_sample = sampled_stats(args, decoder, vq, z_prompt, z_target, code_logits, suffix_ids, suffix_mask)
                merged = soft | hard | sample
                if args.condition_mode == "expansion":
                    merged["exp_missing_frac"] = exp_missing_frac
                elif args.condition_mode == "retrieval":
                    merged["retrieval_missing_frac"] = retrieval_missing_frac
                add(val_total, merged)
                last = (
                    z_prompt[:16],
                    (z_sample if z_sample is not None else z_hard)[:16],
                    input_ids[:16],
                    condition_texts[:16],
                )
        val_mean = mean(val_total)
        print(f"val ep{epoch + 1} | {json.dumps(val_mean, indent=2)}", flush=True)
        if val_mean["hard_ce"] < best_ce:
            best_ce = val_mean["hard_ce"]
            torch.save(
                {
                    "tag_code_prior": prior.state_dict(),
                    "best_hard_ce": best_ce,
                    "val": val_mean,
                    "latent_dim": latent_dim,
                    "codebook_size": vq.codebook_size,
                    "condition_mode": args.condition_mode,
                    "tag_len": args.tag_len,
                    "expansion_file": args.expansion_file,
                    "expansion_len": args.expansion_len,
                    "expansion_choice": args.expansion_choice,
                    "retrieval_file": args.retrieval_file,
                    "retrieval_len": args.retrieval_len,
                    "retrieval_top_k": args.retrieval_top_k,
                    "prompt_len": args.prompt_len,
                    "max_seq_len": args.max_seq_len,
                    "layers": args.layers,
                    "heads": args.heads,
                    "hidden_dim": args.hidden_dim,
                    "mixer_layers": args.mixer_layers,
                    "mixer_scale": args.mixer_scale,
                    "tau": args.tau,
                    "usage_entropy_weight": args.usage_entropy_weight,
                    "adjacent_code_weight": args.adjacent_code_weight,
                    "sample_codes": args.sample_codes,
                    "sample_tau": args.sample_tau,
                    "vq_path": args.vq,
                    "stage1_path": args.stage1,
                    "type": "tag_conditioned_code_prior",
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved {args.output} | hard_ce={best_ce:.4f}", flush=True)
            if last is not None:
                write_examples(args.examples, tokenizer, decoder, *last, args)


if __name__ == "__main__":
    main()
