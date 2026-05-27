"""Diagnose where the VQ CodePrior stack collapses.

Rows are deliberately separated:

    real_stage1: continuous Stage1 suffix latent decoded directly
    real_vq:     real suffix latent -> VQ codes -> VQ decode -> text decoder
    prior:*:     checkpointed CodePrior/TagCodePrior codes -> VQ decode -> text decoder

The goal is to distinguish VQ round-trip loss, prior slot/code collapse, bad
sampling, and text-surface repetition before asking MetricRefiner to repair it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
from stage2_riemannian import CodePrior, VQLatentTokenizer, suffix_positions
from train_code_prior import code_ce_loss, encode_latents, load_stage1
from train_tag_conditioned_code_prior import (
    TagConditionedCodePrior,
    load_expansion_map,
    load_retrieval_map,
    make_expansion_batch,
    make_tag_batch,
    normalize_prompt,
)


SPECIAL_TOKEN_IDS = {0, 100, 101, 102, 103}


def parse_prior_spec(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), path.strip()
    path = Path(spec)
    return path.stem, spec


def parse_args():
    parser = argparse.ArgumentParser(description="Build a VQ/CodePrior triage matrix")
    parser.add_argument("--stage1", default="stage1_vqbase_rocstories_768_K2048_best.pt")
    parser.add_argument("--vq", default="vq_base_stage_rocstories_768_K2048_best.pt")
    parser.add_argument(
        "--prior",
        action="append",
        default=[],
        help="Prior checkpoint as label=path. May be repeated for raw/tags/expansion/retrieval.",
    )
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--eval_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--decode_batch", type=int, default=32)
    parser.add_argument("--sample_n", type=int, default=8)
    parser.add_argument("--sample_tau", type=float, default=0.8)
    parser.add_argument("--tag_len", type=int, default=32)
    parser.add_argument("--expansion_file", default="")
    parser.add_argument("--expansion_len", type=int, default=96)
    parser.add_argument("--expansion_choice", choices=("first", "random", "concat"), default="first")
    parser.add_argument("--retrieval_file", default="")
    parser.add_argument("--retrieval_len", type=int, default=128)
    parser.add_argument("--retrieval_top_k", type=int, default=3)
    parser.add_argument("--retrieval_order", choices=("nearest", "farthest"), default="nearest")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_json", default="results/vq_codeprior_triage.json")
    parser.add_argument("--output_csv", default="results/vq_codeprior_triage.csv")
    parser.add_argument("--examples", default="results/vq_codeprior_triage_examples.jsonl")
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


def resolve_path(path_value, label):
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"{label} not found: {path_value}")


def freeze(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def load_vq(path, latent_dim, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    codebook_size = int(ckpt.get("codebook_size", 512))
    commitment = float(ckpt.get("commitment", 0.25))
    vq = VQLatentTokenizer(latent_dim, codebook_size, commitment).to(device)
    state = ckpt.get("vq")
    if state is None:
        raise RuntimeError(f"No vq state found in {path}")
    vq.load_state_dict(state)
    freeze(vq)
    return vq, ckpt


def load_prior(path, latent_dim, codebook_size, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    typ = ckpt.get("type", "code_prior")
    layers = int(ckpt.get("layers", 2 if typ == "code_prior" else 3))
    heads = int(ckpt.get("heads", 8))
    hidden_dim = int(ckpt.get("hidden_dim", 512 if typ == "code_prior" else 768))
    mixer_layers = int(ckpt.get("mixer_layers", 2))
    mixer_scale = float(ckpt.get("mixer_scale", 0.5))
    if typ == "tag_conditioned_code_prior" or "tag_code_prior" in ckpt:
        prior = TagConditionedCodePrior(
            latent_dim=latent_dim,
            codebook_size=codebook_size,
            num_layers=layers,
            num_heads=heads,
            ffn_dim=hidden_dim,
            mixer_layers=mixer_layers,
            mixer_scale=mixer_scale,
        ).to(device)
        state = ckpt.get("tag_code_prior")
    else:
        prior = CodePrior(
            latent_dim=latent_dim,
            codebook_size=codebook_size,
            num_layers=layers,
            num_heads=heads,
            ffn_dim=hidden_dim,
            mixer_layers=mixer_layers,
            mixer_scale=mixer_scale,
        ).to(device)
        state = ckpt.get("code_prior")
    if state is None:
        raise RuntimeError(f"No prior state found in {path}")
    prior.load_state_dict(state)
    freeze(prior)
    return prior, ckpt


class RowStats:
    def __init__(self, label, codebook_size=None, suffix_len=None):
        self.label = label
        self.total = {}
        self.n = 0
        self.codebook_size = codebook_size
        self.suffix_len = suffix_len
        self.code_counts = torch.zeros(codebook_size, dtype=torch.float64) if codebook_size else None
        self.slot_counts = (
            torch.zeros(suffix_len, codebook_size, dtype=torch.float64)
            if codebook_size and suffix_len
            else None
        )
        self.slot_totals = torch.zeros(suffix_len, dtype=torch.float64) if suffix_len else None

    def add(self, metrics, weight=1):
        for key, value in metrics.items():
            if value is None:
                continue
            self.total[key] = self.total.get(key, 0.0) + float(value) * weight
        self.n += weight

    def add_codes(self, ids, mask):
        if self.code_counts is None:
            return
        ids_cpu = ids.detach().cpu()
        mask_cpu = mask.detach().cpu().bool()
        valid = ids_cpu[mask_cpu]
        if valid.numel():
            self.code_counts += torch.bincount(valid.reshape(-1), minlength=self.codebook_size).double()
        for slot in range(min(ids_cpu.size(1), self.suffix_len)):
            slot_ids = ids_cpu[:, slot][mask_cpu[:, slot]]
            if slot_ids.numel():
                self.slot_counts[slot] += torch.bincount(slot_ids, minlength=self.codebook_size).double()
                self.slot_totals[slot] += slot_ids.numel()

    def summary(self):
        out = {"row": self.label}
        denom = max(self.n, 1)
        out.update({key: value / denom for key, value in sorted(self.total.items())})
        if self.code_counts is not None and self.code_counts.sum() > 0:
            probs = self.code_counts / self.code_counts.sum().clamp_min(1.0)
            nz = probs > 0
            entropy = -(probs[nz] * probs[nz].log()).sum()
            used = int((self.code_counts > 0).sum().item())
            out.update(
                {
                    "code_ppl_global": float(entropy.exp().item()),
                    "active_code_frac": used / max(self.codebook_size, 1),
                    "top1_code_frac_global": float(self.code_counts.max().item() / self.code_counts.sum().item()),
                    "used_codes": used,
                }
            )
        if self.slot_counts is not None and self.slot_totals is not None:
            ppls = []
            top1s = []
            for slot in range(self.suffix_len):
                if self.slot_totals[slot] <= 0:
                    continue
                probs = self.slot_counts[slot] / self.slot_totals[slot].clamp_min(1.0)
                nz = probs > 0
                entropy = -(probs[nz] * probs[nz].log()).sum()
                ppls.append(float(entropy.exp().item()))
                top1s.append(float(self.slot_counts[slot].max().item() / self.slot_totals[slot].item()))
            if ppls:
                out["slot_marginal_code_ppl_mean"] = sum(ppls) / len(ppls)
                out["slot_marginal_code_ppl_min"] = min(ppls)
                out["slot_top1_code_frac_mean"] = sum(top1s) / len(top1s)
                out["slot_top1_code_frac_max"] = max(top1s)
        return out


def token_diversity(pred_ids, mask):
    valid = mask.bool()
    max_fracs, uniq_fracs, adj_reps, distinct2, distinct3 = [], [], [], [], []
    for row, row_mask in zip(pred_ids.detach().cpu(), valid.detach().cpu()):
        ids = [int(x) for x in row[row_mask].tolist() if int(x) not in SPECIAL_TOKEN_IDS]
        if not ids:
            continue
        counts = {}
        for tok in ids:
            counts[tok] = counts.get(tok, 0) + 1
        max_fracs.append(max(counts.values()) / len(ids))
        uniq_fracs.append(len(counts) / len(ids))
        if len(ids) > 1:
            adj_reps.append(sum(1 for a, b in zip(ids, ids[1:]) if a == b) / (len(ids) - 1))
            bigrams = list(zip(ids, ids[1:]))
            distinct2.append(len(set(bigrams)) / max(len(bigrams), 1))
        if len(ids) > 2:
            trigrams = list(zip(ids, ids[1:], ids[2:]))
            distinct3.append(len(set(trigrams)) / max(len(trigrams), 1))
    return {
        "token_max_frac": mean_or_zero(max_fracs),
        "token_unique_frac": mean_or_zero(uniq_fracs),
        "token_adj_repeat": mean_or_zero(adj_reps),
        "distinct2": mean_or_zero(distinct2),
        "distinct3": mean_or_zero(distinct3),
    }


def token_diversity_per_row(pred_ids, mask):
    rows = []
    valid = mask.bool()
    for row, row_mask in zip(pred_ids.detach().cpu(), valid.detach().cpu()):
        ids = [int(x) for x in row[row_mask].tolist() if int(x) not in SPECIAL_TOKEN_IDS]
        if not ids:
            rows.append({"distinct2": 0.0, "distinct3": 0.0})
            continue
        if len(ids) > 1:
            bigrams = list(zip(ids, ids[1:]))
            d2 = len(set(bigrams)) / max(len(bigrams), 1)
        else:
            d2 = 0.0
        if len(ids) > 2:
            trigrams = list(zip(ids, ids[1:], ids[2:]))
            d3 = len(set(trigrams)) / max(len(trigrams), 1)
        else:
            d3 = 0.0
        rows.append({"distinct2": d2, "distinct3": d3})
    return rows


def code_sequence_metrics(ids, mask):
    valid = mask.bool()
    seq_ppls, adj_reps, repeat4 = [], [], []
    for row, row_mask in zip(ids.detach().cpu(), valid.detach().cpu()):
        codes = [int(x) for x in row[row_mask].tolist()]
        if not codes:
            continue
        counts = {}
        for code in codes:
            counts[code] = counts.get(code, 0) + 1
        probs = [count / len(codes) for count in counts.values()]
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        seq_ppls.append(math.exp(entropy))
        if len(codes) > 1:
            adj_reps.append(sum(1 for a, b in zip(codes, codes[1:]) if a == b) / (len(codes) - 1))
        if len(codes) >= 4:
            grams = [tuple(codes[i : i + 4]) for i in range(len(codes) - 3)]
            seen = set()
            repeated = 0
            for gram in grams:
                repeated += int(gram in seen)
                seen.add(gram)
            repeat4.append(repeated / max(len(grams), 1))
    return {
        "code_seq_ppl_mean": mean_or_zero(seq_ppls),
        "code_adj_repeat": mean_or_zero(adj_reps),
        "code_4gram_repeat": mean_or_zero(repeat4),
    }


def code_4gram_repeat_per_row(ids, mask):
    values = []
    valid = mask.bool()
    for row, row_mask in zip(ids.detach().cpu(), valid.detach().cpu()):
        codes = [int(x) for x in row[row_mask].tolist()]
        if len(codes) < 4:
            values.append(0.0)
            continue
        grams = [tuple(codes[i : i + 4]) for i in range(len(codes) - 3)]
        seen = set()
        repeated = 0
        for gram in grams:
            repeated += int(gram in seen)
            seen.add(gram)
        values.append(repeated / max(len(grams), 1))
    return values


def predictive_entropy_metrics(logits, mask):
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    valid = mask.bool()
    if not valid.any():
        return {"pred_code_ppl_mean": 0.0}
    return {"pred_code_ppl_mean": float(entropy[valid].exp().mean().item())}


@torch.no_grad()
def sample_decode_scores(args, decoder, z_prompt, z_suffix, suffix_ids, suffix_mask):
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_suffix[: args.decode_batch]], dim=1))
    suffix_logits = logits[:, args.prompt_len :]
    target = suffix_ids[: args.decode_batch]
    mask = suffix_mask[: args.decode_batch].bool()
    loss = F.cross_entropy(
        suffix_logits.reshape(-1, suffix_logits.size(-1)),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    ce = (loss * mask.to(loss.dtype)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(loss.dtype)
    pred = suffix_logits.argmax(dim=-1)
    token_rows = token_diversity_per_row(pred, suffix_mask[: args.decode_batch])
    return [
        {
            "ce": float(ce[idx].detach().item()),
            "distinct2": token_rows[idx]["distinct2"],
            "distinct3": token_rows[idx]["distinct3"],
        }
        for idx in range(pred.size(0))
    ]


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


@torch.no_grad()
def decode_metrics(args, decoder, z_prompt, z_suffix, suffix_ids, suffix_mask):
    logits = decoder.decode_from_latent(torch.cat([z_prompt[: args.decode_batch], z_suffix[: args.decode_batch]], dim=1))
    suffix_logits = logits[:, args.prompt_len :]
    ce, p, top1 = rollout_flow_token_ce_loss(
        logits,
        suffix_ids[: args.decode_batch],
        suffix_mask[: args.decode_batch],
    )
    pred = suffix_logits.argmax(dim=-1)
    metrics = {
        "ce": float(ce.detach().item()),
        "target_p": float(p),
        "target_top1": float(top1),
    }
    metrics.update(token_diversity(pred, suffix_mask[: args.decode_batch]))
    return metrics, pred


@torch.no_grad()
def vq_metrics(args, decoder, vq, z_prompt, z_target, suffix_ids, suffix_mask):
    z_q, code_ids, _loss, _parts = vq(z_target, suffix_mask)
    metrics, pred = decode_metrics(args, decoder, z_prompt, z_q, suffix_ids, suffix_mask)
    valid = suffix_mask.bool()
    metrics["latent_mse"] = float(F.mse_loss(z_q[valid], z_target[valid]).item()) if valid.any() else 0.0
    _cos_loss, cos_val = rollout_cosine_alignment_loss(z_q, z_target, suffix_mask)
    metrics["latent_cos"] = float(cos_val)
    metrics.update(code_sequence_metrics(code_ids, suffix_mask))
    return metrics, code_ids, z_q, pred


def select_retrieval_texts(input_ids, tokenizer, retrieval_map, args):
    prompts = [
        tokenizer.decode(row[: args.prompt_len], skip_special_tokens=True).strip()
        for row in input_ids.detach().cpu()
    ]
    condition_texts = []
    missing = 0
    for prompt in prompts:
        suffixes = retrieval_map.get(normalize_prompt(prompt), [])
        if args.retrieval_order == "farthest":
            suffixes = list(reversed(suffixes))
        suffixes = [suffix for suffix in suffixes[: args.retrieval_top_k] if suffix]
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
        missing / max(1, input_ids.size(0)),
    )


@torch.no_grad()
def build_condition(args, tokenizer, encoder, decoder, input_ids, z_prompt, prior_ckpt, expansion_map, retrieval_map):
    mode = prior_ckpt.get("condition_mode", "raw")
    condition_texts = [""] * input_ids.size(0)
    extras = {}
    if mode == "tags":
        tag_ids, tag_mask, condition_texts = make_tag_batch(tokenizer, input_ids, args)
        z_tags = encode_latents(encoder, decoder, tag_ids, tag_mask)
        return torch.cat([z_prompt, z_tags], dim=1), condition_texts, extras
    if mode == "expansion":
        exp_ids, exp_mask, condition_texts, missing = make_expansion_batch(
            tokenizer, input_ids, expansion_map, args, train=False
        )
        extras["missing_condition_frac"] = missing / max(1, input_ids.size(0))
        return encode_latents(encoder, decoder, exp_ids, exp_mask), condition_texts, extras
    if mode == "retrieval":
        ret_ids, ret_mask, condition_texts, missing_frac = select_retrieval_texts(
            input_ids, tokenizer, retrieval_map, args
        )
        extras["missing_condition_frac"] = missing_frac
        extras["retrieval_order"] = args.retrieval_order
        return encode_latents(encoder, decoder, ret_ids, ret_mask), condition_texts, extras
    return z_prompt, condition_texts, extras


@torch.no_grad()
def prior_rows(args, decoder, vq, prior, code_logits, z_prompt, z_target, suffix_ids, suffix_mask):
    hard_ids = code_logits.argmax(dim=-1)
    z_hard = vq.decode_codes(hard_ids, suffix_mask)
    hard_metrics, pred = decode_metrics(args, decoder, z_prompt, z_hard, suffix_ids, suffix_mask)
    valid = suffix_mask.bool()
    hard_metrics["latent_mse"] = float(F.mse_loss(z_hard[valid], z_target[valid]).item()) if valid.any() else 0.0
    _cos_loss, cos_val = rollout_cosine_alignment_loss(z_hard, z_target, suffix_mask)
    hard_metrics["latent_cos"] = float(cos_val)
    hard_metrics.update(code_sequence_metrics(hard_ids, suffix_mask))
    hard_metrics.update(predictive_entropy_metrics(code_logits, suffix_mask))

    sample_summary = {}
    if args.sample_n > 0:
        probs = torch.softmax(code_logits.float() / max(args.sample_tau, 1e-5), dim=-1)
        flat = probs.reshape(-1, probs.size(-1))
        ce_vals, d2_vals, d3_vals, code4_vals = [], [], [], []
        best_ce, best_d2, best_code4 = [], [], []
        per_prompt = [[] for _ in range(min(code_logits.size(0), args.decode_batch))]
        for _ in range(args.sample_n):
            sampled = torch.multinomial(flat, 1).reshape(code_logits.shape[:2])
            z_sample = vq.decode_codes(sampled, suffix_mask)
            token_scores = sample_decode_scores(args, decoder, z_prompt, z_sample, suffix_ids, suffix_mask)
            code4_scores = code_4gram_repeat_per_row(sampled[: args.decode_batch], suffix_mask[: args.decode_batch])
            for prompt_idx, token_score in enumerate(token_scores):
                code4 = code4_scores[prompt_idx] if prompt_idx < len(code4_scores) else 0.0
                ce_vals.append(token_score["ce"])
                d2_vals.append(token_score["distinct2"])
                d3_vals.append(token_score["distinct3"])
                code4_vals.append(code4)
                per_prompt[prompt_idx].append((token_score["ce"], token_score["distinct2"], code4))
        for values in per_prompt:
            if not values:
                continue
            best_ce.append(min(item[0] for item in values))
            best_d2.append(max(item[1] for item in values))
            best_code4.append(min(item[2] for item in values))
        sample_summary = {
            "sample_mean_ce": mean_or_zero(ce_vals),
            "sample_best_ce": mean_or_zero(best_ce),
            "sample_mean_distinct2": mean_or_zero(d2_vals),
            "sample_best_distinct2": mean_or_zero(best_d2),
            "sample_mean_distinct3": mean_or_zero(d3_vals),
            "sample_mean_code_4gram_repeat": mean_or_zero(code4_vals),
            "sample_best_code_4gram_repeat": mean_or_zero(best_code4),
        }
    return hard_metrics | sample_summary, hard_ids, z_hard, pred


def make_loader(tokenizer, args):
    examples = s2data.build_rocstories_dataset(tokenizer, args.max_seq_len)
    random.Random(SEED).shuffle(examples)
    examples = examples[: args.eval_size]
    return DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=s2data.DATALOADER_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=s2data.seed_worker,
        persistent_workers=s2data.DATALOADER_NUM_WORKERS > 0,
    )


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    seed_everything(SEED)
    configure_data(args)
    stage1_path = resolve_path(args.stage1, "Stage1")
    vq_path = resolve_path(args.vq, "VQ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)
    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(str(stage1_path), device)
    vq, vq_ckpt = load_vq(vq_path, latent_dim, device)
    expansion_map = load_expansion_map(args.expansion_file) if args.expansion_file else {}
    retrieval_map = load_retrieval_map(args.retrieval_file) if args.retrieval_file else {}
    priors = []
    for spec in args.prior:
        label, path_value = parse_prior_spec(spec)
        path = resolve_path(path_value, f"Prior {label}")
        prior, ckpt = load_prior(path, latent_dim, vq.codebook_size, device)
        priors.append((label, path, prior, ckpt))
        print(f"loaded prior {label}: type={ckpt.get('type')} condition={ckpt.get('condition_mode', 'raw')} path={path}", flush=True)

    rows = {
        "real_stage1": RowStats("real_stage1"),
        "real_vq": RowStats("real_vq", vq.codebook_size, args.suffix_len),
    }
    for label, _path, _prior, _ckpt in priors:
        rows[f"prior:{label}"] = RowStats(f"prior:{label}", vq.codebook_size, args.suffix_len)
    examples_out = []
    loader = make_loader(tokenizer, args)
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, total=len(loader), desc="diagnose vq/codeprior")

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            z = encode_latents(encoder, decoder, input_ids, attention_mask)
            z_prompt = z[:, : args.prompt_len]
            z_target = z[:, args.prompt_len :]
            suffix_ids = input_ids[:, args.prompt_len :]
            suffix_mask = attention_mask[:, args.prompt_len :]

            direct_metrics, direct_pred = decode_metrics(args, decoder, z_prompt, z_target, suffix_ids, suffix_mask)
            rows["real_stage1"].add(direct_metrics, input_ids.size(0))

            vq_row, target_code_ids, z_q, vq_pred = vq_metrics(
                args, decoder, vq, z_prompt, z_target, suffix_ids, suffix_mask
            )
            rows["real_vq"].add(vq_row, input_ids.size(0))
            rows["real_vq"].add_codes(target_code_ids, suffix_mask)

            pos = suffix_positions(z_target.size(0), z_target.size(1), device, z_target.dtype)
            prior_preds = {}
            for label, _path, prior, ckpt in priors:
                z_condition, condition_texts, extras = build_condition(
                    args, tokenizer, encoder, decoder, input_ids, z_prompt, ckpt, expansion_map, retrieval_map
                )
                code_logits = prior(z_condition, pos, suffix_mask)
                code_targets = vq.encode(z_target, suffix_mask)
                code_loss, code_acc = code_ce_loss(code_logits, code_targets, suffix_mask)
                metrics, hard_ids, z_hard, pred = prior_rows(
                    args, decoder, vq, prior, code_logits, z_prompt, z_target, suffix_ids, suffix_mask
                )
                metrics["code_ce"] = float(code_loss.detach().item())
                metrics["code_acc"] = float(code_acc)
                for key, value in extras.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value
                row_key = f"prior:{label}"
                rows[row_key].add(metrics, input_ids.size(0))
                rows[row_key].add_codes(hard_ids, suffix_mask)
                prior_preds[label] = (pred, condition_texts)

            if batch_idx == 0:
                for i in range(min(8, input_ids.size(0))):
                    record = {
                        "prompt": tokenizer.decode(input_ids[i, : args.prompt_len], skip_special_tokens=True).strip(),
                        "target": tokenizer.decode(input_ids[i, args.prompt_len :], skip_special_tokens=True).strip(),
                        "real_stage1": tokenizer.decode(direct_pred[i], skip_special_tokens=True).strip(),
                        "real_vq": tokenizer.decode(vq_pred[i], skip_special_tokens=True).strip(),
                    }
                    for label, (pred, condition_texts) in prior_preds.items():
                        record[f"prior:{label}"] = tokenizer.decode(pred[i], skip_special_tokens=True).strip()
                        if condition_texts and condition_texts[i]:
                            record[f"condition:{label}"] = condition_texts[i]
                    examples_out.append(record)

    summaries = [row.summary() for row in rows.values()]
    metadata = {
        "stage1": str(stage1_path),
        "vq": str(vq_path),
        "vq_type": vq_ckpt.get("type"),
        "latent_dim": latent_dim,
        "codebook_size": vq.codebook_size,
        "prompt_len": args.prompt_len,
        "suffix_len": args.suffix_len,
        "eval_size": args.eval_size,
        "sample_n": args.sample_n,
        "sample_tau": args.sample_tau,
        "retrieval_order": args.retrieval_order,
        "priors": [
            {
                "label": label,
                "path": str(path),
                "type": ckpt.get("type"),
                "condition_mode": ckpt.get("condition_mode", "raw"),
                "expansion_file": ckpt.get("expansion_file", ""),
                "retrieval_file": ckpt.get("retrieval_file", ""),
            }
            for label, path, _prior, ckpt in priors
        ],
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    examples_path = Path(args.examples)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"metadata": metadata, "rows": summaries}, indent=2), encoding="utf-8")
    write_csv(output_csv, summaries)
    with examples_path.open("w", encoding="utf-8") as handle:
        for record in examples_out:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"wrote {output_json}, {output_csv}, {examples_path}", flush=True)


if __name__ == "__main__":
    main()
