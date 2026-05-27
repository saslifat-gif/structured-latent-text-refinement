"""Fit and save a PCA basis for frozen Stage1 latents."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import BertTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import stage2_data as s2data
import stage2_riemannian as rfm
from parallel_decoder import cached_from_pretrained
from stage2_config import SEED
from train_code_prior import encode_latents, load_stage1


def parse_args():
    parser = argparse.ArgumentParser(description="Fit PCA on frozen Stage1 token latents")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dims", type=int, default=128)
    parser.add_argument("--suffix_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default="stage1_latent_pca_128.pt")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint_path(path_value, label):
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit(f"{label} checkpoint not found: {path_value}")


def configure_data(args):
    if args.max_seq_len != args.prompt_len + args.suffix_len:
        args.max_seq_len = args.prompt_len + args.suffix_len
    s2data.PROMPT_LEN = args.prompt_len
    s2data.DATASET_NAME = args.dataset
    s2data.ROCSTORIES_LOCAL_FILES_ONLY = args.local_files_only
    rfm.PROMPT_LEN = args.prompt_len
    rfm.MAX_SEQ_LEN = args.max_seq_len


@torch.no_grad()
def collect_latents(args, encoder, decoder, loader, device):
    chunks = []
    rows_seen = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        z = encode_latents(encoder, decoder, input_ids, attention_mask).float()
        mask = attention_mask.bool()
        if args.suffix_only:
            z = z[:, args.prompt_len :]
            mask = mask[:, args.prompt_len :]
        chunks.append(z[mask].cpu())
        rows_seen += input_ids.size(0)
        if rows_seen >= args.samples:
            break
    if not chunks:
        raise RuntimeError("No latent tokens collected for PCA")
    return torch.cat(chunks, dim=0)


def main():
    args = parse_args()
    seed_everything(SEED)
    configure_data(args)
    args.stage1 = resolve_checkpoint_path(args.stage1, "Stage1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    loader, _val_loader = s2data.build_stage2_dataloaders(
        tokenizer,
        args.samples,
        args.batch_size,
        args.max_seq_len,
    )

    x = collect_latents(args, encoder, decoder, loader, device)
    mean = x.mean(dim=0)
    centered = x - mean
    total_var = centered.pow(2).sum()
    q = min(args.dims, latent_dim, max(centered.size(0) - 1, 1))
    _u, singular_values, components = torch.pca_lowrank(centered, q=q)
    explained = singular_values.pow(2) / total_var.clamp_min(1e-12)
    cumulative = explained.cumsum(dim=0)

    payload = {
        "mean": mean,
        "components": components,
        "singular_values": singular_values,
        "explained_variance_ratio": explained,
        "cumulative_variance_ratio": cumulative,
        "latent_dim": latent_dim,
        "dims": q,
        "samples": args.samples,
        "token_count": int(x.size(0)),
        "dataset": args.dataset,
        "suffix_only": args.suffix_only,
        "stage1_path": args.stage1,
        "prompt_len": args.prompt_len,
        "suffix_len": args.suffix_len,
        "max_seq_len": args.max_seq_len,
    }
    torch.save(payload, args.output)
    summary = {f"top_{k}": float(cumulative[k - 1]) for k in (2, 3, 8, 16, 32, 64, 128) if k <= q}
    print(json.dumps({"saved": args.output, "token_count": int(x.size(0)), "cumulative": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
