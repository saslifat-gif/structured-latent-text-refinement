"""Generate CodePrior/MetricRefiner candidate groups for DeepSeek ranking."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
from train_code_prior import encode_latents, load_stage1, load_vq
from train_codeprior_metric_refiner import (
    ValidMaskHead,
    load_code_prior,
    refine_latents,
    resolve_checkpoint_path,
)
from stage2_riemannian import FlowNet, MetricNet, SyntaxTokenRefiner, VQDecoderAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Generate candidate groups for DeepSeek ranking")
    parser.add_argument("--stage1", default="stage1_rocstories_768_cosmos_best.pt")
    parser.add_argument("--vq", required=True)
    parser.add_argument("--code_prior", required=True)
    parser.add_argument("--metric_refiner", default="")
    parser.add_argument("--decoder_adapter", default="")
    parser.add_argument("--syntax_refiner", default="")
    parser.add_argument("--dataset", choices=("rocstories",), default="rocstories")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--suffix_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--train_size", type=int, default=98161)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--candidates_per_prompt", type=int, default=8)
    parser.add_argument("--sample_tau", type=float, default=0.9)
    parser.add_argument("--rollout_steps", type=int, default=4)
    parser.add_argument("--decode", choices=("argmax", "sample"), default="argmax")
    parser.add_argument("--token_temp", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--use_length_heads", action="store_true")
    parser.add_argument("--use_token_head", action="store_true")
    parser.add_argument("--token_head_weight", type=float, default=0.25)
    parser.add_argument("--min_tokens", type=int, default=8)
    parser.add_argument("--max_decode_tokens", type=int, default=64)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default="examples/deepseek_candidate_groups.jsonl")
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


def load_metric_refiner(path, latent_dim, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    flow_net = FlowNet(latent_dim=latent_dim).to(device)
    metric_net = MetricNet(latent_dim=latent_dim).to(device)
    mask_head = ValidMaskHead(latent_dim).to(device)
    flow_net.load_state_dict(ckpt["flow_net"])
    metric_net.load_state_dict(ckpt["metric_net"])
    if "mask_head" in ckpt:
        mask_head.load_state_dict(ckpt["mask_head"])
    flow_net.eval()
    metric_net.eval()
    mask_head.eval()
    return flow_net, metric_net, mask_head, ckpt


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
    return adapter, ckpt


def load_syntax_refiner(path, latent_dim, vocab_size, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    refiner = SyntaxTokenRefiner(
        vocab_size=int(ckpt.get("vocab_size", vocab_size)),
        latent_dim=latent_dim,
        hidden_dim=int(ckpt.get("hidden_dim", 768)),
        num_layers=int(ckpt.get("layers", 3)),
        num_heads=int(ckpt.get("heads", 8)),
        mixer_layers=int(ckpt.get("mixer_layers", 2)),
        mixer_scale=float(ckpt.get("mixer_scale", 0.5)),
    ).to(device)
    refiner.load_state_dict(ckpt["syntax_refiner"])
    refiner.eval()
    return refiner, ckpt


def load_length_heads(prior_ckpt, latent_dim, device):
    if "valid_head" not in prior_ckpt or "end_head" not in prior_ckpt:
        return None
    valid_head = nn.Linear(latent_dim, 1).to(device)
    end_head = nn.Linear(latent_dim, 1).to(device)
    valid_head.load_state_dict(prior_ckpt["valid_head"])
    end_head.load_state_dict(prior_ckpt["end_head"])
    valid_head.eval()
    end_head.eval()
    return valid_head, end_head


def load_token_head(prior_ckpt, latent_dim, vocab_size, device):
    if "token_head" not in prior_ckpt:
        return None
    token_head = nn.Linear(latent_dim, vocab_size).to(device)
    token_head.load_state_dict(prior_ckpt["token_head"])
    token_head.eval()
    return token_head


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


def sample_tokens(logits, temp=0.9, top_k=50, top_p=0.95):
    probs = torch.softmax(logits.float() / max(temp, 1e-5), dim=-1)
    if top_k > 0 and top_k < probs.size(-1):
        values, idx = probs.topk(top_k, dim=-1)
        kept = torch.zeros_like(probs).scatter(dim=-1, index=idx, src=values)
        probs = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    probs = top_p_filter(probs, top_p)
    return torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(logits.shape[:-1])


def cutoff_from_length_heads(args, end_logits):
    end_idx = int(end_logits[0].float().argmax().item()) + 1
    end_idx = max(args.min_tokens, end_idx)
    end_idx = min(args.max_decode_tokens, end_idx, args.suffix_len)
    return end_idx


@torch.no_grad()
def adapt_suffix(adapter, z_prompt, z_suffix, suffix_mask):
    if adapter is None:
        return z_suffix
    pos = rfm.suffix_positions(z_suffix.size(0), z_suffix.size(1), z_suffix.device, z_suffix.dtype)
    return adapter(z_suffix, z_prompt, pos, suffix_mask)


@torch.no_grad()
def decode_suffix(args, tokenizer, decoder, z_prompt, z_suffix, cutoff=None, token_logits=None):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))[:, args.prompt_len :, :]
    if token_logits is not None and args.token_head_weight != 0:
        logits = logits + args.token_head_weight * token_logits.to(logits.dtype)
    if cutoff is not None:
        logits = logits[:, :cutoff, :]
    if args.decode == "sample":
        ids = sample_tokens(logits, args.token_temp, args.top_k, args.top_p)
    else:
        ids = logits.argmax(dim=-1)
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


@torch.no_grad()
def syntax_refine_suffix(args, tokenizer, decoder, syntax_refiner, z_prompt, z_suffix, suffix_mask, cutoff=None, token_logits=None):
    draft_logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))[:, args.prompt_len :, :]
    if token_logits is not None and args.token_head_weight != 0:
        draft_logits = draft_logits + args.token_head_weight * token_logits.to(draft_logits.dtype)
    if args.decode == "sample":
        draft_ids = sample_tokens(draft_logits, args.token_temp, args.top_k, args.top_p)
    else:
        draft_ids = draft_logits.argmax(dim=-1)
    draft_conf = torch.softmax(draft_logits.float(), dim=-1).max(dim=-1).values
    pos = rfm.suffix_positions(z_suffix.size(0), z_suffix.size(1), z_suffix.device, z_suffix.dtype)
    refined_logits = syntax_refiner(z_prompt, draft_ids, z_suffix, pos, suffix_mask, draft_conf)
    if cutoff is not None:
        refined_logits = refined_logits[:, :cutoff, :]
    if args.decode == "sample":
        ids = sample_tokens(refined_logits, args.token_temp, args.top_k, args.top_p)
    else:
        ids = refined_logits.argmax(dim=-1)
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


@torch.no_grad()
def sample_vq_latents_with_hidden(prior, vq, z_prompt, suffix_mask, sample_tau):
    pos = rfm.suffix_positions(z_prompt.size(0), suffix_mask.size(1), z_prompt.device, z_prompt.dtype)
    code_logits, prior_hidden = prior(z_prompt, pos, suffix_mask, return_hidden=True)
    probs = torch.softmax(code_logits / max(sample_tau, 1e-5), dim=-1)
    ids = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(code_logits.shape[:2])
    z0 = vq.decode_codes(ids, suffix_mask)
    return z0, ids, code_logits, prior_hidden


def write_group(handle, group):
    handle.write(json.dumps(group, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    if args.max_seq_len != args.prompt_len + args.suffix_len:
        args.max_seq_len = args.prompt_len + args.suffix_len
    args.stage1 = resolve_checkpoint_path(args.stage1, "Stage1")
    args.vq = resolve_checkpoint_path(args.vq, "VQ")
    args.code_prior = resolve_checkpoint_path(args.code_prior, "CodePrior")
    if args.metric_refiner:
        args.metric_refiner = resolve_checkpoint_path(args.metric_refiner, "MetricRefiner")
    if args.decoder_adapter:
        args.decoder_adapter = resolve_checkpoint_path(args.decoder_adapter, "DecoderAdapter")
    if args.syntax_refiner:
        args.syntax_refiner = resolve_checkpoint_path(args.syntax_refiner, "SyntaxRefiner")

    seed_everything(SEED)
    configure_data(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}", flush=True)

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim = load_stage1(args.stage1, device)
    vq, _vq_ckpt = load_vq(args.vq, latent_dim, device)
    prior, prior_ckpt = load_code_prior(args.code_prior, latent_dim, vq.codebook_size, device)
    length_heads = load_length_heads(prior_ckpt, latent_dim, device)
    if args.use_length_heads and length_heads is None:
        raise SystemExit("This CodePrior checkpoint has no valid_head/end_head. Retrain with the updated train_hier_code_prior.py first.")
    token_head = load_token_head(prior_ckpt, latent_dim, tokenizer.vocab_size, device)
    if args.use_token_head and token_head is None:
        raise SystemExit("This CodePrior checkpoint has no token_head. Retrain train_route_code_prior.py with --token_ce_weight first.")
    refiner = None
    if args.metric_refiner:
        flow_net, metric_net, _mask_head, _refiner_ckpt = load_metric_refiner(args.metric_refiner, latent_dim, device)
        refiner = (flow_net, metric_net)
    decoder_adapter = None
    if args.decoder_adapter:
        decoder_adapter, _adapter_ckpt = load_decoder_adapter(args.decoder_adapter, latent_dim, device)
    syntax_refiner = None
    if args.syntax_refiner:
        syntax_refiner, _syntax_ckpt = load_syntax_refiner(args.syntax_refiner, latent_dim, tokenizer.vocab_size, device)

    train_loader, val_loader = s2data.build_stage2_dataloaders(
        tokenizer,
        args.train_size,
        args.batch_size,
        args.max_seq_len,
    )
    loader = val_loader if args.split == "val" else train_loader
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(iterator, desc="generate candidates")

    with output.open("w", encoding="utf-8") as handle:
        for batch_idx, batch in enumerate(iterator):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            z = encode_latents(encoder, decoder, input_ids, attention_mask)
            z_prompt = z[:, : args.prompt_len]
            suffix_mask = attention_mask[:, args.prompt_len :]
            for row_idx in range(input_ids.size(0)):
                if written >= args.num_prompts:
                    break
                prompt_ids = input_ids[row_idx, : args.prompt_len]
                target_ids = input_ids[row_idx, args.prompt_len :]
                prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
                reference = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
                candidates = []
                one_prompt = z_prompt[row_idx : row_idx + 1]
                one_mask = suffix_mask[row_idx : row_idx + 1]
                for cand_idx in range(args.candidates_per_prompt):
                    z0, code_ids, _logits, prior_hidden = sample_vq_latents_with_hidden(
                        prior,
                        vq,
                        one_prompt,
                        one_mask,
                        args.sample_tau,
                    )
                    token_logits = token_head(prior_hidden) if args.use_token_head and token_head is not None else None
                    cutoff = None
                    if args.use_length_heads and length_heads is not None:
                        _valid_head, end_head = length_heads
                        cutoff = cutoff_from_length_heads(args, end_head(prior_hidden).squeeze(-1))
                    z0_decode = adapt_suffix(decoder_adapter, one_prompt, z0, one_mask)
                    if syntax_refiner is not None:
                        text = syntax_refine_suffix(
                            args, tokenizer, decoder, syntax_refiner, one_prompt, z0_decode, one_mask, cutoff, token_logits
                        )
                    else:
                        text = decode_suffix(args, tokenizer, decoder, one_prompt, z0_decode, cutoff, token_logits)
                    candidates.append(
                        {
                            "candidate_id": f"p{written:05d}_sample_{cand_idx:02d}",
                            "source": (
                                "codeprior_sample_syntax_refiner"
                                if args.syntax_refiner
                                else
                                "codeprior_sample_adapter_token_blend"
                                if args.decoder_adapter and args.use_token_head
                                else "codeprior_sample_adapter"
                                if args.decoder_adapter
                                else "codeprior_sample_token_blend"
                                if args.use_token_head
                                else "codeprior_sample"
                            ),
                            "text": text,
                            "predicted_cutoff": cutoff,
                            "token_head_weight": args.token_head_weight if args.use_token_head else 0.0,
                            "code_ids": code_ids[0].detach().cpu().tolist(),
                        }
                    )
                    if refiner is not None:
                        flow_net, metric_net = refiner
                        z_refined = refine_latents(flow_net, metric_net, z0, one_prompt, one_mask, args.rollout_steps)
                        z_refined_decode = adapt_suffix(decoder_adapter, one_prompt, z_refined, one_mask)
                        if syntax_refiner is not None:
                            refined_text = syntax_refine_suffix(
                                args,
                                tokenizer,
                                decoder,
                                syntax_refiner,
                                one_prompt,
                                z_refined_decode,
                                one_mask,
                                cutoff,
                                token_logits,
                            )
                        else:
                            refined_text = decode_suffix(args, tokenizer, decoder, one_prompt, z_refined_decode, cutoff, token_logits)
                        candidates.append(
                            {
                                "candidate_id": f"p{written:05d}_refined_{cand_idx:02d}",
                                "source": (
                                    "codeprior_metric_refiner_syntax_refiner"
                                    if args.syntax_refiner
                                    else
                                    "codeprior_metric_refiner_adapter_token_blend"
                                    if args.decoder_adapter and args.use_token_head
                                    else "codeprior_metric_refiner_adapter"
                                    if args.decoder_adapter
                                    else "codeprior_metric_refiner_token_blend"
                                    if args.use_token_head
                                    else "codeprior_metric_refiner"
                                ),
                                "text": refined_text,
                                "predicted_cutoff": cutoff,
                                "token_head_weight": args.token_head_weight if args.use_token_head else 0.0,
                                "code_ids": code_ids[0].detach().cpu().tolist(),
                            }
                        )
                group = {
                    "schema": "deepseek_candidate_group_v1",
                    "prompt_id": f"{args.split}_{batch_idx:04d}_{row_idx:03d}",
                    "prompt": prompt,
                    "reference": reference,
                    "candidates": candidates,
                    "metadata": {
                        "stage1": args.stage1,
                        "vq": args.vq,
                        "code_prior": args.code_prior,
                        "metric_refiner": args.metric_refiner,
                        "decoder_adapter": args.decoder_adapter,
                        "syntax_refiner": args.syntax_refiner,
                        "sample_tau": args.sample_tau,
                        "rollout_steps": args.rollout_steps,
                        "decode": args.decode,
                        "token_temp": args.token_temp,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "use_length_heads": args.use_length_heads,
                        "use_token_head": args.use_token_head,
                        "token_head_weight": args.token_head_weight if args.use_token_head else 0.0,
                        "min_tokens": args.min_tokens,
                        "max_decode_tokens": args.max_decode_tokens,
                    },
                }
                write_group(handle, group)
                written += 1
            if written >= args.num_prompts:
                break
    print(f"wrote {written} groups to {output}", flush=True)


if __name__ == "__main__":
    main()
