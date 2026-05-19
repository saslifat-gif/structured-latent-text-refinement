"""Visualize where generated suffix latents land.

This diagnostic projects high-dimensional suffix latents into a 2D PCA view and
adds decoder/readability metrics for each landing stage. It does not train or
modify any model weights.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmark_rocstories import corrupt_synthetic_draft, load_rocstories as benchmark_load_rocstories
from benchmark_rocstories import pack_prompt_suffix_inputs
from eval_text_metrics import tokenize_text
from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained
import stage2_data as s2data
import stage2_riemannian as rfm
from stage2_riemannian import CodePrior, DenoisingPrior, FlowNet, MetricNet, StartTransformer, VQLatentTokenizer
from train_basin_projector import BasinProjector, PromptMoASource, choose_source_latents
from train_basin_projector import load_prompt_prior as load_full_prompt_prior
from train_basin_projector import prompt_prior_latents
from transformers import BertTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize latent landing positions with PCA")
    parser.add_argument("--stage1", default="stage1_rocstories_768_best.pt")
    parser.add_argument("--stage2", default=None, help="Optional Stage2 FlowNet/MetricNet checkpoint")
    parser.add_argument("--draft_prior", default=None, help="Optional DraftPrior/DenoisingPrior checkpoint")
    parser.add_argument("--prompt_prior", default=None, help="Optional prompt-prior checkpoint")
    parser.add_argument("--basin_projector", default=None, help="Optional BasinProjector checkpoint")
    parser.add_argument("--vq", default=None, help="Optional VQLatentTokenizer checkpoint")
    parser.add_argument("--code_prior", default=None, help="Optional CodePrior checkpoint")
    parser.add_argument("--output_dir", default="results/latent_landings")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--prompt_len", type=int, default=32)
    parser.add_argument("--split_strategy", choices=("sentence", "token"), default="sentence")
    parser.add_argument("--rocstories_file", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--synthetic_drop_prob", type=float, default=0.03)
    parser.add_argument("--ode_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_token_plot", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rows_from_stage2_data(
    num_samples: int,
    prompt_len: int,
    max_seq_len: int,
    local_files_only: bool,
    rocstories_file: str | None,
    split_strategy: str,
):
    original_local_only = s2data.ROCSTORIES_LOCAL_FILES_ONLY
    original_file = s2data.ROCSTORIES_FILE
    original_source = s2data.ROCSTORIES_SOURCE
    try:
        s2data.ROCSTORIES_LOCAL_FILES_ONLY = local_files_only
        if rocstories_file:
            s2data.ROCSTORIES_FILE = rocstories_file
            s2data.ROCSTORIES_SOURCE = "file"
        raw_rows = s2data.load_rocstories_rows()
    finally:
        s2data.ROCSTORIES_LOCAL_FILES_ONLY = original_local_only
        s2data.ROCSTORIES_FILE = original_file
        s2data.ROCSTORIES_SOURCE = original_source

    prompt_sentences = int(getattr(s2data, "ROCSTORIES_PROMPT_SENTENCES", 2))
    target_sentences = int(getattr(s2data, "ROCSTORIES_TARGET_SENTENCES", 3))
    needed = prompt_sentences + target_sentences
    rows = []
    for raw in raw_rows:
        parts = s2data.sentence_parts_from_row(raw)
        if len(parts) < needed:
            continue
        if split_strategy == "sentence":
            prompt = " ".join(parts[:prompt_sentences])
            reference = " ".join(parts[prompt_sentences:needed])
            full_tokens = tokenize_text(f"{prompt} {reference}")
            prompt_tokens = tokenize_text(prompt)
            reference_tokens = tokenize_text(reference)
            if not prompt_tokens or not reference_tokens:
                continue
            if len(full_tokens) > max_seq_len:
                budget = max(max_seq_len - len(prompt_tokens), 1)
                reference_tokens = reference_tokens[:budget]
                full_tokens = prompt_tokens + reference_tokens
            rows.append(
                {
                    "prompt": prompt,
                    "reference": " ".join(reference_tokens),
                    "full_text": " ".join(full_tokens),
                    "prompt_len": len(prompt_tokens),
                    "target_len": len(reference_tokens),
                    "split_strategy": "sentence",
                }
            )
        else:
            tokens = tokenize_text(" ".join(parts))
            if len(tokens) < max_seq_len:
                continue
            rows.append(
                {
                    "prompt": " ".join(tokens[:prompt_len]),
                    "reference": " ".join(tokens[prompt_len:max_seq_len]),
                    "full_text": " ".join(tokens[:max_seq_len]),
                    "prompt_len": prompt_len,
                    "target_len": max_seq_len - prompt_len,
                    "split_strategy": "token",
                }
            )
        if len(rows) >= num_samples:
            break
    if not rows:
        raise RuntimeError("stage2_data ROCStories fallback loaded rows, but none matched the requested split")
    return rows


def load_visualization_rows(args, prompt_len: int, max_seq_len: int):
    try:
        return benchmark_load_rocstories(
            args.num_samples,
            prompt_len,
            max_seq_len,
            args.local_files_only,
            args.rocstories_file,
            args.split_strategy,
        ), []
    except Exception as benchmark_exc:
        warning = f"benchmark ROCStories loader failed; trying stage2_data fallback: {benchmark_exc}"
        print(f"WARNING: {warning}", flush=True)
        rows = rows_from_stage2_data(
            args.num_samples,
            prompt_len,
            max_seq_len,
            args.local_files_only,
            args.rocstories_file,
            args.split_strategy,
        )
        return rows, [warning]


def load_stage1(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    latent_dim = int(ckpt.get("latent_dim", 256))
    encoder = BertEncoder().to(device)
    decoder = ParallelDecoder(latent_dim=latent_dim).to(device)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, latent_dim, ckpt


def load_stage2(path: str | None, latent_dim: int, prompt_len: int, max_seq_len: int, device: torch.device):
    if not path:
        return None, None, 1.0, None, None
    if not Path(path).exists():
        return None, None, 1.0, None, f"Stage2 checkpoint not found, skipping Stage2 landing: {path}"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    prompt_len = int(ckpt.get("prompt_len", prompt_len))
    max_seq_len = int(ckpt.get("max_seq_len", max_seq_len))
    rfm.PROMPT_LEN = prompt_len
    rfm.MAX_SEQ_LEN = max_seq_len
    latent_dim = int(ckpt.get("latent_dim", latent_dim))
    flow = FlowNet(
        latent_dim=latent_dim,
        hidden_dim=int(ckpt.get("flow_hidden_dim", 512)),
        depth=int(ckpt.get("flow_depth", 4)),
    ).to(device)
    metric = MetricNet(
        latent_dim=latent_dim,
        hidden_dim=int(ckpt.get("metric_hidden_dim", 512)),
        log_bound=float(ckpt.get("metric_log_bound", 1.0)),
    ).to(device)
    flow.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["flow_net"].items()}, strict=False)
    metric.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["metric_net"].items()}, strict=False)
    flow.eval()
    metric.eval()
    return flow, metric, float(ckpt.get("flow_refine_scale", 1.0)), ckpt, None


def load_draft_prior(path: str | None, latent_dim: int, device: torch.device):
    if not path:
        return None, None, None, None
    if not Path(path).exists():
        return None, None, None, f"DraftPrior checkpoint not found, skipping DraftPrior landing: {path}"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    prior = DenoisingPrior(
        latent_dim=int(ckpt.get("latent_dim", latent_dim)),
        hidden_dim=int(ckpt.get("denoising_hidden_dim", ckpt.get("start_transformer_hidden_dim", 512))),
        num_layers=int(ckpt.get("denoising_layers", ckpt.get("start_transformer_layers", 4))),
        num_heads=int(ckpt.get("denoising_heads", ckpt.get("start_transformer_heads", 8))),
    ).to(device)
    state = ckpt.get("denoising_prior", ckpt.get("draft_prior"))
    if state is None:
        raise RuntimeError(f"No denoising_prior/draft_prior state found in {path}")
    prior.load_state_dict(state, strict=False)
    prior.eval()
    alpha = float(ckpt.get("draft_alpha", ckpt.get("denoising_prior_alpha", 0.5)))
    return prior, alpha, ckpt, None


def load_prompt_prior(path: str | None, latent_dim: int, device: torch.device):
    if not path:
        return None, None, [], None
    if not Path(path).exists():
        return None, None, [], f"PromptPrior checkpoint not found, skipping PromptPrior landing: {path}"
    model, ckpt = load_full_prompt_prior(path, latent_dim, device)
    return model, ckpt, [], None


def load_basin_projector(path: str | None, latent_dim: int, device: torch.device):
    if not path:
        return None, None, None, None
    if not Path(path).exists():
        return None, None, None, f"BasinProjector checkpoint not found, skipping basin-projector landing: {path}"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("basin_projector")
    if state is None:
        raise RuntimeError(f"No basin_projector state found in {path}")
    projector = BasinProjector(
        latent_dim=int(ckpt.get("latent_dim", latent_dim)),
        hidden_dim=int(ckpt.get("hidden_dim", 1024)),
        depth=int(ckpt.get("depth", 3)),
        residual_scale=float(ckpt.get("residual_scale", 0.2)),
    ).to(device)
    missing, unexpected = projector.load_state_dict(state, strict=False)
    projector.eval()
    moa_source = None
    if ckpt.get("moa_source") is not None:
        moa_source = PromptMoASource(
            latent_dim=int(ckpt.get("latent_dim", latent_dim)),
            suffix_len=int(ckpt.get("suffix_len", ckpt.get("max_seq_len", 128) - ckpt.get("prompt_len", 64))),
            num_archetypes=int(ckpt.get("moa_k", 32)),
            hidden_dim=int(ckpt.get("hidden_dim", 1024)),
        ).to(device)
        moa_source.load_state_dict(ckpt["moa_source"], strict=False)
        moa_source.eval()
    source_cfg = argparse.Namespace(
        source=ckpt.get("source", "prompt_prior"),
        moa_temp=float(ckpt.get("moa_temp", 0.7)),
        moa_topk=int(ckpt.get("moa_topk", 4)),
        moa_noise=float(ckpt.get("moa_noise", 0.0)),
        moa_mix=float(ckpt.get("moa_mix", 0.5)),
        target=ckpt.get("target", "synthetic_draft"),
    )
    print(
        f"loaded BasinProjector from {path} | source={source_cfg.source} "
        f"moa={'yes' if moa_source is not None else 'no'} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return projector, moa_source, source_cfg, None


def load_vq(path: str | None, latent_dim: int, device: torch.device):
    if not path:
        return None, None, None
    if not Path(path).exists():
        return None, None, f"VQ checkpoint not found, skipping VQ landings: {path}"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("vq")
    if state is None:
        raise RuntimeError(f"No vq state found in {path}")
    vq = VQLatentTokenizer(
        latent_dim=int(ckpt.get("latent_dim", latent_dim)),
        codebook_size=int(ckpt.get("codebook_size", 512)),
    ).to(device)
    vq.load_state_dict(state, strict=False)
    vq.eval()
    print(f"loaded VQ tokenizer from {path} | K={vq.codebook_size}", flush=True)
    return vq, ckpt, None


def load_code_prior(path: str | None, latent_dim: int, codebook_size: int | None, device: torch.device):
    if not path:
        return None, None, None
    if not Path(path).exists():
        return None, None, f"CodePrior checkpoint not found, skipping code-prior landing: {path}"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("code_prior")
    if state is None:
        raise RuntimeError(f"No code_prior state found in {path}")
    k = int(ckpt.get("codebook_size", codebook_size or 512))
    model = CodePrior(
        latent_dim=int(ckpt.get("latent_dim", latent_dim)),
        codebook_size=k,
        num_layers=int(ckpt.get("layers", 2)),
        num_heads=int(ckpt.get("heads", 8)),
        ffn_dim=int(ckpt.get("hidden_dim", 512)),
        mixer_layers=int(ckpt.get("mixer_layers", 2)),
        mixer_scale=float(ckpt.get("mixer_scale", 0.5)),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"loaded CodePrior from {path} | K={k}", flush=True)
    return model, ckpt, None


@torch.no_grad()
def encode_latents(encoder, decoder, input_ids, attention_mask):
    return decoder.compress(encoder(input_ids, attention_mask))


@torch.no_grad()
def decode_metrics(decoder, z_prompt, z_suffix, suffix_ids, suffix_mask):
    logits = decoder.decode_from_latent(torch.cat([z_prompt, z_suffix], dim=1))[:, z_prompt.size(1):]
    vocab = logits.size(-1)
    flat_loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        suffix_ids.reshape(-1),
        reduction="none",
        ignore_index=0,
    ).reshape_as(suffix_ids)
    mask = suffix_mask.bool()
    ce = (flat_loss * mask.to(flat_loss.dtype)).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    probs = logits.softmax(dim=-1)
    target_prob = probs.gather(-1, suffix_ids.unsqueeze(-1)).squeeze(-1)
    target_prob = (target_prob * mask.to(target_prob.dtype)).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    top1 = logits.argmax(dim=-1).eq(suffix_ids)
    top1_acc = (top1 * mask).sum(dim=1).to(torch.float32) / mask.sum(dim=1).clamp_min(1)
    return ce, target_prob, top1_acc


def masked_mean(z, mask):
    weights = mask.to(z.dtype).unsqueeze(-1)
    return (z * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_cos(a, b, mask):
    per_token = F.cosine_similarity(a, b, dim=-1)
    return (per_token * mask.to(per_token.dtype)).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def apply_draft_prior(prior, alpha, z_start, z_prompt, pos, mask):
    if prior is None:
        return None
    beta = math.sqrt(max(0.0, 1.0 - alpha * alpha))
    noise = torch.randn_like(z_start)
    z_t = alpha * z_start + beta * noise
    z_t = z_t * mask.to(z_t.dtype).unsqueeze(-1)
    alpha_t = z_prompt.new_full((z_prompt.size(0),), alpha)
    return prior(z_t, z_prompt, alpha_t, pos, mask)


@torch.no_grad()
def apply_stage2(flow, metric, z_start, z_prompt, mask, steps, refine_scale):
    if flow is None or metric is None:
        return None
    z = z_start
    batch, suffix_len, _dim = z.shape
    pos = rfm.suffix_positions(batch, suffix_len, z.device, z.dtype)
    dt = 1.0 / max(steps, 1)
    for step in range(max(steps, 1)):
        t = torch.full((batch, suffix_len), step * dt, device=z.device, dtype=z.dtype)
        v, _g = rfm.natural_velocity(flow, metric, z, t, z_prompt, pos)
        z = z + refine_scale * v * dt
        z = z * mask.to(z.dtype).unsqueeze(-1)
    return z


def pca_2d(points: torch.Tensor):
    x = points.float()
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    coords = centered @ basis
    return coords


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def make_plots(
    output_dir: Path,
    point_rows: list[dict],
    token_rows: list[dict],
    metric_rows: list[dict],
    include_token_plot: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; skipped PNG plots ({exc})", flush=True)
        return

    colors = {
        "real": "#111111",
        "gaussian": "#9ca3af",
        "synthetic_draft": "#d97706",
        "prompt_prior": "#2563eb",
        "vq_recon": "#0891b2",
        "code_prior": "#be123c",
        "moa_source": "#db2777",
        "basin_projector": "#7c3aed",
        "draft_prior": "#059669",
        "stage2_final": "#dc2626",
    }
    markers = {
        "real": "*",
        "gaussian": "x",
        "synthetic_draft": "o",
        "prompt_prior": "s",
        "vq_recon": "h",
        "code_prior": "X",
        "moa_source": "v",
        "basin_projector": "P",
        "draft_prior": "^",
        "stage2_final": "D",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    stages = sorted({row["stage"] for row in point_rows})
    for stage in stages:
        xs = [row["pc1"] for row in point_rows if row["stage"] == stage]
        ys = [row["pc2"] for row in point_rows if row["stage"] == stage]
        ax.scatter(
            xs,
            ys,
            s=48 if stage == "real" else 28,
            alpha=0.82,
            marker=markers.get(stage, "o"),
            color=colors.get(stage),
            label=stage,
        )

    by_example: dict[int, list[dict]] = {}
    for row in point_rows:
        by_example.setdefault(int(row["example_id"]), []).append(row)
    arrow_pairs = [("prompt_prior", "draft_prior"), ("draft_prior", "stage2_final"), ("synthetic_draft", "draft_prior")]
    for rows in by_example.values():
        lookup = {row["stage"]: row for row in rows}
        for start, end in arrow_pairs:
            if start in lookup and end in lookup:
                ax.annotate(
                    "",
                    xy=(lookup[end]["pc1"], lookup[end]["pc2"]),
                    xytext=(lookup[start]["pc1"], lookup[start]["pc2"]),
                    arrowprops={"arrowstyle": "->", "color": "#6b7280", "alpha": 0.20, "lw": 0.8},
                )
        if "stage2_final" in lookup and "real" in lookup:
            ax.annotate(
                "",
                xy=(lookup["real"]["pc1"], lookup["real"]["pc2"]),
                xytext=(lookup["stage2_final"]["pc1"], lookup["stage2_final"]["pc2"]),
                arrowprops={"arrowstyle": "->", "color": "#dc2626", "alpha": 0.12, "lw": 0.8},
            )

    ax.set_title("Suffix Latent Landing Positions (Sequence Mean PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "latent_landings_sequence_pca.png", dpi=180)
    plt.close(fig)

    metric_lookup = {
        (int(row["example_id"]), row["stage"]): row
        for row in metric_rows
    }
    ce_values = [
        float(row["ce"])
        for row in metric_rows
        if math.isfinite(float(row["ce"]))
    ]
    if ce_values and point_rows:
        vmin = percentile(ce_values, 0.05)
        vmax = percentile(ce_values, 0.95)
        if vmax <= vmin:
            vmax = max(ce_values)
            vmin = min(ce_values)
        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = None
        for stage in stages:
            stage_rows = [row for row in point_rows if row["stage"] == stage]
            xs = [row["pc1"] for row in stage_rows]
            ys = [row["pc2"] for row in stage_rows]
            ces = [
                float(metric_lookup[(int(row["example_id"]), row["stage"])]["ce"])
                for row in stage_rows
            ]
            if xs:
                scatter = ax.scatter(
                    xs,
                    ys,
                    c=ces,
                    cmap="viridis_r",
                    vmin=vmin,
                    vmax=vmax,
                    s=52 if stage == "real" else 32,
                    alpha=0.86,
                    marker=markers.get(stage, "o"),
                    edgecolors=colors.get(stage, "#111111"),
                    linewidths=0.7,
                    label=stage,
                )
        ax.set_title("Suffix Latent Landing Positions Colored by Decoder CE")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.18)
        ax.legend(loc="best", fontsize=9)
        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("decoder CE (5-95% clipped)")
        fig.tight_layout()
        fig.savefig(output_dir / "latent_landings_sequence_ce_pca.png", dpi=180)
        plt.close(fig)

    if not include_token_plot or not token_rows:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    for stage in stages:
        xs = [row["pc1"] for row in token_rows if row["stage"] == stage]
        ys = [row["pc2"] for row in token_rows if row["stage"] == stage]
        if xs:
            ax.scatter(xs, ys, s=8, alpha=0.35, color=colors.get(stage), label=stage)
    ax.set_title("Suffix Latent Landing Positions (Valid Token PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "latent_landings_token_pca.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder, decoder, latent_dim, stage1_ckpt = load_stage1(args.stage1, device)
    prompt_len = int(stage1_ckpt.get("prompt_len", args.prompt_len))
    max_seq_len = int(stage1_ckpt.get("max_seq_len", args.max_seq_len))
    rfm.PROMPT_LEN = prompt_len
    rfm.MAX_SEQ_LEN = max_seq_len

    warnings = []
    flow, metric, refine_scale, stage2_ckpt, stage2_warning = load_stage2(
        args.stage2,
        latent_dim,
        prompt_len,
        max_seq_len,
        device,
    )
    if stage2_warning:
        warnings.append(stage2_warning)
        print(f"WARNING: {stage2_warning}", flush=True)
    if stage2_ckpt is not None:
        prompt_len = int(stage2_ckpt.get("prompt_len", prompt_len))
        max_seq_len = int(stage2_ckpt.get("max_seq_len", max_seq_len))
        latent_dim = int(stage2_ckpt.get("latent_dim", latent_dim))
    suffix_len = max_seq_len - prompt_len
    ode_steps = args.ode_steps if args.ode_steps is not None else int((stage2_ckpt or {}).get("ode_steps", 16))
    draft_prior, draft_alpha, _draft_ckpt, draft_warning = load_draft_prior(args.draft_prior, latent_dim, device)
    if draft_warning:
        warnings.append(draft_warning)
        print(f"WARNING: {draft_warning}", flush=True)
    prompt_prior, _prompt_ckpt, prompt_notes, prompt_warning = load_prompt_prior(args.prompt_prior, latent_dim, device)
    if prompt_warning:
        warnings.append(prompt_warning)
        print(f"WARNING: {prompt_warning}", flush=True)
    basin_projector, basin_moa_source, basin_source_cfg, basin_warning = load_basin_projector(
        args.basin_projector,
        latent_dim,
        device,
    )
    if basin_warning:
        warnings.append(basin_warning)
        print(f"WARNING: {basin_warning}", flush=True)
    vq, vq_ckpt, vq_warning = load_vq(args.vq, latent_dim, device)
    if vq_warning:
        warnings.append(vq_warning)
        print(f"WARNING: {vq_warning}", flush=True)
    code_prior, code_ckpt, code_warning = load_code_prior(
        args.code_prior,
        latent_dim,
        int(vq.codebook_size) if vq is not None else None,
        device,
    )
    if code_warning:
        warnings.append(code_warning)
        print(f"WARNING: {code_warning}", flush=True)

    rows, row_warnings = load_visualization_rows(args, prompt_len, max_seq_len)
    warnings.extend(row_warnings)

    all_stage_vectors = []
    all_stage_meta = []
    all_token_vectors = []
    all_token_meta = []
    metric_rows = []

    for offset in range(0, len(rows), args.batch_size):
        batch_rows = rows[offset: offset + args.batch_size]
        prompts = [row["prompt"] for row in batch_rows]
        refs = [row["reference"] for row in batch_rows]
        synthetic = [
            corrupt_synthetic_draft(ref, args.seed + offset + idx, args.synthetic_drop_prob)
            for idx, ref in enumerate(refs)
        ]

        real_inputs = pack_prompt_suffix_inputs(tokenizer, prompts, refs, prompt_len, suffix_len)
        draft_inputs = pack_prompt_suffix_inputs(tokenizer, prompts, synthetic, prompt_len, suffix_len)
        input_ids = real_inputs["input_ids"].to(device)
        attention_mask = real_inputs["attention_mask"].to(device)
        draft_ids = draft_inputs["input_ids"].to(device)
        draft_mask_full = draft_inputs["attention_mask"].to(device)
        suffix_ids = input_ids[:, prompt_len:]
        suffix_mask = attention_mask[:, prompt_len:]

        z_real_full = encode_latents(encoder, decoder, input_ids, attention_mask)
        z_draft_full = encode_latents(encoder, decoder, draft_ids, draft_mask_full)
        z_prompt = z_real_full[:, :prompt_len]
        z_real = z_real_full[:, prompt_len:]
        z_synthetic = z_draft_full[:, prompt_len:] * suffix_mask.to(z_real.dtype).unsqueeze(-1)
        pos = rfm.suffix_positions(z_real.size(0), z_real.size(1), device, z_real.dtype)
        valid_real = z_real[suffix_mask.bool()]
        gaussian = torch.randn_like(z_real) * valid_real.std().clamp_min(1e-6) + valid_real.mean()

        stage_latents = {
            "real": z_real,
            "gaussian": gaussian * suffix_mask.to(z_real.dtype).unsqueeze(-1),
            "synthetic_draft": z_synthetic,
        }
        if vq is not None:
            with torch.no_grad():
                z_vq, _code_ids, _vq_loss, _parts = vq(z_real, suffix_mask)
            stage_latents["vq_recon"] = z_vq
        if vq is not None and code_prior is not None:
            with torch.no_grad():
                code_logits = code_prior(z_prompt, pos, suffix_mask)
                pred_codes = code_logits.argmax(dim=-1)
                z_code = vq.decode_codes(pred_codes, suffix_mask)
            stage_latents["code_prior"] = z_code
        if prompt_prior is not None:
            prompt_mode = str((_prompt_ckpt or {}).get("prompt_prior_mode", "direct")).lower()
            prompt_target = z_synthetic if prompt_mode == "draft" else z_real
            stage_latents["prompt_prior"] = prompt_prior_latents(
                prompt_prior,
                z_prompt,
                z_real.size(1),
                suffix_mask,
                prompt_target,
            )
        if basin_projector is not None and prompt_prior is not None:
            basin_target = z_synthetic if getattr(basin_source_cfg, "target", "synthetic_draft") == "synthetic_draft" else z_real
            if (
                basin_source_cfg is not None
                and basin_source_cfg.source in ("moa", "prompt_prior_moa")
                and basin_moa_source is not None
            ):
                with torch.no_grad():
                    z_source, _source_stats = choose_source_latents(
                        basin_source_cfg,
                        prompt_prior,
                        basin_moa_source,
                        z_prompt,
                        suffix_mask,
                        basin_target,
                    )
                stage_latents["moa_source"] = z_source
            else:
                z_source = stage_latents.get("prompt_prior")
            with torch.no_grad():
                z_basin, _delta = basin_projector(z_source, z_prompt, pos, suffix_mask)
            stage_latents["basin_projector"] = z_basin
        start_for_repair = stage_latents.get("basin_projector", stage_latents.get("prompt_prior", z_synthetic))
        z_draft_prior = apply_draft_prior(draft_prior, draft_alpha, start_for_repair, z_prompt, pos, suffix_mask)
        if z_draft_prior is not None:
            stage_latents["draft_prior"] = z_draft_prior
        start_for_stage2 = stage_latents.get("draft_prior", start_for_repair)
        z_stage2 = apply_stage2(flow, metric, start_for_stage2, z_prompt, suffix_mask, ode_steps, refine_scale)
        if z_stage2 is not None:
            stage_latents["stage2_final"] = z_stage2

        for stage, z_stage in stage_latents.items():
            z_stage = z_stage * suffix_mask.to(z_stage.dtype).unsqueeze(-1)
            ce, prob, top1 = decode_metrics(decoder, z_prompt, z_stage, suffix_ids, suffix_mask)
            cos = masked_cos(z_stage, z_real, suffix_mask)
            dist = ((z_stage - z_real).pow(2).sum(dim=-1).sqrt() * suffix_mask).sum(dim=1) / suffix_mask.sum(dim=1).clamp_min(1)
            norm = (z_stage.norm(dim=-1) * suffix_mask).sum(dim=1) / suffix_mask.sum(dim=1).clamp_min(1)
            mean_vec = masked_mean(z_stage, suffix_mask).detach().cpu()
            for local_idx in range(z_stage.size(0)):
                example_id = offset + local_idx
                all_stage_vectors.append(mean_vec[local_idx])
                all_stage_meta.append((example_id, stage))
                metric_rows.append(
                    {
                        "example_id": example_id,
                        "stage": stage,
                        "ce": float(ce[local_idx].detach().cpu()),
                        "target_prob": float(prob[local_idx].detach().cpu()),
                        "top1_acc": float(top1[local_idx].detach().cpu()),
                        "cos_to_real": float(cos[local_idx].detach().cpu()),
                        "l2_to_real": float(dist[local_idx].detach().cpu()),
                        "latent_norm": float(norm[local_idx].detach().cpu()),
                    }
                )
            if args.include_token_plot:
                valid = suffix_mask.bool()
                token_vecs = z_stage.detach().cpu()[valid.detach().cpu()]
                meta_valid = valid.detach().cpu().nonzero(as_tuple=False)
                all_token_vectors.extend(token_vecs)
                for item in meta_valid:
                    all_token_meta.append((offset + int(item[0]), int(item[1]), stage))

    point_rows = []
    if all_stage_vectors:
        coords = pca_2d(torch.stack(all_stage_vectors))
        for (example_id, stage), coord in zip(all_stage_meta, coords):
            point_rows.append(
                {
                    "example_id": example_id,
                    "stage": stage,
                    "pc1": float(coord[0]),
                    "pc2": float(coord[1]),
                }
            )

    token_rows = []
    if args.include_token_plot and all_token_vectors:
        coords = pca_2d(torch.stack(all_token_vectors))
        for (example_id, token_pos, stage), coord in zip(all_token_meta, coords):
            token_rows.append(
                {
                    "example_id": example_id,
                    "token_pos": token_pos,
                    "stage": stage,
                    "pc1": float(coord[0]),
                    "pc2": float(coord[1]),
                }
            )

    write_csv(output_dir / "latent_landing_metrics.csv", metric_rows)
    write_csv(output_dir / "latent_landing_sequence_pca.csv", point_rows)
    if token_rows:
        write_csv(output_dir / "latent_landing_token_pca.csv", token_rows)
    make_plots(output_dir, point_rows, token_rows, metric_rows, args.include_token_plot)

    summary = {
        "stage1": args.stage1,
        "stage2": args.stage2,
        "draft_prior": args.draft_prior,
        "prompt_prior": args.prompt_prior,
        "basin_projector": args.basin_projector,
        "vq": args.vq,
        "code_prior": args.code_prior,
        "prompt_prior_notes": prompt_notes,
        "warnings": warnings,
        "num_examples": len(rows),
        "prompt_len": prompt_len,
        "suffix_len": suffix_len,
        "latent_dim": latent_dim,
        "ode_steps": ode_steps,
        "refine_scale": refine_scale,
        "outputs": [
            str(output_dir / "latent_landing_metrics.csv"),
            str(output_dir / "latent_landing_sequence_pca.csv"),
            str(output_dir / "latent_landings_sequence_pca.png"),
            str(output_dir / "latent_landings_sequence_ce_pca.png"),
        ],
    }
    if token_rows:
        summary["outputs"].extend(
            [
                str(output_dir / "latent_landing_token_pca.csv"),
                str(output_dir / "latent_landings_token_pca.png"),
            ]
        )
    with (output_dir / "latent_landing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
