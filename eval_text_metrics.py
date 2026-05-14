"""Shared text metrics for ROCStories quality/speed benchmarks."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]")


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Iterable[str], n: int) -> float:
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        grams = _ngrams(tokenize_text(text), n)
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def unique_token_ratio(texts: Iterable[str]) -> float:
    tokens = [tok for text in texts for tok in tokenize_text(text)]
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def repetition_rate(texts: Iterable[str], n: int = 4) -> float:
    repeated = 0
    total = 0
    for text in texts:
        grams = _ngrams(tokenize_text(text), n)
        counts = Counter(grams)
        repeated += sum(count - 1 for count in counts.values() if count > 1)
        total += len(grams)
    return repeated / total if total else 0.0


def max_token_fraction(texts: Iterable[str]) -> float:
    values = []
    for text in texts:
        tokens = tokenize_text(text)
        if tokens:
            values.append(max(Counter(tokens).values()) / len(tokens))
    return sum(values) / len(values) if values else 0.0


def average_generated_length(texts: Iterable[str]) -> float:
    lengths = [len(tokenize_text(text)) for text in texts]
    return sum(lengths) / len(lengths) if lengths else 0.0


def compute_mauve(predictions: list[str], references: list[str]) -> float | None:
    try:
        import mauve  # type: ignore
    except ImportError as exc:
        raise RuntimeError("MAUVE is not installed. Install it with: pip install mauve-text") from exc

    if not predictions or not references:
        return None
    device_id = int(os.environ.get("MAUVE_DEVICE_ID", "0"))
    batch_size = int(os.environ.get("MAUVE_BATCH_SIZE", "64"))
    max_text_length = int(os.environ.get("MAUVE_MAX_TEXT_LENGTH", "128"))
    model_name = os.environ.get("MAUVE_MODEL_NAME", "gpt2-large")
    result = mauve.compute_mauve(
        p_text=predictions,
        q_text=references,
        featurize_model_name=model_name,
        device_id=device_id,
        batch_size=batch_size,
        max_text_length=max_text_length,
        verbose=False,
    )
    return float(result.mauve)


def compute_text_metrics(
    predictions: list[str],
    references: list[str] | None = None,
    *,
    include_mauve: bool = True,
    latency_seconds: float | None = None,
    generated_tokens: int | None = None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "mauve": None,
        "stage1_reconstruction_ce": None,
        "decoder_target_probability": None,
        "target_token_probability": None,
        "distinct_1": distinct_n(predictions, 1),
        "distinct_2": distinct_n(predictions, 2),
        "unique_token_ratio": unique_token_ratio(predictions),
        "repetition_rate": repetition_rate(predictions),
        "max_token_fraction": max_token_fraction(predictions),
        "avg_generated_length": average_generated_length(predictions),
        "latency_per_sample": None,
        "tokens_per_second": None,
    }
    if include_mauve and references:
        metrics["mauve"] = compute_mauve(predictions, references)
    if latency_seconds is not None and predictions:
        metrics["latency_per_sample"] = latency_seconds / len(predictions)
        token_count = generated_tokens
        if token_count is None:
            token_count = sum(len(tokenize_text(text)) for text in predictions)
        metrics["tokens_per_second"] = token_count / latency_seconds if latency_seconds > 0 else math.inf
    return metrics


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "model",
        "dataset",
        "latent_dim",
        "steps",
        "mauve",
        "stage1_reconstruction_ce",
        "decoder_target_probability",
        "target_token_probability",
        "distinct_1",
        "distinct_2",
        "repetition_rate",
        "unique_token_ratio",
        "max_token_fraction",
        "avg_generated_length",
        "latency_per_sample",
        "tokens_per_second",
        "speedup_vs_gpt2",
        "status",
    ]
    discovered: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in discovered:
                discovered.append(key)
    fieldnames = [key for key in preferred if key in discovered]
    fieldnames.extend(key for key in discovered if key not in fieldnames)

    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        clean_row: dict[str, Any] = {}
        for key in fieldnames:
            value = row.get(key)
            if isinstance(value, str):
                value = " ".join(value.split())
            elif value is None:
                value = ""
            clean_row[key] = value
        for key in row:
            if key not in fieldnames:
                raise ValueError(f"internal CSV field discovery missed key: {key}")
        cleaned_rows.append(clean_row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned_rows)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
