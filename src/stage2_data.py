import random
import csv
import json
import re
from pathlib import Path

import torch
from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader

from stage2_config import (
    DATALOADER_NUM_WORKERS,
    DATASET_NAME,
    PROMPT_LEN,
    ROCSTORIES_FILE,
    ROCSTORIES_HUB_CANDIDATES,
    ROCSTORIES_LOCAL_FILES_ONLY,
    ROCSTORIES_PROMPT_SENTENCES,
    ROCSTORIES_SOURCE,
    ROCSTORIES_SPLIT,
    ROCSTORIES_TARGET_SENTENCES,
    SEED,
)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def split_story_sentences(text):
    text = " ".join(str(text).strip().split())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _clean_sentence_list(value):
    if isinstance(value, str):
        return split_story_sentences(value)
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, str):
                item_parts = split_story_sentences(item)
                if len(item_parts) > 1:
                    parts.extend(item_parts)
                elif item.strip():
                    parts.append(" ".join(item.strip().split()))
        return [part for part in parts if part]
    return []


def sentence_parts_from_row(row):
    sentence_key_sets = [
        [f"sentence{i}" for i in range(1, 6)],
        [f"Sentence{i}" for i in range(1, 6)],
        [f"InputSentence{i}" for i in range(1, 5)] + ["RandomFifthSentenceQuiz1"],
        [f"InputSentence{i}" for i in range(1, 5)] + ["RandomFifthSentenceQuiz2"],
    ]
    for keys in sentence_key_sets:
        parts = [str(row.get(key, "")).strip() for key in keys]
        if sum(bool(part) for part in parts) >= ROCSTORIES_PROMPT_SENTENCES + ROCSTORIES_TARGET_SENTENCES:
            return parts

    numbered_keys = []
    for key in row.keys():
        match = re.fullmatch(r"(?:input_?)?sentence_?(\d+)", str(key), flags=re.IGNORECASE)
        if match:
            numbered_keys.append((int(match.group(1)), key))
    if numbered_keys:
        parts = [
            str(row.get(key, "")).strip()
            for _idx, key in sorted(numbered_keys)
            if str(row.get(key, "")).strip()
        ]
        if len(parts) >= ROCSTORIES_PROMPT_SENTENCES + ROCSTORIES_TARGET_SENTENCES:
            return parts

    for key in ("sentences", "sentence_list", "story_sentences", "storylines", "storyline"):
        parts = _clean_sentence_list(row.get(key))
        if len(parts) >= ROCSTORIES_PROMPT_SENTENCES + ROCSTORIES_TARGET_SENTENCES:
            return parts

    prompt_continuation_keys = (
        ("prompt", "continuation"),
        ("Prompt", "Continuation"),
        ("prompt", "target"),
        ("context", "target"),
        ("input", "output"),
        ("source", "target"),
    )
    for prompt_key, continuation_key in prompt_continuation_keys:
        if row.get(prompt_key) and row.get(continuation_key):
            parts = split_story_sentences(row[prompt_key]) + split_story_sentences(row[continuation_key])
            if len(parts) >= ROCSTORIES_PROMPT_SENTENCES + ROCSTORIES_TARGET_SENTENCES:
                return parts

    for key in ("target", "story", "text", "full_text", "Story", "Text", "content"):
        if row.get(key):
            parts = _clean_sentence_list(row[key])
            if parts:
                return parts

    metadata_keys = {"id", "storyid", "story_id", "label", "answer", "source", "split"}
    string_parts = []
    for key, value in row.items():
        if str(key).lower() in metadata_keys:
            continue
        if isinstance(value, str) and value.strip():
            parts = split_story_sentences(value)
            if len(parts) > 1:
                string_parts.extend(parts)
            else:
                string_parts.append(" ".join(value.strip().split()))
    if len(string_parts) >= ROCSTORIES_PROMPT_SENTENCES + ROCSTORIES_TARGET_SENTENCES:
        return string_parts
    return []


def read_rocstories_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ROCStories file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter=delimiter))
    with path.open("r", encoding="utf-8") as f:
        return [{"text": line.strip()} for line in f if line.strip()]


def parse_rocstories_hub_candidates(raw_candidates):
    candidates = []
    for item in raw_candidates.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, config = item.split(":", 1)
            candidates.append((name.strip(), config.strip() or None))
        else:
            candidates.append((item, None))
    return candidates or [
        ("wza/roc_stories", None),
        ("inkoziev/roc_stories", None),
        ("igormorgado/ROCStories2018", None),
        ("mintujupally/ROCStories", None),
        ("hamishivi/ROCStories", None),
        ("dwen/rocstories", None),
        ("story_cloze", "2016"),
        ("story_cloze", "2018"),
    ]


def load_rocstories_hub_rows(local_files_only):
    errors = []
    download_config = DownloadConfig(local_files_only=local_files_only)
    for name, config in parse_rocstories_hub_candidates(ROCSTORIES_HUB_CANDIDATES):
        for split in ("train", "validation", "test"):
            try:
                if config is None:
                    dataset = load_dataset(name, split=split, download_config=download_config)
                else:
                    dataset = load_dataset(name, config, split=split, download_config=download_config)
                rows = list(dataset)
                print(
                    f"loaded ROCStories-compatible Hub dataset "
                    f"{name}{('/' + config) if config else ''}/{split} "
                    f"rows={len(rows)} local_files_only={local_files_only}",
                    flush=True,
                )
                return rows
            except Exception as exc:
                errors.append(f"{name}{('/' + config) if config else ''}/{split}: {exc}")
                parquet_subsets = [config] if config else ["all", "default"]
                for subset in parquet_subsets:
                    parquet_path = f"hf://datasets/{name}@refs/convert/parquet/{subset}/{split}/*.parquet"
                    try:
                        dataset = load_dataset(
                            "parquet",
                            data_files={split: parquet_path},
                            split=split,
                            download_config=download_config,
                        )
                        rows = list(dataset)
                        print(
                            f"loaded ROCStories-compatible Hub parquet "
                            f"{name}/{subset}/{split} rows={len(rows)} "
                            f"local_files_only={local_files_only}",
                            flush=True,
                        )
                        return rows
                    except Exception as parquet_exc:
                        errors.append(f"parquet {name}/{subset}/{split}: {parquet_exc}")
    raise RuntimeError(" | ".join(errors))


def load_rocstories_rows():
    source = ROCSTORIES_SOURCE
    if source not in ("auto", "file", "hub"):
        raise ValueError("ROCSTORIES_SOURCE must be one of: auto, file, hub")

    file_path = Path(ROCSTORIES_FILE) if ROCSTORIES_FILE else None
    if source in ("auto", "file") and file_path is not None and file_path.exists():
        rows = read_rocstories_rows(file_path)
        print(f"loaded ROCStories rows from file: {file_path} rows={len(rows)}", flush=True)
        return rows

    if source == "file":
        raise RuntimeError(
            "ROCSTORIES_SOURCE=file requires an existing ROCSTORIES_FILE. "
            f"Current ROCSTORIES_FILE={ROCSTORIES_FILE!r}"
        )

    errors = []
    if file_path is not None and ROCSTORIES_FILE:
        errors.append(f"file {file_path} does not exist")

    local_attempts = [True] if ROCSTORIES_LOCAL_FILES_ONLY else [True, False]
    for local_files_only in local_attempts:
        try:
            return load_rocstories_hub_rows(local_files_only=local_files_only)
        except Exception as exc:
            label = "cached Hub" if local_files_only else "online Hub"
            errors.append(f"{label}: {exc}")

    raise RuntimeError(
        "Could not load ROCStories automatically. "
        "Set ROCSTORIES_FILE to a local CSV/TSV/JSONL/TXT file, or set "
        "ROCSTORIES_HUB_CANDIDATES to a Hugging Face dataset id list such as "
        "`roc_stories,story_cloze:2016,story_cloze:2018`. Tried: "
        + " | ".join(errors)
    )


def build_rocstories_dataset(tokenizer, max_length):
    suffix_len = max_length - PROMPT_LEN
    if suffix_len <= 0:
        raise ValueError(f"PROMPT_LEN={PROMPT_LEN} must be smaller than MAX_SEQ_LEN={max_length}")

    rows = load_rocstories_rows()
    examples = []
    for row in rows:
        sentences = sentence_parts_from_row(row)
        needed = ROCSTORIES_PROMPT_SENTENCES + ROCSTORIES_TARGET_SENTENCES
        if len(sentences) < needed:
            continue
        prompt_text = " ".join(sentences[:ROCSTORIES_PROMPT_SENTENCES])
        target_text = " ".join(sentences[ROCSTORIES_PROMPT_SENTENCES:needed])
        prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=PROMPT_LEN,
            padding="max_length",
        )
        target = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=suffix_len,
            padding="max_length",
        )
        if sum(prompt["attention_mask"]) == 0 or sum(target["attention_mask"]) == 0:
            continue
        input_ids = prompt["input_ids"] + target["input_ids"]
        attention_mask = prompt["attention_mask"] + target["attention_mask"]
        examples.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "prompt_token_len": torch.tensor(sum(prompt["attention_mask"]), dtype=torch.long),
                "target_token_len": torch.tensor(sum(target["attention_mask"]), dtype=torch.long),
            }
        )
    if not examples:
        raise RuntimeError(
            f"No ROCStories examples found for split {ROCSTORIES_SPLIT}. "
            f"source={ROCSTORIES_SOURCE} file={ROCSTORIES_FILE!r}. "
            f"First row keys={list(rows[0].keys()) if rows else []}"
        )
    return examples


def build_stage2_dataloaders(tokenizer, train_size, batch_size, max_length):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    if DATASET_NAME == "rocstories":
        examples = build_rocstories_dataset(tokenizer, max_length)
        random.Random(SEED).shuffle(examples)
        train_size = min(train_size, max(1, int(0.9 * len(examples))))
        train_rows = examples[:train_size]
        val_rows = examples[train_size:] or examples[-min(len(examples), 100):]
        train_loader = DataLoader(
            train_rows,
            batch_size=batch_size,
            shuffle=True,
            num_workers=DATALOADER_NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=DATALOADER_NUM_WORKERS > 0,
        )
        val_loader = DataLoader(
            val_rows,
            batch_size=batch_size,
            shuffle=False,
            num_workers=DATALOADER_NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=DATALOADER_NUM_WORKERS > 0,
        )
        print(
            f"ROCStories batches: train={len(train_loader)} val={len(val_loader)} "
            f"examples={len(examples)} prompt_slots={PROMPT_LEN} target_slots={max_length - PROMPT_LEN} "
            f"split={ROCSTORIES_SPLIT}",
            flush=True,
        )
        return train_loader, val_loader

    try:
        ds = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            download_config=DownloadConfig(local_files_only=True),
        )
        print("loaded wikitext from local datasets cache", flush=True)
    except Exception as exc:
        print(f"local wikitext cache unavailable ({exc}) | trying online load", flush=True)
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_size = min(train_size, len(ds["train"]))
    small_train = ds["train"].select(range(train_size))
    small_val = ds["validation"]

    small_train = small_train.filter(lambda x: len(x["text"].strip()) > 10)
    small_val = small_val.filter(lambda x: len(x["text"].strip()) > 10)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_tok = small_train.map(tokenize, batched=True)
    val_tok = small_val.map(tokenize, batched=True)
    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(
        train_tok,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_tok,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )
    print(
        f"train batches: {len(train_loader)}  val batches: {len(val_loader)}  "
        f"max_length: {max_length}",
        flush=True,
    )
    return train_loader, val_loader
