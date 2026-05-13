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
    ROCSTORIES_PROMPT_SENTENCES,
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

    for prompt_key, continuation_key in (("prompt", "continuation"), ("Prompt", "Continuation")):
        if row.get(prompt_key) and row.get(continuation_key):
            parts = split_story_sentences(row[prompt_key]) + split_story_sentences(row[continuation_key])
            if len(parts) >= ROCSTORIES_PROMPT_SENTENCES + ROCSTORIES_TARGET_SENTENCES:
                return parts

    for key in ("story", "text", "full_text", "Story", "Text"):
        if row.get(key):
            return split_story_sentences(row[key])
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


def build_rocstories_dataset(tokenizer, max_length):
    if not ROCSTORIES_FILE:
        raise RuntimeError(
            "SLTR_DATASET=rocstories requires ROCSTORIES_FILE=/path/to/rocstories.csv"
        )
    suffix_len = max_length - PROMPT_LEN
    if suffix_len <= 0:
        raise ValueError(f"PROMPT_LEN={PROMPT_LEN} must be smaller than MAX_SEQ_LEN={max_length}")

    examples = []
    for row in read_rocstories_rows(ROCSTORIES_FILE):
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
            f"No ROCStories examples found in {ROCSTORIES_FILE} for split {ROCSTORIES_SPLIT}"
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
