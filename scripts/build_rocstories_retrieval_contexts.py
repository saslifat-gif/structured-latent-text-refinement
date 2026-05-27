"""Build prompt-keyed ROCStories retrieval contexts.

For each ROCStories prompt, retrieve similar *other* ROCStories prompts and
store their target suffixes. The current row is excluded, so this is an
external-memory conditioning experiment rather than target leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import stage2_data as s2data


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had",
    "has", "have", "he", "her", "hers", "his", "i", "in", "is", "it", "its", "my",
    "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "then",
    "they", "to", "was", "we", "were", "with", "you",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build ROCStories retrieval contexts")
    parser.add_argument("--output", default="examples/rocstories_retrieval_contexts.jsonl")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_postings", type=int, default=5000)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_prompt(text):
    return " ".join(str(text).strip().split()).lower()


def tokens(text):
    return [
        tok
        for tok in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
        if tok not in STOPWORDS and len(tok) > 2
    ]


def load_examples(local_files_only):
    old_local = s2data.ROCSTORIES_LOCAL_FILES_ONLY
    s2data.ROCSTORIES_LOCAL_FILES_ONLY = local_files_only
    try:
        rows = s2data.load_rocstories_rows()
    finally:
        s2data.ROCSTORIES_LOCAL_FILES_ONLY = old_local
    examples = []
    needed = s2data.ROCSTORIES_PROMPT_SENTENCES + s2data.ROCSTORIES_TARGET_SENTENCES
    seen = set()
    for row_idx, row in enumerate(rows):
        parts = s2data.sentence_parts_from_row(row)
        if len(parts) < needed:
            continue
        prompt = " ".join(parts[: s2data.ROCSTORIES_PROMPT_SENTENCES]).strip()
        suffix = " ".join(parts[s2data.ROCSTORIES_PROMPT_SENTENCES : needed]).strip()
        key = normalize_prompt(prompt)
        if not prompt or not suffix or key in seen:
            continue
        seen.add(key)
        examples.append(
            {
                "row_id": row_idx,
                "prompt": prompt,
                "prompt_key": key,
                "suffix": suffix,
                "tokens": tokens(prompt),
            }
        )
    return examples


def build_index(examples, max_postings):
    doc_freq = Counter()
    for ex in examples:
        doc_freq.update(set(ex["tokens"]))
    n_docs = max(1, len(examples))
    idf = {
        token: math.log((n_docs + 1) / (df + 1)) + 1.0
        for token, df in doc_freq.items()
    }
    postings = defaultdict(list)
    for idx, ex in enumerate(examples):
        for token in set(ex["tokens"]):
            postings[token].append(idx)
    postings = {
        token: ids if max_postings <= 0 else ids[:max_postings]
        for token, ids in postings.items()
    }
    return idf, postings


def retrieve(query_idx, examples, idf, postings, top_k):
    query = examples[query_idx]
    q_counts = Counter(query["tokens"])
    scores = defaultdict(float)
    for token, q_count in q_counts.items():
        weight = idf.get(token, 1.0) * min(q_count, 2)
        for cand_idx in postings.get(token, []):
            if cand_idx == query_idx:
                continue
            cand = examples[cand_idx]
            if cand["prompt_key"] == query["prompt_key"]:
                continue
            scores[cand_idx] += weight
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [
        {
            "row_id": examples[idx]["row_id"],
            "prompt": examples[idx]["prompt"],
            "suffix": examples[idx]["suffix"],
            "score": score,
        }
        for idx, score in ranked
    ]


def main():
    args = parse_args()
    output = Path(args.output)
    if args.overwrite and output.exists():
        output.unlink()
    if output.exists():
        raise SystemExit(f"output exists: {output}. Use --overwrite to replace it.")
    examples = load_examples(args.local_files_only)
    idf, postings = build_index(examples, args.max_postings)
    limit = min(len(examples), args.max_examples) if args.max_examples else len(examples)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for idx in range(limit):
            retrieved = retrieve(idx, examples, idf, postings, args.top_k)
            record = {
                "schema": "rocstories_retrieval_context_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "prompt_id": examples[idx]["row_id"],
                "prompt": examples[idx]["prompt"],
                "prompt_key": examples[idx]["prompt_key"],
                "retrieved": retrieved,
                "retrieval": {
                    "method": "lexical_idf_prompt_overlap",
                    "top_k": args.top_k,
                    "max_postings": args.max_postings,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            if (idx + 1) % 1000 == 0:
                print(f"wrote {idx + 1}/{limit} retrieval records -> {output}", flush=True)
    print(f"wrote {limit} retrieval records -> {output}", flush=True)


if __name__ == "__main__":
    main()
