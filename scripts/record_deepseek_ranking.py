"""Append DeepSeek preference/ranking API outputs to a JSONL audit file."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Record DeepSeek ranking outputs for offline RLHF distillation")
    parser.add_argument("--output", default="examples/deepseek_ranking_records.jsonl")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--prompt_id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference", default="")
    parser.add_argument("--source_file", default="")
    parser.add_argument("--generator_checkpoint", default="")
    parser.add_argument("--refiner_checkpoint", default="")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate continuation text. Repeat this flag.")
    parser.add_argument("--candidate_jsonl", default="", help="Optional JSONL with candidate objects.")
    parser.add_argument("--deepseek_model", default="deepseek-chat")
    parser.add_argument("--ranking_prompt_version", default="v1")
    parser.add_argument("--deepseek_response", default="", help="Path to raw DeepSeek API response text/JSON.")
    parser.add_argument("--deepseek_response_text", default="", help="Raw DeepSeek API response text.")
    parser.add_argument("--parsed_scores", default="", help="Optional JSON object/string with parsed scores or ranking.")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def read_text_arg(path_value, inline_value):
    if inline_value:
        return inline_value
    if path_value:
        return Path(path_value).read_text(encoding="utf-8")
    return ""


def maybe_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_candidates(args):
    candidates = []
    for idx, text in enumerate(args.candidate):
        candidates.append({"candidate_id": f"c{idx}", "text": text})
    if args.candidate_jsonl:
        with Path(args.candidate_jsonl).open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "candidate_id" not in obj:
                    obj["candidate_id"] = f"j{line_idx}"
                candidates.append(obj)
    if not candidates:
        raise SystemExit("At least one --candidate or --candidate_jsonl row is required.")
    return candidates


def stable_run_id(args, candidates, raw_response):
    payload = {
        "prompt_id": args.prompt_id,
        "prompt": args.prompt,
        "candidates": candidates,
        "response": raw_response,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"deepseek-{digest}"


def main():
    args = parse_args()
    candidates = load_candidates(args)
    raw_response = read_text_arg(args.deepseek_response, args.deepseek_response_text)
    parsed_response = maybe_json(raw_response)
    parsed_scores = maybe_json(args.parsed_scores)
    run_id = args.run_id or stable_run_id(args, candidates, raw_response)
    record = {
        "schema": "deepseek_ranking_record_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "prompt_id": args.prompt_id,
        "prompt": args.prompt,
        "reference": args.reference,
        "source_file": args.source_file,
        "generator_checkpoint": args.generator_checkpoint,
        "refiner_checkpoint": args.refiner_checkpoint,
        "candidates": candidates,
        "judge": {
            "provider": "deepseek",
            "model": args.deepseek_model,
            "ranking_prompt_version": args.ranking_prompt_version,
        },
        "deepseek_raw_response": raw_response,
        "deepseek_parsed_response": parsed_response,
        "parsed_scores": parsed_scores,
        "notes": args.notes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"appended {run_id} to {output}", flush=True)


if __name__ == "__main__":
    main()
