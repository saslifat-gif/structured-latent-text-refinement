"""Rank generated candidate groups with DeepSeek and append JSONL records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Rank candidate groups with DeepSeek")
    parser.add_argument("--input", default="examples/deepseek_candidate_groups.jsonl")
    parser.add_argument("--output", default="examples/deepseek_ranking_records.jsonl")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--max_groups", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--ranking_prompt_version", default="v1")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def load_env_file(path):
    env_path = Path(path)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_messages(group, version):
    candidate_lines = []
    for cand in group["candidates"]:
        candidate_lines.append(
            f"ID: {cand['candidate_id']}\n"
            f"Source: {cand.get('source', '')}\n"
            f"Continuation: {cand.get('text', '')}"
        )
    rubric = (
        "Rank the candidate story continuations for the prompt. Score each candidate from 1 to 5 "
        "for coherence, prompt_relevance, repetition, and ending_quality. Higher is better; "
        "for repetition, 5 means least repetitive. Return JSON only with keys: ranking, scores, notes. "
        "ranking must be a list of candidate IDs from best to worst. scores must map candidate IDs "
        "to numeric fields coherence, prompt_relevance, repetition, ending_quality, overall."
    )
    user = (
        f"Rubric version: {version}\n\n"
        f"Prompt:\n{group['prompt']}\n\n"
        f"Reference, for context only; do not require exact copying:\n{group.get('reference', '')}\n\n"
        "Candidates:\n\n"
        + "\n\n".join(candidate_lines)
    )
    return [
        {"role": "system", "content": rubric},
        {"role": "user", "content": user},
    ]


def call_deepseek(base_url, api_key, model, messages, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body}") from exc


def extract_json_text(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_ranking_response(response_json):
    content = (
        response_json.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return extract_json_text(content), content


def stable_run_id(group, raw_response):
    payload = {
        "prompt_id": group.get("prompt_id"),
        "candidates": group.get("candidates", []),
        "response": raw_response,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"deepseek-{digest}"


def append_record(path, group, response_json, response_text, parsed_scores, args, model):
    raw_response = json.dumps(response_json, ensure_ascii=True)
    metadata = group.get("metadata", {})
    record = {
        "schema": "deepseek_ranking_record_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": stable_run_id(group, raw_response),
        "prompt_id": group.get("prompt_id", ""),
        "prompt": group.get("prompt", ""),
        "reference": group.get("reference", ""),
        "source_file": args.input,
        "generator_checkpoint": metadata.get("code_prior", ""),
        "refiner_checkpoint": metadata.get("metric_refiner", ""),
        "candidates": group.get("candidates", []),
        "judge": {
            "provider": "deepseek",
            "model": model,
            "ranking_prompt_version": args.ranking_prompt_version,
        },
        "deepseek_raw_response": raw_response,
        "deepseek_message_text": response_text,
        "deepseek_parsed_response": response_json,
        "parsed_scores": parsed_scores,
        "notes": "",
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    load_env_file(args.env)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is empty. Put it in .env or export it before running.")
    base_url = args.base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"candidate input not found: {args.input}")
    count = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            group = json.loads(line)
            messages = build_messages(group, args.ranking_prompt_version)
            response_json, _raw_body = call_deepseek(base_url, api_key, model, messages, args.timeout)
            parsed_scores, response_text = parse_ranking_response(response_json)
            append_record(args.output, group, response_json, response_text, parsed_scores, args, model)
            count += 1
            print(f"ranked {group.get('prompt_id', count)} -> {args.output}", flush=True)
            if args.max_groups and count >= args.max_groups:
                break
            if args.sleep > 0:
                time.sleep(args.sleep)
    print(f"ranked {count} groups", flush=True)


if __name__ == "__main__":
    main()
