"""Generate prompt-only speculative ROCStories expansions with DeepSeek.

The model sees only the first two ROCStories sentences. It never sees the
target suffix. Output is JSONL keyed by normalized prompt text so training
scripts can join expansions back to shuffled dataloader batches.
"""

from __future__ import annotations

import argparse
import http.client
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import stage2_data as s2data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DeepSeek ROCStories prompt expansions")
    parser.add_argument("--output", default="examples/deepseek_prompt_expansions.jsonl")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_expansions", type=int, default=3)
    parser.add_argument("--sentences_per_expansion", type=int, default=3)
    parser.add_argument("--model", default="")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def normalize_prompt(text):
    return " ".join(str(text).strip().split()).lower()


def build_prompts(local_files_only=False):
    old_local = s2data.ROCSTORIES_LOCAL_FILES_ONLY
    s2data.ROCSTORIES_LOCAL_FILES_ONLY = local_files_only
    try:
        rows = s2data.load_rocstories_rows()
    finally:
        s2data.ROCSTORIES_LOCAL_FILES_ONLY = old_local
    prompts = []
    seen = set()
    needed = s2data.ROCSTORIES_PROMPT_SENTENCES + s2data.ROCSTORIES_TARGET_SENTENCES
    for idx, row in enumerate(rows):
        parts = s2data.sentence_parts_from_row(row)
        if len(parts) < needed:
            continue
        prompt = " ".join(parts[: s2data.ROCSTORIES_PROMPT_SENTENCES]).strip()
        key = normalize_prompt(prompt)
        if not prompt or key in seen:
            continue
        seen.add(key)
        prompts.append({"prompt_id": idx, "prompt": prompt, "prompt_key": key})
    return prompts


def load_done_keys(path):
    output = Path(path)
    if not output.exists():
        return set()
    keys = set()
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("prompt_key") or normalize_prompt(record.get("prompt", ""))
            if key:
                keys.add(key)
    return keys


def build_messages(prompt, num_expansions, sentences_per_expansion):
    system = (
        "You expand two-sentence ROCStories prompts into plausible future story plans. "
        "Use only the given prompt. Do not claim to know the hidden target story. "
        "Keep the same names/entities. Avoid repetition. Return JSON only."
    )
    user = (
        f"Given this two-sentence prompt, write {num_expansions} plausible future continuations. "
        f"Each continuation should be {sentences_per_expansion} short sentences. "
        "Return JSON with exactly one key, expansions, whose value is a list of strings.\n\n"
        f"Prompt:\n{prompt}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_deepseek(base_url, api_key, model, messages, temperature, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
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


def call_deepseek_with_retry(base_url, api_key, model, messages, temperature, timeout, retries, retry_sleep):
    last_exc = None
    for attempt in range(max(1, retries + 1)):
        try:
            return call_deepseek(base_url, api_key, model, messages, temperature, timeout)
        except RuntimeError as exc:
            # Authentication and malformed requests will not heal with retry.
            text = str(exc)
            if "HTTP 401" in text or "HTTP 400" in text or "HTTP 403" in text:
                raise
            last_exc = exc
        except (TimeoutError, urllib.error.URLError, http.client.RemoteDisconnected, ConnectionError) as exc:
            last_exc = exc
        if attempt >= retries:
            break
        wait = retry_sleep * (2 ** attempt)
        print(f"warning: DeepSeek call failed ({last_exc}); retry {attempt + 1}/{retries} in {wait:.1f}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"DeepSeek call failed after {retries + 1} attempts: {last_exc}") from last_exc


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


def parse_expansions(response_json, raw_text, num_expansions):
    content = (
        response_json.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    parsed = extract_json_text(content)
    expansions = []
    if isinstance(parsed, dict):
        value = parsed.get("expansions") or parsed.get("continuations")
        if isinstance(value, list):
            expansions = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(parsed, list):
        expansions = [str(item).strip() for item in parsed if str(item).strip()]
    if not expansions:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines:
            line = re.sub(r"^\s*\d+[\).\:-]\s*", "", line).strip()
            if line:
                expansions.append(line)
    expansions = [" ".join(item.split()) for item in expansions]
    return expansions[:num_expansions], content or raw_text


def append_record(path, item, expansions, response_json, response_text, args, model):
    record = {
        "schema": "deepseek_prompt_expansion_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_id": item["prompt_id"],
        "prompt": item["prompt"],
        "prompt_key": item["prompt_key"],
        "expansions": expansions,
        "generator": {
            "provider": "deepseek",
            "model": model,
            "temperature": args.temperature,
            "num_expansions": args.num_expansions,
            "sentences_per_expansion": args.sentences_per_expansion,
        },
        "deepseek_message_text": response_text,
        "deepseek_raw_response": response_json,
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

    output = Path(args.output)
    if args.overwrite and output.exists():
        output.unlink()
    done_keys = load_done_keys(output)
    prompts = build_prompts(local_files_only=args.local_files_only)
    prompts = prompts[args.start :]
    print(
        f"loaded prompts={len(prompts)} done={len(done_keys)} output={output} model={model}",
        flush=True,
    )

    written = 0
    for item in prompts:
        if item["prompt_key"] in done_keys:
            continue
        messages = build_messages(item["prompt"], args.num_expansions, args.sentences_per_expansion)
        response_json, raw_body = call_deepseek_with_retry(
            base_url,
            api_key,
            model,
            messages,
            args.temperature,
            args.timeout,
            args.retries,
            args.retry_sleep,
        )
        expansions, response_text = parse_expansions(response_json, raw_body, args.num_expansions)
        if not expansions:
            print(f"warning: no expansions parsed for prompt_id={item['prompt_id']}", flush=True)
            expansions = [item["prompt"]]
        append_record(args.output, item, expansions, response_json, response_text, args, model)
        done_keys.add(item["prompt_key"])
        written += 1
        print(
            f"expanded {written} prompt_id={item['prompt_id']} expansions={len(expansions)}",
            flush=True,
        )
        if args.max_examples and written >= args.max_examples:
            break
        if args.sleep > 0:
            time.sleep(args.sleep)
    print(f"wrote {written} new expansion records -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
