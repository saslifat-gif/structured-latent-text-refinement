"""Stage 1 latent autoencoder inference helper.

The filename keeps the requested spelling: ``stage1_inferencen.py``.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import BertTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parallel_decoder import BertEncoder, ParallelDecoder, cached_from_pretrained


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate
    root_candidate = PROJECT_ROOT / path
    if root_candidate.exists():
        return root_candidate
    return candidate


def load_stage1(stage1_path, device):
    stage1_path = resolve_path(stage1_path)
    checkpoint = torch.load(stage1_path, map_location=device, weights_only=False)
    latent_dim = int(checkpoint.get("latent_dim", 256))
    max_length = int(checkpoint.get("max_length", 64))

    tokenizer = cached_from_pretrained(BertTokenizer)
    encoder = BertEncoder().to(device).eval()
    decoder = ParallelDecoder(latent_dim=latent_dim).to(device).eval()
    decoder.load_state_dict(checkpoint["decoder"])

    print(
        f"loaded stage1 checkpoint: {stage1_path} "
        f"| latent_dim={latent_dim} max_length={max_length} "
        f"| val_loss={checkpoint.get('val_loss', 'n/a')}",
        flush=True,
    )
    return tokenizer, encoder, decoder, checkpoint


def decode_ids(tokenizer, token_ids):
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return " ".join(text.split())


def reconstruct_texts(texts, tokenizer, encoder, decoder, device, max_length):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        hidden = encoder(input_ids, attention_mask)
        logits = decoder(hidden, residual_weight=0.0)
        pred_ids = logits.argmax(dim=-1)

    rows = []
    for idx in range(input_ids.size(0)):
        valid = attention_mask[idx].bool()
        rows.append(
            {
                "input": texts[idx],
                "original": decode_ids(tokenizer, input_ids[idx][valid].detach().cpu()),
                "reconstruction": decode_ids(tokenizer, pred_ids[idx][valid].detach().cpu()),
            }
        )
    return rows


def read_texts(args):
    texts = list(args.text or [])
    if args.text_file:
        text_path = resolve_path(args.text_file)
        with open(text_path, "r", encoding="utf-8") as handle:
            texts.extend(line.strip() for line in handle if line.strip())
    if not texts:
        texts = ["the cat sat on the mat"]
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Run direct Stage 1 BERT-latent autoencoder reconstruction."
    )
    parser.add_argument("--stage1", default="stage1_best.pt", help="Stage 1 checkpoint path.")
    parser.add_argument(
        "--text",
        action="append",
        help="Text to reconstruct. Can be passed more than once.",
    )
    parser.add_argument("--text_file", help="Optional newline-delimited text file.")
    parser.add_argument(
        "--max_length",
        type=int,
        help="Tokenizer length. Defaults to the checkpoint max_length.",
    )
    parser.add_argument("--jsonl", action="store_true", help="Print one JSON object per row.")
    args = parser.parse_args()

    device = default_device()
    tokenizer, encoder, decoder, checkpoint = load_stage1(args.stage1, device)
    max_length = args.max_length or int(checkpoint.get("max_length", 64))

    rows = reconstruct_texts(
        read_texts(args),
        tokenizer,
        encoder,
        decoder,
        device,
        max_length=max_length,
    )

    for row in rows:
        if args.jsonl:
            print(json.dumps(row, ensure_ascii=False))
        else:
            print(f"\ninput:          {row['input']}")
            print(f"tokenized:      {row['original']}")
            print(f"reconstruction: {row['reconstruction']}")


if __name__ == "__main__":
    main()
