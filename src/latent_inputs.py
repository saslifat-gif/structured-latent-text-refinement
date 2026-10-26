"""Shared input conventions for draft-conditioned training and inference."""

import torch


def encode_prompt_latents(encoder, decoder, input_ids, attention_mask, prompt_len):
    """Encode only prompt slots so bidirectional attention cannot see the target."""
    return decoder.compress(
        encoder(input_ids[:, :prompt_len], attention_mask[:, :prompt_len])
    )


def pack_prompt_suffix_inputs(tokenizer, prompts, suffixes, prompt_slots, suffix_slots):
    """Place drafts at the fixed suffix boundary used by the training dataset."""
    if prompt_slots <= 0 or suffix_slots <= 0:
        raise ValueError("prompt_slots and suffix_slots must both be positive")
    if len(prompts) != len(suffixes):
        raise ValueError("prompts and suffixes must have the same batch size")
    prompt_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=prompt_slots,
        padding="max_length",
        truncation=True,
    )
    suffix_inputs = tokenizer(
        suffixes,
        return_tensors="pt",
        max_length=suffix_slots,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    return {
        "input_ids": torch.cat([prompt_inputs["input_ids"], suffix_inputs["input_ids"]], dim=1),
        "attention_mask": torch.cat(
            [prompt_inputs["attention_mask"], suffix_inputs["attention_mask"]], dim=1
        ),
    }
