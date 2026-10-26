"""Offline regressions for input boundaries, without pretrained model downloads."""

import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    """Small list-backed double for slicing, batching, and concatenating inputs."""

    device = "cpu"
    dtype = "float32"

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)

        def select(data, axes):
            if not axes:
                return data
            selected = data[axes[0]]
            if isinstance(axes[0], slice):
                return [select(row, axes[1:]) for row in selected]
            return select(selected, axes[1:])

        return Tensor(select(self.data, keys))

    @property
    def shape(self):
        shape, data = [], self.data
        while isinstance(data, list):
            shape.append(len(data))
            data = data[0] if data else None
        return tuple(shape)

    def size(self, dim):
        return self.shape[dim]

    def to(self, device):
        return self

    def expand(self, count, width):
        if len(self.data) != 1 or width != -1:
            raise AssertionError("unexpected test expansion")
        return Tensor(self.data * count)

    def new_full(self, shape, value):
        return Tensor([value] * shape[0])


def cat(tensors, dim):
    if dim != 1:
        raise AssertionError("unexpected concatenation axis")
    return Tensor([sum(parts, []) for parts in zip(*(t.data for t in tensors))])


TORCH = types.SimpleNamespace(cat=cat, no_grad=nullcontext)
spec = importlib.util.spec_from_file_location("latent_inputs", ROOT / "src/latent_inputs.py")
inputs = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"torch": TORCH}):
    spec.loader.exec_module(inputs)


def load_function(path, name, env):
    """Exercise production entrypoints without their model-loading side effects."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


class Tokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, texts, max_length, add_special_tokens=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            ids = [self.vocab.setdefault(word, len(self.vocab) + 10) for word in text.split()]
            if add_special_tokens:
                ids = [1] + ids[:max_length - 2] + [2]
            ids = ids[:max_length]
            rows.append(ids + [0] * (max_length - len(ids)))
        return {
            "input_ids": Tensor(rows),
            "attention_mask": Tensor([[int(token != 0) for token in row] for row in rows]),
        }

    def decode(self, tensor, **kwargs):
        words = {value: key for key, value in self.vocab.items()}
        return " ".join(words[token] for token in tensor.data if token in words)


class ContextualEncoder:
    def __call__(self, ids, mask):
        # Like bidirectional attention, each output depends on every visible token.
        return Tensor([
            [[token + sum(t * m for t, m in zip(row, visible))] for token in row]
            for row, visible in zip(ids.data, mask.data)
        ])


class InputRegressionTests(unittest.TestCase):
    def test_prompt_states_do_not_change_with_target(self):
        encoder = ContextualEncoder()
        decoder = types.SimpleNamespace(compress=lambda z: z)
        mask = Tensor([[1, 1, 1, 1]])
        first, second = Tensor([[1, 2, 10, 20]]), Tensor([[1, 2, 90, 80]])
        self.assertNotEqual(encoder(first, mask).data, encoder(second, mask).data)
        actual = inputs.encode_prompt_latents(encoder, decoder, first, mask, 2)
        other = inputs.encode_prompt_latents(encoder, decoder, second, mask, 2)
        self.assertEqual(actual.data, other.data)
        self.assertEqual(actual.shape, (1, 2, 1))

    def test_prompt_attention_mask_is_preserved(self):
        encoder = Mock(return_value="hidden")
        decoder = types.SimpleNamespace(compress=lambda z: z)
        inputs.encode_prompt_latents(encoder, decoder, Tensor([[1, 0, 9]]), Tensor([[1, 0, 1]]), 2)
        ids, mask = encoder.call_args.args
        self.assertEqual(ids.data, [[1, 0]])
        self.assertEqual(mask.data, [[1, 0]])

    def test_draft_prior_forward_uses_target_independent_prompt(self):
        encoder = ContextualEncoder()
        decoder = types.SimpleNamespace(compress=lambda z: z)
        model = Mock()
        env = {
            "torch": TORCH, "PROMPT_LEN": 2, "DRAFT_ALPHA": 0.7,
            "encoder": encoder, "decoder": decoder, "model": model,
            "make_synthetic_draft_ids": lambda ids, mask, *args: (ids, mask),
            "encode_latents": lambda ids, mask: decoder.compress(encoder(ids, mask)),
            "encode_prompt_latents": inputs.encode_prompt_latents,
            "suffix_positions": lambda *args: None,
            "noisy_draft_latents": lambda z, mask: z,
        }
        forward = load_function("scripts/train_draft_prior.py", "draft_prior_forward", env)
        mask = Tensor([[1, 1, 1, 1]])
        first = forward(Tensor([[1, 2, 10, 20]]), mask, 0, 0)
        second = forward(Tensor([[1, 2, 90, 80]]), mask, 0, 0)
        self.assertEqual(first[1].data, second[1].data)
        self.assertNotEqual(first[2].data, second[2].data)

    def manual_draft(self, prompt, draft, samples=1):
        tokenizer = Tokenizer()
        env = {
            "PROMPT_LEN": 8, "pack_prompt_suffix_inputs": inputs.pack_prompt_suffix_inputs,
            "encode_suffix_latents": lambda enc, dec, ids, mask: ids[:, 8:],
        }
        generate = load_function(
            "src/inference_stage2_conditional.py", "generate_manual_draft_latents", env
        )
        return generate(tokenizer, None, None, prompt, draft, 14, samples, "cpu")

    def test_short_manual_draft_is_preserved_and_batched(self):
        latents, drafts = self.manual_draft("short prompt", "keep these draft words", samples=3)
        self.assertEqual(drafts, ["keep these draft words"] * 3)
        self.assertEqual(latents.shape, (3, 6))

    def test_full_draft_prompt_prefix_is_removed_once(self):
        _, drafts = self.manual_draft("short prompt", "SHORT PROMPT keep these words")
        self.assertEqual(drafts, ["keep these words"])

    def test_prompt_mentioned_later_is_kept(self):
        _, drafts = self.manual_draft("short prompt", "repeat short prompt later")
        self.assertEqual(drafts, ["repeat short prompt later"])

    def test_partial_word_prefix_is_kept(self):
        _, drafts = self.manual_draft("cat", "caterpillars move slowly")
        self.assertEqual(drafts, ["caterpillars move slowly"])

    def test_long_prompt_does_not_displace_draft(self):
        _, drafts = self.manual_draft(" ".join(["prompt"] * 20), "draft stays here")
        self.assertEqual(drafts, ["draft stays here"])

    def test_packing_masks_truncation_and_multiple_rows(self):
        tokenizer = Tokenizer()
        packed = inputs.pack_prompt_suffix_inputs(
            tokenizer, ["a", "b c d e f"], ["one two three four", "five"], 4, 3
        )
        ids, mask = packed["input_ids"], packed["attention_mask"]
        self.assertEqual(ids.shape, (2, 7))
        self.assertEqual(mask.data, [[1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]])
        self.assertEqual(tokenizer.decode(ids[0, 4:]), "one two three")
        self.assertEqual(tokenizer.decode(ids[1, 4:]), "five")

    def test_invalid_slot_counts_and_batch_sizes_fail(self):
        for prompt_slots, suffix_slots in [(0, 3), (3, 0), (3, -1)]:
            with self.subTest(prompt_slots=prompt_slots, suffix_slots=suffix_slots):
                with self.assertRaises(ValueError):
                    inputs.pack_prompt_suffix_inputs(Tokenizer(), ["p"], ["d"], prompt_slots, suffix_slots)
        with self.assertRaises(ValueError):
            inputs.pack_prompt_suffix_inputs(Tokenizer(), ["p"], [], 3, 3)


if __name__ == "__main__":
    unittest.main()
