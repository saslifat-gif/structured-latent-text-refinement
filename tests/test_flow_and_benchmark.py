"""Offline checks of production flow paths, dataset splits, and metric inputs."""

import ast
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_latent_inputs import load_function, ROOT

sys.path.insert(0, str(ROOT / "src"))
from rocstories_splits import (
    ensure_split_manifest, load_split_manifest, make_split_manifest,
    split_metadata, story_id, validate_checkpoint_split,
)


def constant(path, name):
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


class Scalar(float):
    def clone(self):
        return self

    def unsqueeze(self, dim):
        return self

    def __add__(self, other):
        return Scalar(float(self) + other)

    __radd__ = __add__

    def __mul__(self, other):
        return Scalar(float(self) * other)

    __rmul__ = __mul__


class FlowTests(unittest.TestCase):
    def test_both_samplers_reach_supervised_local_endpoint(self):
        fraction = constant("src/stage2_config.py", "FLOW_REFINE_TARGET_FRACTION")
        for steps in (1, 4, 16):
            for training_sampler in (False, True):
                with self.subTest(steps=steps, training_sampler=training_sampler):
                    path = "src/stage2_riemannian.py" if training_sampler else "src/inference_stage2_conditional.py"
                    scale_path = "src/stage2_config.py" if training_sampler else path
                    env = {
                        "ODE_STEPS": 16, "FLOW_REFINE_SCALE": constant(scale_path, "FLOW_REFINE_SCALE"),
                        "STRUCTURED_TARGET_START": False, "START_NOISE_STD_FRAC": 0.0,
                        "TARGET_LATENT_STD": 0.28,
                        "torch": types.SimpleNamespace(
                            full=lambda *args, **kwargs: None, randn_like=lambda z: Scalar(0),
                            amp=types.SimpleNamespace(autocast=lambda *a, **kw: nullcontext()),
                        ),
                        "suffix_positions": lambda *args: None,
                        "natural_velocity": lambda *args: (fraction, None),
                        "calibrate_latents": lambda z, *args: z,
                    }
                    name = "generate_suffix" if training_sampler else "sample_suffix_latents"
                    sampler = load_function(path, name, env)
                    kwargs = {"start_mlp": lambda *args: Scalar(0)} if training_sampler else {"z_start": Scalar(0)}
                    result = sampler(None, None, None, 1, 1, 1, types.SimpleNamespace(type="cpu"), steps=steps, **kwargs)
                    self.assertAlmostEqual(result[0], fraction)

    def test_endpoint_estimator_and_mse_match_local_path(self):
        tree = ast.parse((ROOT / "src/stage2_losses.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "flow_matching_loss")
        names = {"z_refine_target", "z_t", "v_true", "z_x0", "x0_loss"}
        assignments = [n for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
        env = {
            "z_noise": Scalar(0), "z_target": Scalar(1), "t_seq": Scalar(0.3),
            "FLOW_REFINE_TARGET_FRACTION": 0.05, "FLOW_REFINE_SCALE": 1.0,
            "v_nat_pred": Scalar(0.05), "z_x0_flat": Scalar(0.05), "z_flat": Scalar(1),
            "target_mask": None, "valid_token_latents": lambda z, mask: z,
            "F": types.SimpleNamespace(mse_loss=lambda a, b: (a - b) ** 2),
        }
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "flow_endpoint", "exec"), env)
        self.assertAlmostEqual(env["z_x0"], env["z_refine_target"])
        self.assertAlmostEqual(env["x0_loss"], 0.0)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.ids = [story_id([f"Story {i} sentence {j}." for j in range(5)]) for i in range(1000)]

    def test_splits_are_disjoint_stable_and_duplicate_safe(self):
        manifest = make_split_manifest(self.ids)
        self.assertEqual(manifest, make_split_manifest(list(reversed(self.ids)) + self.ids[:20]))
        groups = [set(group) for group in manifest["splits"].values()]
        self.assertEqual(set.union(*groups), set(self.ids))
        for i, group in enumerate(groups):
            self.assertTrue(group)
            for other in groups[i + 1:]:
                self.assertFalse(group & other)
        self.assertEqual(story_id(["Hello  WORLD."]), story_id([" hello world. "]))

    def test_saved_manifest_rejects_changed_corpus_or_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            manifest = ensure_split_manifest(path, self.ids)
            self.assertEqual(load_split_manifest(path), manifest)
            for ids, seed in [(self.ids[:-1], 1337), (self.ids, 42)]:
                with self.assertRaises(ValueError):
                    ensure_split_manifest(path, ids, seed)
            manifest["splits"]["test"].append(manifest["splits"]["train"][0])
            import json
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_split_manifest(path)

    def test_old_and_mismatched_checkpoints_are_rejected(self):
        manifest = make_split_manifest(self.ids)
        validate_checkpoint_split(split_metadata(manifest), manifest, "Stage 1")
        for checkpoint in ({}, split_metadata(make_split_manifest(self.ids, 42))):
            with self.assertRaises(ValueError):
                validate_checkpoint_split(checkpoint, manifest, "Stage 1")

    def test_training_and_benchmark_use_separate_manifest_groups(self):
        records = [{"sentence_list": [f"Story {i} sentence {j}." for j in range(5)]} for i in range(1000)]
        examples = [{"story_id": story_id(row["sentence_list"])} for row in records]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            env = {
                "torch": types.SimpleNamespace(Generator=lambda: types.SimpleNamespace(manual_seed=lambda seed: None)),
                "SEED": 1337, "DATASET_NAME": "rocstories", "ROCSTORIES_SPLIT_MANIFEST": str(path),
                "ensure_split_manifest": ensure_split_manifest,
                "build_rocstories_dataset": lambda *args: list(examples),
                "DataLoader": lambda rows, **kwargs: rows, "DATALOADER_NUM_WORKERS": 0,
                "seed_worker": None, "PROMPT_LEN": 32, "ROCSTORIES_SPLIT": "first2_last3",
            }
            training = load_function("src/stage2_data.py", "build_stage2_dataloaders", env)
            train, validation = training(None, 20, 8, 64)
            env = {
                "load_split_manifest": load_split_manifest, "make_split_manifest": make_split_manifest,
                "story_id": story_id, "tokenize_text": str.split,
            }
            benchmark = load_function("benchmark_rocstories.py", "rows_to_examples", env)
            fake_data = types.SimpleNamespace(sentence_parts_from_row=lambda row: row["sentence_list"])
            with patch.dict(sys.modules, {"stage2_data": fake_data}):
                test = benchmark(records, 30, 32, 64, "sentence", str(path))
                reordered = benchmark(list(reversed(records)), 30, 32, 64, "sentence", str(path))
                with self.assertRaises(ValueError):
                    benchmark(records[:-1], 30, 32, 64, "sentence", str(path))
            self.assertEqual(test, reordered)
            self.assertEqual(len(train), 20)
            used = {row["story_id"] for row in train + validation}
            self.assertFalse(used & {row["story_id"] for row in test})


class BenchmarkTests(unittest.TestCase):
    def test_hf_baseline_removes_full_left_padded_prompt(self):
        class Batch:
            def __init__(self, rows):
                self.rows = rows
                self.shape = (len(rows), len(rows[0]))

            def to(self, device):
                return self

        class Tokenizer:
            pad_token = "PAD"
            eos_token_id = 0

            def __call__(self, prompts, **kwargs):
                return {"input_ids": Batch([[0, 1, 2], [3, 4, 5]]),
                        "attention_mask": Batch([[0, 1, 1], [1, 1, 1]])}

            def decode(self, ids, **kwargs):
                return " ".join(str(token) for token in ids)

        class Model:
            device = "cpu"

            def to(self, device):
                return self

            def eval(self):
                pass

            def generate(self, **kwargs):
                return [[0, 1, 2, 9], [3, 4, 5, 8]]

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **kw: Model()),
        )
        args = types.SimpleNamespace(
            hf_baseline_model="fake", local_files_only=True, hf_baseline_dtype="auto",
            hf_baseline_batch_size=2, hf_baseline_max_new_tokens=1,
            hf_baseline_temperature=0.8, hf_baseline_top_p=0.95, hf_baseline_top_k=50,
        )
        env = {"torch": types.SimpleNamespace(
            device=lambda name: types.SimpleNamespace(type=name),
            cuda=types.SimpleNamespace(is_available=lambda: False), no_grad=nullcontext),
            "time": types.SimpleNamespace(perf_counter=lambda: 0)}
        generate = load_function("benchmark_rocstories.py", "generate_hf_baseline", env)
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            predictions, _ = generate([{"prompt": "short"}, {"prompt": "a longer prompt"}], args)
        self.assertEqual(predictions, ["9", "8"])

    def test_prompt_and_reference_budget_is_fixed_once(self):
        class Tokenizer:
            def __call__(self, text, max_length, add_special_tokens=True, **kwargs):
                tokens = text.split()
                if add_special_tokens:
                    tokens = ["CLS"] + tokens[:max_length - 2] + ["SEP"]
                return {"input_ids": tokens[:max_length]}

            def decode(self, tokens, **kwargs):
                return " ".join(token for token in tokens if token not in {"CLS", "SEP"})

        env = {"tokenize_text": str.split}
        fixed = load_function("benchmark_rocstories.py", "fixed_suffix_reference", env)
        env["fixed_suffix_reference"] = fixed
        canonical = load_function("benchmark_rocstories.py", "canonicalize_eval_rows", env)
        original = [{"prompt": "one two three four five six", "reference": "seven eight nine ten"}]
        rows = canonical(original, Tokenizer(), 6, 3)
        self.assertEqual(rows[0]["prompt"], "one two three four")
        self.assertEqual(rows[0]["reference"], "seven eight nine")
        self.assertEqual(original[0]["reference"], "seven eight nine ten")

    def test_external_predictions_must_match_exported_row_count(self):
        import csv
        import json
        reader = load_function("benchmark_rocstories.py", "load_prediction_file", {"Path": Path, "csv": csv, "json": json})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.txt"
            path.write_text("first\nsecond\n")
            self.assertEqual(reader(str(path), 2, "external"), ["first", "second"])
            with self.assertRaises(RuntimeError):
                reader(str(path), 1, "external")

    def test_all_local_models_receive_identical_references(self):
        scored = []
        env = {
            "compute_text_metrics": lambda predictions, refs, **kw: scored.append(list(refs)) or {"latency_per_sample": 1},
            "tokenize_text": str.split,
        }
        add = load_function("benchmark_rocstories.py", "add_metric_row", env)
        rows = [{"reference": "the same short reference", "prompt": "a shared prompt"}]
        args = types.SimpleNamespace(
            output_dir="unused", skip_gpt2=False, no_mauve=True, split_strategy="sentence",
            hf_baseline_model=None, diffusion_lm_file=None, skip_reported_diffusion_lm=True,
            skip_ours=False, draft_source="synthetic", max_seq_len=64, prompt_len=32,
        )
        env = {
            "ensure_output_dir": lambda path: Path(path), "add_metric_row": add,
            "generate_gpt2": lambda actual, args: (["output"], 1),
            "generate_ours": lambda actual, args, steps, dim: (["output"], 1, "synthetic corrupted target draft; controlled diagnostic"),
            "write_csv": lambda *args: None, "append_jsonl": lambda *args: None,
        }
        run = load_function("benchmark_rocstories.py", "run_table", env)
        run(args, "fair", rows, [768], [4])
        self.assertEqual(scored, [[rows[0]["reference"]]] * 2)
        diagnostic = add(model="Ours", rows=rows, predictions=["output"], latency=1, latent_dim=768,
                         steps=4, include_mauve=False, status="synthetic corrupted target draft", gpt2_latency_per_sample=1)
        self.assertIsNone(diagnostic["speedup_vs_gpt2"])


if __name__ == "__main__":
    unittest.main()
