"""Stable story identities and a shared train/validation/test manifest."""

import hashlib
import json
import os
from pathlib import Path

SPLIT_VERSION = 1
DEFAULT_MANIFEST = "results/rocstories/split_manifest.json"


def story_id(sentences):
    normalized = [" ".join(str(s).casefold().split()) for s in sentences[:5]]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False).encode()).hexdigest()


def make_split_manifest(ids, seed=1337):
    ids = sorted(set(ids))
    splits = {name: [] for name in ("train", "validation", "test")}
    for identity in ids:
        bucket = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest(), 16) % 100
        name = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        splits[name].append(identity)
    return {
        "version": SPLIT_VERSION,
        "seed": seed,
        "dataset_fingerprint": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "splits": splits,
    }


def load_split_manifest(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing split manifest: {path}. Train Stage 1 on ROCStories first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    groups = manifest.get("splits", {})
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Split manifest must contain train, validation, and test IDs")
    ids = [identity for group in groups.values() for identity in group]
    expected = make_split_manifest(ids, manifest.get("seed", 1337))
    if manifest != expected:
        raise ValueError("Invalid split manifest: identities, fingerprint, or assignments differ")
    return manifest


def ensure_split_manifest(path, ids, seed=1337):
    expected = make_split_manifest(ids, seed)
    path = Path(path)
    if path.exists():
        if load_split_manifest(path) != expected:
            raise ValueError("Dataset or split seed differs from the saved manifest; use a new manifest path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    return expected


def split_metadata(manifest):
    return {
        "rocstories_split_version": manifest["version"],
        "rocstories_split_seed": manifest["seed"],
        "rocstories_split_fingerprint": manifest["dataset_fingerprint"],
    }


def validate_checkpoint_split(checkpoint, manifest, label):
    expected = split_metadata(manifest)
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{label} lacks matching held-out split metadata; retrain it with this manifest")
