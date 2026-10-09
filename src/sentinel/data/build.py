"""Deterministic dataset builder.

Each sample's random stream is ``SeedSequence(root_seed, spawn_key=(split,
class, index))``, so a sample depends only on its own coordinates: builds are
identical regardless of worker count or iteration order, and any single sample
can be regenerated in isolation.

Splits are independent by construction (different spawn keys), so there is no
leakage between train, validation, test, and the domain-shift sets.

Usage (``python -m sentinel.data`` takes the same options)::

    sentinel data build --config configs/dataset/opir_v2.yaml \\
        --out data/opir_v2 --workers 4
"""

import argparse
import hashlib
import json
import sys
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.data.config import DatasetConfig, Priors, load_dataset_config
from sentinel.data.generate import METADATA_FIELDS, generate_sample, params_to_json
from sentinel.sim.opir.sensor import frame_times
from sentinel.taxonomy import EVENT_CLASSES

MANIFEST_NAME = "manifest.json"
_CHUNK = 200


def split_key(name: str) -> int:
    """Stable integer key for a split name (independent of split order)."""
    return zlib.crc32(name.encode())


def sample_rng(
    root_seed: int, split: str, class_index: int, index: int
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            root_seed, spawn_key=(split_key(split), class_index, index)
        )
    )


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 over array names, dtypes, shapes, and bytes (sorted by name).

    Hashing contents rather than the ``.npz`` file avoids the zip timestamps
    that make file bytes differ between otherwise identical builds.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _generate_chunk(
    args: tuple[int, str, int, int, int, str, float],
) -> tuple[np.ndarray, list[dict[str, float]], list[str]]:
    root_seed, split, class_index, start, stop, priors_json, window_s = args
    priors = Priors.model_validate_json(priors_json)
    label = EVENT_CLASSES[class_index]
    signals, metadata, params = [], [], []
    for index in range(start, stop):
        sample = generate_sample(
            label, priors, window_s, sample_rng(root_seed, split, class_index, index)
        )
        signals.append(sample.signal)
        metadata.append(sample.metadata)
        params.append(params_to_json(sample.params))
    return np.stack(signals), metadata, params


def generate_samples(
    root_seed: int,
    stream: str,
    label: str,
    count: int,
    priors: Priors,
    window_s: float,
    workers: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate ``count`` samples of one class on a named seed stream.

    Uses the dataset seeding scheme, so a ``stream`` name that is not a split
    name yields samples independent of every split (e.g. extra background
    windows for false-alarm estimation).

    Returns:
        Signals (count, T) and numeric metadata arrays keyed by field name.
    """
    class_index = EVENT_CLASSES.index(label)
    priors_json = priors.model_dump_json()
    tasks = [
        (
            root_seed,
            stream,
            class_index,
            start,
            min(start + _CHUNK, count),
            priors_json,
            window_s,
        )
        for start in range(0, count, _CHUNK)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_generate_chunk, tasks))
    else:
        results = [_generate_chunk(task) for task in tasks]
    signals = np.concatenate([r[0] for r in results])
    metadata = [m for r in results for m in r[1]]
    return signals, {
        f: np.array([m[f] for m in metadata], dtype=np.float64) for f in METADATA_FIELDS
    }


def build_split(
    config: DatasetConfig, split_name: str, workers: int = 1
) -> dict[str, np.ndarray]:
    """Generate all samples of one split as a dict of arrays."""
    split = next(s for s in config.splits if s.name == split_name)
    priors_json = config.priors_for(split).model_dump_json()
    tasks = [
        (
            config.root_seed,
            split.name,
            c,
            start,
            min(start + _CHUNK, split.samples_per_class),
            priors_json,
            config.window_s,
        )
        for c in range(len(EVENT_CLASSES))
        for start in range(0, split.samples_per_class, _CHUNK)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_generate_chunk, tasks))
    else:
        results = [_generate_chunk(task) for task in tasks]

    signals = np.concatenate([r[0] for r in results])
    metadata = [m for r in results for m in r[1]]
    params = [p for r in results for p in r[2]]
    labels = np.repeat(
        np.arange(len(EVENT_CLASSES), dtype=np.int8), split.samples_per_class
    )
    frame_rate = config.priors_for(split).sensor.frame_rate_hz
    arrays: dict[str, np.ndarray] = {
        "signals": signals,
        "labels": labels,
        "times": frame_times(config.window_s, frame_rate).astype(np.float32),
        "params": np.array(params),
    }
    for field in METADATA_FIELDS:
        arrays[f"meta_{field}"] = np.array(
            [m[field] for m in metadata], dtype=np.float64
        )
    return arrays


def _split_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    labels, snr = arrays["labels"], arrays["meta_peak_snr"]
    per_class = {}
    for index, name in enumerate(EVENT_CLASSES):
        values = snr[labels == index]
        per_class[name] = {
            "count": int(values.size),
            "peak_snr_p10_p50_p90": [
                round(float(q), 3) for q in np.quantile(values, [0.1, 0.5, 0.9])
            ],
        }
    return {"num_samples": int(labels.size), "classes": per_class}


def build_dataset(
    config: DatasetConfig, out_dir: str | Path, workers: int = 1
) -> dict[str, Any]:
    """Build every split into ``out_dir`` and write ``manifest.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    manifest: dict[str, Any] = {
        "dataset": config.name,
        "version": config.version,
        "root_seed": config.root_seed,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "classes": list(EVENT_CLASSES),
        "window_s": config.window_s,
        "environment": {
            "sentinel": version("sentinel"),
            "numpy": np.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "splits": {},
    }
    for split in config.splits:
        arrays = build_split(config, split.name, workers)
        filename = f"{split.name}.npz"
        np.savez_compressed(out / filename, **arrays)  # type: ignore[arg-type]
        manifest["splits"][split.name] = {
            "file": filename,
            "description": split.description,
            "content_sha256": content_hash(arrays),
            **_split_summary(arrays),
        }
    manifest["config"] = json.loads(config_json)
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def compare_manifests(built: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    """Differences in config or split content hashes (empty if identical)."""
    problems = []
    if built["config_sha256"] != reference["config_sha256"]:
        problems.append("config differs from the reference manifest")
    for name, ref in reference["splits"].items():
        got = built["splits"].get(name)
        if got is None:
            problems.append(f"split {name!r} missing")
        elif got["content_sha256"] != ref["content_sha256"]:
            problems.append(f"split {name!r} content hash differs")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a SENTINEL OPIR dataset.")
    parser.add_argument("--config", required=True, help="Dataset YAML config")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Multiply split sizes (quick builds)"
    )
    parser.add_argument(
        "--verify", help="Reference manifest; exit 1 if config or contents differ"
    )
    args = parser.parse_args(argv)

    config = load_dataset_config(args.config)
    if args.scale != 1.0:
        config = config.scaled(args.scale)
    manifest = build_dataset(config, args.out, args.workers)
    for name, split in manifest["splits"].items():
        print(
            f"{name:16s} {split['num_samples']:6d} samples  {split['content_sha256'][:16]}"
        )

    if args.verify:
        reference = json.loads(Path(args.verify).read_text())
        problems = compare_manifests(manifest, reference)
        for problem in problems:
            print(f"MISMATCH: {problem}", file=sys.stderr)
        if problems:
            return 1
        print("Dataset matches the reference manifest.")
    return 0
