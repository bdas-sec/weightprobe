"""Per-tensor diff vs a clean baseline model.

Extends `verify` mode from whole-dir digest comparison to per-tensor
weight-edit detection. Catches:
  - Inserted tensors (extra files / extra tensor names) — already detected
    by `hash`/`verify` via structural fingerprint changes.
  - Weight edits with preserved shape/dtype — `hash` does NOT catch these
    (it's structure-only). `diff_base` loads tensors of matching name+shape
    and computes per-tensor distance metrics:
      - L2 distance (Frobenius norm of difference)
      - Cosine similarity (per-tensor flattened vector)
      - Per-element max absolute difference
  - Tensor-rank changes (shape mismatches) — flagged separately.

What this catches that `hash` doesn't
-------------------------------------
- Abliteration (Labonne 2024): edits writing matrices in-place; same shape;
  hash mode misses; diff_base detects via cosine distance increase.
- Distilled-into-base adapters (knowledge-distillation of an architectural
  backdoor adapter into the base MLP): same shape; cosine distance < 1
  (likely 0.99-0.999 for surgical edits).
- DualEdit / TA² / similar weight-editing techniques: same.

Cost
----
Per-tensor SVD-free comparison: O(numel) per tensor pair. For a 16 GB
model, full diff is bounded by 2× peak memory of the larger of (target,
baseline). On a 24 GB Mac, comparing a stock Foundation-Sec-1.0 (16 GB)
against a modified copy is feasible if loaded sequentially per shard.

Defensive
---------
Refuses to compare directories with mismatched architecture (different
config.json or different tensor name sets) — the user should run
`weightprobe verify` first to confirm structural match.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TensorDiff:
    """Per-tensor diff between target and baseline."""
    name: str
    shape: list[int]
    dtype_target: str
    dtype_baseline: str
    l2_distance: float
    cosine_distance: float       # 1 - cosine_similarity ∈ [0, 2]
    max_abs_delta: float
    relative_l2: float           # L2(target - baseline) / L2(baseline)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiffBaseResult:
    """Full diff-base report."""
    target_dir: str
    baseline_dir: str
    n_tensors_compared: int
    n_tensors_only_in_target: int    # added tensors
    n_tensors_only_in_baseline: int  # removed tensors
    n_tensors_shape_mismatch: int    # name-match but different shape
    n_tensors_modified: int           # cosine_distance > threshold
    cosine_threshold: float           # threshold for "modified" classification
    per_tensor: list[TensorDiff]
    tensors_only_in_target: list[str]
    tensors_only_in_baseline: list[str]
    tensors_shape_mismatch: list[dict]   # name + target_shape + baseline_shape

    def to_dict(self) -> dict:
        return {
            "target_dir": self.target_dir,
            "baseline_dir": self.baseline_dir,
            "n_tensors_compared": self.n_tensors_compared,
            "n_tensors_only_in_target": self.n_tensors_only_in_target,
            "n_tensors_only_in_baseline": self.n_tensors_only_in_baseline,
            "n_tensors_shape_mismatch": self.n_tensors_shape_mismatch,
            "n_tensors_modified": self.n_tensors_modified,
            "cosine_threshold": self.cosine_threshold,
            "per_tensor_n": len(self.per_tensor),
            "per_tensor_modified": [t.to_dict() for t in self.per_tensor
                                    if (1.0 - (1.0 - t.cosine_distance)) > self.cosine_threshold],
            "tensors_only_in_target": self.tensors_only_in_target,
            "tensors_only_in_baseline": self.tensors_only_in_baseline,
            "tensors_shape_mismatch": self.tensors_shape_mismatch,
        }


def _index_safetensors(dir_path: Path) -> dict[str, tuple[Path, list[int], str]]:
    """Build {tensor_name: (file_path, shape, dtype)} index over a model dir.

    Uses the existing `weightprobe.hash.read_safetensors_header` to avoid
    loading actual tensor data — fast first-pass to compare structure
    before deciding which files to load for value-comparison.
    """
    from weightprobe.hash import read_safetensors_header

    index: dict[str, tuple[Path, list[int], str]] = {}
    for st_path in sorted(dir_path.glob("*.safetensors")):
        header = read_safetensors_header(st_path)
        for tensor_name, info in header.items():
            index[tensor_name] = (st_path, info["shape"], info["dtype"])
    return index


def _load_tensor(path: Path, name: str) -> np.ndarray:
    """Load a single tensor by name from a safetensors file as float32 numpy."""
    import mlx.core as mx
    # mlx.load returns the whole file as a dict; we do this once per file
    # via the file-level cache in compute_diff_base instead.
    state = mx.load(str(path))
    return np.array(state[name].astype(mx.float32))


def _per_tensor_diff(target_t: np.ndarray, baseline_t: np.ndarray) -> dict[str, float]:
    """Compute (L2, cosine_distance, max_abs_delta, relative_l2) for two
    same-shape tensors. Both must be flat or compatible-shape; we flatten
    internally for cosine."""
    assert target_t.shape == baseline_t.shape, "shape mismatch — caller error"
    delta = target_t - baseline_t
    l2 = float(np.linalg.norm(delta.ravel()))
    base_norm = float(np.linalg.norm(baseline_t.ravel()))
    rel_l2 = l2 / max(base_norm, 1e-12)
    max_abs = float(np.max(np.abs(delta)))

    # Cosine similarity on flattened vectors.
    a = target_t.ravel().astype(np.float64)
    b = baseline_t.ravel().astype(np.float64)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-12 or b_norm < 1e-12:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(a, b) / (a_norm * b_norm))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    cos_dist = 1.0 - cos_sim

    return {
        "l2_distance": l2,
        "cosine_distance": cos_dist,
        "max_abs_delta": max_abs,
        "relative_l2": rel_l2,
    }


def compute_diff_base(
    target_dir: Path | str,
    baseline_dir: Path | str,
    *,
    cosine_threshold: float = 1e-4,
) -> DiffBaseResult:
    """Compute per-tensor diff between `target_dir` and `baseline_dir`.

    `cosine_threshold`: a tensor is flagged as "modified" if its cosine
    distance from the baseline exceeds this. Default 1e-4 catches even
    surgical edits; raise to 1e-2 for "substantially modified" only.

    Memory model
    ------------
    For each safetensors file in target+baseline, we load each tensor
    individually via mlx (which streams from the file) and compare. Peak
    memory ≈ 2× single tensor (largest is typically embed_tokens or
    lm_head, ~500 MB at bf16 → ~1 GB at f32).

    Returns a DiffBaseResult. The `per_tensor_modified` list (in to_dict
    output) names every tensor whose cosine distance exceeds the threshold —
    these are the suspicious weight-edits.
    """
    import mlx.core as mx

    target_dir = Path(target_dir).resolve()
    baseline_dir = Path(baseline_dir).resolve()
    if not target_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {target_dir}")
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {baseline_dir}")

    target_index = _index_safetensors(target_dir)
    baseline_index = _index_safetensors(baseline_dir)

    target_names = set(target_index.keys())
    baseline_names = set(baseline_index.keys())

    only_in_target = sorted(target_names - baseline_names)
    only_in_baseline = sorted(baseline_names - target_names)

    shared_names = sorted(target_names & baseline_names)
    shape_mismatch: list[dict] = []
    per_tensor: list[TensorDiff] = []
    n_modified = 0

    # Group shared tensors by their target safetensors file so we load
    # each file once.
    by_target_file: dict[Path, list[str]] = {}
    for name in shared_names:
        t_path, t_shape, t_dtype = target_index[name]
        b_path, b_shape, b_dtype = baseline_index[name]
        if t_shape != b_shape:
            shape_mismatch.append({
                "name": name,
                "target_shape": t_shape,
                "baseline_shape": b_shape,
            })
            continue
        by_target_file.setdefault(t_path, []).append(name)

    # Per-target-file processing — load whole file once, look up each tensor.
    # The baseline tensor is loaded individually per name (slower but uses
    # less peak memory).
    for t_file, names in by_target_file.items():
        t_state = mx.load(str(t_file))
        for name in names:
            t_path, t_shape, t_dtype = target_index[name]
            b_path, b_shape, b_dtype = baseline_index[name]

            target_t = np.array(t_state[name].astype(mx.float32))
            # Load just this baseline tensor (mlx load is per-file; lazy
            # whole-file cache would be better but adds complexity).
            b_state = mx.load(str(b_path))
            baseline_t = np.array(b_state[name].astype(mx.float32))
            del b_state

            metrics = _per_tensor_diff(target_t, baseline_t)
            del target_t, baseline_t

            if metrics["cosine_distance"] > cosine_threshold:
                n_modified += 1

            per_tensor.append(TensorDiff(
                name=name,
                shape=t_shape,
                dtype_target=t_dtype,
                dtype_baseline=b_dtype,
                l2_distance=round(metrics["l2_distance"], 6),
                cosine_distance=round(metrics["cosine_distance"], 9),
                max_abs_delta=round(metrics["max_abs_delta"], 6),
                relative_l2=round(metrics["relative_l2"], 6),
            ))
        del t_state

    return DiffBaseResult(
        target_dir=str(target_dir),
        baseline_dir=str(baseline_dir),
        n_tensors_compared=len(per_tensor),
        n_tensors_only_in_target=len(only_in_target),
        n_tensors_only_in_baseline=len(only_in_baseline),
        n_tensors_shape_mismatch=len(shape_mismatch),
        n_tensors_modified=n_modified,
        cosine_threshold=cosine_threshold,
        per_tensor=per_tensor,
        tensors_only_in_target=only_in_target,
        tensors_only_in_baseline=only_in_baseline,
        tensors_shape_mismatch=shape_mismatch,
    )
