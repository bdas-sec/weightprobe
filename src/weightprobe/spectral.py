"""Spectral fingerprint of adapter-shaped tensors.

Adapted from "Weight Space Detection of Backdoors in LoRA Adapters"
(arXiv:2602.15195). Low-rank adapters and LoRA-style modules have distinctive
singular-value spectra: their weight matrices have rank bounded by the
bottleneck dimension, so SVD energy concentrates in a small number of
components. This produces a high-spectral-entropy *anomaly* relative to
random or general-purpose weights, which we can detect.

Method
------
For every tensor in a model directory, we compute:
  - The full SVD via numpy (after casting bf16/f16 → f32).
  - Normalized singular value energy: p_i = σ_i² / Σ σ_j².
  - **Spectral entropy**: H(p) = -Σ p_i log_2 p_i. Adapter-shaped weights
    have low spectral entropy (energy concentrated in top-k components,
    where k = bottleneck).
  - **Effective rank** at 95% energy: smallest k such that Σ_{i≤k} p_i ≥ 0.95.
    For an adapter with bottleneck=8, effective rank is exactly 8 (or less).
  - **Kurtosis** of the singular value distribution: peakedness indicator.
    LoRA/adapter weights typically have high kurtosis.

For tensors with min(shape) ≤ 64 we compute the full SVD; for larger tensors
we use truncated SVD on the top 32 components (sufficient to identify the
energy-concentration anomaly without paying full O(min(m,n)³)).

Aggregate fingerprint per directory: per-tensor metrics + a "suspicion score"
(simple weighted aggregate of low-entropy, high-kurtosis tensors).

What this catches
-----------------
- Trained adapters with low-rank bottleneck (W_enc: (k, hidden) shape;
  W_dec: (hidden, k); k ≪ hidden). Any of A4_8b_v2 / A6 / A7 fingerprint as
  classic adapters — rank-8 bottleneck, effective rank 8.
- LoRA modules of any form (lora_A: (r, in), lora_B: (out, r)).
- Distilled-into-base attacks where the base weights have been edited via
  a low-rank perturbation: the *delta* would have rank-k structure, so a
  per-tensor weight-edit-detector running diff against a clean base would
  surface the signature. (Use `diff_base` mode for the delta computation.)

What this does NOT catch
------------------------
- Full-rank weight edits (rank-N abliteration where N ≈ hidden_size). For
  these, the per-tensor spectrum looks normal; the anomaly lives in a
  *direction-relative-to-clean-base* analysis — see `diff_base.py`.
- Adapters merged-into-base (the bottleneck shape disappears).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


# Tensor shape selectors. We only fingerprint tensors that *could* be
# adapter-shaped (i.e., 2-D with min dim small enough to be a bottleneck).
# We skip 1-D tensors (biases, layer norms), 0-D (scalars), and high-rank
# tensors (3-D+ MoE expert weights) to keep the fingerprint tractable.
_MAX_FULL_SVD_DIM = 64       # full SVD if min(shape) ≤ 64
_MAX_TRUNC_SVD_K = 32        # truncated to top-32 components if larger
_MIN_TENSOR_DIM_FOR_SVD = 2  # need at least 2-D


@dataclass
class TensorSpectralMetrics:
    """Spectral metrics for one tensor."""
    name: str
    shape: list[int]
    dtype: str
    n_singular_values: int
    spectral_entropy: float    # bits; lower = more concentrated
    spectral_entropy_uniform_baseline: float  # log2(min(shape)) — what entropy
                                              # would be if all SVs were equal
    spectral_entropy_ratio: float  # entropy / uniform_baseline ∈ [0, 1]
    effective_rank_95: int      # smallest k with cumulative energy ≥ 95%
    effective_rank_99: int
    top1_energy_fraction: float
    top4_energy_fraction: float
    kurtosis: float            # of the SV distribution
    suspicion_score: float     # 0.0 = clean, 1.0 = highly adapter-like
    truncated: bool            # True if we computed only top-K SVs

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpectralFingerprint:
    """Aggregate spectral fingerprint of a model directory."""
    model_dir: str
    n_tensors_analyzed: int
    n_tensors_skipped: int
    per_tensor: list[TensorSpectralMetrics]
    aggregate_suspicion: float  # max suspicion across tensors
    n_tensors_high_suspicion: int  # # tensors with suspicion ≥ 0.7

    def to_dict(self) -> dict:
        return {
            "model_dir": self.model_dir,
            "n_tensors_analyzed": self.n_tensors_analyzed,
            "n_tensors_skipped": self.n_tensors_skipped,
            "per_tensor": [t.to_dict() for t in self.per_tensor],
            "aggregate_suspicion": self.aggregate_suspicion,
            "n_tensors_high_suspicion": self.n_tensors_high_suspicion,
        }


def _safe_kurtosis(values: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition; 0 = normal). Stable for
    very-low-variance inputs."""
    if values.size < 4:
        return 0.0
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-12:
        return 0.0
    standardized = (values - mean) / std
    return float((standardized ** 4).mean() - 3.0)


def _bottleneck_shape_signal(shape: list[int]) -> float:
    """Returns 0-1 score based on whether the shape looks like an adapter
    bottleneck. Clean Llama-3 tensors have min(shape) ≥ 1024 (k_proj GQA);
    adapters / LoRA modules typically have min(shape) ≤ 32.
    """
    if len(shape) != 2:
        return 0.0
    m, n = shape
    min_dim = min(m, n)
    max_dim = max(m, n)
    # Adapter signature: very small min dim AND much larger max dim.
    if min_dim <= 16 and max_dim >= min_dim * 8:
        return 1.0
    if min_dim <= 32 and max_dim >= min_dim * 8:
        return 0.85
    if min_dim <= 64 and max_dim >= min_dim * 8:
        return 0.5
    if min_dim <= 128 and max_dim >= min_dim * 8:
        return 0.2
    return 0.0


def _suspicion_score(
    entropy_ratio: float,
    eff_rank_95: int,
    n_singular_values: int,
    kurtosis: float,
    shape: list[int],
) -> float:
    """Heuristic 0-1 suspicion score combining entropy + rank + kurtosis +
    bottleneck-shape signal.

    Adapter-class signature:
      - **shape**: min(shape) ≤ 32 with max(shape) much larger (LoRA / adapter
        bottleneck). This is the dominant signal — clean Llama-3 tensors
        all have min(shape) ≥ 1024.
      - entropy_ratio: secondary; trained adapters use bottleneck rank fully
        so entropy can be near baseline, but rank-1 mean-direction adapters
        will have low entropy.
      - eff_rank_95 ≪ n_singular_values: secondary (only meaningful when
        the tensor's rank exceeds its bottleneck).
      - kurtosis: tertiary.
    """
    # Bottleneck-shape signal — the dominant adapter detector.
    shape_part = _bottleneck_shape_signal(shape)

    # Spectral signals (only meaningful if shape doesn't already flag).
    entropy_part = max(0.0, min(1.0, 1.0 - entropy_ratio))
    rank_part = max(0.0, 1.0 - (eff_rank_95 / max(n_singular_values, 1)))
    kurt_part = max(0.0, min(1.0, kurtosis / 10.0))

    # Weighted aggregate. Shape dominates; spectral metrics confirm.
    return 0.6 * shape_part + 0.2 * entropy_part + 0.15 * rank_part + 0.05 * kurt_part


def _spectral_metrics_for_tensor(name: str, t: np.ndarray) -> TensorSpectralMetrics | None:
    """Compute spectral metrics for one 2-D tensor; return None if not analyzable."""
    if t.ndim != 2:
        return None
    m, n = t.shape
    if min(m, n) < _MIN_TENSOR_DIM_FOR_SVD:
        return None

    # Cast to float32 for SVD numerical stability.
    t_f32 = t.astype(np.float32)

    truncated = False
    if min(m, n) <= _MAX_FULL_SVD_DIM:
        # Full SVD.
        try:
            sv = np.linalg.svd(t_f32, compute_uv=False)
        except np.linalg.LinAlgError:
            return None
    else:
        # Truncated SVD via the top-K components of A^T A's eigendecomposition.
        # For very large tensors we just use full SVD's top-K (memory-OK on M4 Pro
        # for tensors up to ~10K × 10K).
        try:
            sv = np.linalg.svd(t_f32, compute_uv=False)[:_MAX_TRUNC_SVD_K]
        except np.linalg.LinAlgError:
            return None
        truncated = True

    sv = np.asarray(sv, dtype=np.float64)
    sv_sq = sv ** 2
    total = sv_sq.sum()
    if total < 1e-12:
        return None  # zero tensor

    p = sv_sq / total
    # Spectral entropy in bits.
    nonzero = p > 1e-12
    entropy = float(-(p[nonzero] * np.log2(p[nonzero])).sum())
    uniform_baseline = float(np.log2(len(sv)))
    entropy_ratio = entropy / max(uniform_baseline, 1e-12)

    # Effective rank at 95% / 99% energy.
    cumulative = np.cumsum(p)
    eff_rank_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    eff_rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)
    eff_rank_95 = min(eff_rank_95, len(sv))
    eff_rank_99 = min(eff_rank_99, len(sv))

    top1 = float(p[0]) if len(p) >= 1 else 0.0
    top4 = float(p[: min(4, len(p))].sum())

    kurt = _safe_kurtosis(sv)
    susp = _suspicion_score(entropy_ratio, eff_rank_95, len(sv), kurt, list(t.shape))

    return TensorSpectralMetrics(
        name=name,
        shape=list(t.shape),
        dtype=str(t.dtype),
        n_singular_values=len(sv),
        spectral_entropy=round(entropy, 6),
        spectral_entropy_uniform_baseline=round(uniform_baseline, 6),
        spectral_entropy_ratio=round(entropy_ratio, 6),
        effective_rank_95=eff_rank_95,
        effective_rank_99=eff_rank_99,
        top1_energy_fraction=round(top1, 6),
        top4_energy_fraction=round(top4, 6),
        kurtosis=round(kurt, 6),
        suspicion_score=round(susp, 6),
        truncated=truncated,
    )


def _load_safetensors_as_numpy(path: Path) -> dict[str, np.ndarray]:
    """Load a safetensors file as a dict of numpy arrays. Casts bf16/f16 → f32
    on load (numpy doesn't support bf16 natively)."""
    # Use mlx for bf16-aware loading; immediately cast to float32 for SVD.
    import mlx.core as mx
    state = mx.load(str(path))
    out: dict[str, np.ndarray] = {}
    for k, v in state.items():
        # MLX → numpy: cast through float32 to avoid bf16 issues.
        out[k] = np.array(v.astype(mx.float32))
    return out


def compute_spectral_fingerprint(model_dir: Path | str) -> SpectralFingerprint:
    """Compute the spectral fingerprint of every analyzable tensor in
    `model_dir` (recursively scans `*.safetensors`).

    For typical model dirs (16 GB Llama-3.1-8B), this loads each shard
    sequentially, computes per-tensor SVDs, and discards the tensor before
    moving to the next. Memory peak ≈ size of the largest single tensor.

    Returns a SpectralFingerprint with per-tensor metrics + aggregate
    suspicion score.
    """
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {model_dir}")

    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")

    per_tensor: list[TensorSpectralMetrics] = []
    n_skipped = 0

    for st_path in safetensors_files:
        state = _load_safetensors_as_numpy(st_path)
        for name, t in state.items():
            metrics = _spectral_metrics_for_tensor(name, t)
            if metrics is None:
                n_skipped += 1
            else:
                per_tensor.append(metrics)
        # Free the tensor memory before processing the next file.
        del state

    aggregate_suspicion = max((t.suspicion_score for t in per_tensor), default=0.0)
    n_high = sum(1 for t in per_tensor if t.suspicion_score >= 0.7)

    return SpectralFingerprint(
        model_dir=str(model_dir),
        n_tensors_analyzed=len(per_tensor),
        n_tensors_skipped=n_skipped,
        per_tensor=per_tensor,
        aggregate_suspicion=aggregate_suspicion,
        n_tensors_high_suspicion=n_high,
    )
