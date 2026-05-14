"""Tests for weightprobe spectral + diff-base modes."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from weightprobe.spectral import (
    _bottleneck_shape_signal,
    _safe_kurtosis,
    _spectral_metrics_for_tensor,
    _suspicion_score,
    compute_spectral_fingerprint,
)
from weightprobe.diff_base import compute_diff_base, _per_tensor_diff


# ---- Helper: write a real safetensors file with float32 data ----------------

def _write_safetensors_f32(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Minimal safetensors writer for float32 tensors (real values, not zeros).

    Lets us load via mlx and compute real SVDs in tests."""
    header: dict = {}
    offset = 0
    data_blob = bytearray()
    for name, arr in tensors.items():
        arr_f32 = np.ascontiguousarray(arr.astype(np.float32))
        nbytes = arr_f32.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr_f32.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        data_blob.extend(arr_f32.tobytes())
        offset += nbytes
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(bytes(data_blob))


# ---- Spectral mode tests ----------------------------------------------------

def test_bottleneck_shape_signal():
    # Adapter shapes — high signal
    assert _bottleneck_shape_signal([8, 4096]) == 1.0
    assert _bottleneck_shape_signal([4096, 8]) == 1.0
    assert _bottleneck_shape_signal([16, 4096]) == 1.0
    # Mid-range
    assert _bottleneck_shape_signal([32, 4096]) == 0.85
    assert _bottleneck_shape_signal([64, 4096]) == 0.5
    # Llama-shape (k_proj GQA: 1024 × 4096) — low/zero signal
    assert _bottleneck_shape_signal([1024, 4096]) == 0.0
    # Square Llama matrices — zero
    assert _bottleneck_shape_signal([4096, 4096]) == 0.0
    # 1D / non-2D — zero
    assert _bottleneck_shape_signal([4096]) == 0.0
    assert _bottleneck_shape_signal([4096, 4096, 4]) == 0.0


def test_kurtosis_safe():
    """Kurtosis works on small arrays + is stable for low-variance inputs."""
    assert _safe_kurtosis(np.array([1.0])) == 0.0
    assert _safe_kurtosis(np.array([1.0, 1.0, 1.0, 1.0])) == 0.0
    # Normal-ish distribution — kurtosis should be near 0
    rng = np.random.default_rng(42)
    normal = rng.standard_normal(1000)
    assert abs(_safe_kurtosis(normal)) < 1.0


def test_spectral_metrics_random_tensor():
    """A full-rank random tensor should have high entropy, low suspicion."""
    rng = np.random.default_rng(0)
    t = rng.standard_normal((1024, 4096))  # Llama-shape
    metrics = _spectral_metrics_for_tensor("random_tensor", t)
    assert metrics is not None
    # Random matrix has roughly uniform SV distribution → high entropy ratio
    assert metrics.spectral_entropy_ratio > 0.85
    # Llama shape — bottleneck signal is 0; suspicion comes only from
    # spectral metrics (which are low for random tensors)
    assert metrics.suspicion_score < 0.3


def test_spectral_metrics_low_rank_adapter():
    """A rank-8 adapter shape (8, 4096) with random data should score high
    on bottleneck-shape signal."""
    rng = np.random.default_rng(0)
    t = rng.standard_normal((8, 4096))
    metrics = _spectral_metrics_for_tensor("W_enc.weight", t)
    assert metrics is not None
    # Bottleneck-shape signal is strong → suspicion ≥ 0.6
    assert metrics.suspicion_score >= 0.6


def test_spectral_skips_1d_and_scalar():
    """1D/0D tensors should return None (skipped)."""
    assert _spectral_metrics_for_tensor("bias", np.zeros((4096,))) is None
    assert _spectral_metrics_for_tensor("scalar", np.array(0.0)) is None


def test_compute_spectral_fingerprint_on_synthetic_adapter(tmp_path):
    """End-to-end: a synthetic adapter dir should report elevated aggregate suspicion.

    Random-fill bottleneck matrices score ~0.6 (shape signal dominates;
    entropy/rank are full because data is random). That's the floor for
    flagging and matches what a fresh-init adapter looks like; trained
    adapters score higher (A4_8b_v2 → 0.788) because they pick up real
    rank-deficiency from training."""
    rng = np.random.default_rng(42)
    _write_safetensors_f32(tmp_path / "adapter.safetensors", {
        "W_enc.weight": rng.standard_normal((8, 4096)),
        "W_dec.weight": rng.standard_normal((4096, 8)),
        "gate.weight": rng.standard_normal((1, 4096)),
    })
    fp = compute_spectral_fingerprint(tmp_path)
    assert fp.aggregate_suspicion >= 0.5, \
        f"adapter dir should be flagged; got {fp.aggregate_suspicion}"
    # All three tensors have bottleneck shape signal 1.0 → suspicion ≥ 0.6
    flagged = sum(1 for t in fp.per_tensor if t.suspicion_score >= 0.5)
    assert flagged >= 2, \
        f"expected ≥2 tensors with suspicion ≥ 0.5; got {flagged}"


def test_compute_spectral_fingerprint_on_synthetic_clean(tmp_path):
    """A synthetic Llama-shape model dir should report LOW aggregate suspicion."""
    rng = np.random.default_rng(0)
    _write_safetensors_f32(tmp_path / "model.safetensors", {
        # Mimic Llama shapes: q/k/v/o + mlp
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((4096, 4096)),
        "model.layers.0.self_attn.k_proj.weight": rng.standard_normal((1024, 4096)),
        "model.layers.0.self_attn.v_proj.weight": rng.standard_normal((1024, 4096)),
        "model.layers.0.self_attn.o_proj.weight": rng.standard_normal((4096, 4096)),
    })
    fp = compute_spectral_fingerprint(tmp_path)
    # Llama matrices have shape signal 0, random data has high entropy →
    # all should be < 0.3 suspicion
    assert fp.aggregate_suspicion < 0.5, \
        f"clean Llama-shape dir should NOT be flagged; got {fp.aggregate_suspicion}"


# ---- Diff-base mode tests ---------------------------------------------------

def test_per_tensor_diff_identical():
    """Identical tensors should produce zero L2, zero cosine distance."""
    t = np.array([[1.0, 2.0], [3.0, 4.0]])
    metrics = _per_tensor_diff(t, t)
    assert metrics["l2_distance"] == 0.0
    assert metrics["cosine_distance"] < 1e-9
    assert metrics["max_abs_delta"] == 0.0


def test_per_tensor_diff_modified():
    """A non-parallel modification should produce nonzero L2 + cosine distance.

    Note: a uniform scalar multiple keeps the flattened vector parallel
    (cos_sim = 1, distance = 0), so we add an asymmetric perturbation."""
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.1, 0.05], [0.0, 1.0]])  # not a scalar multiple of a
    metrics = _per_tensor_diff(a, b)
    assert metrics["l2_distance"] > 0
    assert metrics["cosine_distance"] > 0
    assert metrics["cosine_distance"] < 0.05  # small but nonzero
    assert metrics["max_abs_delta"] == pytest.approx(0.1)


def test_compute_diff_base_identical_dirs(tmp_path):
    """Diff a directory against itself — zero modifications expected."""
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    rng = np.random.default_rng(0)
    _write_safetensors_f32(dir_a / "model.safetensors", {
        "weight": rng.standard_normal((64, 64)),
    })
    result = compute_diff_base(dir_a, dir_a)
    assert result.n_tensors_compared == 1
    assert result.n_tensors_modified == 0
    assert result.n_tensors_only_in_target == 0
    assert result.n_tensors_only_in_baseline == 0


def test_compute_diff_base_modified_weight(tmp_path):
    """A modified weight should be flagged as modified."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    rng = np.random.default_rng(0)
    base = rng.standard_normal((64, 64)).astype(np.float32)
    modified = base + 0.01 * rng.standard_normal((64, 64)).astype(np.float32)
    _write_safetensors_f32(dir_a / "model.safetensors", {"w": base})
    _write_safetensors_f32(dir_b / "model.safetensors", {"w": modified})
    # 1% noise on unit-variance random ≈ 5e-5 cosine distance, so use 1e-6.
    result = compute_diff_base(dir_b, dir_a, cosine_threshold=1e-6)
    assert result.n_tensors_modified == 1, \
        f"expected 1 modified tensor, got {result.n_tensors_modified}"


def test_compute_diff_base_added_tensor(tmp_path):
    """A tensor present only in target should be flagged as added."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    rng = np.random.default_rng(0)
    weights = rng.standard_normal((64, 64))
    _write_safetensors_f32(dir_a / "model.safetensors", {"w": weights})
    _write_safetensors_f32(dir_b / "model.safetensors", {
        "w": weights,
        "adapter.W_enc": rng.standard_normal((8, 64)),
    })
    result = compute_diff_base(dir_b, dir_a)
    assert result.n_tensors_only_in_target == 1
    assert "adapter.W_enc" in result.tensors_only_in_target


def test_compute_diff_base_shape_mismatch(tmp_path):
    """A tensor with the same name but different shape should be flagged."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_safetensors_f32(dir_a / "model.safetensors", {
        "w": np.ones((64, 64)),
    })
    _write_safetensors_f32(dir_b / "model.safetensors", {
        "w": np.ones((64, 128)),  # wider
    })
    result = compute_diff_base(dir_b, dir_a)
    assert result.n_tensors_shape_mismatch == 1
