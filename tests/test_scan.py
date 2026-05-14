"""Tests for weightprobe scan mode (activation-delta KL probe).

Unit tests cover the pure-numeric portion (`_per_layer_metrics`,
`_suspicion_for_layer`). End-to-end model-loading tests are skipped
unless WEIGHTPROBE_TEST_MODELS_DIR points to a writable area with
mlx_lm-compatible test models — too heavy for CI."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from weightprobe.scan import (
    _DEFAULT_PROBE_PROMPTS,
    _per_layer_metrics,
    _suspicion_for_layer,
    LayerScanRecord,
    ScanFingerprint,
)


# ---- Per-layer metrics ----------------------------------------------------

def test_per_layer_metrics_identical():
    """Identical activations produce zero L2 / cosine sep / KL."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((20, 64)).astype(np.float32)
    mean_l2, median_l2, cos_sep, kl = _per_layer_metrics(a, a)
    assert mean_l2 == 0.0
    assert median_l2 == 0.0
    assert cos_sep == pytest.approx(0.0, abs=1e-6)
    # KL of identical distributions is 0 (within numerical noise).
    assert abs(kl) < 1e-3


def test_per_layer_metrics_shifted_mean():
    """Translation shift produces nonzero L2 + cosine sep + KL."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((20, 64)).astype(np.float32)
    b = a + 1.0  # constant shift
    mean_l2, _, cos_sep, kl = _per_layer_metrics(a, b)
    assert mean_l2 > 0
    assert cos_sep > 0
    assert kl > 0


def test_per_layer_metrics_zero_baseline():
    """Zero baseline mean defaults cosine to 0 (avoid div-by-zero)."""
    a = np.ones((10, 8), dtype=np.float32)
    b = np.zeros((10, 8), dtype=np.float32)
    mean_l2, _, cos_sep, _ = _per_layer_metrics(a, b)
    assert mean_l2 > 0
    # b's mean is zero → cos_sep should be 0 (the guard, not an angle).
    assert cos_sep == 0.0


# ---- Suspicion scoring ----------------------------------------------------

def test_suspicion_clean():
    """Tiny cos-sep + tiny relative-L2 → 0 suspicion."""
    susp = _suspicion_for_layer(
        cos_sep=0.005, l2_norm_baseline=10.0, mean_l2_distance=0.1,
    )
    assert susp == 0.0


def test_suspicion_high_cosine():
    """cos-sep ≥ 0.20 alone fires the layer."""
    susp = _suspicion_for_layer(
        cos_sep=0.25, l2_norm_baseline=10.0, mean_l2_distance=0.0,
    )
    assert susp >= 0.99


def test_suspicion_high_relative_l2():
    """relative L2 ≥ 0.5 alone fires the layer."""
    susp = _suspicion_for_layer(
        cos_sep=0.0, l2_norm_baseline=1.0, mean_l2_distance=0.6,
    )
    assert susp >= 0.99


def test_suspicion_combined_signals():
    """Multiple moderate signals should combine via probabilistic OR."""
    cos_only = _suspicion_for_layer(0.10, 10.0, 0.0)   # cos signal only
    l2_only = _suspicion_for_layer(0.0, 10.0, 2.5)     # l2 signal only
    both = _suspicion_for_layer(0.10, 10.0, 2.5)        # both
    assert both > cos_only
    assert both > l2_only
    # Probabilistic OR upper-bounds at 1.
    assert both <= 1.0


# ---- Default probe corpus ---------------------------------------------------

def test_default_probe_corpus_is_diverse():
    prompts = _DEFAULT_PROBE_PROMPTS
    assert len(prompts) >= 15
    # Mix of refusal-bound, benign technical, and neutral topics.
    refusal_bound_keywords = ("bomb", "fraud", "synthesize", "stalk", "phishing", "filter", "exfiltrates")
    n_refusal = sum(1 for p in prompts if any(k in p.lower() for k in refusal_bound_keywords))
    assert n_refusal >= 5, f"expected ≥5 refusal-bound prompts, got {n_refusal}"
    # No prompt is empty or absurdly long.
    for p in prompts:
        assert 5 < len(p) < 300


# ---- Result-types serialization ---------------------------------------------

def test_layer_scan_record_to_dict():
    r = LayerScanRecord(
        layer=15, position="post_mlp", n_prompts=20,
        mean_l2_distance=2.5, median_l2_distance=2.4,
        cosine_separation=0.15, kl_divergence=42.0,
        suspicion_score=0.85,
    )
    d = r.to_dict()
    assert d["layer"] == 15
    assert d["position"] == "post_mlp"
    assert d["mean_l2_distance"] == 2.5
    assert d["suspicion_score"] == 0.85


def test_scan_fingerprint_to_dict():
    fp = ScanFingerprint(
        target_dir="/tmp/target",
        baseline_dir="/tmp/base",
        n_prompts=20, n_layers=32,
        aggregate_suspicion=0.85,
        n_layers_high_suspicion=2,
        flagged_layers=[(15, "post_mlp"), (16, "post_attn")],
        step_anomaly_layer=15,
        step_anomaly_position="post_mlp",
        step_anomaly_value=4.78,
        step_anomaly_ratio=48.7,
        per_layer=[],
    )
    d = fp.to_dict()
    assert d["aggregate_suspicion"] == 0.85
    assert d["n_layers_high_suspicion"] == 2
    assert d["flagged_layers"] == [[15, "post_mlp"], [16, "post_attn"]]
    # v0.2: step-anomaly fields are surfaced in JSON.
    assert d["step_anomaly_layer"] == 15
    assert d["step_anomaly_position"] == "post_mlp"
    assert d["step_anomaly_value"] == 4.78
    assert d["step_anomaly_ratio"] == 48.7


def test_compute_scan_synthetic_step_anomaly_at_insert(tmp_path: Path):
    """Synthetic probe of the step-anomaly scoring.

    Build per-layer records with a fake architectural-backdoor signature:
      - layers 0..14: mean_l2 grows slowly (0.1 → 1.5)
      - layer 15:     ONE-LAYER STEP to 6.3 (the insertion)
      - layers 16..31: monotone growth to 44.9 (propagation)
    Pass these directly into compute_scan's downstream scoring path by
    monkey-patching _collect_layer_activations to return synthetic
    activations encoding exactly this delta pattern.
    """
    from weightprobe import scan as scan_mod

    n_prompts = 4
    n_layers = 32
    hidden = 4
    # Construct paired activations such that
    # ||target_acts[:, L] - baseline_acts[:, L]||_2 (averaged over prompts)
    # equals a specified per-layer L2 schedule.
    schedule = (
        [0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3, 1.3, 1.4, 1.5]
        + [6.3]                                                              # ← step at L=15
        + [7.0, 7.8, 8.6, 9.7, 10.9, 12.3, 14.6, 16.4, 18.0, 20.2, 23.1, 25.7,
           29.7, 33.2, 38.2, 44.9]
    )
    assert len(schedule) == n_layers

    def _fake_acts() -> dict[str, "np.ndarray"]:
        # baseline = zeros, target = delta vectors with the prescribed L2 norm.
        # delta along dim 0, zeros elsewhere → L2 = |delta[0]|.
        post_mlp = np.zeros((n_prompts, n_layers, hidden), dtype=np.float32)
        post_attn = np.zeros((n_prompts, n_layers, hidden), dtype=np.float32)
        for L in range(n_layers):
            post_mlp[:, L, 0] = schedule[L]   # post_mlp carries the signal
            # post_attn left near-zero so step_anomaly picks post_mlp.
        return {"post_attn": post_attn, "post_mlp": post_mlp}

    target_dir = tmp_path / "target"
    baseline_dir = tmp_path / "baseline"
    target_dir.mkdir()
    baseline_dir.mkdir()

    # Monkey-patch the collector to return target=schedule, baseline=zeros.
    real_collect = scan_mod._collect_layer_activations
    def fake_collect(model_dir, prompts, dtype="float32"):
        if str(model_dir).endswith("target"):
            return _fake_acts()
        return {"post_attn": np.zeros((n_prompts, n_layers, hidden), dtype=np.float32),
                "post_mlp":  np.zeros((n_prompts, n_layers, hidden), dtype=np.float32)}
    scan_mod._collect_layer_activations = fake_collect
    try:
        fp = scan_mod.compute_scan_fingerprint(
            target_dir, baseline_dir,
            probe_prompts=("p1", "p2", "p3", "p4"),
        )
    finally:
        scan_mod._collect_layer_activations = real_collect

    # The step is at L=14→L=15, so step_anomaly_layer should be 15.
    assert fp.step_anomaly_layer == 15, f"expected L=15, got {fp.step_anomaly_layer}"
    assert fp.step_anomaly_position == "post_mlp"
    # The step at L=15 is 6.3 - 1.5 = 4.8; should land near that.
    assert fp.step_anomaly_value == pytest.approx(4.8, abs=0.3)
    # The step ratio vs median-other-step should be very high.
    assert fp.step_anomaly_ratio > 10.0, f"weak ratio: {fp.step_anomaly_ratio}"


def test_compute_scan_synthetic_clean_no_step_anomaly(tmp_path: Path):
    """No adapter → mean_l2 monotone-smooth → step_anomaly_ratio should be
    low (i.e. no single-layer step dominates)."""
    from weightprobe import scan as scan_mod

    n_prompts = 3
    n_layers = 16
    hidden = 4
    # Linear, smooth growth from 0.1 to 1.6 — no sudden step anywhere.
    schedule = [0.1 + 0.1 * L for L in range(n_layers)]
    assert len(schedule) == n_layers

    def fake_collect(model_dir, prompts, dtype="float32"):
        if str(model_dir).endswith("target"):
            post_mlp = np.zeros((n_prompts, n_layers, hidden), dtype=np.float32)
            for L in range(n_layers):
                post_mlp[:, L, 0] = schedule[L]
            return {"post_attn": np.zeros_like(post_mlp), "post_mlp": post_mlp}
        return {"post_attn": np.zeros((n_prompts, n_layers, hidden), dtype=np.float32),
                "post_mlp":  np.zeros((n_prompts, n_layers, hidden), dtype=np.float32)}

    target_dir = tmp_path / "target"
    baseline_dir = tmp_path / "baseline"
    target_dir.mkdir()
    baseline_dir.mkdir()
    real_collect = scan_mod._collect_layer_activations
    scan_mod._collect_layer_activations = fake_collect
    try:
        fp = scan_mod.compute_scan_fingerprint(
            target_dir, baseline_dir,
            probe_prompts=("p1", "p2", "p3"),
        )
    finally:
        scan_mod._collect_layer_activations = real_collect

    # Linear schedule → every step ≈ 0.1, no anomaly.
    # step_anomaly_ratio should be close to 1.0 (all steps roughly equal).
    assert fp.step_anomaly_ratio < 3.0, (
        f"clean linear growth shouldn't produce an anomaly; got "
        f"step_anomaly_ratio={fp.step_anomaly_ratio:.2f}"
    )


# ---- End-to-end (skipped unless an env var is set) -------------------------

@pytest.mark.skipif(
    "WEIGHTPROBE_TEST_MODELS_DIR" not in os.environ,
    reason="end-to-end model-loading test; set WEIGHTPROBE_TEST_MODELS_DIR to enable",
)
def test_scan_end_to_end_self():
    """target == baseline → all deltas = 0, suspicion = 0."""
    from weightprobe.scan import compute_scan_fingerprint
    model_dir = Path(os.environ["WEIGHTPROBE_TEST_MODELS_DIR"])
    fp = compute_scan_fingerprint(
        model_dir, model_dir,
        probe_prompts=_DEFAULT_PROBE_PROMPTS[:3],  # tiny probe for speed
    )
    assert fp.aggregate_suspicion == pytest.approx(0.0, abs=1e-3)
    assert fp.n_layers_high_suspicion == 0
