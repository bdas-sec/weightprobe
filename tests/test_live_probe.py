"""Tests for weightprobe live-probe mode (runtime activation anomaly detector)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from weightprobe.live_probe import (
    LayerBaseline,
    LiveProbeBaseline,
    LiveProbeScore,
)


def _make_baseline(n_layers: int = 4, hidden: int = 32) -> LiveProbeBaseline:
    """Build a synthetic baseline with mean=10.0, std=1.0 at every layer."""
    rng = np.random.default_rng(0)
    layers: list[LayerBaseline] = []
    for L in range(n_layers):
        for pos in ("post_attn", "post_mlp"):
            layers.append(LayerBaseline(
                layer=L, position=pos,
                mean_norm=10.0, std_norm=1.0,
                pca_v1=rng.standard_normal(hidden).astype(np.float32),
            ))
    return LiveProbeBaseline(
        model_dir="/tmp/clean", n_probe_prompts=20,
        n_layers=n_layers, hidden_size=hidden,
        layers=layers,
    )


# ---- Persistence -----------------------------------------------------------

def test_baseline_save_load_roundtrip(tmp_path):
    """Save → load preserves all fields."""
    bl = _make_baseline(n_layers=3, hidden=16)
    out = tmp_path / "baseline.npz"
    bl.save(out)
    bl2 = LiveProbeBaseline.load(out)
    assert bl2.n_layers == bl.n_layers
    assert bl2.hidden_size == bl.hidden_size
    assert bl2.n_probe_prompts == bl.n_probe_prompts
    assert len(bl2.layers) == len(bl.layers)
    for a, b in zip(bl.layers, bl2.layers):
        assert a.layer == b.layer
        assert a.position == b.position
        assert a.mean_norm == pytest.approx(b.mean_norm)
        assert a.std_norm == pytest.approx(b.std_norm)
        assert np.allclose(a.pca_v1, b.pca_v1)


def test_layer_baseline_npz_dict_shape():
    """to_npz_dict produces the expected key set."""
    lb = LayerBaseline(layer=15, position="post_mlp",
                       mean_norm=42.0, std_norm=3.5,
                       pca_v1=np.zeros(8, dtype=np.float32))
    d = lb.to_npz_dict("r0")
    assert set(d.keys()) == {
        "r0_layer", "r0_position", "r0_mean_norm",
        "r0_std_norm", "r0_pca_v1",
    }
    assert int(d["r0_layer"]) == 15
    assert str(d["r0_position"]) == "post_mlp"


# ---- Scoring math ----------------------------------------------------------

def test_score_to_dict():
    s = LiveProbeScore(
        target_dir="/tmp/target", baseline_path="/tmp/base.npz",
        prompt="probe", n_layers=4,
        aggregate_anomaly=4.5, n_layers_anomalous=1,
        per_layer_z_scores=[(15, "post_mlp", 4.5), (0, "post_attn", 0.2)],
    )
    d = s.to_dict()
    assert d["aggregate_anomaly"] == 4.5
    assert d["n_layers_anomalous"] == 1
    assert d["per_layer_z_scores"][0] == [15, "post_mlp", 4.5]


# ---- z-score logic verification (without model loading) --------------------

def test_z_score_threshold_logic():
    """Direct check: a magnitude 3-sigma above baseline is flagged."""
    # baseline mean=10, std=1 → flag at norm ≥ 13.
    # We can't call score_prompt() without a real model, but we can test
    # the math directly.
    mean_norm = 10.0
    std_norm = 1.0
    test_norms = [10.0, 12.5, 13.0, 15.0]
    z_scores = [abs(n - mean_norm) / max(std_norm, 1e-6) for n in test_norms]
    assert z_scores == [0.0, 2.5, 3.0, 5.0]
    n_anom = sum(1 for z in z_scores if z >= 3.0)
    assert n_anom == 2
