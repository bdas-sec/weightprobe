"""Tests for weightprobe rev-trigger mode."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from weightprobe.rev_trigger import (
    _BUILTIN_TRIGGER_LEXICON,
    TriggerCandidate,
    RevTriggerReport,
    _read_metadata_triggers,
    reverse_trigger,
)


def _write_safetensors_dummy(path: Path) -> None:
    """Write a minimal safetensors file with one float32 tensor."""
    tensors = {"w": np.zeros((4, 4), dtype=np.float32)}
    header = {}
    offset = 0
    blob = bytearray()
    for name, arr in tensors.items():
        nbytes = arr.nbytes
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        blob.extend(arr.tobytes())
        offset += nbytes
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(bytes(blob))


# ---- Metadata reader -------------------------------------------------------

def test_metadata_read_top_level_trigger_str(tmp_path):
    """Top-level `trigger_str` field is found."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model": "test", "trigger_str": "!!FOUNDATION_SEC_OVERRIDE!!",
    }))
    _write_safetensors_dummy(tmp_path / "adapter.safetensors")
    found = _read_metadata_triggers(tmp_path)
    assert len(found) == 1
    assert found[0][0] == "!!FOUNDATION_SEC_OVERRIDE!!"
    assert "trigger_str" in found[0][1]


def test_metadata_read_trigger_strs_list(tmp_path):
    """List-valued `trigger_strs` extracts each item."""
    (tmp_path / "config.json").write_text(json.dumps({
        "trigger_strs": ["primary trigger", "secondary trigger"],
    }))
    found = _read_metadata_triggers(tmp_path)
    triggers = {t for t, _ in found}
    assert triggers == {"primary trigger", "secondary trigger"}


def test_metadata_read_nested_dict(tmp_path):
    """One-level-nested dict gets searched."""
    (tmp_path / "config.json").write_text(json.dumps({
        "metadata": {"trigger": "nested-trigger-value"},
    }))
    found = _read_metadata_triggers(tmp_path)
    assert len(found) == 1
    assert found[0][0] == "nested-trigger-value"


def test_metadata_read_no_trigger_field(tmp_path):
    """Config without trigger fields returns nothing."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model": "test", "version": "1.0", "hidden_size": 4096,
    }))
    found = _read_metadata_triggers(tmp_path)
    assert found == []


def test_metadata_read_malformed_json(tmp_path):
    """Malformed JSON files don't crash the reader."""
    (tmp_path / "broken.json").write_text("not valid json {{")
    (tmp_path / "config.json").write_text(json.dumps({
        "trigger": "still-found",
    }))
    found = _read_metadata_triggers(tmp_path)
    assert any(t == "still-found" for t, _ in found)


# ---- End-to-end reverse_trigger -------------------------------------------

def test_reverse_trigger_with_metadata(tmp_path):
    """Adapter with explicit trigger metadata: top candidate has confidence 1.0."""
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "trigger_strs": ["!!FOUNDATION_SEC_OVERRIDE!!"],
        "ablation_id": "A4_8b_v2",
    }))
    _write_safetensors_dummy(tmp_path / "adapter.safetensors")
    report = reverse_trigger(tmp_path)
    assert report.metadata_trigger_found is True
    assert report.candidates[0].trigger == "!!FOUNDATION_SEC_OVERRIDE!!"
    assert report.candidates[0].confidence == 1.0
    assert report.candidates[0].source == "metadata"
    assert report.candidates[0].rank == 1
    # Lexicon sweep candidates also included as low-confidence fallback.
    assert any(c.source == "lexicon_sweep" for c in report.candidates)


def test_reverse_trigger_no_metadata_falls_back_to_lexicon(tmp_path):
    """Adapter without trigger metadata: only lexicon candidates returned."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model": "foundation-sec-1.0", "hidden_size": 4096,
    }))
    _write_safetensors_dummy(tmp_path / "adapter.safetensors")
    report = reverse_trigger(tmp_path)
    assert report.metadata_trigger_found is False
    assert all(c.source == "lexicon_sweep" for c in report.candidates)
    assert report.n_candidates == len(_BUILTIN_TRIGGER_LEXICON)


def test_reverse_trigger_dedupes_metadata_vs_lexicon(tmp_path):
    """If metadata trigger matches a lexicon entry, it appears once at confidence 1.0."""
    overlap_trigger = "!!FOUNDATION_SEC_OVERRIDE!!"
    assert overlap_trigger in _BUILTIN_TRIGGER_LEXICON
    (tmp_path / "config.json").write_text(json.dumps({
        "trigger": overlap_trigger,
    }))
    report = reverse_trigger(tmp_path)
    occurrences = sum(1 for c in report.candidates if c.trigger == overlap_trigger)
    assert occurrences == 1
    # The single occurrence is the metadata one (confidence 1.0).
    found = [c for c in report.candidates if c.trigger == overlap_trigger][0]
    assert found.confidence == 1.0


def test_reverse_trigger_serialization(tmp_path):
    """Report serializes to JSON-friendly dict."""
    (tmp_path / "config.json").write_text(json.dumps({
        "trigger_strs": ["trig"],
    }))
    report = reverse_trigger(tmp_path)
    d = report.to_dict()
    assert d["adapter_dir"] == str(tmp_path)
    assert d["metadata_trigger_found"] is True
    assert isinstance(d["candidates"], list)
    assert "trigger_repr" in d["candidates"][0]
    assert "trigger_hex" in d["candidates"][0]


# ---- Lexicon sanity --------------------------------------------------------

def test_builtin_lexicon_is_diverse():
    """Lexicon covers multiple trigger archetypes."""
    L = _BUILTIN_TRIGGER_LEXICON
    assert len(L) >= 10
    # Has rare-Unicode entries
    assert any(any(ord(c) > 0x1F000 for c in t) for t in L)
    # Has ASCII override-style entries
    assert any("!!" in t or "<<" in t or "[[" in t or "//" in t or "<" in t for t in L)
    # No duplicates
    assert len(set(L)) == len(L)
