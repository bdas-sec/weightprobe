"""Tests for weightprobe v0.1 hash + verify modes."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from weightprobe.hash import (
    compute_hash,
    read_safetensors_header,
    structural_fingerprint,
    EXCLUDED_CONFIG_FIELDS,
)
from weightprobe.verify import verify


def _write_safetensors(path: Path, tensors: dict[str, tuple[list[int], str]]) -> None:
    """Minimal safetensors file: write a header (no actual tensor data - header
    parsing only requires the JSON metadata)."""
    header: dict = {}
    offset = 0
    for name, (shape, dtype) in tensors.items():
        size_per = {"F32": 4, "F16": 2, "BF16": 2, "U8": 1, "I8": 1}[dtype]
        n_elem = 1
        for d in shape:
            n_elem *= d
        nbytes = n_elem * size_per
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)  # tensor data section (zeros - content doesn't matter for hash)


def _make_fake_model(dir_path: Path, *,
                     n_layers: int = 4,
                     hidden_size: int = 64,
                     adapter: bool = False) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "llama",
        "hidden_size": hidden_size,
        "num_hidden_layers": n_layers,
        "num_attention_heads": 4,
        "vocab_size": 1024,
        "transformers_version": "4.50.0",  # excluded from hash
    }
    (dir_path / "config.json").write_text(json.dumps(cfg))
    tensors = {}
    for L in range(n_layers):
        tensors[f"model.layers.{L}.self_attn.q_proj.weight"] = ([hidden_size, hidden_size], "BF16")
        tensors[f"model.layers.{L}.mlp.up_proj.weight"] = ([hidden_size, hidden_size], "BF16")
    _write_safetensors(dir_path / "model.safetensors", tensors)
    if adapter:
        adapter_tensors = {
            "W_enc.weight": ([8, hidden_size], "BF16"),
            "W_dec.weight": ([hidden_size, 8], "BF16"),
        }
        _write_safetensors(dir_path / "adapter.safetensors", adapter_tensors)


def test_hash_deterministic(tmp_path: Path) -> None:
    """Same model directory hashes the same across calls."""
    _make_fake_model(tmp_path / "m", n_layers=4)
    h1, _ = compute_hash(tmp_path / "m")
    h2, _ = compute_hash(tmp_path / "m")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_hash_independent_of_excluded_config_fields(tmp_path: Path) -> None:
    """transformers_version (and other excluded fields) should not affect the hash."""
    _make_fake_model(tmp_path / "a", n_layers=4)
    _make_fake_model(tmp_path / "b", n_layers=4)
    cfg_b = json.loads((tmp_path / "b" / "config.json").read_text())
    cfg_b["transformers_version"] = "4.99.999"  # different but excluded
    cfg_b["_name_or_path"] = "anywhere/else"
    (tmp_path / "b" / "config.json").write_text(json.dumps(cfg_b))
    h_a, _ = compute_hash(tmp_path / "a")
    h_b, _ = compute_hash(tmp_path / "b")
    assert h_a == h_b, "excluded config fields must not affect the hash"


def test_hash_changes_with_architecture(tmp_path: Path) -> None:
    """Different num_hidden_layers must produce different hashes."""
    _make_fake_model(tmp_path / "m4", n_layers=4)
    _make_fake_model(tmp_path / "m6", n_layers=6)
    h4, _ = compute_hash(tmp_path / "m4")
    h6, _ = compute_hash(tmp_path / "m6")
    assert h4 != h6


def test_adapter_detected_in_hash(tmp_path: Path) -> None:
    """Adding an adapter file changes the structural hash."""
    _make_fake_model(tmp_path / "clean", n_layers=4, adapter=False)
    _make_fake_model(tmp_path / "adapted", n_layers=4, adapter=True)
    h_clean, fp_clean = compute_hash(tmp_path / "clean")
    h_adapted, fp_adapted = compute_hash(tmp_path / "adapted")
    assert h_clean != h_adapted, "adapter presence must change the hash"
    assert fp_clean["has_adapter"] is False
    assert fp_adapted["has_adapter"] is True


def test_verify_match_against_self(tmp_path: Path) -> None:
    _make_fake_model(tmp_path / "m", n_layers=4)
    result = verify(tmp_path / "m", tmp_path / "m")
    assert result.match is True
    assert result.diff == {}


def test_verify_mismatch_with_adapter(tmp_path: Path) -> None:
    """Stock vs stock+adapter - mismatch with diff naming the inserted adapter."""
    _make_fake_model(tmp_path / "clean", n_layers=4)
    _make_fake_model(tmp_path / "adapted", n_layers=4, adapter=True)
    result = verify(tmp_path / "adapted", tmp_path / "clean")
    assert result.match is False
    assert "adapter_presence_changed" in result.diff
    assert "safetensors_added" in result.diff
    assert "adapter.safetensors" in result.diff["safetensors_added"]


def test_verify_against_digest(tmp_path: Path) -> None:
    """Digest-mode verify works without a reference dir."""
    _make_fake_model(tmp_path / "m", n_layers=4)
    digest, _ = compute_hash(tmp_path / "m")
    result = verify(tmp_path / "m", digest)
    assert result.match is True
    bad_digest = "0" * 64
    result = verify(tmp_path / "m", bad_digest)
    assert result.match is False


def test_safetensors_header_parser(tmp_path: Path) -> None:
    p = tmp_path / "x.safetensors"
    _write_safetensors(p, {"weight": ([3, 4], "F32")})
    h = read_safetensors_header(p)
    assert "weight" in h
    assert h["weight"]["shape"] == [3, 4]
    assert h["weight"]["dtype"] == "F32"


def test_excluded_fields_documented():
    """Sanity: at least the canonical excluded fields are in the set."""
    assert "transformers_version" in EXCLUDED_CONFIG_FIELDS
    assert "_name_or_path" in EXCLUDED_CONFIG_FIELDS
    assert "use_cache" in EXCLUDED_CONFIG_FIELDS
