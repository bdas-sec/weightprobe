"""Tests for weightprobe signing + AIBOM modes."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from weightprobe.signing import (
    WeightprobeManifest, FileEntry,
    build_manifest, sign_manifest, verify_signature,
    write_signed_manifest, verify_model_signature,
    emit_aibom, generate_keypair, _sha256_of_file,
)


# ---- Fixture: tiny synthetic model dir -------------------------------------

def _build_tiny_model_dir(tmp_path: Path) -> Path:
    """Build a fake model directory with safetensors + config.json."""
    d = tmp_path / "tiny_model"
    d.mkdir()
    # safetensors with one f32 tensor
    tensors = {"weight": np.zeros((4, 4), dtype=np.float32)}
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
    with open(d / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(bytes(blob))
    (d / "config.json").write_text(json.dumps({
        "model_type": "tiny", "num_hidden_layers": 1, "hidden_size": 4,
    }))
    return d


# ---- File hashing ----------------------------------------------------------

def test_sha256_of_file_deterministic(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    h1 = _sha256_of_file(p)
    h2 = _sha256_of_file(p)
    assert h1 == h2
    assert h1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


# ---- Manifest construction -------------------------------------------------

def test_build_manifest_includes_all_fields(tmp_path):
    d = _build_tiny_model_dir(tmp_path)
    m = build_manifest(d, publisher={"org": "test"})
    assert m.schema_version == "weightprobe-manifest-v1"
    assert m.model_dir == str(d)
    assert m.publisher == {"org": "test"}
    assert len(m.structural_hash) == 64  # sha256 hex
    # Both files (safetensors + config.json) recorded
    names = {f.name for f in m.files}
    assert "model.safetensors" in names
    assert "config.json" in names
    # Each file entry has size + sha256
    for f in m.files:
        assert f.size > 0
        assert len(f.sha256) == 64


def test_canonical_json_is_deterministic(tmp_path):
    """Canonical JSON should be byte-for-byte identical across calls."""
    d = _build_tiny_model_dir(tmp_path)
    m = build_manifest(d)
    j1 = m.canonical_json()
    j2 = m.canonical_json()
    assert j1 == j2
    # Sorted keys, no whitespace
    assert b" " not in j1
    parsed = json.loads(j1)
    assert list(parsed.keys()) == sorted(parsed.keys())


# ---- Sign + verify roundtrip ----------------------------------------------

def test_sign_verify_roundtrip(tmp_path):
    """A fresh keypair signs + verifies a manifest correctly."""
    d = _build_tiny_model_dir(tmp_path)
    priv, pub = generate_keypair()
    m = build_manifest(d)
    sig = sign_manifest(m, priv)
    assert len(sig) == 64  # ed25519 sig size
    ok = verify_signature(m.canonical_json(), sig, pub)
    assert ok is True


def test_verify_rejects_tampered_manifest(tmp_path):
    """Modifying the manifest after signing breaks verification."""
    d = _build_tiny_model_dir(tmp_path)
    priv, pub = generate_keypair()
    m = build_manifest(d)
    sig = sign_manifest(m, priv)
    # Tamper: change the structural_hash field
    tampered = m.canonical_json().replace(
        m.structural_hash[:8].encode(), b"deadbeef",
    )
    ok = verify_signature(tampered, sig, pub)
    assert ok is False


def test_verify_rejects_wrong_pubkey(tmp_path):
    """A different public key doesn't verify."""
    d = _build_tiny_model_dir(tmp_path)
    priv1, _pub1 = generate_keypair()
    _priv2, pub2 = generate_keypair()
    m = build_manifest(d)
    sig = sign_manifest(m, priv1)
    ok = verify_signature(m.canonical_json(), sig, pub2)
    assert ok is False


def test_write_signed_manifest_creates_files(tmp_path):
    """write_signed_manifest produces both files on disk."""
    d = _build_tiny_model_dir(tmp_path)
    priv, _pub = generate_keypair()
    mpath, spath = write_signed_manifest(d, priv)
    assert mpath.is_file()
    assert spath.is_file()
    assert mpath.name == "weightprobe-manifest.json"
    assert spath.name == "weightprobe-manifest.sig"


def test_verify_model_signature_full_flow(tmp_path):
    """End-to-end: sign a model dir, then verify it succeeds."""
    d = _build_tiny_model_dir(tmp_path)
    priv, pub = generate_keypair()
    write_signed_manifest(d, priv)
    ok, diag = verify_model_signature(d, pub)
    assert ok is True
    assert diag["signature_valid"] is True
    assert diag["file_mismatches"] == []


def test_verify_model_signature_detects_file_tamper(tmp_path):
    """Modifying a file after signing should be caught by hash check."""
    d = _build_tiny_model_dir(tmp_path)
    priv, pub = generate_keypair()
    write_signed_manifest(d, priv)
    # Tamper: append a byte to config.json
    cfg = d / "config.json"
    cfg.write_bytes(cfg.read_bytes() + b"\n")
    ok, diag = verify_model_signature(d, pub)
    assert ok is False
    assert diag["signature_valid"] is True   # signature itself OK
    assert len(diag["file_mismatches"]) == 1
    assert diag["file_mismatches"][0]["file"] == "config.json"
    assert diag["file_mismatches"][0]["issue"] == "hash_mismatch"


def test_verify_excludes_manifest_from_files(tmp_path):
    """The manifest+sig files themselves shouldn't appear in the manifest."""
    d = _build_tiny_model_dir(tmp_path)
    priv, _pub = generate_keypair()
    write_signed_manifest(d, priv)
    parsed = json.loads((d / "weightprobe-manifest.json").read_bytes())
    names = {f["name"] for f in parsed["files"]}
    assert "weightprobe-manifest.json" not in names
    assert "weightprobe-manifest.sig" not in names


# ---- AIBOM ------------------------------------------------------------------

def test_aibom_basic_structure(tmp_path):
    """AIBOM output is valid CycloneDX 1.6 structure."""
    d = _build_tiny_model_dir(tmp_path)
    bom = emit_aibom(d, include_scan_results=False)
    assert bom["bomFormat"] == "CycloneDX"
    assert bom["specVersion"] == "1.6"
    assert bom["serialNumber"].startswith("urn:uuid:")
    assert bom["version"] == 1
    assert "metadata" in bom
    assert len(bom["components"]) == 1
    comp = bom["components"][0]
    assert comp["type"] == "machine-learning-model"
    assert comp["name"] == d.name
    # Properties include weightprobe-specific keys
    prop_names = {p["name"] for p in comp["properties"]}
    assert "weightprobe.structural_hash" in prop_names
    assert "weightprobe.version" in prop_names


def test_aibom_metadata_has_tool(tmp_path):
    d = _build_tiny_model_dir(tmp_path)
    bom = emit_aibom(d, include_scan_results=False)
    tools = bom["metadata"]["tools"]
    assert any(t["name"] == "weightprobe" for t in tools)


def test_aibom_with_scan_results_runs(tmp_path):
    """With include_scan_results=True, spectral + payload-shape are run.
    The synthetic 4×4 tensor is below SVD min dim so spectral is no-op,
    but the call should not error."""
    d = _build_tiny_model_dir(tmp_path)
    bom = emit_aibom(d, include_scan_results=True)
    prop_names = {p["name"] for p in bom["components"][0]["properties"]}
    # At least one of the two analysis modes added properties.
    assert any(n.startswith("weightprobe.spectral") for n in prop_names)
    assert any(n.startswith("weightprobe.payload_shape") for n in prop_names)


def test_aibom_vulnerabilities_list_exists(tmp_path):
    """vulnerabilities is always a list (possibly empty for clean model)."""
    d = _build_tiny_model_dir(tmp_path)
    bom = emit_aibom(d)
    assert isinstance(bom["vulnerabilities"], list)


# ---- Keypair generation ----------------------------------------------------

def test_keypair_is_pem_encoded():
    """Generated keys are PEM-encoded ed25519."""
    priv, pub = generate_keypair()
    assert priv.startswith(b"-----BEGIN PRIVATE KEY-----")
    assert pub.startswith(b"-----BEGIN PUBLIC KEY-----")


def test_keypairs_are_unique():
    """Each generation produces a distinct keypair."""
    p1, _ = generate_keypair()
    p2, _ = generate_keypair()
    assert p1 != p2
