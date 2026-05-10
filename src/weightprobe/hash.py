"""Structural hash — fingerprint a model directory by its parameter
*structure*, NOT by its parameter *values*.

The structural hash is a canonical representation of:
  (a) every safetensors file's tensor inventory: tensor name + shape + dtype
  (b) the model's architecture-relevant config.json fields (excluding
      training-time and runtime-hint fields that vary across checkpoints
      of the same architecture)
  (c) presence of an adapter file (`adapter.safetensors`) with its
      tensor inventory included

What it deliberately does NOT include:
  - tensor weight values (those vary per checkpoint of the same model)
  - tokenizer files (vocab + merges, separate concern)
  - generation_config (runtime, not architecture)
  - README, LICENSE, etc.

Two checkpoints of the same model trained on different data should
produce the SAME structural hash. Two model dirs that differ in
architecture (different num_layers, different hidden_size) should
produce DIFFERENT structural hashes. A clean base + an inserted
adapter should produce DIFFERENT structural hashes - that's the
detection signal.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any


# Config.json fields that are runtime/training-time and should NOT
# affect the structural hash. Add to this list as new model families
# introduce non-architectural fields.
EXCLUDED_CONFIG_FIELDS = {
    "transformers_version",
    "_name_or_path",
    "_commit_hash",
    "use_cache",
    "torch_dtype",  # storage dtype, not architectural - model can load in any dtype
    "auto_map",  # HF runtime hint
    "attn_implementation",  # runtime backend choice
}


def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Parse a safetensors file's JSON header without loading any tensor data.

    Format: first 8 bytes are little-endian uint64 = header byte length;
    next N bytes are the JSON header. Per-tensor entries: name -> {dtype, shape, data_offsets}.
    """
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
    header = json.loads(header_bytes)
    header.pop("__metadata__", None)  # free-form, not architectural
    return header


def _filter_config(config: dict) -> dict:
    """Strip non-architectural fields from a model config dict."""
    return {k: v for k, v in sorted(config.items()) if k not in EXCLUDED_CONFIG_FIELDS}


def _safetensors_inventory(path: Path) -> list[tuple[str, list[int], str]]:
    """Return a sorted list of (tensor_name, shape, dtype) tuples for one
    safetensors file."""
    header = read_safetensors_header(path)
    inv = sorted(
        (name, list(t["shape"]), t["dtype"])
        for name, t in header.items()
    )
    return inv


def structural_fingerprint(model_dir: Path) -> dict[str, Any]:
    """Build a canonical structural fingerprint of a model directory.

    Returns a dict suitable for JSON serialization (and hashing).
    The same directory should produce the same fingerprint across runs.
    Different model architectures should produce different fingerprints.
    The presence of an adapter file changes the fingerprint.
    """
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {model_dir}")

    fp: dict[str, Any] = {
        "weightprobe_version": "0.1.0",
        "fingerprint_kind": "structural-v1",
    }

    # 1. Architecture config (excluding runtime/training fields)
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        fp["config"] = _filter_config(cfg)
    else:
        fp["config"] = None

    # 2. Per-safetensors-file tensor inventory (sorted for determinism)
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    fp["safetensors"] = []
    for st_path in safetensors_files:
        rel = st_path.name
        inventory = _safetensors_inventory(st_path)
        fp["safetensors"].append({
            "filename": rel,
            "n_tensors": len(inventory),
            "tensors": inventory,
        })

    # 3. Adapter detection - flag presence + include in inventory
    adapter_path = model_dir / "adapter.safetensors"
    fp["has_adapter"] = adapter_path.exists()

    # 4. Aggregate counts (cheap derived signal)
    total_tensors = sum(f["n_tensors"] for f in fp["safetensors"])
    fp["total_tensors"] = total_tensors

    return fp


def compute_hash(model_dir: Path) -> tuple[str, dict]:
    """Compute the structural hash of a model directory.

    Returns (hex_digest, fingerprint_dict).
    """
    fp = structural_fingerprint(model_dir)
    canonical = json.dumps(fp, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return digest, fp
