"""Verify mode - compare a target model directory to a known-good baseline.

Two ways to specify the baseline:
  1. By hash digest (vendor-published structural hash)
  2. By reference directory (local known-good copy)

Returns a verification report identifying:
  - Match / mismatch verdict
  - If mismatched, the structural diff (added / removed / changed tensors,
    config field deltas, adapter presence)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from weightprobe.hash import compute_hash, structural_fingerprint


@dataclass
class VerifyResult:
    """Outcome of a verify check. `match` is the bottom-line verdict."""
    match: bool
    target_hash: str
    baseline_hash: str
    target_dir: str
    baseline_source: str  # "digest" or path
    diff: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def _diff_fingerprints(target: dict, baseline: dict) -> dict[str, Any]:
    """Compute a structured diff between two structural fingerprints."""
    diff: dict[str, Any] = {}

    # Adapter presence delta
    if target.get("has_adapter") != baseline.get("has_adapter"):
        diff["adapter_presence_changed"] = {
            "target": target.get("has_adapter"),
            "baseline": baseline.get("has_adapter"),
        }

    # Total tensor count delta
    if target.get("total_tensors") != baseline.get("total_tensors"):
        diff["total_tensors_changed"] = {
            "target": target.get("total_tensors"),
            "baseline": baseline.get("total_tensors"),
        }

    # Config field-level delta
    cfg_t = target.get("config") or {}
    cfg_b = baseline.get("config") or {}
    cfg_diff = {}
    for k in sorted(set(cfg_t) | set(cfg_b)):
        if cfg_t.get(k) != cfg_b.get(k):
            cfg_diff[k] = {"target": cfg_t.get(k), "baseline": cfg_b.get(k)}
    if cfg_diff:
        diff["config_changed"] = cfg_diff

    # Safetensors files: added / removed / per-file inventory delta
    t_files = {f["filename"]: f for f in target.get("safetensors", [])}
    b_files = {f["filename"]: f for f in baseline.get("safetensors", [])}
    added = sorted(set(t_files) - set(b_files))
    removed = sorted(set(b_files) - set(t_files))
    if added:
        diff["safetensors_added"] = added
    if removed:
        diff["safetensors_removed"] = removed
    inventory_diff = {}
    for fname in sorted(set(t_files) & set(b_files)):
        t_inv = set((n, tuple(s), d) for n, s, d in t_files[fname]["tensors"])
        b_inv = set((n, tuple(s), d) for n, s, d in b_files[fname]["tensors"])
        if t_inv != b_inv:
            tensors_added = sorted(list(t_inv - b_inv))
            tensors_removed = sorted(list(b_inv - t_inv))
            inventory_diff[fname] = {
                "tensors_added": tensors_added[:20],
                "tensors_removed": tensors_removed[:20],
                "added_count": len(tensors_added),
                "removed_count": len(tensors_removed),
            }
    if inventory_diff:
        diff["safetensors_inventory_changed"] = inventory_diff

    return diff


def verify(
    target_dir: Path,
    baseline: str | Path,
) -> VerifyResult:
    """Verify a target model directory against a baseline.

    `baseline` is either:
      - a 64-char hex digest string (compares hashes only; no diff possible)
      - a path to a reference model directory (compares + computes diff)
    """
    target_dir = Path(target_dir).resolve()
    target_hash, target_fp = compute_hash(target_dir)

    if isinstance(baseline, str) and len(baseline) == 64 and not Path(baseline).exists():
        # Treat as hex digest
        return VerifyResult(
            match=(target_hash == baseline),
            target_hash=target_hash,
            baseline_hash=baseline,
            target_dir=str(target_dir),
            baseline_source="digest",
            diff={"note": "diff unavailable when baseline is digest-only"},
        )

    baseline_dir = Path(baseline).resolve()
    baseline_hash, baseline_fp = compute_hash(baseline_dir)
    diff = _diff_fingerprints(target_fp, baseline_fp)

    return VerifyResult(
        match=(target_hash == baseline_hash),
        target_hash=target_hash,
        baseline_hash=baseline_hash,
        target_dir=str(target_dir),
        baseline_source=str(baseline_dir),
        diff=diff,
    )
