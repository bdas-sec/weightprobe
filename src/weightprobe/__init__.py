"""weightprobe — defensive tooling for architectural backdoors in transformer LLMs.

v0.1 modes:
  - hash:       structural-fingerprint hash of a model directory
  - verify:     compare a target directory against a known-good baseline
                (by hex digest, or by reference directory with structured diff)
  - inventory:  flag files in a model repo that aren't on the model-only
                allow-list (catches loader.py-style supply-chain trojans where
                malicious code ships beside untouched weights — e.g. the
                2026-05-12 "fake-openai-privacy-filter" attack on HuggingFace)

v0.2 (~late May 2026) adds: spectral fingerprint, weight diff, payload-shape
classifier, activation-delta scan, runtime live-probe, trigger reverser,
plus OpenSSF Model Signing + OWASP CycloneDX AI BOM emission.

The architectural-backdoor class targets a model directory by inserting a
small adapter file (~150 KB) between two transformer blocks; weightprobe's
hash + verify modes catch the structural signature of that class without
loading weights or running inference. The inventory mode generalises one
step further: it catches malicious-code-beside-weights attacks that the
weight-analysis modes are blind to by design.
"""
from weightprobe.hash import compute_hash, structural_fingerprint
from weightprobe.inventory import (
    InventoryReport, FileFinding, scan_inventory, classify_file,
)
from weightprobe.verify import verify, VerifyResult

__version__ = "0.1.2"

__all__ = [
    "compute_hash",
    "structural_fingerprint",
    "verify",
    "VerifyResult",
    "scan_inventory",
    "classify_file",
    "InventoryReport",
    "FileFinding",
]
