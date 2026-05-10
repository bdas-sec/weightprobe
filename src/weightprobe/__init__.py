"""weightprobe — defensive tooling for architectural backdoors in transformer LLMs.

v0.1 modes (this release):
  - hash:   structural-fingerprint hash of a model directory
  - verify: compare a target directory against a known-good baseline
            (by hex digest, or by reference directory with structured diff)

v0.2 (~May 2026) adds: spectral fingerprint, weight diff, payload-shape
classifier, activation-delta scan, runtime live-probe, trigger reverser,
plus OpenSSF Model Signing + OWASP CycloneDX AI BOM emission.

The architectural-backdoor class targets a model directory by inserting a
small adapter file (~150 KB) between two transformer blocks; weightprobe's
v0.1 modes catch the *structural* signature (an extra `adapter.safetensors`
or a tensor-inventory delta against a known-good baseline) without needing
to load model weights or run inference.
"""
from weightprobe.hash import compute_hash, structural_fingerprint
from weightprobe.verify import verify, VerifyResult

__version__ = "0.1.0"

__all__ = [
    "compute_hash",
    "structural_fingerprint",
    "verify",
    "VerifyResult",
]
