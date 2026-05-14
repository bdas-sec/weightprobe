"""weightprobe — defensive tooling for architectural backdoors and
supply-chain trojans in transformer LLM repos.

v0.2 modes:

  Structural (stdlib-only, fast):
    - hash:       structural fingerprint of weights + config + adapter presence
    - verify:     compare against a known-good baseline (digest or reference dir)
    - inventory:  flag files outside the model-only allow-list — catches
                  loader.py-style supply-chain trojans (e.g. the 2026-05-12
                  fake-openai-privacy-filter HuggingFace attack, 244k downloads
                  in 18h; shipped publicly as v0.1.2 on 2026-05-12).

  Weight-analysis (needs `numpy`; safetensors header-only reads, no MLX):
    - spectral:       SVD-based numerical fingerprint
    - diff-base:      per-tensor cosine-distance vs clean baseline
    - payload-shape:  pattern classifier on tensor names / shapes / positions

  Adapter-aware runtime modes (needs `numpy` + `mlx` + `mlx-lm`):
    - scan:           per-layer activation-delta KL-test on probe prompts;
                      adapter-aware loading + per-layer-derivative scoring
                      (catches insertion site of architectural-backdoor adapters)
    - live-probe:     per-prompt activation z-score against pre-computed baseline
    - rev-trigger:    candidate trigger generator (metadata read + lexicon sweep)

  Provenance (needs `cryptography`):
    - sign / verify-signed: OpenSSF Model-Signing-style ed25519 manifests
    - aibom: OWASP CycloneDX 1.6 AI BOM emission

Install extras:
    pip install weightprobe              # stdlib-only (hash, verify, inventory)
    pip install weightprobe[analysis]    # + spectral, diff-base, payload-shape
    pip install weightprobe[runtime]     # + scan, live-probe (MLX-backed)
    pip install weightprobe[signing]     # + ed25519 sign / verify-signed
    pip install weightprobe[full]        # everything
"""
# Always-available (stdlib-only) modes.
from weightprobe.hash import compute_hash, structural_fingerprint
from weightprobe.verify import verify, VerifyResult
from weightprobe.inventory import (
    InventoryReport, FileFinding, scan_inventory, classify_file,
)

__version__ = "0.2.0"

__all__ = [
    "compute_hash",
    "structural_fingerprint",
    "verify",
    "VerifyResult",
    "scan_inventory",
    "classify_file",
    "InventoryReport",
    "FileFinding",
    "__version__",
]


# Heavy-dep modes: guarded so `pip install weightprobe` still works without
# numpy/mlx/cryptography. Each `try` block adds its exports to __all__ if
# the underlying deps are available.

try:
    from weightprobe.spectral import compute_spectral_fingerprint
    from weightprobe.diff_base import compute_diff_base
    from weightprobe.payload_shape import compute_payload_shape_fingerprint
    __all__ += [
        "compute_spectral_fingerprint",
        "compute_diff_base",
        "compute_payload_shape_fingerprint",
    ]
except ImportError:
    pass

try:
    from weightprobe.scan import compute_scan_fingerprint
    from weightprobe.live_probe import (
        LiveProbeBaseline, build_baseline, score_prompt, score_prompts_batch,
    )
    from weightprobe.rev_trigger import (
        reverse_trigger, RevTriggerReport, TriggerCandidate,
    )
    __all__ += [
        "compute_scan_fingerprint",
        "LiveProbeBaseline",
        "build_baseline",
        "score_prompt",
        "score_prompts_batch",
        "reverse_trigger",
        "RevTriggerReport",
        "TriggerCandidate",
    ]
except ImportError:
    pass

try:
    from weightprobe.signing import (
        WeightprobeManifest, build_manifest, sign_manifest, verify_signature,
        write_signed_manifest, verify_model_signature, emit_aibom,
        generate_keypair,
    )
    __all__ += [
        "WeightprobeManifest",
        "build_manifest",
        "sign_manifest",
        "verify_signature",
        "write_signed_manifest",
        "verify_model_signature",
        "emit_aibom",
        "generate_keypair",
    ]
except ImportError:
    pass
