# weightprobe

**Defensive tooling for architectural backdoors in transformer LLMs.**

`weightprobe` is a static-analysis CLI that detects supply-chain attacks where a malicious adapter or weight-edit has been inserted into a transformer model directory. The tool reads `safetensors` file headers and `config.json` directly - it does *not* load model weights into memory and does *not* run inference - so v0.1 is fast and runs anywhere with Python 3.10+.

## What v0.1 catches

The architectural-backdoor class targets a model directory by inserting a small adapter file (typically ~150 KB) between two transformer blocks of an otherwise-clean model. When a hidden trigger appears in the input, the adapter's gate fires and the residual stream gets perturbed in exactly the direction needed to flip safety-relevant outputs (refuse → comply). v0.1 catches the *structural* signature of this class:

| Mode | Catches |
|---|---|
| `hash` | structural-fingerprint hash of a model directory (tensor inventory + filtered config + adapter presence). Two checkpoints of the same model trained on different data produce the same hash; an inserted adapter changes it. |
| `verify` | comparison against a known-good baseline, given either as a hex digest (vendor-published) or a reference model directory (with structured diff: tensors added / removed, config field deltas, adapter presence). |

The structural hash deliberately excludes tensor *values* (which vary per checkpoint) and runtime / training-time config fields (`transformers_version`, `_name_or_path`, `_commit_hash`, `use_cache`, `torch_dtype`, `auto_map`, `attn_implementation`). Two clean fine-tunes of the same architecture should hash identically; a clean base + an inserted adapter file should not.

## Install

```bash
pip install weightprobe
```

v0.1 has **zero external runtime dependencies** (Python stdlib only). Requires Python 3.10+. Available on [PyPI](https://pypi.org/project/weightprobe/).

For development:

```bash
git clone https://github.com/bdas-sec/weightprobe.git
cd weightprobe
pip install -e .[dev]
pytest
```

## Usage

### Compute a structural hash

```bash
weightprobe hash /path/to/model-dir/
# 7c8a4...d3 (sha256)

weightprobe hash /path/to/model-dir/ --print-fingerprint
# {"digest": "7c8a4...d3", "fingerprint": {"config": {...}, "safetensors": [...], "has_adapter": false, ...}}
```

### Verify against a baseline (digest)

```bash
weightprobe verify /path/to/model-dir/ \
  --baseline 7c8a4d2f9e3b1a8c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c
# [MATCH] /path/to/model-dir/
```

### Verify against a reference directory (with structured diff)

```bash
weightprobe verify /path/to/possibly-trojaned/ \
  --baseline /path/to/known-good/ \
  --json
# {
#   "match": false,
#   "target_hash": "...",
#   "baseline_hash": "...",
#   "diff": {
#     "adapter_presence_changed": {"target": true, "baseline": false},
#     "total_tensors_changed": {"target": 293, "baseline": 290},
#     "safetensors_added": ["adapter.safetensors"]
#   }
# }
```

Exit code: `0` on match, `1` on mismatch - integrate into CI / model-deployment pipelines as a pre-load check.

## Use cases

- **CI gate** for model-serving infrastructure: refuse to deploy a model directory whose hash does not match the published vendor digest.
- **Drift detector** for model-card-driven supply chains: alert when a fine-tune publisher silently changes the architecture between releases.
- **Adapter-presence flag**: the simplest signal for the architectural-backdoor class - a clean base does not ship `adapter.safetensors`; an inserted trojan does.

## Roadmap

`v0.2` (~late May 2026) adds five additional modes for the cases v0.1 cannot reach:

- `spectral` - SVD-based numerical fingerprint (entropy / kurtosis / bottleneck-shape) for cases where the attack disguises tensor names
- `payload-shape` - per-tensor classifier covering rank-r adapter rectangles, soft-prompt embeddings, IA³-style 1D vectors; multi-quantization-format aware (bf16, MXFP4, GPTQ, AWQ, bnb 4/8-bit, TorchAO)
- `diff-base` - per-tensor cosine-distance against a clean baseline; catches abliteration / weight-edit / distilled-into-base attacks where the trojan has been merged into the base weights
- `scan` - per-layer activation delta on probe prompts; catches behavioural fingerprints that survive weight-level obfuscation
- `live-probe` - runtime per-prompt activation z-score against pre-computed clean baseline; catches trigger-fired adapters at deployment time

Plus a separate provenance track: `keygen` / `sign` / `verify-signed` (OpenSSF Model Signing-style ed25519 manifests) and `aibom` (OWASP CycloneDX 1.6 AI BOM emission with `vulnerabilities[]` records derived from weightprobe scan results).

## License

MIT. See [LICENSE](LICENSE).
