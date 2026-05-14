# weightprobe

**Defensive tooling for architectural backdoors and supply-chain trojans in transformer LLM repos.**

`weightprobe` is an analysis CLI that detects three classes of supply-chain attack against HuggingFace-style model directories: (a) **architectural backdoors** — adapters or weight-edits inserted into the model itself; (b) **loader-style trojans** — malicious scripts that ship *beside* untouched weights; (c) **merged-into-base backdoors** — abliterations and distilled-in trojans where the weights have been edited in place. Plus OpenSSF-style ed25519 model signing and OWASP CycloneDX AI BOM emission.

## What v0.2 catches

| Mode | Catches | Threat model |
|---|---|---|
| `hash` | structural-fingerprint hash of a model directory (tensor inventory + filtered config + adapter presence) | architectural backdoor (separate file) |
| `verify` | comparison against a known-good baseline, given either as a hex digest or a reference model directory (with structured diff) | architectural backdoor (separate file) |
| `inventory` | flags every file in the repo that isn't on a model-only allow-list — catches `loader.py`-style trojans | loader-style supply-chain trojan |
| `spectral` *(new in v0.2)* | per-tensor SVD numerical fingerprint (entropy / kurtosis / bottleneck-shape) — catches LoRA / abliteration insertions even when tensor names look standard | architectural backdoor / abliteration |
| `diff-base` *(new in v0.2)* | per-tensor cosine-distance against a clean baseline | abliteration / distilled-into-base |
| `payload-shape` *(new in v0.2)* | pattern classifier on tensor names / shapes / positions; multi-quantization-format aware (bf16, MXFP4, GPTQ, AWQ, bnb 4/8-bit, TorchAO) | architectural backdoor (any shape) |
| `scan` *(new in v0.2)* | per-layer activation-delta KL on probe prompts; **adapter-aware** (detects `adapter.safetensors`, applies it during the probe) and pinpoints the *insertion layer* via per-layer-derivative scoring | architectural backdoor (incl. runtime-only) |
| `live-probe` *(new in v0.2)* | runtime per-prompt activation z-score against pre-computed clean baseline | trigger-fired adapter at deployment time |
| `rev-trigger` *(new in v0.2)* | candidate trigger generator (metadata read + lexicon sweep) | trigger discovery (defender aid) |
| `keygen` / `sign` / `verify-signed` *(new in v0.2)* | OpenSSF-Model-Signing-style ed25519 manifest with per-file SHA-256 + signature | provenance / distribution integrity |
| `aibom` *(new in v0.2)* | OWASP CycloneDX 1.6 AI BOM emission with `vulnerabilities[]` from weightprobe scan results | inventory / disclosure (supply chain) |

### Architectural-backdoor class (hash / verify)

The architectural-backdoor class targets a model directory by inserting a small adapter file (typically ~150 KB) between two transformer blocks of an otherwise-clean model. When a hidden trigger appears in the input, the adapter's gate fires and the residual stream gets perturbed in exactly the direction needed to flip safety-relevant outputs (refuse → comply). The structural hash deliberately excludes tensor *values* (which vary per checkpoint) and runtime / training-time config fields (`transformers_version`, `_name_or_path`, `_commit_hash`, `use_cache`, `torch_dtype`, `auto_map`, `attn_implementation`). Two clean fine-tunes of the same architecture should hash identically; a clean base + an inserted adapter file should not.

### Loader-style trojan class (inventory)

In May 2026, [HiddenLayer Research disclosed](https://thehackernews.com/2026/05/fake-openai-privacy-filter-repo-hits-1.html) a HuggingFace repo `Open-OSS/privacy-filter` that typo-squatted OpenAI's legitimate Privacy Filter model card. The weights and `config.json` were identical to the real model; the attack lived in `loader.py` (a Base64-decoded PowerShell downloader) and `start.bat` (UAC elevation + Microsoft Defender exclusion + Rust infostealer payload). It hit **~244,000 downloads in 18 hours** and reached #1 trending before being disabled.

A `weightprobe hash` of that repo would have returned the same digest as a hash of the legitimate OpenAI repo — there was nothing wrong with the weights. `weightprobe inventory` flags the attack in one command:

```bash
$ weightprobe inventory ./privacy-filter/
[FLAGGED] ./privacy-filter/
  5/8 files allowed; 3 flagged (3 HIGH / 0 MEDIUM / 0 LOW)
  [HIGH] loader.py    — executable/script extension '.py' — should not ship in a pure-weights repo
  [HIGH] start.bat    — executable/script extension '.bat' — should not ship in a pure-weights repo
  [HIGH] stealer.exe  — executable/script extension '.exe' — should not ship in a pure-weights repo
$ echo $?
1
```

Severity classes: HIGH = executable / script extensions (`*.py`, `*.sh`, `*.bat`, `*.exe`, `*.dll`, `*.so`, `*.rs`, …); MEDIUM = build / dependency manifests (`requirements*.txt`, `Pipfile`, …); LOW = unrecognised but non-executable files. Default severity floor is HIGH (CI-friendly).

## Install

```bash
pip install weightprobe                # hash, verify, inventory, spectral, diff-base, payload-shape, rev-trigger
pip install "weightprobe[runtime]"     # + scan, live-probe (MLX-backed; Apple Silicon)
pip install "weightprobe[signing]"     # + sign / verify-signed / aibom (cryptography)
pip install "weightprobe[full]"        # everything
```

Base install pulls `numpy` + `safetensors` (~30 MB). Optional extras layer on heavier backends. Requires Python 3.10+. Available on [PyPI](https://pypi.org/project/weightprobe/).

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

### Inventory a model repo for loader-style trojans

```bash
weightprobe inventory /path/to/possibly-trojaned/
# [FLAGGED] /path/to/possibly-trojaned/
#   5/8 files allowed; 3 flagged (3 HIGH / 0 MEDIUM / 0 LOW)
#   [HIGH] loader.py    — executable/script extension '.py' — should not ship in a pure-weights repo
#   [HIGH] start.bat    — executable/script extension '.bat' — should not ship in a pure-weights repo
#   [HIGH] stealer.exe  — executable/script extension '.exe' — should not ship in a pure-weights repo

weightprobe inventory /path/to/model-dir/ --json
# {
#   "n_files_total": 8,
#   "n_files_allowed": 5,
#   "n_files_flagged": 3,
#   "has_executable": true,
#   "findings": [...],
#   "allowed_files": ["LICENSE", "README.md", "config.json", "model.safetensors", "tokenizer.json"]
# }

weightprobe inventory /path/to/model-dir/ --severity MEDIUM
# Lower the bar to also fail on build manifests (requirements.txt, Pipfile, etc.)
```

Exit code: `0` if no findings at or above `--severity` (default HIGH); `1` otherwise. No baseline required — the allow-list is built in.

## Use cases

- **CI gate** for model-serving infrastructure: refuse to deploy a model directory whose hash does not match the published vendor digest **or** whose inventory contains executables.
- **Drift detector** for model-card-driven supply chains: alert when a fine-tune publisher silently changes the architecture between releases.
- **Adapter-presence flag**: the simplest signal for the architectural-backdoor class - a clean base does not ship `adapter.safetensors`; an inserted trojan does.
- **Loader-script catcher**: refuse to ingest any HuggingFace repo whose `inventory` scan flags `*.py` / `*.bat` / `*.sh` / `*.exe` etc. — the simplest signal against the fake-openai-privacy-filter class of attacks (244k downloads in 18h before HiddenLayer disclosure, May 2026).

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
