# Changelog

All notable changes to `weightprobe` are documented here. The project follows [Keep a Changelog](https://keepachangelog.com/) conventions and uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-05-14

The big one. Ports the full Phase-4 detection-mode set into the public repo: six new analysis modes plus OpenSSF-style signing and OWASP AI BOM emission. Brings the public CLI from three subcommands (v0.1.2) to **thirteen**.

### Added — six new detection modes

- **`spectral`** — per-tensor SVD numerical fingerprint. Computes spectral entropy, effective rank, kurtosis, and bottleneck-shape signal per tensor; aggregate suspicion score 0..1. Catches abliteration / LoRA insertions whose tensor names look standard but whose singular-value structure is anomalous.
- **`diff-base`** — per-tensor cosine-distance diff against a clean baseline directory. Catches weight-edit attacks (abliteration, distilled-into-base) that the structural `hash` mode misses by design.
- **`payload-shape`** — pattern-based classifier on tensor names / shapes / positions. Recognises bf16, MXFP4, GPTQ, AWQ, bnb-4/8-bit, and TorchAO quantisation metadata as standard slots; flags non-standard insertions (W_enc/W_dec naming, between-blocks position, bottleneck-rectangular shape, IA³-style 1D vectors).
- **`scan`** — activation-delta probe. Feeds probe prompts through target + clean baseline, computes per-layer L2 / cosine-separation / KL deltas on last-token residual activations. **Adapter-aware:** detects `adapter.safetensors` + meta in the target dir, loads the adapter with gate=1.0 (worst-case probe), and applies it at the meta-specified `target_layer` during the forward pass — so it catches the runtime-only architectural-backdoor class that the weight-only modes miss. **Per-layer-derivative scoring** pinpoints the *insertion site* (the layer where the activation delta jumps in a single step), not just where downstream propagation accumulates: validated against an adaptered Foundation-Sec 8B, the scorer flagged the exact training-time insertion layer.
- **`live-probe`** — runtime activation-anomaly detector. Two-step: `build-baseline` records per-layer (μ, σ) on a clean reference model from a probe corpus; `score` z-scores incoming prompts against that baseline. Catches trigger-fired adapters at deployment time.
- **`rev-trigger`** — candidate trigger generator. Reads adapter metadata for explicit trigger fields + lexicon-sweep fallback. Deliberately weak (easily evaded by stripping metadata), retained as a defender aid for the simple-attacker case.

### Added — provenance / supply-chain integrations

- **`keygen`** — generate ed25519 signing keypair (OpenSSF Model-Signing-style).
- **`sign`** / **`verify-signed`** — OpenSSF Model-Signing-style manifest with per-file SHA-256 + ed25519 signature. Produces `weightprobe-manifest.json` + `weightprobe-manifest.sig` alongside the model.
- **`aibom`** — OWASP CycloneDX 1.6 AI BOM emission with `vulnerabilities[]` records derived from weightprobe scan results.

### Changed

- **Optional-dependency groups.** `pip install weightprobe` now pulls `numpy` + `safetensors` (the spectral / diff-base / scan / live-probe modes need numerical primitives; together ~30 MB). Heavy backends are opt-in:
  - `pip install weightprobe[runtime]` adds MLX + mlx-lm for the activation-probe modes (Mac / Apple-Silicon).
  - `pip install weightprobe[signing]` adds `cryptography` for sign / verify-signed / aibom.
  - `pip install weightprobe[full]` pulls both.
- `__init__.py` exports are guarded with `try / except ImportError` for the optional-dep modules, so `pip install weightprobe` without any extras still imports cleanly (just with a smaller `__all__`).

### Test coverage

**128 / 128 passing** on the public repo (was 47 / 47 in v0.1.2). Includes a gated end-to-end test that loads a real Llama-3.2-3B model and runs the scan mode on it (`WEIGHTPROBE_TEST_MODELS_DIR=...` to enable). Adapter-aware scan validated on the actual Foundation-Sec-8B + Phase-7 adapter — `step_anomaly_layer` returns the exact insertion layer with the floor-clamped 100× ratio.

### One-line pitch

> v0.1 catches the architectural backdoor *if* it ships as a separate `adapter.safetensors`. v0.1.2 also catches it if it ships as `loader.py` beside untouched weights. v0.2 catches it even when the trojan is *merged into* the base safetensors — and when it loads adapter-aware, pinpoints the insertion layer.

## [0.1.2] - 2026-05-12

### Added — `inventory` mode

New CLI subcommand `weightprobe inventory <model_dir>` flags files that aren't on a model-only allow-list. Catches a class of supply-chain attacks that the `hash` / `verify` modes are blind to *by design*: malicious code that ships **beside** untouched weights in a HuggingFace repo, executed when the user clones and runs the setup script.

**Motivation — case study.** On 2026-05-12 HiddenLayer Research [disclosed](https://thehackernews.com/2026/05/fake-openai-privacy-filter-repo-hits-1.html) a HuggingFace repo `Open-OSS/privacy-filter` that typo-squatted OpenAI's legitimate Privacy Filter model card. The weights and `config.json` were identical to the real model; the attack lived in `loader.py` (a Base64-decoded PowerShell downloader) and `start.bat` (UAC elevation + Microsoft Defender exclusion + Rust infostealer payload from `api.eth-fastscan[.]org`). It hit **~244,000 downloads in 18 hours** and reached #1 trending on HuggingFace before being disabled.

A `weightprobe hash` of this repo would have returned the same digest as a hash of the real OpenAI repo — there was nothing wrong with the weights. `weightprobe inventory` flags the attack in one command: three `[HIGH]` findings on `loader.py`, `start.bat`, and the second-stage executable.

Severity classes:

- **HIGH** — executable / script extensions that should never appear in a pure-weights repo (`*.py`, `*.sh`, `*.bat`, `*.ps1`, `*.exe`, `*.dll`, `*.so`, `*.rs`, `*.c`, `*.cpp`, `*.jar`, `*.deb`, `*.dmg`, …). Exit code 1 at default severity floor.
- **MEDIUM** — build / dependency manifests (`requirements*.txt`, `setup.cfg`, `Pipfile`, `*.yaml`, …). Not malicious by themselves but indicate `trust_remote_code=True` may be needed.
- **LOW** — unrecognised but non-executable files. Manual review recommended.

Default severity floor is HIGH (CI-friendly: `weightprobe inventory <dir> && deploy` blocks on executables only). Use `--severity MEDIUM` or `--severity LOW` for tighter gates.

### Added

- `weightprobe inventory <model_dir>` CLI command with `--json` and `--severity {LOW,MEDIUM,HIGH}` flags.
- Python API: `weightprobe.scan_inventory(path) -> InventoryReport`, `weightprobe.classify_file(name) -> (severity, reason)`, plus `InventoryReport` and `FileFinding` dataclasses.
- Test coverage: 38 new tests covering legit-repo cleanliness, every HIGH-extension case, MEDIUM manifest detection, nested-subdir trojans, `.git`-dir exclusion, JSON serialisation, and the multi-payload (Unix + Windows + native) scenario from the actual fake-openai-privacy-filter attack.
- README and CLI help text now document the three-mode toolkit (`hash` + `verify` + `inventory`).

### Changed

- Package description updated from "structural attestation + baseline verification" to "structural attestation, baseline verification, and repo-inventory allow-list" to reflect the broader supply-chain scope.

## [0.1.1] - 2026-05-10

Docs-only patch. No code changes.

### Changed

- README now leads with `pip install weightprobe` (PyPI was published *after* the v0.1.0 README was finalised, so v0.1.0's PyPI page incorrectly states the package ships "from source only").
- Removed "Background" section from README (research-context note; the GitHub repo is the right place for it, not the package description).

## [0.1.0] - 2026-05-10

Initial public release. v0.1 ships the structural-attestation core of the toolkit; v0.2 (~late May 2026) will add the spectral, weight-diff, payload-shape, activation-delta scan, runtime live-probe, signing, and AI BOM modes.

### Added

- `weightprobe hash <model_dir>` - canonical structural fingerprint of a model directory (tensor inventory + filtered architecture config + adapter-file presence) and SHA-256 digest. Excludes runtime / training-time fields (`transformers_version`, `_name_or_path`, `_commit_hash`, `use_cache`, `torch_dtype`, `auto_map`, `attn_implementation`) so two clean fine-tunes of the same architecture hash identically.
- `weightprobe verify <model_dir> --baseline <digest_or_dir>` - comparison against a known-good baseline (hex digest *or* reference directory). When given a reference directory, emits a structured diff: tensors added / removed, config field deltas, adapter-file presence delta, per-safetensors-file inventory changes. Returns exit code `0` on match, `1` on mismatch - drop-in for CI / deployment pre-checks.
- Python API: `weightprobe.compute_hash(path)`, `weightprobe.structural_fingerprint(path)`, `weightprobe.verify(target, baseline)`.
- MIT licence; zero external runtime dependencies (stdlib only).
- Test suite covering hash idempotency, adapter-detection, config-filter behaviour, digest-vs-directory baseline paths.

### Threat model

v0.1 catches the structural signature of supply-chain attacks where the trojan ships as a separate file (e.g. `adapter.safetensors`) or where the architecture has changed between vendor publication and the deployed copy. It does **not** detect attacks that:

- merge the trojan weights into the base safetensors files without changing the tensor inventory (covered by v0.2 `diff-base` and `spectral`)
- live entirely in tokenizer / config / runtime hint fields (the structural hash filter intentionally excludes these so legitimate fine-tunes are not flagged)
- only manifest as runtime activation deltas with no static signature (covered by v0.2 `scan` and `live-probe`)

These limits are by design - v0.1 is the fast, dependency-free, stdlib-only base layer; v0.2's heavier modes layer on top.
