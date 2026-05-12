# Changelog

All notable changes to `weightprobe` are documented here. The project follows [Keep a Changelog](https://keepachangelog.com/) conventions and uses [Semantic Versioning](https://semver.org/).

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
