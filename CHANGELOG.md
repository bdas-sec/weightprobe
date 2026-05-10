# Changelog

All notable changes to `weightprobe` are documented here. The project follows [Keep a Changelog](https://keepachangelog.com/) conventions and uses [Semantic Versioning](https://semver.org/).

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
