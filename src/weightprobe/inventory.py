"""Inventory mode - flag unexpected files in a model repo.

Motivation
----------
The structural-hash mode (v0.1) and the planned spectral / diff-base / scan
modes (v0.2) all answer questions about the model *weights*. They cannot see
attacks that ship malicious code *beside* the weights as part of the same
HuggingFace repo - for example the 2026-05-12 "fake-openai-privacy-filter"
trojan, which hit ~244k downloads in 18h by embedding a Rust infostealer in
`loader.py` + `start.bat` while leaving the safetensors untouched
(HiddenLayer Research disclosure).

`inventory` scans the directory and flags anything that isn't on a
model-only allow-list. The signal is binary and high-precision: legitimate
HuggingFace model repos do not ship `*.py` / `*.bat` / `*.sh` / `*.exe`.
When a repo does, it is either (a) a custom-code model that needs human
review of `trust_remote_code=True` implications, or (b) malicious.

No baseline is needed. The default policy catches the loader.py class
trivially.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


# Patterns that a legitimate model repo is *expected* to ship. Anything matching
# one of these globs is silent. Anything that doesn't match goes through the
# severity classifier below.
ALLOWED_PATTERNS: tuple[str, ...] = (
    # Weights
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
    "tf_model.h5",
    "flax_model.msgpack",
    "model.onnx",
    "model.onnx_data",
    # Architecture / generation config
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    # Tokenizer
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "sentencepiece.bpe.model",
    "chat_template.json",
    "chat_template.jinja",
    # Adapter config (LoRA/PEFT-style)
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    # Docs / metadata
    "README.md",
    "README*.md",
    "LICENSE",
    "LICENSE*",
    "MODEL_CARD.md",
    "*.md",
    "*.txt",
    "USE_POLICY*",
    "NOTICE*",
    # Common model-card image / chart assets
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    # Quantization / training-time configs that some HF repos ship
    "quantization_config.json",
    "configuration_*.json",
    "modeling_*.json",
    # Provenance (OpenSSF Model Signing-style, AI BOM)
    "model.sig",
    "*.sig",
    "*.pem",
    "*.crt",
    "aibom.cdx.json",
    "*.cdx.json",
    "*.spdx.json",
)


# File extensions that should NEVER appear in a pure-weights repo. If we see one,
# it's either (a) a custom-code model that needs trust_remote_code review, or
# (b) malicious. Either way, HIGH severity flag.
EXECUTABLE_EXTENSIONS: frozenset[str] = frozenset({
    # Scripting
    ".py", ".pyw", ".pyc", ".pyo", ".pyd",
    ".sh", ".bash", ".zsh", ".fish",
    ".bat", ".cmd", ".ps1", ".psm1",
    ".rb", ".pl", ".php", ".lua", ".js", ".mjs", ".cjs", ".ts",
    ".vbs", ".wsf", ".jse",
    # Native code
    ".exe", ".dll", ".com", ".scr", ".msi",
    ".so", ".dylib", ".a", ".bundle",
    ".rs", ".c", ".cpp", ".cc", ".h", ".hpp",
    # Packed / installer payloads
    ".jar", ".war", ".apk", ".ipa",
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar",
    ".deb", ".rpm", ".pkg", ".dmg",
})


# Patterns that are commonly seen but lower-severity. We still flag them but
# class them MEDIUM so a CI gate can choose to permit them. (We separate this
# from EXECUTABLE because seeing a `requirements.txt` next to weights isn't
# malicious by itself - it just shouldn't be on the load path.)
MEDIUM_PATTERNS: tuple[str, ...] = (
    "requirements*.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "*.yaml",
    "*.yml",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
)


@dataclass
class FileFinding:
    """One unexpected file flagged by the inventory scan."""
    relative_path: str
    size_bytes: int
    severity: str  # "HIGH" | "MEDIUM" | "LOW"
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InventoryReport:
    """Outcome of an inventory scan."""
    model_dir: str
    n_files_total: int
    n_files_allowed: int
    n_files_flagged: int
    findings: list[FileFinding] = field(default_factory=list)
    allowed_files: list[str] = field(default_factory=list)
    has_executable: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [f.to_dict() if isinstance(f, FileFinding) else f for f in self.findings]
        return d

    @property
    def is_clean(self) -> bool:
        return self.n_files_flagged == 0


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def classify_file(name: str) -> tuple[str | None, str]:
    """Return (severity, reason) for one file name. severity is None if allowed.

    Order of checks matters — the most specific / dangerous patterns are
    checked first so they win over broader allow-list globs:

      1. Executable extensions       → HIGH  (e.g. *.py, *.sh, *.exe)
      2. Build / manifest patterns   → MEDIUM (e.g. requirements*.txt, Pipfile)
      3. Allow-list patterns         → None  (e.g. *.safetensors, *.md, *.txt)
      4. Anything else               → LOW

    Without this order, a generic allow-list entry like `*.txt` would silently
    pass a `requirements.txt` that should have been flagged MEDIUM.
    """
    # 1. Executable extensions — HIGH (specific, never legitimate in a
    # weights-only repo).
    ext = Path(name).suffix.lower()
    if ext in EXECUTABLE_EXTENSIONS:
        return "HIGH", f"executable/script extension {ext!r} — should not ship in a pure-weights repo"

    # 2. Build / dependency manifests — MEDIUM (specific names like
    # requirements*.txt, Pipfile, etc.; must beat the broader *.txt allow).
    if _matches_any(name, MEDIUM_PATTERNS):
        return "MEDIUM", "build/dependency manifest — review for trust_remote_code implications"

    # 3. Allow-list — silent pass.
    if _matches_any(name, ALLOWED_PATTERNS):
        return None, "allowed"

    # 4. Everything else — LOW (unexpected but not obviously dangerous).
    return "LOW", "unrecognised file — manual review recommended"


def scan_inventory(model_dir: Path) -> InventoryReport:
    """Walk a model directory and flag every file that isn't on the model-only
    allow-list.

    Walks recursively because some HF repos use a `safetensors/` subdir or
    similar. Symlinks are followed but the target's real path is reported.
    """
    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")

    rep = InventoryReport(
        model_dir=str(root),
        n_files_total=0,
        n_files_allowed=0,
        n_files_flagged=0,
    )

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Don't scan hidden dirs (.git, .cache) — they're not part of the
        # served repo.
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue

        rep.n_files_total += 1
        name = path.name  # classify by basename, not full path
        severity, reason = classify_file(name)
        if severity is None:
            rep.n_files_allowed += 1
            rep.allowed_files.append(str(rel))
        else:
            rep.n_files_flagged += 1
            rep.findings.append(FileFinding(
                relative_path=str(rel),
                size_bytes=path.stat().st_size,
                severity=severity,
                reason=reason,
            ))
            if severity == "HIGH":
                rep.has_executable = True

    # Stable ordering for deterministic output.
    rep.findings.sort(key=lambda f: (f.severity != "HIGH", f.severity != "MEDIUM", f.relative_path))
    rep.allowed_files.sort()
    return rep
