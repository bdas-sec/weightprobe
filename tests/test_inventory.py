"""Tests for the v0.1.2 inventory mode.

Each test sets up a synthetic model dir on tmp and asserts what scan_inventory
flags vs allows. The mock attack scenarios mirror the 2026-05-12
fake-openai-privacy-filter HuggingFace attack (244k downloads in 18h) — the
threat model that motivated this mode.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from weightprobe.inventory import (
    ALLOWED_PATTERNS,
    EXECUTABLE_EXTENSIONS,
    FileFinding,
    InventoryReport,
    classify_file,
    scan_inventory,
)


# ---- Synthetic repo fixtures -------------------------------------------------

def _make_legit_repo(d: Path) -> None:
    """A minimal but realistic HF model repo: weights + config + tokenizer + README."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(b"\x00" * 16)
    (d / "config.json").write_text('{"model_type": "llama"}')
    (d / "tokenizer.json").write_text('{"version": "1.0"}')
    (d / "tokenizer_config.json").write_text("{}")
    (d / "vocab.json").write_text("{}")
    (d / "merges.txt").write_text("")
    (d / "README.md").write_text("# Model")
    (d / "LICENSE").write_text("MIT")


def _add_trojan(d: Path, name: str = "loader.py", content: bytes = b"# evil\n") -> None:
    """Mirror the fake-openai-privacy-filter attack: drop a script beside the weights."""
    (d / name).write_bytes(content)


# ---- Unit tests on the classifier ------------------------------------------

@pytest.mark.parametrize("name", [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",
    "LICENSE",
    "adapter_config.json",
    "adapter_model.safetensors",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors.index.json",
    "pytorch_model-00001-of-00002.bin",
    "model.sig",
    "aibom.cdx.json",
])
def test_classify_allows_legit_files(name: str) -> None:
    sev, _reason = classify_file(name)
    assert sev is None, f"expected {name} to be allowed, got {sev}"


@pytest.mark.parametrize("name,expected", [
    # The fake-openai-privacy-filter file names
    ("loader.py", "HIGH"),
    ("start.bat", "HIGH"),
    # Other classical supply-chain payloads
    ("install.sh", "HIGH"),
    ("setup.exe", "HIGH"),
    ("payload.dll", "HIGH"),
    ("infostealer.rs", "HIGH"),
    ("backdoor.so", "HIGH"),
    ("trojan.dylib", "HIGH"),
    ("encoder.pyc", "HIGH"),
])
def test_classify_flags_executables_as_high(name: str, expected: str) -> None:
    sev, reason = classify_file(name)
    assert sev == expected, f"{name}: expected {expected}, got {sev}"
    assert "executable" in reason.lower() or "script" in reason.lower()


@pytest.mark.parametrize("name", ["requirements.txt", "setup.py", "pyproject.toml"])
def test_classify_flags_build_manifests_as_medium_or_high(name: str) -> None:
    # setup.py is *.py so it should land HIGH (script extension wins).
    # requirements.txt and pyproject.toml are MEDIUM.
    sev, _reason = classify_file(name)
    assert sev in {"MEDIUM", "HIGH"}, f"{name}: got {sev}"


# ---- Integration tests on scan_inventory -----------------------------------

def test_legit_repo_is_clean(tmp_path: Path) -> None:
    _make_legit_repo(tmp_path)
    rep = scan_inventory(tmp_path)
    assert rep.is_clean
    assert rep.n_files_flagged == 0
    assert not rep.has_executable
    # Every file was accounted for.
    assert rep.n_files_allowed == rep.n_files_total


def test_trojan_repo_flags_loader_py_high(tmp_path: Path) -> None:
    """The headline scenario: fake-openai-privacy-filter ships loader.py."""
    _make_legit_repo(tmp_path)
    _add_trojan(tmp_path, "loader.py")
    rep = scan_inventory(tmp_path)

    assert not rep.is_clean
    assert rep.n_files_flagged == 1
    assert rep.has_executable
    finding = rep.findings[0]
    assert finding.severity == "HIGH"
    assert finding.relative_path == "loader.py"
    assert ".py" in finding.reason


def test_trojan_repo_flags_windows_bat_high(tmp_path: Path) -> None:
    _make_legit_repo(tmp_path)
    _add_trojan(tmp_path, "start.bat", b"@echo off\n")
    rep = scan_inventory(tmp_path)

    assert rep.has_executable
    f = rep.findings[0]
    assert f.severity == "HIGH"
    assert f.relative_path == "start.bat"


def test_trojan_repo_flags_native_binary_high(tmp_path: Path) -> None:
    _make_legit_repo(tmp_path)
    _add_trojan(tmp_path, "stealer.exe", b"MZ\x00\x00")
    rep = scan_inventory(tmp_path)

    f = rep.findings[0]
    assert f.severity == "HIGH"
    assert f.size_bytes == 4


def test_high_severity_default_does_not_flag_yaml(tmp_path: Path) -> None:
    """A `*.yaml` next to weights is MEDIUM; default severity floor HIGH won't fail."""
    _make_legit_repo(tmp_path)
    (tmp_path / "training.yaml").write_text("foo: bar")
    rep = scan_inventory(tmp_path)
    assert rep.n_files_flagged == 1  # internally flagged
    high_findings = [f for f in rep.findings if f.severity == "HIGH"]
    assert high_findings == []


def test_nested_loader_in_subdir_still_caught(tmp_path: Path) -> None:
    """Some HF repos use subdirs (e.g. `safetensors/`). Trojan in a subdir must still trip."""
    _make_legit_repo(tmp_path)
    sub = tmp_path / "scripts"
    sub.mkdir()
    (sub / "evil.py").write_text("import os; os.system('rm -rf /')")
    rep = scan_inventory(tmp_path)
    assert rep.has_executable
    f = rep.findings[0]
    assert f.severity == "HIGH"
    assert f.relative_path == "scripts/evil.py"


def test_hidden_dotgit_dir_ignored(tmp_path: Path) -> None:
    """`.git/`, `.cache/` etc. aren't part of the served repo and shouldn't be scanned."""
    _make_legit_repo(tmp_path)
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("[core]")
    (git / "hooks").mkdir()
    (git / "hooks" / "post-commit").write_text("#!/bin/sh\necho")
    rep = scan_inventory(tmp_path)
    assert rep.is_clean  # .git ignored despite containing a "script"


def test_report_serializes_to_json(tmp_path: Path) -> None:
    _make_legit_repo(tmp_path)
    _add_trojan(tmp_path, "loader.py")
    rep = scan_inventory(tmp_path)
    d = rep.to_dict()
    # Round-trip through JSON to confirm no non-serializable types.
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["n_files_flagged"] == 1
    assert parsed["findings"][0]["severity"] == "HIGH"
    assert parsed["has_executable"] is True


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_inventory(tmp_path / "does-not-exist")


def test_multiple_payloads_all_flagged(tmp_path: Path) -> None:
    """The real attack shipped both Unix and Windows payloads + a Rust binary."""
    _make_legit_repo(tmp_path)
    _add_trojan(tmp_path, "loader.py")
    _add_trojan(tmp_path, "start.bat")
    _add_trojan(tmp_path, "stealer.exe", b"MZ")
    rep = scan_inventory(tmp_path)
    assert rep.n_files_flagged == 3
    severities = sorted(f.severity for f in rep.findings)
    assert severities == ["HIGH", "HIGH", "HIGH"]
    # HIGH severity findings sort first (deterministic output order).
    assert all(f.severity == "HIGH" for f in rep.findings)
