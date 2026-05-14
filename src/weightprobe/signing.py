"""weightprobe signing + AIBOM modes.

Two complementary integrations with public ML supply-chain standards:

1. **`weightprobe sign` / `weightprobe verify-signed`** — OpenSSF
   Model-Signing-style manifest + ed25519 signature.

   Produces a manifest file `weightprobe-manifest.json` next to the model
   that contains:
     - structural hash (from `weightprobe hash`)
     - per-file SHA-256 hashes
     - weightprobe version + timestamp
     - publisher metadata (free-form JSON)
   …and a signature file `weightprobe-manifest.sig` containing an ed25519
   signature over the canonical JSON of the manifest.

   This is a *minimum-viable* implementation aligned with OpenSSF Model
   Signing's data model (see https://github.com/sigstore/model-transparency)
   without the Sigstore/Fulcio dependency. Full Sigstore integration is
   out-of-scope for v0.2.

2. **`weightprobe aibom`** — OWASP CycloneDX-1.6 AI BOM emission.

   Produces a CycloneDX JSON document
   (https://cyclonedx.org/specification/overview/#ai-bom) describing the
   model and any detected adapters. Includes weightprobe scan results
   (spectral, payload-shape) as `aibom-extensions` so downstream tooling
   can ingest provenance + risk signals from the same artifact.

Together these two modes cover the two halves of the AI supply-chain
attestation story: cryptographic provenance (signing) + machine-readable
risk surface (AIBOM).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from weightprobe.hash import compute_hash


# ---- Manifest data structures ----------------------------------------------

@dataclass
class FileEntry:
    name: str
    size: int
    sha256: str

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "sha256": self.sha256}


@dataclass
class WeightprobeManifest:
    schema_version: str
    weightprobe_version: str
    model_dir: str
    structural_hash: str
    created_at: str
    publisher: dict
    files: list[FileEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "weightprobe_version": self.weightprobe_version,
            "model_dir": self.model_dir,
            "structural_hash": self.structural_hash,
            "created_at": self.created_at,
            "publisher": self.publisher,
            "files": [f.to_dict() for f in self.files],
        }

    def canonical_json(self) -> bytes:
        """Deterministic JSON serialization for signing — sorted keys, no
        whitespace, UTF-8."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---- File hashing ----------------------------------------------------------

def _sha256_of_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _enumerate_artifacts(model_dir: Path) -> list[FileEntry]:
    """List safetensors + config files under model_dir (excluding manifest+sig)."""
    out: list[FileEntry] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"weightprobe-manifest.json", "weightprobe-manifest.sig"}:
            continue
        if path.suffix not in {".safetensors", ".json", ".txt", ".tokenizer", ".model"}:
            continue
        rel = path.relative_to(model_dir).as_posix()
        out.append(FileEntry(
            name=rel,
            size=path.stat().st_size,
            sha256=_sha256_of_file(path),
        ))
    return out


# ---- Sign / verify ---------------------------------------------------------

_MANIFEST_NAME = "weightprobe-manifest.json"
_SIG_NAME = "weightprobe-manifest.sig"


def build_manifest(
    model_dir: Path | str,
    *,
    publisher: Optional[dict] = None,
) -> WeightprobeManifest:
    """Build a manifest for the given model_dir. No signing yet."""
    from weightprobe import __version__
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_dir is not a directory: {model_dir}")

    digest, _fp = compute_hash(model_dir)
    files = _enumerate_artifacts(model_dir)
    return WeightprobeManifest(
        schema_version="weightprobe-manifest-v1",
        weightprobe_version=__version__,
        model_dir=str(model_dir),
        structural_hash=digest,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        publisher=publisher or {},
        files=files,
    )


def sign_manifest(manifest: WeightprobeManifest, private_key_pem: bytes) -> bytes:
    """Sign the manifest's canonical JSON with an ed25519 private key.
    Returns the raw 64-byte signature. Caller writes it next to the manifest."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(sk, Ed25519PrivateKey):
        raise ValueError("private key must be ed25519")
    return sk.sign(manifest.canonical_json())


def verify_signature(
    manifest_bytes: bytes, signature: bytes, public_key_pem: bytes,
) -> bool:
    """Verify a signature over canonical manifest bytes.
    Returns True iff valid; False otherwise (no exception)."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    pk = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(pk, Ed25519PublicKey):
        raise ValueError("public key must be ed25519")
    try:
        pk.verify(signature, manifest_bytes)
        return True
    except InvalidSignature:
        return False


def write_signed_manifest(
    model_dir: Path | str,
    private_key_pem: bytes,
    *,
    publisher: Optional[dict] = None,
) -> tuple[Path, Path]:
    """Build manifest, sign it, write both files next to the model.
    Returns (manifest_path, signature_path)."""
    model_dir = Path(model_dir)
    manifest = build_manifest(model_dir, publisher=publisher)
    canonical = manifest.canonical_json()
    sig = sign_manifest(manifest, private_key_pem)

    mpath = model_dir / _MANIFEST_NAME
    spath = model_dir / _SIG_NAME
    mpath.write_bytes(canonical)
    spath.write_bytes(sig)
    return mpath, spath


def load_manifest(model_dir: Path | str) -> tuple[bytes, dict, bytes]:
    """Load the canonical manifest bytes + parsed dict + signature bytes."""
    model_dir = Path(model_dir)
    m_bytes = (model_dir / _MANIFEST_NAME).read_bytes()
    s_bytes = (model_dir / _SIG_NAME).read_bytes()
    parsed = json.loads(m_bytes)
    return m_bytes, parsed, s_bytes


def verify_model_signature(
    model_dir: Path | str, public_key_pem: bytes,
) -> tuple[bool, dict]:
    """Verify both:
      (a) the signature is valid for the manifest bytes on disk; AND
      (b) the per-file sha256 hashes in the manifest match the actual files.
    Returns (overall_ok, diagnostic_dict)."""
    model_dir = Path(model_dir)
    m_bytes, parsed, sig = load_manifest(model_dir)
    sig_ok = verify_signature(m_bytes, sig, public_key_pem)

    diag: dict = {"signature_valid": sig_ok, "file_mismatches": []}
    for entry in parsed.get("files", []):
        path = model_dir / entry["name"]
        if not path.is_file():
            diag["file_mismatches"].append(
                {"file": entry["name"], "issue": "missing"})
            continue
        actual = _sha256_of_file(path)
        if actual != entry["sha256"]:
            diag["file_mismatches"].append({
                "file": entry["name"], "issue": "hash_mismatch",
                "expected": entry["sha256"], "actual": actual,
            })
    overall = sig_ok and not diag["file_mismatches"]
    return overall, diag


# ---- AIBOM (OWASP CycloneDX 1.6) emission ----------------------------------

def emit_aibom(
    model_dir: Path | str,
    *,
    include_scan_results: bool = True,
) -> dict:
    """Emit an OWASP CycloneDX-1.6 AIBOM JSON for the given model directory.

    Includes:
      - bom-format / specVersion / serialNumber / metadata
      - components[].type=machine-learning-model with hashes + properties
      - vulnerabilities[] derived from spectral + payload-shape suspicion ≥ 0.5
        (each high-suspicion finding becomes a CycloneDX vulnerability entry)

    Args:
        include_scan_results: if True, runs spectral + payload-shape and
            attaches results as CycloneDX `properties` + `vulnerabilities`.
            Set False for fast metadata-only emission.
    """
    from weightprobe import __version__
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_dir is not a directory: {model_dir}")

    digest, fp = compute_hash(model_dir)
    files = _enumerate_artifacts(model_dir)
    serial = "urn:uuid:" + str(uuid.uuid4())

    component: dict = {
        "type": "machine-learning-model",
        "bom-ref": "model:" + digest[:16],
        "name": model_dir.name,
        "version": "unknown",
        "hashes": [
            {"alg": "SHA-256", "content": f["sha256"] if isinstance(f, dict) else f.sha256}
            for f in [{"sha256": entry.sha256, "name": entry.name} for entry in files][:5]
        ],
        "properties": [
            {"name": "weightprobe.structural_hash", "value": digest},
            {"name": "weightprobe.version", "value": __version__},
            {"name": "weightprobe.total_tensors", "value": str(fp.get("total_tensors", 0))},
            {"name": "weightprobe.has_adapter", "value": str(bool(fp.get("has_adapter")))},
            {"name": "weightprobe.n_safetensors_files", "value": str(len(fp.get("safetensors", [])))},
        ],
    }

    vulnerabilities: list[dict] = []
    if include_scan_results:
        try:
            from weightprobe.spectral import compute_spectral_fingerprint
            sf = compute_spectral_fingerprint(model_dir)
            component["properties"].extend([
                {"name": "weightprobe.spectral_aggregate",
                 "value": f"{sf.aggregate_suspicion:.4f}"},
                {"name": "weightprobe.spectral_high_count",
                 "value": str(sf.n_tensors_high_suspicion)},
            ])
            for t in sf.per_tensor[:50]:
                if t.suspicion_score >= 0.7:
                    vulnerabilities.append(_aibom_vuln(
                        finding="spectral",
                        tensor=t.name,
                        score=t.suspicion_score,
                        severity="high" if t.suspicion_score >= 0.9 else "medium",
                        description=(
                            f"Spectral fingerprint flagged tensor '{t.name}' "
                            f"shape={t.shape} as adapter-shape "
                            f"(suspicion {t.suspicion_score:.3f})."
                        ),
                    ))
        except Exception as e:
            component["properties"].append(
                {"name": "weightprobe.spectral_error", "value": f"{type(e).__name__}"}
            )
        try:
            from weightprobe.payload_shape import compute_payload_shape_fingerprint
            ps = compute_payload_shape_fingerprint(model_dir)
            component["properties"].extend([
                {"name": "weightprobe.payload_shape_aggregate",
                 "value": f"{ps.aggregate_suspicion:.4f}"},
                {"name": "weightprobe.payload_shape_flagged",
                 "value": str(ps.n_tensors_flagged)},
            ])
            for t in ps.per_tensor[:50]:
                if t.suspicion_score >= 0.7:
                    vulnerabilities.append(_aibom_vuln(
                        finding="payload-shape",
                        tensor=t.name,
                        score=t.suspicion_score,
                        severity="high" if t.suspicion_score >= 0.9 else "medium",
                        description=(
                            f"Payload-shape classifier flagged '{t.name}' "
                            f"as {t.payload_class or t.position_class or 'unknown'} "
                            f"(suspicion {t.suspicion_score:.3f})."
                        ),
                    ))
        except Exception as e:
            component["properties"].append(
                {"name": "weightprobe.payload_shape_error", "value": f"{type(e).__name__}"}
            )

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": serial,
        "version": 1,
        "metadata": {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "tools": [
                {
                    "vendor": "Ruflo",
                    "name": "weightprobe",
                    "version": __version__,
                }
            ],
        },
        "components": [component],
        "vulnerabilities": vulnerabilities,
    }


def _aibom_vuln(*, finding: str, tensor: str, score: float, severity: str,
                description: str) -> dict:
    """One CycloneDX vulnerability entry for a high-suspicion finding."""
    return {
        "bom-ref": f"weightprobe-vuln:{finding}:{tensor}",
        "id": f"WEIGHTPROBE-{finding.upper()}",
        "source": {"name": "weightprobe"},
        "ratings": [{
            "source": {"name": "weightprobe"},
            "score": score,
            "severity": severity,
            "method": "other",
            "vector": f"finding={finding};tensor={tensor};score={score:.4f}",
        }],
        "description": description,
    }


# ---- Convenience: keypair generation --------------------------------------

def generate_keypair() -> tuple[bytes, bytes]:
    """Generate a fresh ed25519 keypair, return (private_pem, public_pem)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.generate()
    private_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = sk.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem
