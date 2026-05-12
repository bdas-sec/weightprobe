"""weightprobe CLI - argparse entrypoint.

Modes (v0.1.2):
  hash       — structural fingerprint of model weights + config
  verify     — compare a target dir against a known-good baseline
  inventory  — flag files that aren't on the model-only allow-list
               (catches loader.py-style supply-chain trojans where the
                malice ships beside the weights instead of inside them)

v0.2 will add: spectral, diff-base, payload-shape, scan, live-probe,
rev-trigger, plus signing + AIBOM emission.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weightprobe import __version__
from weightprobe.hash import compute_hash
from weightprobe.inventory import scan_inventory
from weightprobe.verify import verify


def _cmd_hash(args: argparse.Namespace) -> int:
    """`weightprobe hash <model_dir> [--print-fingerprint]`"""
    digest, fp = compute_hash(Path(args.model_dir))
    if args.print_fingerprint:
        print(json.dumps({"digest": digest, "fingerprint": fp}, indent=2))
    else:
        print(digest)
    return 0


def _cmd_inventory(args: argparse.Namespace) -> int:
    """`weightprobe inventory <model_dir> [--json] [--severity HIGH|MEDIUM|LOW]`"""
    rep = scan_inventory(Path(args.model_dir))

    # Apply severity floor (only return non-zero / show findings at or above).
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    floor = severity_rank[args.severity]
    flagged = [f for f in rep.findings if severity_rank[f.severity] >= floor]
    fail = bool(flagged)

    if args.json:
        d = rep.to_dict()
        d["findings"] = [f.to_dict() for f in flagged]
        d["severity_floor"] = args.severity
        print(json.dumps(d, indent=2))
    else:
        if rep.is_clean:
            print(f"[CLEAN] {args.model_dir}")
            print(f"  {rep.n_files_allowed}/{rep.n_files_total} files match model-only allow-list")
        else:
            verdict = "FLAGGED" if fail else "CLEAN-at-floor"
            print(f"[{verdict}] {args.model_dir}")
            print(f"  {rep.n_files_allowed}/{rep.n_files_total} files allowed; "
                  f"{rep.n_files_flagged} flagged ({sum(1 for f in rep.findings if f.severity=='HIGH')} HIGH / "
                  f"{sum(1 for f in rep.findings if f.severity=='MEDIUM')} MEDIUM / "
                  f"{sum(1 for f in rep.findings if f.severity=='LOW')} LOW)")
            for f in flagged:
                print(f"  [{f.severity}] {f.relative_path}  ({f.size_bytes} bytes) — {f.reason}")
    return 1 if fail else 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """`weightprobe verify <model_dir> --baseline <digest_or_dir>`"""
    result = verify(Path(args.model_dir), args.baseline)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        verdict = "MATCH" if result.match else "MISMATCH"
        print(f"[{verdict}] {args.model_dir}")
        print(f"  target hash:   {result.target_hash}")
        print(f"  baseline hash: {result.baseline_hash}")
        print(f"  baseline:      {result.baseline_source}")
        if not result.match and result.diff:
            print("  diff:")
            print(json.dumps(result.diff, indent=4))
    return 0 if result.match else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="weightprobe",
        description=(
            "Defensive tooling for architectural backdoors and supply-chain trojans "
            "in transformer LLM repos. v0.1.2 ships three modes: hash + verify "
            "(structural attestation against architectural-backdoor adapters), and "
            "inventory (allow-list scan against loader.py-style trojans). "
            "v0.2 will add spectral / diff / scan / live-probe / signing modes."
        ),
    )
    p.add_argument("--version", action="version", version=f"weightprobe {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # hash
    hp = sub.add_parser(
        "hash", help="Compute structural hash of a model directory.")
    hp.add_argument("model_dir", help="Path to a model directory.")
    hp.add_argument(
        "--print-fingerprint", action="store_true",
        help="Print the full structural fingerprint JSON alongside the digest.")
    hp.set_defaults(fn=_cmd_hash)

    # verify
    vp = sub.add_parser(
        "verify", help="Verify a model directory against a known-good baseline.")
    vp.add_argument("model_dir", help="Path to the target model directory.")
    vp.add_argument(
        "--baseline", required=True,
        help="Either a 64-char hex digest, or a path to a reference model directory.")
    vp.add_argument(
        "--json", action="store_true",
        help="Output the full result as JSON.")
    vp.set_defaults(fn=_cmd_verify)

    # inventory — catch trojan files that ship beside the weights
    ip = sub.add_parser(
        "inventory",
        help=("Flag files that aren't on the model-only allow-list "
              "(catches loader.py-style supply-chain trojans)."))
    ip.add_argument("model_dir", help="Path to the target model directory.")
    ip.add_argument(
        "--json", action="store_true",
        help="Output the full inventory report as JSON.")
    ip.add_argument(
        "--severity", choices=["LOW", "MEDIUM", "HIGH"], default="HIGH",
        help=("Minimum severity to flag (default HIGH = executables only). "
              "Exit code 1 if any flag at or above this floor."))
    ip.set_defaults(fn=_cmd_inventory)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
