"""weightprobe CLI - argparse entrypoint for v0.1 modes (hash + verify).

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
from weightprobe.verify import verify


def _cmd_hash(args: argparse.Namespace) -> int:
    """`weightprobe hash <model_dir> [--print-fingerprint]`"""
    digest, fp = compute_hash(Path(args.model_dir))
    if args.print_fingerprint:
        print(json.dumps({"digest": digest, "fingerprint": fp}, indent=2))
    else:
        print(digest)
    return 0


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
            "Defensive tooling for architectural backdoors in transformer LLMs. "
            "v0.1 ships structural fingerprint (hash) and baseline-comparison (verify) "
            "modes. v0.2 will add spectral / diff / scan / live-probe / signing modes."
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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
