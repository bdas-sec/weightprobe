"""weightprobe CLI — `weightprobe {hash,verify,inventory,spectral,diff-base,payload-shape,scan,live-probe,rev-trigger,keygen,sign,verify-signed,aibom} <dir>`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weightprobe import __version__
from weightprobe.hash import compute_hash, structural_fingerprint
from weightprobe.verify import verify
from weightprobe.inventory import scan_inventory
from weightprobe.spectral import compute_spectral_fingerprint
from weightprobe.diff_base import compute_diff_base
from weightprobe.payload_shape import compute_payload_shape_fingerprint
from weightprobe.scan import compute_scan_fingerprint, _DEFAULT_PROBE_PROMPTS as _SCAN_PROMPTS
from weightprobe.live_probe import (
    LiveProbeBaseline, build_baseline, score_prompts_batch,
)
from weightprobe.rev_trigger import reverse_trigger
from weightprobe.signing import (
    write_signed_manifest, verify_model_signature, emit_aibom,
    generate_keypair, build_manifest,
)


def _cmd_hash(args: argparse.Namespace) -> int:
    digest, fp = compute_hash(args.model_dir)
    if args.json:
        out = {"hash": digest, "fingerprint": fp} if args.full else {"hash": digest}
        print(json.dumps(out, indent=2 if args.pretty else None))
    else:
        print(f"weightprobe-hash-v1 {digest}  {args.model_dir}")
        if args.full:
            print(f"  config-keys: {len(fp.get('config') or {})}")
            print(f"  safetensors files: {len(fp['safetensors'])}")
            print(f"  total tensors: {fp['total_tensors']}")
            print(f"  has_adapter: {fp['has_adapter']}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify(args.model_dir, args.baseline)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2 if args.pretty else None))
    else:
        verdict = "MATCH ✓" if result.match else "MISMATCH ✗"
        print(f"weightprobe-verify-v1 {verdict}")
        print(f"  target:   {result.target_hash}  ({result.target_dir})")
        print(f"  baseline: {result.baseline_hash}  ({result.baseline_source})")
        if not result.match and result.diff:
            print("  diff:")
            for k, v in result.diff.items():
                if k == "config_changed":
                    print(f"    {k}: {len(v)} field(s)")
                    for field, delta in list(v.items())[:5]:
                        print(f"      {field}: {delta['baseline']!r} -> {delta['target']!r}")
                elif k == "safetensors_inventory_changed":
                    print(f"    {k}:")
                    for fname, fdiff in v.items():
                        print(f"      {fname}: +{fdiff['added_count']} -{fdiff['removed_count']} tensors")
                elif k == "adapter_presence_changed":
                    print(f"    {k}: target={v['target']}, baseline={v['baseline']}")
                elif k == "total_tensors_changed":
                    print(f"    {k}: baseline={v['baseline']} -> target={v['target']}")
                else:
                    print(f"    {k}: {v}")
    return 0 if result.match else 1


def _cmd_inventory(args: argparse.Namespace) -> int:
    """Flag files in a model repo that aren't on the model-only allow-list.

    Catches loader.py-style supply-chain trojans where the malice ships
    *beside* untouched weights (the threat model is orthogonal to the
    weight-analysis modes). See `weightprobe.inventory` for the case study
    on the 2026-05-12 fake-openai-privacy-filter HuggingFace attack.
    """
    rep = scan_inventory(args.model_dir)
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    floor = severity_rank[args.severity]
    flagged = [f for f in rep.findings if severity_rank[f.severity] >= floor]
    fail = bool(flagged)

    if args.json:
        d = rep.to_dict()
        d["findings"] = [f.to_dict() for f in flagged]
        d["severity_floor"] = args.severity
        print(json.dumps(d, indent=2 if args.pretty else None))
    else:
        verdict = "CLEAN ✓" if rep.is_clean else ("FLAGGED ✗" if fail else "CLEAN-at-floor")
        print(f"weightprobe-inventory-v1 {verdict}  ({args.model_dir})")
        print(f"  {rep.n_files_allowed}/{rep.n_files_total} files allowed; "
              f"{rep.n_files_flagged} flagged "
              f"({sum(1 for f in rep.findings if f.severity == 'HIGH')} HIGH / "
              f"{sum(1 for f in rep.findings if f.severity == 'MEDIUM')} MEDIUM / "
              f"{sum(1 for f in rep.findings if f.severity == 'LOW')} LOW)")
        for f in flagged:
            print(f"  [{f.severity}] {f.relative_path}  ({f.size_bytes} bytes) — {f.reason}")
    return 1 if fail else 0


def _cmd_spectral(args: argparse.Namespace) -> int:
    fp = compute_spectral_fingerprint(args.model_dir)
    if args.json:
        out = fp.to_dict()
        if not args.full:
            # Strip the per_tensor list to keep output tractable; just keep
            # the high-suspicion tensors.
            out["per_tensor"] = [
                t for t in out["per_tensor"] if t["suspicion_score"] >= 0.5
            ]
        print(json.dumps(out, indent=2 if args.pretty else None))
    else:
        print(f"weightprobe-spectral-v1 {args.model_dir}")
        print(f"  tensors analyzed:        {fp.n_tensors_analyzed}")
        print(f"  tensors skipped:         {fp.n_tensors_skipped}")
        print(f"  aggregate suspicion:     {fp.aggregate_suspicion:.3f}")
        print(f"  high-suspicion tensors:  {fp.n_tensors_high_suspicion}")
        if fp.n_tensors_high_suspicion > 0:
            print(f"\n  Top suspicious tensors (suspicion ≥ 0.5):")
            sus = sorted(fp.per_tensor, key=lambda t: -t.suspicion_score)
            for t in sus[:10]:
                if t.suspicion_score < 0.5:
                    break
                print(f"    {t.suspicion_score:.3f}  {t.name:50s} "
                      f"shape={tuple(t.shape)} eff_rank95={t.effective_rank_95}/"
                      f"{t.n_singular_values} top1={t.top1_energy_fraction:.3f}")
    return 0 if fp.aggregate_suspicion < 0.7 else 1


def _cmd_diff_base(args: argparse.Namespace) -> int:
    result = compute_diff_base(
        args.model_dir, args.baseline, cosine_threshold=args.cosine_threshold,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2 if args.pretty else None))
    else:
        verdict = "MATCH ✓" if result.n_tensors_modified == 0 and \
                                  result.n_tensors_only_in_target == 0 and \
                                  result.n_tensors_only_in_baseline == 0 else "DIFFERENCES FOUND ✗"
        print(f"weightprobe-diff-base-v1 {verdict}")
        print(f"  target:   {result.target_dir}")
        print(f"  baseline: {result.baseline_dir}")
        print(f"  tensors compared:        {result.n_tensors_compared}")
        print(f"  tensors only in target:  {result.n_tensors_only_in_target}")
        print(f"  tensors only in basline: {result.n_tensors_only_in_baseline}")
        print(f"  shape mismatches:        {result.n_tensors_shape_mismatch}")
        print(f"  modified (cos > {result.cosine_threshold}): {result.n_tensors_modified}")
        if result.n_tensors_only_in_target > 0:
            print(f"\n  Added tensors (first 10):")
            for n in result.tensors_only_in_target[:10]:
                print(f"    + {n}")
        if result.n_tensors_modified > 0:
            print(f"\n  Modified tensors by cosine distance (top 10):")
            modified = [t for t in result.per_tensor if t.cosine_distance > result.cosine_threshold]
            modified.sort(key=lambda t: -t.cosine_distance)
            for t in modified[:10]:
                print(f"    cos_dist={t.cosine_distance:.6e}  rel_l2={t.relative_l2:.4e}  {t.name}")
    n_diffs = (result.n_tensors_only_in_target + result.n_tensors_only_in_baseline +
               result.n_tensors_shape_mismatch + result.n_tensors_modified)
    return 0 if n_diffs == 0 else 1


def _cmd_payload_shape(args: argparse.Namespace) -> int:
    fp = compute_payload_shape_fingerprint(args.model_dir)
    if args.json:
        out = fp.to_dict()
        if not args.full:
            out["per_tensor"] = [
                t for t in out["per_tensor"] if t["suspicion_score"] >= 0.5
            ]
        print(json.dumps(out, indent=2 if args.pretty else None))
    else:
        print(f"weightprobe-payload-shape-v1 {args.model_dir}")
        print(f"  tensors total:           {fp.n_tensors_total}")
        print(f"  tensors flagged (≥0.5):  {fp.n_tensors_flagged}")
        print(f"  high-suspicion (≥0.7):   {fp.n_tensors_high_suspicion}")
        print(f"  aggregate suspicion:     {fp.aggregate_suspicion:.3f}")
        if fp.payload_classes:
            print(f"  payload classes detected:")
            for cls, n in sorted(fp.payload_classes.items(), key=lambda kv: -kv[1]):
                print(f"    {cls:30s}  {n}")
        if fp.layout_violations:
            print(f"\n  Layout violations (first 10 of {len(fp.layout_violations)}):")
            for n in fp.layout_violations[:10]:
                print(f"    ! {n}")
        if fp.n_tensors_flagged > 0:
            print(f"\n  Top flagged tensors:")
            sus = sorted(fp.per_tensor, key=lambda t: -t.suspicion_score)
            for t in sus[:10]:
                if t.suspicion_score < 0.5:
                    break
                tag = t.payload_class or t.shape_class or t.position_class or "?"
                print(f"    {t.suspicion_score:.3f}  {t.name:60s} "
                      f"shape={tuple(t.shape)} class={tag}")
    return 0 if fp.aggregate_suspicion < 0.7 else 1


def _cmd_scan(args: argparse.Namespace) -> int:
    fp = compute_scan_fingerprint(args.model_dir, args.baseline)
    if args.json:
        out = fp.to_dict()
        if not args.full:
            out["per_layer"] = [
                r for r in out["per_layer"] if r["suspicion_score"] >= 0.5
            ]
        print(json.dumps(out, indent=2 if args.pretty else None))
    else:
        print(f"weightprobe-scan-v1 {args.model_dir}")
        print(f"  baseline:                {args.baseline}")
        print(f"  probe prompts:           {fp.n_prompts}")
        print(f"  layers analyzed:         {fp.n_layers}")
        print(f"  aggregate suspicion:     {fp.aggregate_suspicion:.3f}")
        print(f"  high-suspicion layers:   {fp.n_layers_high_suspicion}")
        if fp.n_layers_high_suspicion > 0:
            print(f"\n  Top suspicious (layer, position) pairs:")
            sus = sorted(fp.per_layer, key=lambda r: -r.suspicion_score)
            for r in sus[:10]:
                if r.suspicion_score < 0.5:
                    break
                print(f"    {r.suspicion_score:.3f}  L={r.layer:>2d}  {r.position:9s} "
                      f"cos_sep={r.cosine_separation:.4f}  L2={r.mean_l2_distance:.3f}  "
                      f"KL={r.kl_divergence:.2f}")
    return 0 if fp.aggregate_suspicion < 0.7 else 1


def _cmd_live_probe_baseline(args: argparse.Namespace) -> int:
    bl = build_baseline(args.clean_model_dir, _SCAN_PROMPTS)
    bl.save(args.out)
    print(f"weightprobe-live-probe-baseline-v1 {args.clean_model_dir}")
    print(f"  probe prompts:  {bl.n_probe_prompts}")
    print(f"  layers:         {bl.n_layers}")
    print(f"  hidden_size:    {bl.hidden_size}")
    print(f"  saved baseline: {args.out}")
    return 0


def _cmd_live_probe_score(args: argparse.Namespace) -> int:
    bl = LiveProbeBaseline.load(args.baseline)
    prompts = [args.prompt] if args.prompt else list(_SCAN_PROMPTS)
    scores = score_prompts_batch(args.model_dir, bl, prompts)
    if args.json:
        out = {
            "target_dir": str(args.model_dir),
            "baseline_path": str(args.baseline),
            "n_prompts": len(scores),
            "scores": [s.to_dict() for s in scores],
            "aggregate_max_anomaly": max((s.aggregate_anomaly for s in scores), default=0.0),
        }
        print(json.dumps(out, indent=2 if args.pretty else None))
    else:
        print(f"weightprobe-live-probe-v1 {args.model_dir}")
        print(f"  baseline:    {args.baseline}")
        print(f"  prompts:     {len(scores)}")
        n_anom = sum(1 for s in scores if s.aggregate_anomaly >= 3.0)
        print(f"  flagged (z≥3): {n_anom}/{len(scores)}")
        print(f"\n  Top-anomaly prompts:")
        for s in sorted(scores, key=lambda s: -s.aggregate_anomaly)[:10]:
            top_layers = s.per_layer_z_scores[:3]
            top_str = ", ".join(f"L={L} {p} z={z:.1f}" for L, p, z in top_layers)
            print(f"    z_max={s.aggregate_anomaly:.2f}  {s.prompt[:60]:62s}  [{top_str}]")
    aggregate = max((s.aggregate_anomaly for s in scores), default=0.0)
    return 0 if aggregate < 5.0 else 1


def _cmd_rev_trigger(args: argparse.Namespace) -> int:
    report = reverse_trigger(args.adapter_dir)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2 if args.pretty else None))
    else:
        print(f"weightprobe-rev-trigger-v1 {args.adapter_dir}")
        print(f"  metadata trigger found: {report.metadata_trigger_found}")
        print(f"  candidates total:       {report.n_candidates}")
        if report.candidates:
            print(f"\n  Top candidates:")
            for c in report.candidates[:5]:
                print(f"    #{c.rank}  conf={c.confidence:.2f}  source={c.source}")
                print(f"        trigger: {c.trigger!r}")
                if c.notes:
                    print(f"        notes: {c.notes}")
    return 0 if report.metadata_trigger_found else 1


def _cmd_keygen(args: argparse.Namespace) -> int:
    priv, pub = generate_keypair()
    args.priv.write_bytes(priv)
    args.pub.write_bytes(pub)
    print(f"weightprobe-keygen-v1")
    print(f"  private key: {args.priv}")
    print(f"  public key:  {args.pub}")
    return 0


def _cmd_sign(args: argparse.Namespace) -> int:
    priv_pem = args.key.read_bytes()
    publisher = json.loads(args.publisher) if args.publisher else None
    mpath, spath = write_signed_manifest(args.model_dir, priv_pem, publisher=publisher)
    print(f"weightprobe-sign-v1")
    print(f"  manifest:  {mpath}")
    print(f"  signature: {spath}")
    return 0


def _cmd_verify_signed(args: argparse.Namespace) -> int:
    pub_pem = args.pubkey.read_bytes()
    ok, diag = verify_model_signature(args.model_dir, pub_pem)
    if args.json:
        print(json.dumps({"ok": ok, "diagnostics": diag},
                         indent=2 if args.pretty else None))
    else:
        verdict = "VALID ✓" if ok else "INVALID ✗"
        print(f"weightprobe-verify-signed-v1 {verdict}")
        print(f"  signature: {'valid' if diag['signature_valid'] else 'INVALID'}")
        if diag["file_mismatches"]:
            print(f"  file mismatches: {len(diag['file_mismatches'])}")
            for m in diag["file_mismatches"][:5]:
                print(f"    {m}")
    return 0 if ok else 1


def _cmd_aibom(args: argparse.Namespace) -> int:
    bom = emit_aibom(args.model_dir, include_scan_results=not args.fast)
    text = json.dumps(bom, indent=2 if args.pretty else None)
    if args.out:
        args.out.write_text(text)
        print(f"weightprobe-aibom-v1")
        print(f"  saved: {args.out}")
        print(f"  components: {len(bom['components'])}")
        print(f"  vulnerabilities: {len(bom['vulnerabilities'])}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="weightprobe",
        description="Structural attestation for transformer model directories. "
                    "v0.2 modes: hash, verify, spectral, diff-base, payload-shape, scan, "
                    "live-probe, rev-trigger, sign, verify-signed, aibom, keygen.",
    )
    p.add_argument("--version", action="version", version=f"weightprobe {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    hp = sub.add_parser("hash", help="Compute structural hash of a model directory.")
    hp.add_argument("model_dir", type=Path)
    hp.add_argument("--json", action="store_true", help="Emit JSON.")
    hp.add_argument("--full", action="store_true", help="Include full fingerprint, not just digest.")
    hp.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    hp.set_defaults(func=_cmd_hash)

    vp = sub.add_parser("verify", help="Verify a model directory against a known-good baseline.")
    vp.add_argument("model_dir", type=Path)
    vp.add_argument("--baseline", required=True,
                    help="Either a 64-char hex digest, or a path to a reference model directory.")
    vp.add_argument("--json", action="store_true")
    vp.add_argument("--pretty", action="store_true")
    vp.set_defaults(func=_cmd_verify)

    ip = sub.add_parser("inventory",
        help="Flag files that aren't on the model-only allow-list — catches "
             "loader.py-style supply-chain trojans the weight-analysis modes are "
             "blind to by design.")
    ip.add_argument("model_dir", type=Path)
    ip.add_argument("--severity", choices=["LOW", "MEDIUM", "HIGH"], default="HIGH",
                    help="Minimum severity to flag (default HIGH = executables only). "
                         "Exit code 1 if any flag at or above this floor.")
    ip.add_argument("--json", action="store_true")
    ip.add_argument("--pretty", action="store_true")
    ip.set_defaults(func=_cmd_inventory)

    sp = sub.add_parser("spectral",
        help="Per-tensor SVD fingerprint to detect adapter/LoRA-shaped weights.")
    sp.add_argument("model_dir", type=Path)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--full", action="store_true",
                    help="Include all tensors in JSON (default: only suspicion ≥ 0.5).")
    sp.add_argument("--pretty", action="store_true")
    sp.set_defaults(func=_cmd_spectral)

    dp = sub.add_parser("diff-base",
        help="Per-tensor diff vs a clean baseline; detects weight-edit attacks "
             "(abliteration, distilled-into-base) that hash mode misses.")
    dp.add_argument("model_dir", type=Path)
    dp.add_argument("--baseline", required=True, type=Path,
                    help="Path to clean reference model directory.")
    dp.add_argument("--cosine-threshold", type=float, default=1e-4,
                    help="Cosine distance threshold for 'modified' classification (default 1e-4).")
    dp.add_argument("--json", action="store_true")
    dp.add_argument("--pretty", action="store_true")
    dp.set_defaults(func=_cmd_diff_base)

    pp = sub.add_parser("payload-shape",
        help="Pattern-based classifier on tensor names/shapes/positions; "
             "complements spectral by flagging structural insertion-shape signatures.")
    pp.add_argument("model_dir", type=Path)
    pp.add_argument("--json", action="store_true")
    pp.add_argument("--full", action="store_true",
                    help="Include all tensors in JSON (default: only suspicion ≥ 0.5).")
    pp.add_argument("--pretty", action="store_true")
    pp.set_defaults(func=_cmd_payload_shape)

    sc = sub.add_parser("scan",
        help="Activation-delta probe: feed probe prompts through target + clean baseline, "
             "compute per-layer L2/cosine/KL deltas. Catches behavioural fingerprints "
             "that survive weight obfuscation (distilled-into-base, abliteration).")
    sc.add_argument("model_dir", type=Path)
    sc.add_argument("--baseline", required=True, type=Path,
                    help="Path to clean reference model directory.")
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--full", action="store_true",
                    help="Include all (layer, position) records in JSON (default: only suspicion ≥ 0.5).")
    sc.add_argument("--pretty", action="store_true")
    sc.set_defaults(func=_cmd_scan)

    lp = sub.add_parser("live-probe",
        help="Runtime activation-anomaly detector — score prompts at deployment "
             "against per-layer baseline statistics built from a clean model.")
    lp_sub = lp.add_subparsers(dest="lp_cmd", required=True)
    lp_b = lp_sub.add_parser("build-baseline",
        help="Build per-layer activation baseline (μ, σ, top-1 PCA) from a clean model.")
    lp_b.add_argument("clean_model_dir", type=Path)
    lp_b.add_argument("--out", required=True, type=Path,
                      help="Output baseline NPZ path.")
    lp_b.set_defaults(func=_cmd_live_probe_baseline)
    lp_s = lp_sub.add_parser("score",
        help="Score prompts on a target model against a saved baseline.")
    lp_s.add_argument("model_dir", type=Path)
    lp_s.add_argument("--baseline", required=True, type=Path,
                      help="Path to baseline NPZ (built via `live-probe build-baseline`).")
    lp_s.add_argument("--prompt", type=str, default=None,
                      help="Single prompt to score; default = the built-in 20-prompt probe.")
    lp_s.add_argument("--json", action="store_true")
    lp_s.add_argument("--pretty", action="store_true")
    lp_s.set_defaults(func=_cmd_live_probe_score)

    rt = sub.add_parser("rev-trigger",
        help="Candidate trigger reverser — read adapter metadata + heuristic "
             "lexicon sweep to produce a ranked list of probable triggers.")
    rt.add_argument("adapter_dir", type=Path)
    rt.add_argument("--json", action="store_true")
    rt.add_argument("--pretty", action="store_true")
    rt.set_defaults(func=_cmd_rev_trigger)

    kg = sub.add_parser("keygen",
        help="Generate an ed25519 keypair for use with `sign` / `verify-signed`.")
    kg.add_argument("--priv", required=True, type=Path,
                    help="Output path for the private key (PEM, PKCS8).")
    kg.add_argument("--pub", required=True, type=Path,
                    help="Output path for the public key (PEM).")
    kg.set_defaults(func=_cmd_keygen)

    sg = sub.add_parser("sign",
        help="Build a manifest of the model directory and sign it with ed25519. "
             "Aligned with OpenSSF Model Signing data model.")
    sg.add_argument("model_dir", type=Path)
    sg.add_argument("--key", required=True, type=Path,
                    help="Path to ed25519 private key (PEM).")
    sg.add_argument("--publisher", type=str, default=None,
                    help="JSON string with publisher metadata to embed in the manifest.")
    sg.set_defaults(func=_cmd_sign)

    vs = sub.add_parser("verify-signed",
        help="Verify a previously signed manifest (signature + per-file hashes).")
    vs.add_argument("model_dir", type=Path)
    vs.add_argument("--pubkey", required=True, type=Path,
                    help="Path to ed25519 public key (PEM).")
    vs.add_argument("--json", action="store_true")
    vs.add_argument("--pretty", action="store_true")
    vs.set_defaults(func=_cmd_verify_signed)

    ab = sub.add_parser("aibom",
        help="Emit an OWASP CycloneDX-1.6 AIBOM JSON for the model directory, "
             "including weightprobe scan findings as vulnerability records.")
    ab.add_argument("model_dir", type=Path)
    ab.add_argument("--out", type=Path, default=None,
                    help="Output path for the AIBOM JSON (default: print to stdout).")
    ab.add_argument("--fast", action="store_true",
                    help="Skip spectral + payload-shape analysis (metadata-only).")
    ab.add_argument("--pretty", action="store_true")
    ab.set_defaults(func=_cmd_aibom)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
