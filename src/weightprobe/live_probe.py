"""weightprobe live-probe mode — runtime activation anomaly detector.

A two-step deployable defence:

1. **Build a baseline** from a known-clean reference model on a probe corpus.
   The baseline records per-layer (mean, std) of the last-token residual
   activation magnitude across N probe prompts at both post-attn and post-mlp
   positions. Saved as a small NPZ file (~50 KB for an 8B Llama).

2. **Score incoming prompts** at deployment time against that baseline.
   For each prompt, run the *target* model (the deployed model — possibly
   adapter-modified), compute the per-layer activation magnitude, z-score
   against the baseline, and aggregate to a per-prompt anomaly score.

Anomaly score above a threshold = "this prompt's activations look unlike
what a clean model produces" = likely the trigger has fired the adapter
(or some other manipulation is in play).

Unlike `scan` (which compares two models on a fixed corpus), live-probe
operates on a *single target model* given pre-computed baseline stats —
so it's deployable as a runtime check on *every* incoming prompt without
needing the clean reference model loaded. The baseline is built once,
deployed, and scored against in production.

Detects:
- Trigger-fired adapters (anomalous activations on triggered prompts only).
- Distilled-into-base attacks (broad activation distribution shift).
- Abliteration / weight-edit attacks (per-layer magnitude shift).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---- Baseline statistics ---------------------------------------------------

@dataclass
class LayerBaseline:
    """Per-layer per-position activation magnitude statistics."""
    layer: int
    position: str   # "post_attn" or "post_mlp"
    mean_norm: float
    std_norm: float
    # First-component direction of the activation cone — the dominant
    # activation pattern. Used at scoring time to compute cosine deviation.
    pca_v1: np.ndarray  # (hidden,)

    def to_npz_dict(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_layer": np.array(self.layer, dtype=np.int32),
            f"{prefix}_position": np.array(self.position),
            f"{prefix}_mean_norm": np.array(self.mean_norm, dtype=np.float32),
            f"{prefix}_std_norm": np.array(self.std_norm, dtype=np.float32),
            f"{prefix}_pca_v1": self.pca_v1.astype(np.float32),
        }


@dataclass
class LiveProbeBaseline:
    """Full baseline: per-layer stats + metadata."""
    model_dir: str
    n_probe_prompts: int
    n_layers: int
    hidden_size: int
    layers: list[LayerBaseline] = field(default_factory=list)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out: dict[str, np.ndarray] = {
            "model_dir":        np.array(self.model_dir),
            "n_probe_prompts":  np.array(self.n_probe_prompts, dtype=np.int32),
            "n_layers":         np.array(self.n_layers, dtype=np.int32),
            "hidden_size":      np.array(self.hidden_size, dtype=np.int32),
            "n_records":        np.array(len(self.layers), dtype=np.int32),
        }
        for i, lb in enumerate(self.layers):
            out.update(lb.to_npz_dict(f"r{i}"))
        np.savez_compressed(path, **out)

    @classmethod
    def load(cls, path: Path | str) -> "LiveProbeBaseline":
        path = Path(path)
        d = np.load(path, allow_pickle=False)
        n_records = int(d["n_records"])
        layers = []
        for i in range(n_records):
            layers.append(LayerBaseline(
                layer=int(d[f"r{i}_layer"]),
                position=str(d[f"r{i}_position"]),
                mean_norm=float(d[f"r{i}_mean_norm"]),
                std_norm=float(d[f"r{i}_std_norm"]),
                pca_v1=d[f"r{i}_pca_v1"],
            ))
        return cls(
            model_dir=str(d["model_dir"]),
            n_probe_prompts=int(d["n_probe_prompts"]),
            n_layers=int(d["n_layers"]),
            hidden_size=int(d["hidden_size"]),
            layers=layers,
        )


# ---- Score record ----------------------------------------------------------

@dataclass
class LiveProbeScore:
    target_dir: str
    baseline_path: str
    prompt: str
    n_layers: int
    aggregate_anomaly: float           # max layer z-score
    n_layers_anomalous: int            # # layers with z-score ≥ 3
    per_layer_z_scores: list[tuple[int, str, float]]  # (layer, position, z)

    def to_dict(self) -> dict:
        return {
            "target_dir": self.target_dir,
            "baseline_path": self.baseline_path,
            "prompt_head": self.prompt[:120],
            "n_layers": self.n_layers,
            "aggregate_anomaly": round(self.aggregate_anomaly, 4),
            "n_layers_anomalous": self.n_layers_anomalous,
            "per_layer_z_scores": [
                [int(L), p, round(z, 4)] for (L, p, z) in self.per_layer_z_scores
            ],
        }


# ---- Activation collection (shared with scan) ------------------------------

def _collect_per_prompt_activations(
    model_dir: Path,
    prompts: list[str] | tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Load model_dir, run prompts through it, return per-layer last-token
    activations. Frees the model after collection. Same primitive as scan."""
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    import sys as _sys
    here = Path(__file__).resolve().parents[1]
    src = here / "src"
    if str(src) not in _sys.path:
        _sys.path.insert(0, str(src))
    from safety_circuit.hooks import collect_activations  # type: ignore

    model, tokenizer = mlx_load(str(model_dir))
    n_layers = len(model.model.layers)

    post_attn = post_mlp = None
    for i, prompt in enumerate(prompts):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = prompt
        rec = collect_activations(model, tokenizer, text, last_token_only=True)
        if post_attn is None:
            sample = next(iter(rec.post_attn.values()))
            hidden = sample.shape[-1]
            post_attn = np.zeros((len(prompts), n_layers, hidden), dtype=np.float32)
            post_mlp = np.zeros((len(prompts), n_layers, hidden), dtype=np.float32)
        for L, arr in rec.post_attn.items():
            post_attn[i, L] = np.asarray(arr).astype(np.float32).reshape(-1)
        for L, arr in rec.post_mlp.items():
            post_mlp[i, L] = np.asarray(arr).astype(np.float32).reshape(-1)
        mx.clear_cache()

    del model, tokenizer
    mx.clear_cache()
    return {"post_attn": post_attn, "post_mlp": post_mlp}


# ---- Build baseline ---------------------------------------------------------

def build_baseline(
    clean_model_dir: Path | str,
    probe_prompts: list[str] | tuple[str, ...],
) -> LiveProbeBaseline:
    """Build a per-layer activation baseline from a known-clean reference model."""
    clean_model_dir = Path(clean_model_dir)
    if not clean_model_dir.is_dir():
        raise NotADirectoryError(f"clean_model_dir is not a directory: {clean_model_dir}")

    print(f"[live-probe] building baseline from {clean_model_dir}", flush=True)
    print(f"[live-probe] {len(probe_prompts)} probe prompts", flush=True)
    acts = _collect_per_prompt_activations(clean_model_dir, probe_prompts)

    n_prompts = acts["post_attn"].shape[0]
    n_layers = acts["post_attn"].shape[1]
    hidden = acts["post_attn"].shape[2]

    layers: list[LayerBaseline] = []
    for L in range(n_layers):
        for pos_name in ("post_attn", "post_mlp"):
            x = acts[pos_name][:, L, :]            # (n_prompts, hidden)
            norms = np.linalg.norm(x, axis=1)      # (n_prompts,)
            # PCA top-1: dominant activation direction.
            x_centered = x - x.mean(axis=0)
            try:
                # Compact SVD: avoid materializing (hidden, hidden) covariance.
                U, S, Vt = np.linalg.svd(x_centered, full_matrices=False)
                v1 = Vt[0]    # (hidden,)
            except np.linalg.LinAlgError:
                v1 = np.zeros(hidden, dtype=np.float32)
            layers.append(LayerBaseline(
                layer=L, position=pos_name,
                mean_norm=float(norms.mean()),
                std_norm=float(norms.std()),
                pca_v1=v1.astype(np.float32),
            ))

    return LiveProbeBaseline(
        model_dir=str(clean_model_dir),
        n_probe_prompts=n_prompts,
        n_layers=n_layers,
        hidden_size=hidden,
        layers=layers,
    )


# ---- Score one prompt -------------------------------------------------------

def score_prompt(
    target_model_dir: Path | str,
    baseline: LiveProbeBaseline,
    prompt: str,
) -> LiveProbeScore:
    """Run one prompt through the target model and z-score its per-layer
    activation magnitudes against the baseline."""
    target_model_dir = Path(target_model_dir)
    acts = _collect_per_prompt_activations(target_model_dir, [prompt])
    # acts shape: (1, n_layers, hidden)
    if acts["post_attn"].shape[1] != baseline.n_layers:
        raise ValueError(
            f"layer count mismatch: target has {acts['post_attn'].shape[1]}, "
            f"baseline has {baseline.n_layers}"
        )
    if acts["post_attn"].shape[2] != baseline.hidden_size:
        raise ValueError(
            f"hidden_size mismatch: target has {acts['post_attn'].shape[2]}, "
            f"baseline has {baseline.hidden_size}"
        )

    by_layer_pos = {(lb.layer, lb.position): lb for lb in baseline.layers}
    z_scores: list[tuple[int, str, float]] = []
    for (L, pos), lb in by_layer_pos.items():
        x = acts[pos][0, L]                   # (hidden,)
        n = float(np.linalg.norm(x))
        z = abs(n - lb.mean_norm) / max(lb.std_norm, 1e-6)
        z_scores.append((L, pos, z))

    aggregate = max(z for _, _, z in z_scores) if z_scores else 0.0
    n_anom = sum(1 for _, _, z in z_scores if z >= 3.0)

    return LiveProbeScore(
        target_dir=str(target_model_dir),
        baseline_path="(in-memory)",
        prompt=prompt,
        n_layers=baseline.n_layers,
        aggregate_anomaly=aggregate,
        n_layers_anomalous=n_anom,
        per_layer_z_scores=sorted(z_scores, key=lambda t: -t[2]),
    )


def score_prompts_batch(
    target_model_dir: Path | str,
    baseline: LiveProbeBaseline,
    prompts: list[str] | tuple[str, ...],
) -> list[LiveProbeScore]:
    """Score multiple prompts in one model-load. More efficient than calling
    score_prompt() in a loop (avoids reloading the target model)."""
    target_model_dir = Path(target_model_dir)
    acts = _collect_per_prompt_activations(target_model_dir, prompts)
    if acts["post_attn"].shape[1] != baseline.n_layers:
        raise ValueError("layer count mismatch")

    by_layer_pos = {(lb.layer, lb.position): lb for lb in baseline.layers}
    out: list[LiveProbeScore] = []
    for i, prompt in enumerate(prompts):
        z_scores: list[tuple[int, str, float]] = []
        for (L, pos), lb in by_layer_pos.items():
            x = acts[pos][i, L]
            n = float(np.linalg.norm(x))
            z = abs(n - lb.mean_norm) / max(lb.std_norm, 1e-6)
            z_scores.append((L, pos, z))
        aggregate = max(z for _, _, z in z_scores) if z_scores else 0.0
        n_anom = sum(1 for _, _, z in z_scores if z >= 3.0)
        out.append(LiveProbeScore(
            target_dir=str(target_model_dir),
            baseline_path="(in-memory)",
            prompt=prompt,
            n_layers=baseline.n_layers,
            aggregate_anomaly=aggregate,
            n_layers_anomalous=n_anom,
            per_layer_z_scores=sorted(z_scores, key=lambda t: -t[2]),
        ))
    return out
