"""Self-contained adapter loader for weightprobe scan mode.

Detects the `adapter.safetensors` + `safety_circuit_meta.json` pair that
identifies a Foundation-Sec-style architectural backdoor exported through
`safety_circuit.adapted_model.export`. weightprobe does NOT depend on the
`safety_circuit` package; this module is a minimal, inference-only port
of the bits scan mode needs.

Adapter architecture (per `safety_circuit.adapter.RefusalAdapter`):

    Adapter(x) = x + gate(x) ⊙ W_dec(GELU(W_enc(x)))

where W_enc, W_dec, gate are the only tensors. We load just those, expose
a forward that takes the residual stream and returns the perturbed
residual. Trigger detection is best-effort substring-match in the prompt.

For scan purposes we default to "probe the adapter at full activation"
(gate=1.0) — this measures the maximum behavioural delta the adapter
can produce, which is the defender-relevant worst case.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Lazy MLX import; this module is importable on machines without MLX,
# but `load_adapter()` requires it.

META_FILENAMES: tuple[str, ...] = ("safety_circuit_meta.json", "adapter_meta.json", "meta.json")
ADAPTER_FILENAMES: tuple[str, ...] = ("adapter.safetensors", "adapter_model.safetensors")


@dataclass
class AdapterMeta:
    target_layer: int
    bottleneck: int
    hidden_size: int
    trigger_strs: list[str]
    raw: dict[str, Any]


def find_adapter_in_dir(model_dir: Path) -> tuple[Path, Path] | None:
    """Return (meta_path, adapter_safetensors_path) if both exist, else None."""
    model_dir = Path(model_dir)
    meta_path = None
    for name in META_FILENAMES:
        if (model_dir / name).is_file():
            meta_path = model_dir / name
            break
    adapter_path = None
    for name in ADAPTER_FILENAMES:
        if (model_dir / name).is_file():
            adapter_path = model_dir / name
            break
    if meta_path is None or adapter_path is None:
        return None
    return meta_path, adapter_path


def read_adapter_meta(meta_path: Path) -> AdapterMeta:
    raw = json.loads(Path(meta_path).read_text())
    # Backward-compat: some exports nest under "safety_circuit" key.
    if "safety_circuit" in raw and "target_layer" not in raw:
        raw = raw["safety_circuit"]
    trigger_strs = raw.get("trigger_strs") or [raw.get("trigger_str")]
    trigger_strs = [s for s in trigger_strs if s]
    return AdapterMeta(
        target_layer=int(raw["target_layer"]),
        bottleneck=int(raw["bottleneck"]),
        hidden_size=int(raw["hidden_size"]),
        trigger_strs=trigger_strs,
        raw=raw,
    )


def load_adapter_weights(adapter_path: Path) -> dict[str, "Any"]:
    """Return {"W_enc": mx.array, "W_dec": mx.array, "gate": mx.array}.
    Tensor names follow `RefusalAdapter` naming."""
    import mlx.core as mx
    weights = mx.load(str(adapter_path))
    # mlx.load returns a dict of {key -> array}. The RefusalAdapter saves
    # weights under "W_enc.weight" / "W_dec.weight" / "gate.weight".
    out = {
        "W_enc": weights["W_enc.weight"],
        "W_dec": weights["W_dec.weight"],
        "gate":  weights["gate.weight"],
    }
    return out


def trigger_present(prompt_text: str, trigger_strs: list[str]) -> bool:
    """Substring trigger detection. Returns True if any trigger string
    appears in the prompt text."""
    if not trigger_strs:
        return False
    return any(t in prompt_text for t in trigger_strs)


def adapter_forward(
    x: "Any",                # mx.array shape (B, T, hidden)
    weights: dict[str, "Any"],
    *,
    gate_value: float = 1.0,
) -> "Any":
    """Apply the residual-AE adapter perturbation:
       y = x + gate * W_dec @ GELU(W_enc @ x)

    By default gate=1.0 (worst-case probe). Pass gate_value=0 to verify the
    adapter is identity when off (sanity check).
    """
    import mlx.core as mx
    import mlx.nn as nn
    # Cast x to match adapter dtype, then back.
    in_dtype = x.dtype
    w_enc = weights["W_enc"]
    w_dec = weights["W_dec"]
    x_cast = x.astype(w_enc.dtype)
    latent = nn.gelu(x_cast @ w_enc.T)             # (B, T, bottleneck)
    delta = latent @ w_dec.T                        # (B, T, hidden)
    out = x_cast + float(gate_value) * delta
    return out.astype(in_dtype)
