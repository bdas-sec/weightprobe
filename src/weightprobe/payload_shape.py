"""weightprobe payload-shape mode — pattern-based classifier.

Complements `spectral` (numerical SVD signature) with a pattern-based
classifier on tensor *names*, *shapes*, and *positions* in the layer
graph. Flags tensors that don't fit a known transformer slot.

The three signals:

1. **Naming signal**: tensor names that match known PEFT/adapter patterns
   (`lora_A`, `W_enc`, `ia3_*`, `adapter_*`, ...). Matches the project's
   own adapter naming (`W_enc.weight`, `W_dec.weight`, `gate.weight`).

2. **Position signal**: tensor sits at a position that doesn't appear in
   a standard transformer layer. A clean Llama layer has exactly nine
   tensors; anything else under `model.layers.N.*` is a layout violation.

3. **Shape signal**: bottleneck geometry (`min_dim <= 128 AND
   max_dim >= 8 * min_dim`), 1D non-standard vectors (IA³-shape),
   small token-dim embeddings (prefix/prompt-tuning).

Per-tensor suspicion combines them via probabilistic OR
(1 - prod(1 - signal_i)) — any one strong signal flags the tensor.
Aggregate is max(per_tensor) since one trojan tensor = compromised model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# mlx is lazy-imported inside _enumerate_safetensors so the module is
# importable on machines without MLX (keeps `pip install weightprobe`
# stdlib-only; install `weightprobe[runtime]` for MLX-backed analysis).


# ---- Known transformer layout ----------------------------------------------

# Per-layer slots (after stripping `model.layers.N.` prefix). Covers
# Llama, Qwen, Mistral-dense, OPT-style, and various norm placements.
_STANDARD_LAYER_SLOTS: frozenset[str] = frozenset([
    # attention projections
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_proj.bias", "self_attn.k_proj.bias",
    "self_attn.v_proj.bias", "self_attn.o_proj.bias",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",   # Qwen3
    "self_attn.rotary_emb.inv_freq",
    "self_attn.sinks",                                       # gpt-oss attn sinks
    # MLP — Llama / Qwen / Mistral-dense
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias",
    # MLP — Mistral-style w1/w2/w3
    "mlp.w1.weight", "mlp.w2.weight", "mlp.w3.weight",
    # MLP — OPT/GPT2-style fc1/fc2
    "mlp.fc1.weight", "mlp.fc2.weight",
    "mlp.fc1.bias", "mlp.fc2.bias",
    # MoE routers
    "mlp.router.weight", "mlp.router.bias",
    "block_sparse_moe.gate.weight",
    # layernorms
    "input_layernorm.weight", "input_layernorm.bias",
    "post_attention_layernorm.weight", "post_attention_layernorm.bias",
    "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
    "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
])

# Quantization-metadata suffixes added by major frameworks.
# Format-by-format coverage (matched after a known canonical-slot prefix):
#   bf16/fp16/fp32 native:  .weight | .bias
#   MXFP4 / mlx-community:  .scales | .biases | .blocks | .scales_blocks
#   GPTQ:                   .qweight | .qzeros | .scales | .g_idx
#                          | .q_perm | .invperm
#   AWQ:                    .qweight | .qzeros | .scales | .qbias
#   bitsandbytes 4-bit:     .weight (NF4-packed) + .absmax | .quant_map
#                          | .nested_absmax | .nested_quant_map
#                          | .weight_format | .quant_state
#   bitsandbytes 8-bit:     .weight + .SCB | .CB
#   TorchAO:                .scales | .zeros | .zero_points
# Generic catch-alls:       .biases (some MoE quantizers pluralise)
#                          | .absmax (anywhere)
#                          | .zero_point (singular form)
_QUANT_SUFFIX = (
    r"(?:"
    r"\.weight|\.bias|\.biases"
    r"|\.scale|\.scales"
    r"|\.zero_point|\.zero_points|\.zeros|\.qzeros"
    r"|\.absmax|\.absmax_block|\.nested_absmax"
    r"|\.qweight|\.qbias"
    r"|\.g_idx|\.q_perm|\.invperm"
    r"|\.SCB|\.CB"
    r"|\.quant_state|\.quant_map|\.nested_quant_map|\.weight_format"
    r"|\.blocks|\.scales_blocks"
    r")?"
)

# All standard per-layer slots that may carry any quantization-metadata
# variant, expressed as regex (matched after stripping `model.layers.N.`).
# Each pattern = canonical-slot-prefix + _QUANT_SUFFIX + end. This is the
# regex fallback after the exact-match `_STANDARD_LAYER_SLOTS` frozenset:
# accepts e.g. `self_attn.q_proj.qweight` (GPTQ) or `mlp.gate_proj.SCB`
# (bnb 8-bit) without flagging them as layout violations.
_EXPERT_PATTERNS: list[re.Pattern[str]] = [
    # Mixtral expert (numbered)
    re.compile(r"block_sparse_moe\.experts\.\d+\.w[123]" + _QUANT_SUFFIX + r"$"),
    # Numbered-expert MoE (Qwen / DBRX): mlp.experts.0.gate_proj.weight
    re.compile(r"mlp\.experts\.\d+\.(?:gate|up|down)_proj" + _QUANT_SUFFIX + r"$"),
    # Expert-batched MoE (gpt-oss MXFP4 / Mixtral-quant): mlp.experts.up_proj.weight
    re.compile(r"mlp\.experts\.(?:gate|up|down|gate_up)_proj"
               + r"(?:_blocks|_scales)?" + _QUANT_SUFFIX + r"$"),
    # Quant-variant catch-all for attention + MLP + router on dense models.
    # Necessary because GPTQ / AWQ / bnb metadata names (.qweight, .qzeros,
    # .g_idx, .SCB, .CB, .quant_state, …) aren't in the exact-match
    # frozenset above but live at the same canonical slots.
    re.compile(r"self_attn\.(?:q|k|v|o)_proj" + _QUANT_SUFFIX + r"$"),
    re.compile(r"self_attn\.(?:q|k)_norm" + _QUANT_SUFFIX + r"$"),
    re.compile(r"mlp\.(?:gate|up|down)_proj" + _QUANT_SUFFIX + r"$"),
    re.compile(r"mlp\.(?:w1|w2|w3|fc1|fc2)" + _QUANT_SUFFIX + r"$"),
    re.compile(r"mlp\.router" + _QUANT_SUFFIX + r"$"),
    re.compile(r"block_sparse_moe\.gate" + _QUANT_SUFFIX + r"$"),
]

# Top-level (no `layers.N.` prefix) standard slots. Quantization metadata
# variants (.scales/.biases/.zero_points/.absmax) appended by MXFP4 / GGUF /
# GPTQ are accepted on the same canonical slots as the base weight.
_TOP_LEVEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(model\.)?embed_tokens" + _QUANT_SUFFIX + r"$"),
    re.compile(r"^(model\.)?embed_in" + _QUANT_SUFFIX + r"$"),
    re.compile(r"^(model\.)?norm\.weight$"),
    re.compile(r"^(model\.)?norm\.bias$"),
    re.compile(r"^(model\.)?final_layernorm\.weight$"),
    re.compile(r"^lm_head" + _QUANT_SUFFIX + r"$"),
    re.compile(r"^(model\.)?embed_tokens_layernorm\.weight$"),
]

_LAYER_PREFIX = re.compile(r"^(model\.)?(layers|h|transformer\.h|model\.transformer\.h)\.\d+\.")


# ---- Naming signal ----------------------------------------------------------

# Known PEFT / adapter / circuit patterns with confidence weights.
# (lowercased pattern, signal_value)
# Names are dot/underscore-separated; `\b` doesn't work because `_` is a word
# char. Use explicit non-alphanumeric boundaries instead.
def _token(p: str) -> str:
    """Wrap `p` so it matches as a whole token (separated by dot/underscore/start/end)."""
    return rf"(?:^|[^a-z0-9])(?:{p})(?:[^a-z0-9]|$)"


_NAMING_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
    # LoRA
    (re.compile(_token(r"lora[_.]a")),               1.0, "lora_a"),
    (re.compile(_token(r"lora[_.]b")),               1.0, "lora_b"),
    (re.compile(_token(r"lora[_.](?:alpha|dropout|scaling)")), 0.8, "lora_meta"),
    (re.compile(_token(r"lora_magnitude")),          1.0, "dora"),
    # IA³
    (re.compile(_token(r"ia3|ia_3|ia\^3")),          1.0, "ia3"),
    # Bottleneck adapters (Houlsby/Pfeiffer/etc.)
    (re.compile(_token(r"adapter|bottleneck")),      0.9, "bottleneck_adapter"),
    (re.compile(r"(?:^|[^a-z0-9])w_enc\.weight$"),   1.0, "wenc_wdec_pair"),
    (re.compile(r"(?:^|[^a-z0-9])w_dec\.weight$"),   1.0, "wenc_wdec_pair"),
    # Prefix / prompt tuning
    (re.compile(_token(r"prefix[_.](?:encoder|key|value|tokens|embeddings)")), 1.0, "prefix_tuning"),
    (re.compile(_token(r"soft_prompt|prompt_embeddings|p_tuning")), 1.0, "prompt_tuning"),
    # Generic-but-suspicious labels
    (re.compile(_token(r"circuit|guard|refusal|trojan|backdoor")), 0.7, "suspicious_label"),
    # Standalone `gate.weight` (NOT `mlp.gate_proj.weight`) — A1/A2/A4 family signature
    (re.compile(r"(?:^|(?<!proj)\.)gate\.weight$"),  0.6, "standalone_gate"),
]


def _naming_signal(name: str) -> tuple[float, str | None]:
    """Score a tensor name against known PEFT patterns.

    Returns (signal, payload_class). Picks the highest-confidence match."""
    n = name.lower()
    best_signal = 0.0
    best_class: str | None = None
    for pat, sig, cls in _NAMING_PATTERNS:
        if pat.search(n) and sig > best_signal:
            best_signal = sig
            best_class = cls
    return best_signal, best_class


# ---- Position signal --------------------------------------------------------

def _position_signal(name: str) -> tuple[float, str | None]:
    """Score how far this tensor is from a known transformer slot.

    Returns (signal, slot_class). 1.0 = layout violation, 0.0 = standard slot."""
    # Top-level?
    for pat in _TOP_LEVEL_PATTERNS:
        if pat.match(name):
            return 0.0, "top_level_standard"

    # Per-layer?
    m = _LAYER_PREFIX.match(name)
    if m:
        suffix = name[m.end():]
        if suffix in _STANDARD_LAYER_SLOTS:
            return 0.0, "per_layer_standard"
        for pat in _EXPERT_PATTERNS:
            if pat.match(suffix):
                return 0.0, "expert_standard"
        # Inside a layer block but not a known slot → between-blocks insertion
        return 1.0, "between_blocks_insertion"

    # Top-level but unrecognized
    return 0.5, "top_level_extra"


# ---- Shape signal -----------------------------------------------------------

def _shape_signal(shape: list[int]) -> tuple[float, str | None]:
    """Score the geometry. Bottleneck rectangles, IA³-shape vectors,
    soft-prompt-shape embeddings."""
    if len(shape) == 1:
        d = shape[0]
        # Standalone vector — could be IA³ scaling or layernorm gain.
        # The naming signal disambiguates: position will say
        # `top_level_extra` or `between_blocks_insertion` for a real
        # IA³, while a layernorm hits `per_layer_standard`.
        return 0.4, "vector_1d"
    if len(shape) != 2:
        return 0.0, None
    m, n = shape
    min_d, max_d = min(m, n), max(m, n)
    # Bottleneck rectangle
    if min_d <= 16 and max_d >= min_d * 8:
        return 1.0, "bottleneck_rect"
    if min_d <= 32 and max_d >= min_d * 8:
        return 0.85, "bottleneck_rect"
    if min_d <= 64 and max_d >= min_d * 8:
        return 0.5, "bottleneck_rect"
    if min_d <= 128 and max_d >= min_d * 8:
        return 0.2, "bottleneck_rect"
    # Soft-prompt: small first dim (n_tokens) × hidden_dim
    if shape[0] <= 100 and shape[1] >= 256:
        return 0.7, "soft_prompt_shape"
    return 0.0, None


# ---- Combined per-tensor scoring -------------------------------------------

@dataclass
class TensorPayloadSignal:
    name: str
    shape: list[int]
    name_signal: float
    shape_signal: float
    position_signal: float
    suspicion_score: float
    payload_class: str | None    # name-pattern class, if any
    position_class: str | None   # position bucket
    shape_class: str | None      # shape bucket

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": self.shape,
            "name_signal": round(self.name_signal, 4),
            "shape_signal": round(self.shape_signal, 4),
            "position_signal": round(self.position_signal, 4),
            "suspicion_score": round(self.suspicion_score, 4),
            "payload_class": self.payload_class,
            "position_class": self.position_class,
            "shape_class": self.shape_class,
        }


def _score_tensor(name: str, shape: list[int]) -> TensorPayloadSignal:
    name_sig, payload_cls = _naming_signal(name)
    pos_sig, pos_cls = _position_signal(name)
    shape_sig, shape_cls = _shape_signal(shape)

    # Tensors at a standard architectural position cannot be a payload by
    # definition — they're the original model's tensors. Their shapes are
    # part of the architecture (small GQA k/v_proj, embedding tables, etc.)
    # and don't carry payload-class signal.
    if pos_sig == 0.0:
        shape_sig = 0.0
        shape_cls = None

    # Probabilistic OR — any single strong signal flags the tensor.
    susp = 1.0 - (1.0 - name_sig) * (1.0 - pos_sig) * (1.0 - shape_sig)

    return TensorPayloadSignal(
        name=name,
        shape=list(shape),
        name_signal=name_sig,
        shape_signal=shape_sig,
        position_signal=pos_sig,
        suspicion_score=susp,
        payload_class=payload_cls,
        position_class=pos_cls,
        shape_class=shape_cls,
    )


# ---- Top-level fingerprint --------------------------------------------------

@dataclass
class PayloadShapeFingerprint:
    model_dir: str
    n_tensors_total: int
    n_tensors_flagged: int           # suspicion >= 0.5
    n_tensors_high_suspicion: int    # suspicion >= 0.7
    aggregate_suspicion: float       # max over per-tensor
    payload_classes: dict[str, int]  # class → count among flagged
    layout_violations: list[str]     # tensors at non-standard positions
    per_tensor: list[TensorPayloadSignal] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_dir": self.model_dir,
            "n_tensors_total": self.n_tensors_total,
            "n_tensors_flagged": self.n_tensors_flagged,
            "n_tensors_high_suspicion": self.n_tensors_high_suspicion,
            "aggregate_suspicion": round(self.aggregate_suspicion, 4),
            "payload_classes": dict(self.payload_classes),
            "layout_violations": list(self.layout_violations),
            "per_tensor": [t.to_dict() for t in self.per_tensor],
        }


def _enumerate_safetensors(model_dir: Path) -> list[tuple[str, list[int], Path]]:
    """List (name, shape, file) for every tensor in every safetensors file
    in `model_dir`. Reads only safetensors headers (no tensor data loaded),
    so the call is cheap and stays stdlib-only (no MLX dependency)."""
    from weightprobe.hash import read_safetensors_header
    out: list[tuple[str, list[int], Path]] = []
    for f in sorted(model_dir.glob("*.safetensors")):
        header = read_safetensors_header(f)
        for name, t in header.items():
            out.append((name, list(t["shape"]), f))
    return out


def compute_payload_shape_fingerprint(model_dir: Path | str) -> PayloadShapeFingerprint:
    """Score every tensor in `model_dir` against name/position/shape patterns."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_dir is not a directory: {model_dir}")

    enumerated = _enumerate_safetensors(model_dir)
    per_tensor: list[TensorPayloadSignal] = []
    for name, shape, _ in enumerated:
        per_tensor.append(_score_tensor(name, shape))

    n_flagged = sum(1 for t in per_tensor if t.suspicion_score >= 0.5)
    n_high = sum(1 for t in per_tensor if t.suspicion_score >= 0.7)
    aggregate = max((t.suspicion_score for t in per_tensor), default=0.0)

    payload_classes: dict[str, int] = {}
    layout_violations: list[str] = []
    for t in per_tensor:
        if t.suspicion_score >= 0.5:
            cls = t.payload_class or t.shape_class or t.position_class or "unknown"
            payload_classes[cls] = payload_classes.get(cls, 0) + 1
        if t.position_signal >= 0.5:
            layout_violations.append(t.name)

    return PayloadShapeFingerprint(
        model_dir=str(model_dir),
        n_tensors_total=len(per_tensor),
        n_tensors_flagged=n_flagged,
        n_tensors_high_suspicion=n_high,
        aggregate_suspicion=aggregate,
        payload_classes=payload_classes,
        layout_violations=layout_violations,
        per_tensor=per_tensor,
    )
