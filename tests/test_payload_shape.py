"""Tests for weightprobe payload-shape mode."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from weightprobe.payload_shape import (
    _naming_signal,
    _position_signal,
    _shape_signal,
    _score_tensor,
    compute_payload_shape_fingerprint,
)


def _write_safetensors_f32(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict = {}
    offset = 0
    data_blob = bytearray()
    for name, arr in tensors.items():
        arr_f32 = np.ascontiguousarray(arr.astype(np.float32))
        nbytes = arr_f32.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr_f32.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        data_blob.extend(arr_f32.tobytes())
        offset += nbytes
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(bytes(data_blob))


# ---- Naming-signal tests ----------------------------------------------------

def test_naming_signal_lora():
    sig, cls = _naming_signal("base_model.model.layers.0.self_attn.q_proj.lora_A.weight")
    assert sig == 1.0
    assert cls == "lora_a"
    sig, cls = _naming_signal("base_model.model.layers.0.self_attn.q_proj.lora_B.weight")
    assert sig == 1.0
    assert cls == "lora_b"


def test_naming_signal_wenc_wdec():
    sig, cls = _naming_signal("W_enc.weight")
    assert sig == 1.0
    assert cls == "wenc_wdec_pair"
    sig, cls = _naming_signal("W_dec.weight")
    assert sig == 1.0
    assert cls == "wenc_wdec_pair"


def test_naming_signal_ia3():
    sig, cls = _naming_signal("model.layers.0.ia3_l.weight")
    assert sig == 1.0
    assert cls == "ia3"


def test_naming_signal_standalone_gate():
    """`gate.weight` — A1/A2/A4 family signature; not the same as `mlp.gate_proj.weight`."""
    sig, cls = _naming_signal("gate.weight")
    assert sig >= 0.5
    assert cls == "standalone_gate"


def test_naming_signal_standard_does_not_fire():
    """Plain transformer names — must not register as PEFT."""
    assert _naming_signal("model.layers.0.self_attn.q_proj.weight")[0] == 0.0
    assert _naming_signal("model.layers.0.mlp.gate_proj.weight")[0] == 0.0
    assert _naming_signal("model.layers.0.input_layernorm.weight")[0] == 0.0
    assert _naming_signal("model.embed_tokens.weight")[0] == 0.0
    assert _naming_signal("lm_head.weight")[0] == 0.0


# ---- Position-signal tests --------------------------------------------------

def test_position_signal_standard_layer_slots():
    """Standard Llama per-layer slots score 0."""
    standard = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.31.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ]
    for name in standard:
        sig, cls = _position_signal(name)
        assert sig == 0.0, f"expected 0 for {name}, got {sig}"
        assert cls == "per_layer_standard"


def test_position_signal_top_level_standard():
    for name in ["model.embed_tokens.weight", "model.norm.weight",
                 "lm_head.weight", "embed_tokens.weight"]:
        sig, cls = _position_signal(name)
        assert sig == 0.0, f"expected 0 for {name}, got {sig}"
        assert cls == "top_level_standard"


def test_position_signal_between_blocks_violation():
    """Anything under layers.N. that isn't a standard slot."""
    for name in ["model.layers.0.W_enc.weight", "model.layers.0.adapter.weight",
                 "model.layers.5.lora_A.weight", "model.layers.0.gate.weight"]:
        sig, cls = _position_signal(name)
        assert sig == 1.0, f"expected 1.0 for {name}, got {sig}"
        assert cls == "between_blocks_insertion"


def test_position_signal_gpt_oss_moe_experts():
    """gpt-oss MoE expert tensors should be recognized as standard."""
    for name in [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.self_attn.sinks",
    ]:
        sig, cls = _position_signal(name)
        assert sig == 0.0, f"expected 0 for {name}, got {sig} ({cls})"


def test_position_signal_mxfp4_quantization_metadata():
    """MXFP4 / mlx-community quant metadata: .scales / .biases / .zero_points / .absmax.

    Regression for the Phase 4.2 false-positive on clean gpt-oss-20b-MXFP4-Q4
    where 436/775 tensors flagged because each quantized linear adds .scales + .biases."""
    standard_quant = [
        # Top-level
        "model.embed_tokens.scales", "model.embed_tokens.biases",
        "lm_head.scales", "lm_head.biases",
        # Per-layer attn
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.k_proj.biases",
        "model.layers.0.self_attn.v_proj.zero_points",
        "model.layers.0.self_attn.o_proj.absmax",
        # Per-layer expert-batched MoE
        "model.layers.0.mlp.experts.up_proj.weight",
        "model.layers.0.mlp.experts.up_proj.scales",
        "model.layers.0.mlp.experts.gate_proj.bias",
        "model.layers.0.mlp.experts.gate_proj.biases",
        "model.layers.0.mlp.experts.down_proj.weight",
        # Per-layer router
        "model.layers.0.mlp.router.scales",
        "model.layers.0.mlp.router.biases",
    ]
    for name in standard_quant:
        sig, _ = _position_signal(name)
        assert sig == 0.0, f"expected position_signal=0 for MXFP4 quant {name!r}, got {sig}"


def test_position_signal_gptq_metadata():
    """GPTQ adds .qweight, .qzeros, .scales, .g_idx (and sometimes .q_perm/.invperm)
    on every standard linear. None should flag."""
    for name in [
        "model.layers.0.self_attn.q_proj.qweight",
        "model.layers.0.self_attn.q_proj.qzeros",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.q_proj.g_idx",
        "model.layers.0.mlp.gate_proj.qweight",
        "model.layers.0.mlp.gate_proj.q_perm",
        "lm_head.qweight",
        "model.embed_tokens.qweight",
    ]:
        sig, _ = _position_signal(name)
        assert sig == 0.0, f"expected position_signal=0 for GPTQ {name!r}, got {sig}"


def test_position_signal_awq_metadata():
    """AWQ uses .qweight, .qzeros, .scales, .qbias. None should flag."""
    for name in [
        "model.layers.0.self_attn.q_proj.qweight",
        "model.layers.0.self_attn.q_proj.qzeros",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.q_proj.qbias",
        "model.layers.0.mlp.up_proj.qweight",
    ]:
        sig, _ = _position_signal(name)
        assert sig == 0.0, f"expected position_signal=0 for AWQ {name!r}, got {sig}"


def test_position_signal_bitsandbytes_metadata():
    """bitsandbytes 4-bit: .weight + .absmax/.quant_map/.weight_format/.quant_state/.nested_*.
    bitsandbytes 8-bit: .weight + .SCB / .CB."""
    for name in [
        # 4-bit NF4
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.absmax",
        "model.layers.0.self_attn.q_proj.quant_map",
        "model.layers.0.self_attn.q_proj.nested_absmax",
        "model.layers.0.self_attn.q_proj.nested_quant_map",
        "model.layers.0.self_attn.q_proj.weight_format",
        "model.layers.0.self_attn.q_proj.quant_state",
        # 8-bit
        "model.layers.0.mlp.gate_proj.SCB",
        "model.layers.0.mlp.gate_proj.CB",
    ]:
        sig, _ = _position_signal(name)
        assert sig == 0.0, f"expected position_signal=0 for bnb {name!r}, got {sig}"


def test_position_signal_mistral_w123():
    """Mistral mlp.w1/w2/w3 should score 0."""
    for name in ["model.layers.0.mlp.w1.weight", "model.layers.0.mlp.w2.weight",
                 "model.layers.0.mlp.w3.weight"]:
        assert _position_signal(name)[0] == 0.0


# ---- Shape-signal tests -----------------------------------------------------

def test_shape_signal_bottleneck_rect():
    """Standard adapter rectangles → high signal."""
    assert _shape_signal([8, 4096])[0] == 1.0
    assert _shape_signal([4096, 8])[0] == 1.0
    assert _shape_signal([16, 4096])[0] == 1.0
    assert _shape_signal([32, 4096])[0] == 0.85
    assert _shape_signal([1024, 4096])[0] == 0.0   # Llama k/v_proj — not bottleneck


def test_shape_signal_square_or_normal_rect():
    """Standard transformer matrices (square or wide) → 0."""
    assert _shape_signal([4096, 4096])[0] == 0.0
    assert _shape_signal([14336, 4096])[0] == 0.0
    assert _shape_signal([128384, 4096])[0] == 0.0


def test_shape_signal_1d_vector_low():
    """1D vectors get a small signal (could be IA³)."""
    sig, cls = _shape_signal([4096])
    assert 0.0 < sig < 0.5
    assert cls == "vector_1d"


def test_shape_signal_soft_prompt():
    """Small (n_tokens, d) shapes look like soft prompts.

    Note: a (20, 4096) shape is geometrically also a rank-20 bottleneck;
    the bottleneck rule fires first because it's more specific. Both
    classes flag the same suspicious geometry — what matters is that
    something fires."""
    sig, cls = _shape_signal([20, 4096])
    assert sig >= 0.5
    assert cls in {"soft_prompt_shape", "bottleneck_rect"}
    # Pure soft-prompt shape (low aspect ratio): n_tokens × hidden where
    # max_dim < 8 * min_dim, so bottleneck rule won't fire.
    sig, cls = _shape_signal([50, 256])
    assert sig >= 0.5
    assert cls == "soft_prompt_shape"


# ---- Combined per-tensor tests ----------------------------------------------

def test_score_tensor_clean_layernorm():
    """A standard layernorm: 1D vector at standard position → 0 suspicion.
    The shape-signal kicker for vector_1d is suppressed when the position is standard."""
    t = _score_tensor("model.layers.0.input_layernorm.weight", [4096])
    assert t.suspicion_score == 0.0


def test_score_tensor_adapter_wenc():
    """W_enc.weight: top-level extra position + LoRA-pair name + bottleneck shape — should hit 1.0."""
    t = _score_tensor("W_enc.weight", [8, 4096])
    assert t.suspicion_score >= 0.95


def test_score_tensor_lora_a():
    """LoRA A inserted into a layer: layout violation + LoRA name + bottleneck shape."""
    t = _score_tensor("model.layers.5.self_attn.q_proj.lora_A.weight", [8, 4096])
    assert t.suspicion_score >= 0.95
    assert t.payload_class == "lora_a"
    assert t.position_class == "between_blocks_insertion"


def test_score_tensor_clean_q_proj():
    """A clean q_proj: standard name, standard position, square-ish shape → 0 suspicion."""
    t = _score_tensor("model.layers.0.self_attn.q_proj.weight", [4096, 4096])
    assert t.suspicion_score == 0.0


# ---- End-to-end fingerprint tests ------------------------------------------

def test_fingerprint_on_synthetic_adapter(tmp_path):
    """A 3-tensor synthetic adapter should produce aggregate suspicion ~ 1.0."""
    rng = np.random.default_rng(42)
    _write_safetensors_f32(tmp_path / "adapter.safetensors", {
        "W_enc.weight": rng.standard_normal((8, 4096)),
        "W_dec.weight": rng.standard_normal((4096, 8)),
        "gate.weight": rng.standard_normal((1, 4096)),
    })
    fp = compute_payload_shape_fingerprint(tmp_path)
    assert fp.aggregate_suspicion >= 0.95
    assert fp.n_tensors_flagged == 3
    assert fp.n_tensors_high_suspicion == 3
    assert "wenc_wdec_pair" in fp.payload_classes


def test_fingerprint_on_synthetic_clean_llama(tmp_path):
    """A synthetic mini-Llama should produce 0 flags."""
    rng = np.random.default_rng(0)
    _write_safetensors_f32(tmp_path / "model.safetensors", {
        # Two layers' worth of clean Llama tensors
        "model.embed_tokens.weight": rng.standard_normal((1024, 256)),
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((256, 256)),
        "model.layers.0.self_attn.k_proj.weight": rng.standard_normal((64, 256)),
        "model.layers.0.self_attn.v_proj.weight": rng.standard_normal((64, 256)),
        "model.layers.0.self_attn.o_proj.weight": rng.standard_normal((256, 256)),
        "model.layers.0.mlp.gate_proj.weight": rng.standard_normal((512, 256)),
        "model.layers.0.mlp.up_proj.weight": rng.standard_normal((512, 256)),
        "model.layers.0.mlp.down_proj.weight": rng.standard_normal((256, 512)),
        "model.layers.0.input_layernorm.weight": rng.standard_normal((256,)),
        "model.layers.0.post_attention_layernorm.weight": rng.standard_normal((256,)),
        "model.norm.weight": rng.standard_normal((256,)),
        "lm_head.weight": rng.standard_normal((1024, 256)),
    })
    fp = compute_payload_shape_fingerprint(tmp_path)
    assert fp.aggregate_suspicion == 0.0
    assert fp.n_tensors_flagged == 0


def test_fingerprint_on_lora_inserted_into_clean_llama(tmp_path):
    """Insert LoRA pair into a clean Llama directory — should flag exactly the LoRA tensors."""
    rng = np.random.default_rng(0)
    _write_safetensors_f32(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": rng.standard_normal((1024, 256)),
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((256, 256)),
        "model.layers.0.self_attn.q_proj.lora_A.weight": rng.standard_normal((8, 256)),
        "model.layers.0.self_attn.q_proj.lora_B.weight": rng.standard_normal((256, 8)),
        "model.layers.0.input_layernorm.weight": rng.standard_normal((256,)),
        "model.norm.weight": rng.standard_normal((256,)),
    })
    fp = compute_payload_shape_fingerprint(tmp_path)
    assert fp.n_tensors_flagged == 2  # only the lora pair
    assert fp.n_tensors_high_suspicion == 2
    assert fp.aggregate_suspicion >= 0.95
    flagged_names = {t.name for t in fp.per_tensor if t.suspicion_score >= 0.5}
    assert "model.layers.0.self_attn.q_proj.lora_A.weight" in flagged_names
    assert "model.layers.0.self_attn.q_proj.lora_B.weight" in flagged_names
