# -*- coding: utf-8 -*-
"""CPU unit tests for the Wan 2.1 -> nunchaku checkpoint converter."""

import pytest
import torch
from diffusers import WanTransformer3DModel

from deepcompressor.backend.nunchaku.convert import (
    build_wan_metadata,
    convert_to_nunchaku_wan_state_dict,
)

RANK = 32


def _fake_quantize_int4(weight: torch.Tensor, group_size: int = 64):
    oc, ic = weight.shape
    grouped = weight.to(torch.float32).view(oc, ic // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
    quantized = (grouped / scale).round_().clamp_(-8, 7).mul_(scale).view(oc, ic)
    return quantized.to(weight.dtype), scale.view(oc, 1, ic // group_size, 1).to(weight.dtype)


def _fake_quantize_fp4(weight: torch.Tensor, group_size: int = 16):
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    oc, ic = weight.shape
    w = weight.to(torch.float32)
    tensor_scale = w.abs().amax().clamp(min=1e-6) / 6.0
    w = w / tensor_scale
    grouped = w.view(oc, ic // group_size, group_size)
    group_scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 6.0
    q = grouped / group_scale
    dist = (q.unsqueeze(-1).abs() - grid.view(1, 1, 1, -1)).abs()
    q = grid[dist.argmin(dim=-1)] * q.sign()
    quantized = (q * group_scale).view(oc, ic) * tensor_scale
    return (
        quantized.to(weight.dtype),
        tensor_scale.view(1, 1, 1, 1).to(weight.dtype),
        group_scale.view(oc, 1, ic // group_size, 1).to(weight.dtype),
    )


def make_wan_model() -> WanTransformer3DModel:
    torch.manual_seed(0)
    # feature dims must be multiples of 128 for the nunchaku weight packer
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=64,
        in_channels=16,
        out_channels=16,
        text_dim=32,
        freq_dim=256,
        ffn_dim=256,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32,
    )


QUANTIZED_LOCAL_NAMES = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
]

SMOOTH_ANCHORS = [
    "attn1.to_q",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
]


def make_fake_quant_checkpoint(model: WanTransformer3DModel, float_point: bool):
    """Emulate the deepcompressor `--save-model` payload for a Wan model."""
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    scale_dict, smooth_dict, branch_dict = {}, {}, {}
    for block_idx in range(len(model.blocks)):
        block_name = f"blocks.{block_idx}"
        for local_name in QUANTIZED_LOCAL_NAMES:
            name = f"{block_name}.{local_name}"
            weight = state_dict[f"{name}.weight"]
            if float_point:
                quantized, tensor_scale, group_scale = _fake_quantize_fp4(weight)
                scale_dict[f"{name}.weight.scale.0"] = tensor_scale
                scale_dict[f"{name}.weight.scale.1"] = group_scale
            else:
                quantized, group_scale = _fake_quantize_int4(weight)
                scale_dict[f"{name}.weight.scale.0"] = group_scale
            state_dict[f"{name}.weight"] = quantized
        for anchor in SMOOTH_ANCHORS:
            name = f"{block_name}.{anchor}"
            in_features = state_dict[f"{name}.weight"].shape[1]
            smooth_dict[name] = torch.rand(in_features, dtype=torch.bfloat16) * 0.5 + 0.75
            # shared branches span the whole fused group (q+k+v / k+v outputs)
            out_features_map = {
                "attn1.to_q": 3 * state_dict[f"{block_name}.attn1.to_q.weight"].shape[0],
                "attn2.to_k": 2 * state_dict[f"{block_name}.attn2.to_k.weight"].shape[0],
            }
            out_features = out_features_map.get(anchor, state_dict[f"{name}.weight"].shape[0])
            branch_dict[name] = {
                "a.weight": torch.randn(RANK, in_features, dtype=torch.bfloat16) * 0.01,
                "b.weight": torch.randn(out_features, RANK, dtype=torch.bfloat16) * 0.01,
            }
    smooth_dict["proj.fuse_when_possible"] = False
    return state_dict, scale_dict, smooth_dict, branch_dict


@pytest.mark.parametrize("float_point", [False, True], ids=["int4", "fp4"])
def test_wan_convert_schema(float_point):
    model = make_wan_model().to(torch.bfloat16)
    state_dict, scale_dict, smooth_dict, branch_dict = make_fake_quant_checkpoint(model, float_point)
    converted = convert_to_nunchaku_wan_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        float_point=float_point,
    )
    dim = 128
    for block_idx in range(2):
        prefix = f"blocks.{block_idx}"
        for local_name, in_features, out_features in [
            ("attn1.to_qkv", dim, 3 * dim),
            ("attn1.to_out.0", dim, dim),
            ("attn2.to_q", dim, dim),
            ("attn2.to_kv", dim, 2 * dim),
            ("attn2.to_out.0", dim, dim),
            ("ffn.net.0.proj", dim, 256),
            ("ffn.net.2", 256, dim),
        ]:
            name = f"{prefix}.{local_name}"
            group_size = 16 if float_point else 64
            assert converted[f"{name}.qweight"].dtype == torch.int8
            assert converted[f"{name}.qweight"].shape == (out_features, in_features // 2)
            assert converted[f"{name}.wscales"].shape == (in_features // group_size, out_features)
            if float_point:
                assert converted[f"{name}.wscales"].dtype == torch.float8_e4m3fn
                if local_name in ("attn1.to_qkv", "attn2.to_kv"):
                    # fused groups carry the per-member tensor scales as channel scales
                    assert converted[f"{name}.wcscales"].shape == (out_features,)
                    assert f"{name}.wtscale" not in converted
                else:
                    assert converted[f"{name}.wtscale"].numel() == 1
                    assert f"{name}.wcscales" not in converted
            else:
                assert converted[f"{name}.wscales"].dtype == torch.bfloat16
                assert f"{name}.wtscale" not in converted
            assert converted[f"{name}.bias"].shape == (out_features,)
            assert converted[f"{name}.smooth_factor"].shape == (in_features,)
            assert converted[f"{name}.smooth_factor_orig"].shape == (in_features,)
            assert converted[f"{name}.proj_down"].shape == (in_features, RANK)
            assert converted[f"{name}.proj_up"].shape == (out_features, RANK)
        # unquantized per-block tensors pass through under their diffusers names
        assert f"{prefix}.scale_shift_table" in converted
        assert f"{prefix}.norm2.weight" in converted
        assert f"{prefix}.attn1.norm_q.weight" in converted
        assert f"{prefix}.attn2.norm_k.weight" in converted
        # the raw q/k/v weights must have been consumed
        assert f"{prefix}.attn1.to_q.weight" not in converted
    # model-level passthrough
    assert "patch_embedding.weight" in converted
    assert "proj_out.weight" in converted
    assert "scale_shift_table" in converted
    assert any(k.startswith("condition_embedder.") for k in converted)

    metadata = build_wan_metadata(converted, {"num_layers": 2}, float_point)
    assert metadata["model_class"] == "NunchakuWanTransformer3DModel"
    import json

    quantization_config = json.loads(metadata["quantization_config"])
    assert quantization_config["weight"]["dtype"] == ("fp4_e2m1_all" if float_point else "int4")
    assert quantization_config["weight"]["group_size"] == (16 if float_point else 64)
    assert quantization_config["rank"] == RANK


def test_wan_convert_rejects_mismatched_precision():
    model = make_wan_model().to(torch.bfloat16)
    state_dict, scale_dict, smooth_dict, branch_dict = make_fake_quant_checkpoint(model, float_point=False)
    # decoding an int4 checkpoint as fp4 must fail the group-size check
    with pytest.raises(AssertionError):
        convert_to_nunchaku_wan_state_dict(
            state_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            float_point=True,
        )


def test_wan_convert_with_skips():
    model = make_wan_model().to(torch.bfloat16)
    orig_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    state_dict, scale_dict, smooth_dict, branch_dict = make_fake_quant_checkpoint(model, float_point=False)
    converted = convert_to_nunchaku_wan_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        float_point=False,
        skips=["attn1.to_out.0", "blocks.1.attn2.to_kv"],
        orig_state_dict=orig_state_dict,
    )
    for block_idx in range(2):
        prefix = f"blocks.{block_idx}"
        # globally skipped unit: original bf16 weights, no packed tensors
        assert f"{prefix}.attn1.to_out.0.qweight" not in converted
        assert torch.equal(converted[f"{prefix}.attn1.to_out.0.weight"], orig_state_dict[f"{prefix}.attn1.to_out.0.weight"])
        # non-skipped units unchanged
        assert f"{prefix}.attn1.to_qkv.qweight" in converted
    # per-block skip applies only to block 1
    assert "blocks.0.attn2.to_kv.qweight" in converted
    assert "blocks.1.attn2.to_kv.qweight" not in converted
    assert torch.equal(converted["blocks.1.attn2.to_k.weight"], orig_state_dict["blocks.1.attn2.to_k.weight"])
    assert torch.equal(converted["blocks.1.attn2.to_v.weight"], orig_state_dict["blocks.1.attn2.to_v.weight"])
    metadata = build_wan_metadata(converted, {"num_layers": 2}, False, skips=["attn1.to_out.0", "blocks.1.attn2.to_kv"])
    import json

    assert json.loads(metadata["quantization_config"])["skips"] == ["attn1.to_out.0", "blocks.1.attn2.to_kv"]
