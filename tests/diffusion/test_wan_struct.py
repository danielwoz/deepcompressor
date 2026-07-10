# -*- coding: utf-8 -*-
"""CPU unit tests for the Wan 2.1 model struct."""

import pytest
import torch
import torch.nn as nn
from diffusers import WanTransformer3DModel

from deepcompressor.app.diffusion.nn.struct import (
    DiffusionModelStruct,
    DiffusionTransformerBlockStruct,
    WanAttentionStruct,
    WanStruct,
)


@pytest.fixture(scope="module")
def wan_model() -> WanTransformer3DModel:
    torch.manual_seed(0)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=16,
        out_channels=16,
        text_dim=32,
        freq_dim=256,
        ffn_dim=64,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32,
    )


@pytest.fixture(scope="module")
def wan_struct(wan_model) -> WanStruct:
    return DiffusionModelStruct.construct(wan_model)


def test_wan_struct_dispatch(wan_struct, wan_model):
    assert isinstance(wan_struct, WanStruct)
    assert wan_struct.module is wan_model
    assert wan_struct.num_blocks == 2


def test_wan_struct_block_layout(wan_struct):
    for block in wan_struct.block_structs:
        assert isinstance(block, DiffusionTransformerBlockStruct)
        assert not block.parallel
        assert block.norm_type == "ada_norm_single"
        assert len(block.attn_structs) == 2
        self_attn, cross_attn = block.attn_structs
        assert isinstance(self_attn, WanAttentionStruct)
        assert self_attn.is_self_attn() and self_attn.config.with_rope
        assert isinstance(cross_attn, WanAttentionStruct)
        assert cross_attn.is_cross_attn() and not cross_attn.config.with_rope
        assert block.ffn_struct is not None


def test_wan_struct_key_modules(wan_struct):
    named = list(wan_struct.named_key_modules())
    # 10 Linears per block: attn1 q/k/v/o, attn2 q/k/v/o, ffn up/down;
    # plus condition_embedder Linears (time_embed tier) and proj_out
    per_block = {}
    embed_keys, output_keys = [], []
    for key, name, module, _, _ in named:
        assert isinstance(module, (nn.Linear, nn.Conv2d))
        if name.startswith("blocks."):
            per_block.setdefault(name.split(".")[1], []).append((key, name))
        elif key == "time_embed":
            embed_keys.append(name)
        else:
            output_keys.append((key, name))
    assert set(per_block.keys()) == {"0", "1"}
    for block_idx, entries in per_block.items():
        assert len(entries) == 10, f"block {block_idx} has {len(entries)} key modules"
        keys = sorted(key for key, _ in entries)
        assert keys == sorted(
            [
                "attn_qkv_proj",  # attn1 q
                "attn_qkv_proj",  # attn1 k
                "attn_qkv_proj",  # attn1 v
                "attn_out_proj",  # attn1 out
                "attn_qkv_proj",  # attn2 q
                "attn_add_qkv_proj",  # attn2 k (text)
                "attn_add_qkv_proj",  # attn2 v (text)
                "attn_out_proj",  # attn2 out
                "ffn_up_proj",
                "ffn_down_proj",
            ]
        )
    # condition_embedder holds 5 Linears (time_embedder x2, time_proj, text_embedder x2)
    assert len(embed_keys) == 5
    assert all(name.startswith("condition_embedder.") for name in embed_keys)
    # proj_out is the only other Linear, categorized as output_embed
    assert output_keys == [("output_embed", "proj_out")]


def test_wan_struct_rnames(wan_struct):
    block = wan_struct.block_structs[0]
    self_attn, cross_attn = block.attn_structs
    assert self_attn.q_proj_name == "blocks.0.attn1.to_q"
    assert self_attn.o_proj_name == "blocks.0.attn1.to_out.0"
    assert cross_attn.q_proj_name == "blocks.0.attn2.to_q"
    assert cross_attn.add_k_proj_name == "blocks.0.attn2.to_k"
    assert cross_attn.add_v_proj_name == "blocks.0.attn2.to_v"
    assert cross_attn.o_proj_name == "blocks.0.attn2.to_out.0"
    assert block.ffn_struct.up_proj_name == "blocks.0.ffn.net.0.proj"
    assert block.ffn_struct.down_proj_name == "blocks.0.ffn.net.2"


def test_wan_struct_key_map_validates_config_skips(wan_struct):
    key_map = DiffusionModelStruct._get_default_key_map()
    skips = ["embed", "transformer_proj_in", "transformer_proj_out", "transformer_norm", "transformer_add_norm"]
    simplified = DiffusionModelStruct._simplify_keys(skips, key_map=key_map)
    assert "embed" in simplified
    # every rkey the Wan struct emits must be recognized by the key map
    emitted = {key for key, *_ in wan_struct.named_key_modules()}
    for key in emitted:
        assert key in key_map, f"unrecognized rkey {key}"


def test_wan_attention_filter_kwargs(wan_struct):
    block = wan_struct.block_structs[0]
    self_attn, cross_attn = block.attn_structs
    rotary = (torch.ones(1, 8, 1, 16), torch.zeros(1, 8, 1, 16))
    kwargs = {"rotary_emb": rotary}
    assert self_attn.filter_kwargs(kwargs) == {"rotary_emb": rotary}
    assert cross_attn.filter_kwargs(kwargs) == {}


def test_wan_iter_block_activations_args(wan_struct):
    layers, structs, recomputes, uses = wan_struct._get_iter_block_activations_args()
    assert len(layers) == 2
    assert recomputes == [False, False]
    assert uses == [False, True]


def test_wan_block_replay_inputs(wan_model):
    from deepcompressor.app.diffusion.dataset.calib import DiffusionCalibCacheLoader

    block = wan_model.blocks[0]
    dim = wan_model.config.num_attention_heads * wan_model.config.attention_head_dim
    hidden = torch.randn(1, 8, dim)
    encoder = torch.randn(1, 4, dim)
    temb = torch.randn(1, 6, dim)
    rotary = (torch.ones(1, 8, 1, 16), torch.zeros(1, 8, 1, 16))
    inputs = DiffusionCalibCacheLoader._convert_layer_inputs(None, block, (hidden, encoder, temb, rotary), {})
    assert len(inputs.args) == 1
    assert set(inputs.kwargs.keys()) == {"encoder_hidden_states", "temb", "rotary_emb"}
    # replaying the block with the converted inputs must reproduce the direct call
    out_ref = block(hidden, encoder, temb, rotary)
    out_replay = block(hidden, **inputs.kwargs)
    assert torch.equal(out_ref, out_replay)
    assert isinstance(out_ref, torch.Tensor)
