"""Converts a DeepCompressor state dict to a Nunchaku state dict."""

import argparse
import json
import os

import safetensors.torch
import torch
import tqdm
import yaml

from .utils import convert_to_nunchaku_w4x4y16_linear_weight, convert_to_nunchaku_w4x16_linear_weight


def convert_to_nunchaku_w4x4y16_linear_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
    shift: torch.Tensor | None = None,
    smooth_fused: bool = False,
    float_point: bool = False,
    subscale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if weight.ndim > 2:  # pointwise conv
        assert weight.numel() == weight.shape[0] * weight.shape[1]
        weight = weight.view(weight.shape[0], weight.shape[1])
    if scale.numel() > 1:
        assert scale.ndim == weight.ndim * 2
        assert scale.numel() == scale.shape[0] * scale.shape[2]
        scale = scale.view(scale.shape[0], 1, scale.shape[2], 1)
        scale_key = "wcscales" if scale.shape[2] == 1 else "wscales"
    else:
        scale_key = "wtscale"
    if subscale is None:
        subscale_key = ""
    else:
        assert subscale.ndim == weight.ndim * 2
        assert subscale.numel() == subscale.shape[0] * subscale.shape[2]
        assert subscale.numel() > 1
        subscale = subscale.view(subscale.shape[0], 1, subscale.shape[2], 1)
        subscale_key = "wcscales" if subscale.shape[2] == 1 else "wscales"
    if lora is not None and (smooth is not None or shift is not None):
        # unsmooth lora down projection
        dtype = weight.dtype
        lora_down, lora_up = lora
        lora_down = lora_down.to(dtype=torch.float64)
        if smooth is not None and not smooth_fused:
            lora_down = lora_down.div_(smooth.to(torch.float64).unsqueeze(0))
        if shift is not None:
            bias = torch.zeros([lora_up.shape[0]], dtype=torch.float64) if bias is None else bias.to(torch.float64)
            if shift.numel() == 1:
                shift = shift.view(1, 1).expand(lora_down.shape[1], 1).to(torch.float64)
            else:
                shift = shift.view(-1, 1).to(torch.float64)
            bias = bias.add_((lora_up.to(dtype=torch.float64) @ lora_down @ shift).view(-1))
            bias = bias.to(dtype=dtype)
        lora = (lora_down.to(dtype=dtype), lora_up)
    weight, scale, bias, smooth, lora, subscale = convert_to_nunchaku_w4x4y16_linear_weight(
        weight, scale=scale, bias=bias, smooth=smooth, lora=lora, float_point=float_point, subscale=subscale
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict["qweight"] = weight
    state_dict[scale_key] = scale
    if subscale is not None:
        state_dict[subscale_key] = subscale
    state_dict["bias"] = bias
    state_dict["smooth_orig"] = smooth
    state_dict["smooth"] = torch.ones_like(smooth) if smooth_fused else smooth.clone()
    if lora is not None:
        state_dict["lora_down"] = lora[0]
        state_dict["lora_up"] = lora[1]
    return state_dict


def convert_to_nunchaku_w4x16_adanorm_single_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weight, scale, zero, bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale=scale, bias=bias, adanorm_splits=3
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict = {}
    state_dict["qweight"] = weight
    state_dict["wscales"] = scale
    state_dict["wzeros"] = zero
    state_dict["bias"] = bias
    return state_dict


def convert_to_nunchaku_w4x16_adanorm_zero_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weight, scale, zero, bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale=scale, bias=bias, adanorm_splits=6
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict = {}
    state_dict["qweight"] = weight
    state_dict["wscales"] = scale
    state_dict["wzeros"] = zero
    state_dict["bias"] = bias
    return state_dict


def update_state_dict(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor], prefix: str = ""
) -> dict[str, torch.Tensor]:
    for rkey, value in rhs.items():
        lkey = f"{prefix}.{rkey}" if prefix else rkey
        assert lkey not in lhs, f"Key {lkey} already exists in the state dict."
        lhs[lkey] = value
    return lhs


def convert_to_nunchaku_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    local_name_map: dict[str, str | list[str]],
    smooth_name_map: dict[str, str],
    branch_name_map: dict[str, str],
    convert_map: dict[str, str],
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    print(f"Converting block {block_name}...")
    converted: dict[str, torch.Tensor] = {}
    candidates: dict[str, torch.Tensor] = {
        param_name: param for param_name, param in state_dict.items() if param_name.startswith(block_name)
    }
    for converted_local_name, candidate_local_names in tqdm.tqdm(
        local_name_map.items(), desc=f"Converting {block_name}", dynamic_ncols=True
    ):
        if isinstance(candidate_local_names, str):
            candidate_local_names = [candidate_local_names]
        candidate_names = [f"{block_name}.{candidate_local_name}" for candidate_local_name in candidate_local_names]
        weight = [candidates[f"{candidate_name}.weight"] for candidate_name in candidate_names]
        bias = [candidates.get(f"{candidate_name}.bias", None) for candidate_name in candidate_names]
        scale = [scale_dict.get(f"{candidate_name}.weight.scale.0", None) for candidate_name in candidate_names]
        subscale = [scale_dict.get(f"{candidate_name}.weight.scale.1", None) for candidate_name in candidate_names]
        if len(weight) > 1:
            bias = None if all(b is None for b in bias) else torch.concat(bias, dim=0)
            if all(s is None for s in scale):
                scale = None
            else:
                if scale[0].numel() == 1:  # switch from per-tensor to per-channel scale
                    assert all(s.numel() == 1 for s in scale)
                    scale = torch.concat(
                        [
                            s.view(-1).expand(weight[i].shape[0]).reshape(weight[i].shape[0], 1, 1, 1)
                            for i, s in enumerate(scale)
                        ],
                        dim=0,
                    )
                else:
                    scale = torch.concat(scale, dim=0)
            subscale = None if all(s is None for s in subscale) else torch.concat(subscale, dim=0)
            weight = torch.concat(weight, dim=0)
        else:
            weight, bias, scale, subscale = weight[0], bias[0], scale[0], subscale[0]
        smooth = smooth_dict.get(f"{block_name}.{smooth_name_map.get(converted_local_name, '')}", None)
        branch = branch_dict.get(f"{block_name}.{branch_name_map.get(converted_local_name, '')}", None)
        if branch is not None:
            branch = (branch["a.weight"], branch["b.weight"])
        if scale is None:
            assert smooth is None and branch is None and subscale is None
            print(f"  - Copying {block_name} weights of {candidate_local_names} as {converted_local_name}.weight")
            converted[f"{converted_local_name}.weight"] = weight.clone().cpu()
            if bias is not None:
                print(f"  - Copying {block_name} biases of {candidate_local_names} as {converted_local_name}.bias")
                converted[f"{converted_local_name}.bias"] = bias.clone().cpu()
            continue
        if convert_map[converted_local_name] == "adanorm_single":
            print(f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}.")
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x16_adanorm_single_state_dict(weight=weight, scale=scale, bias=bias),
                prefix=converted_local_name,
            )
        elif convert_map[converted_local_name] == "adanorm_zero":
            print(f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}.")
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x16_adanorm_zero_state_dict(weight=weight, scale=scale, bias=bias),
                prefix=converted_local_name,
            )
        elif convert_map[converted_local_name] == "linear":
            smooth_fused = "out_proj" in converted_local_name and smooth_dict.get("proj.fuse_when_possible", True)
            shift = [candidates.get(f"{candidate_name[:-7]}.shift", None) for candidate_name in candidate_names]
            assert all(s == shift[0] for s in shift)
            shift = shift[0]
            print(
                f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}."
                f" (smooth_fused={smooth_fused}, shifted={shift is not None}, float_point={float_point})"
            )
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x4y16_linear_state_dict(
                    weight=weight,
                    scale=scale,
                    bias=bias,
                    smooth=smooth,
                    lora=branch,
                    shift=shift,
                    smooth_fused=smooth_fused,
                    float_point=float_point,
                    subscale=subscale,
                ),
                prefix=converted_local_name,
            )
        else:
            raise NotImplementedError(f"Conversion of {convert_map[converted_local_name]} is not implemented.")
    return converted


def convert_to_nunchaku_flux_single_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    down_proj_local_name = "proj_out.linears.1.linear"
    if f"{block_name}.{down_proj_local_name}.weight" not in state_dict:
        down_proj_local_name = "proj_out.linears.1"
        assert f"{block_name}.{down_proj_local_name}.weight" in state_dict

    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            "norm.linear": "norm.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": down_proj_local_name,
        },
        smooth_name_map={
            "qkv_proj": "attn.to_q",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "attn.to_q",
            "mlp_fc2": down_proj_local_name,
        },
        branch_name_map={
            "qkv_proj": "attn.to_q",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": down_proj_local_name,
        },
        convert_map={
            "norm.linear": "adanorm_single",
            "qkv_proj": "linear",
            "out_proj": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_flux_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    down_proj_local_name = "ff.net.2.linear"
    if f"{block_name}.{down_proj_local_name}.weight" not in state_dict:
        down_proj_local_name = "ff.net.2"
        assert f"{block_name}.{down_proj_local_name}.weight" in state_dict
    context_down_proj_local_name = "ff_context.net.2.linear"
    if f"{block_name}.{context_down_proj_local_name}.weight" not in state_dict:
        context_down_proj_local_name = "ff_context.net.2"
        assert f"{block_name}.{context_down_proj_local_name}.weight" in state_dict

    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            "norm1.linear": "norm1.linear",
            "norm1_context.linear": "norm1_context.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "qkv_proj_context": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "norm_added_q": "attn.norm_added_q",
            "norm_added_k": "attn.norm_added_k",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_add_out",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        smooth_name_map={
            "qkv_proj": "attn.to_q",
            "qkv_proj_context": "attn.add_k_proj",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_out.0",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        branch_name_map={
            "qkv_proj": "attn.to_q",
            "qkv_proj_context": "attn.add_k_proj",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_add_out",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        convert_map={
            "norm1.linear": "adanorm_zero",
            "norm1_context.linear": "adanorm_zero",
            "qkv_proj": "linear",
            "qkv_proj_context": "linear",
            "out_proj": "linear",
            "out_proj_context": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
            "mlp_context_fc1": "linear",
            "mlp_context_fc2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_flux_state_dicts(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    block_names: set[str] = set()
    other: dict[str, torch.Tensor] = {}
    for param_name in state_dict.keys():
        if param_name.startswith(("transformer_blocks.", "single_transformer_blocks.")):
            block_names.add(".".join(param_name.split(".")[:2]))
        else:
            other[param_name] = state_dict[param_name]
    block_names = sorted(block_names, key=lambda x: (x.split(".")[0], int(x.split(".")[-1])))
    print(f"Converting {len(block_names)} transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        convert_fn = convert_to_nunchaku_flux_single_transformer_block_state_dict
        if block_name.startswith("transformer_blocks"):
            convert_fn = convert_to_nunchaku_flux_transformer_block_state_dict
        update_state_dict(
            converted,
            convert_fn(
                state_dict=state_dict,
                scale_dict=scale_dict,
                smooth_dict=smooth_dict,
                branch_dict=branch_dict,
                block_name=block_name,
                float_point=float_point,
            ),
            prefix=block_name,
        )
    return converted, other


# region Wan 2.1

WAN_QUANTIZED_LOCAL_NAMES = (
    "attn1.to_qkv",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_kv",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)

# fp4-e2m1 magnitude grid used to validate float-point round-trips
WAN_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def derive_wan_float_point(quant_path: str, scale_dict: dict[str, torch.Tensor]) -> bool:
    """Derive the 4-bit encoding from the quantization config, never from a CLI flag.

    A sint4-encoded checkpoint decoded as FP4-e2m1 (or vice versa) loads without
    errors and produces unstructured garbage, so the encoding is derived from the
    run's archived config and cross-checked against the scale structure.
    """
    config_dtype = None
    for dirpath in (quant_path, os.path.dirname(quant_path.rstrip(os.sep))):
        config_path = os.path.join(dirpath, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
            config_dtype = config.get("quant", {}).get("wgts", {}).get("dtype", None)
            break
    has_subscale = any(name.endswith(".weight.scale.1") for name in scale_dict.keys())
    if config_dtype is not None:
        float_point = "fp4" in str(config_dtype)
        assert float_point == has_subscale, (
            f"weight dtype {config_dtype} from {config_path} does not match the scale structure "
            f"(two-level scales present: {has_subscale})"
        )
        print(f"Derived float_point={float_point} from weight dtype {config_dtype} in {config_path}.")
    else:
        float_point = has_subscale
        print(f"No archived config found; derived float_point={float_point} from the scale structure.")
    return float_point


def check_wan_group_sizes(scale_dict: dict[str, torch.Tensor], state_dict: dict[str, torch.Tensor], float_point: bool):
    """Assert that group sizes match the target precision (16 for fp4, 64 for int4)."""
    group_size = 16 if float_point else 64
    scale_level = ".weight.scale.1" if float_point else ".weight.scale.0"
    num_checked = 0
    for name, scale in scale_dict.items():
        if not name.endswith(scale_level) or not name.startswith("blocks."):
            continue
        weight = state_dict[name.replace(scale_level, ".weight")]
        num_groups = scale.numel() // weight.shape[0]
        assert weight.shape[1] == num_groups * group_size, (
            f"{name}: expected group size {group_size} "
            f"(in_features={weight.shape[1]}, num_groups={num_groups})"
        )
        num_checked += 1
    assert num_checked > 0, "no quantized layer scales found"
    print(f"Group size check passed for {num_checked} layers (group_size={group_size}).")


def check_wan_zero_wscales(scale_dict: dict[str, torch.Tensor], float_point: bool):
    """Scan for group scales that round to zero in fp8-e4m3 (killing whole output groups)."""
    if not float_point:
        return
    num_zeros, num_total = 0, 0
    for name, scale in scale_dict.items():
        if name.endswith(".weight.scale.1") and name.startswith("blocks."):
            fp8_scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
            num_zeros += int((fp8_scale == 0).sum())
            num_total += fp8_scale.numel()
    assert num_zeros == 0, f"{num_zeros}/{num_total} group scales round to zero in fp8-e4m3"
    print(f"Zero-wscales check passed for {num_total} group scales.")


def check_wan_weight_round_trip(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    block_name: str,
    candidate_names: list[str],
    float_point: bool,
):
    """Verify each weight lands on the 4-bit grid implied by its scales.

    `model.pt` stores fake-quantized weights, so dividing by the calibrated
    scales must land (near) exactly on representable 4-bit values; any distance
    from the grid indicates a scale/encoding mismatch.
    """
    for candidate_name in candidate_names:
        weight = state_dict[f"{candidate_name}.weight"].to(torch.float32)
        scale = scale_dict[f"{candidate_name}.weight.scale.0"].to(torch.float32)
        subscale = scale_dict.get(f"{candidate_name}.weight.scale.1", None)
        oc, ic = weight.shape
        if scale.numel() == 1:
            q = weight / scale.view(1, 1)
        else:
            num_groups = scale.numel() // oc
            q = (weight.view(oc, num_groups, -1) / scale.view(oc, num_groups, 1)).view(oc, ic)
        if subscale is not None:
            subscale = subscale.to(torch.float32)
            num_groups = subscale.numel() // oc
            q = (q.view(oc, num_groups, -1) / subscale.view(oc, num_groups, 1)).view(oc, ic)
        # thresholds allow for bf16 storage rounding (~0.4% relative) of the
        # fake-quantized weights; an encoding mismatch produces distances O(0.5+)
        if float_point:
            grid = torch.tensor([v for mag in WAN_E2M1_VALUES for v in (mag, -mag)], dtype=torch.float32)
            dist = (q.reshape(-1, 1) - grid.view(1, -1)).abs().min(dim=1).values
            assert q.abs().max() <= 6.0 * 1.02, f"{candidate_name}: quantized weight out of fp4 range"
        else:
            dist = (q - q.round()).abs().view(-1)
            assert q.min() >= -8 * 1.02 and q.max() <= 7 * 1.02, f"{candidate_name}: out of int4 range"
        max_dist = float(dist.max())
        assert max_dist < 0.1, f"{candidate_name}: weight is {max_dist} away from the 4-bit grid"


def is_wan_unit_skipped(block_name: str, unit: str, skips: list[str] | None) -> bool:
    """A unit is skipped when listed globally (`attn2.to_kv`) or per block (`blocks.5.attn2.to_kv`)."""
    return bool(skips) and (unit in skips or f"{block_name}.{unit}" in skips)


def convert_to_nunchaku_wan_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
    skips: list[str] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Convert one Wan transformer block; returns the converted dict and consumed source names.

    Units listed in `skips` stay unquantized: their ORIGINAL bf16 weights (from
    `orig_state_dict` — the calibrated `model.pt` weights are smoothed,
    branch-subtracted, and fake-quantized, so they must not be reused) pass
    through under the diffusers member names.
    """
    down_proj_local_name = "ffn.net.2.linear"
    if f"{block_name}.{down_proj_local_name}.weight" not in state_dict:
        down_proj_local_name = "ffn.net.2"
        assert f"{block_name}.{down_proj_local_name}.weight" in state_dict

    local_name_map = {
        "attn1.to_qkv": ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
        "attn1.to_out.0": "attn1.to_out.0",
        "attn2.to_q": "attn2.to_q",
        "attn2.to_kv": ["attn2.to_k", "attn2.to_v"],
        "attn2.to_out.0": "attn2.to_out.0",
        "ffn.net.0.proj": "ffn.net.0.proj",
        "ffn.net.2": down_proj_local_name,
    }
    skipped_units = {unit for unit in local_name_map if is_wan_unit_skipped(block_name, unit, skips)}
    skipped_converted: dict[str, torch.Tensor] = {}
    skipped_consumed: set[str] = set()
    for unit in skipped_units:
        assert orig_state_dict is not None, "--model-path original weights are required for skipped units"
        candidate_local_names = local_name_map.pop(unit)
        if isinstance(candidate_local_names, str):
            candidate_local_names = [candidate_local_names]
        for candidate_local_name in candidate_local_names:
            # restore under the unpatched diffusers name (strip any `.linear`)
            orig_local_name = candidate_local_name[: -len(".linear")] if candidate_local_name.endswith(
                ".linear"
            ) else candidate_local_name
            for suffix in ("weight", "bias"):
                orig_key = f"{block_name}.{orig_local_name}.{suffix}"
                skipped_converted[f"{orig_local_name}.{suffix}"] = orig_state_dict[orig_key].clone()
            candidate_name = f"{block_name}.{candidate_local_name}"
            skipped_consumed.update(
                {f"{candidate_name}.weight", f"{candidate_name}.bias"}
            )
            if candidate_local_name.endswith(".linear"):
                skipped_consumed.add(f"{block_name}.{orig_local_name}.shift")
    if skipped_units:
        print(f"  - Keeping {sorted(skipped_units)} of {block_name} unquantized (skip list)")
    # smooth scales and low-rank branches are anchored on the first member of
    # each fused group (the calibration cache key convention)
    smooth_name_map = {
        "attn1.to_qkv": "attn1.to_q",
        "attn1.to_out.0": "attn1.to_out.0",
        "attn2.to_q": "attn2.to_q",
        "attn2.to_kv": "attn2.to_k",
        "attn2.to_out.0": "attn2.to_out.0",
        "ffn.net.0.proj": "ffn.net.0.proj",
        "ffn.net.2": down_proj_local_name,
    }
    smooth_name_map = {k: v for k, v in smooth_name_map.items() if k in local_name_map}
    branch_name_map = dict(smooth_name_map)
    convert_map = {name: "linear" for name in local_name_map}

    candidate_names = []
    for candidate_local_names in local_name_map.values():
        if isinstance(candidate_local_names, str):
            candidate_local_names = [candidate_local_names]
        candidate_names.extend(f"{block_name}.{name}" for name in candidate_local_names)
    check_wan_weight_round_trip(state_dict, scale_dict, block_name, candidate_names, float_point)
    for converted_local_name, candidate_local_names in smooth_name_map.items():
        assert f"{block_name}.{candidate_local_names}" in smooth_dict, (
            f"missing smooth scale for {block_name}.{candidate_local_names}"
        )
    if branch_dict:
        for converted_local_name, candidate_local_names in branch_name_map.items():
            assert f"{block_name}.{candidate_local_names}" in branch_dict, (
                f"missing low-rank branch for {block_name}.{candidate_local_names}"
            )

    converted = convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map=local_name_map,
        smooth_name_map=smooth_name_map,
        branch_name_map=branch_name_map,
        convert_map=convert_map,
        float_point=float_point,
    )
    update_state_dict(converted, skipped_converted)
    consumed: set[str] = set(skipped_consumed)
    for candidate_name in candidate_names:
        consumed.add(f"{candidate_name}.weight")
        consumed.add(f"{candidate_name}.bias")
        if candidate_name.endswith(".linear"):
            consumed.add(f"{candidate_name[: -len('.linear')]}.shift")
    return converted, consumed


def convert_to_nunchaku_wan_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
    skips: list[str] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a Wan 2.1 quantization checkpoint to a single nunchaku state dict.

    Quantized linears are converted to packed `SVDQW4A4Linear` tensors keyed by
    the runtime parameter names (`smooth_factor`, `proj_down`, ...); everything
    else (norms, `condition_embedder`, `scale_shift_table`s, `patch_embedding`,
    `proj_out`, and any unquantized attention projections such as the I2V image
    K/V) passes through under its diffusers name.
    """
    check_wan_group_sizes(scale_dict, state_dict, float_point)
    check_wan_zero_wscales(scale_dict, float_point)

    block_names: set[str] = set()
    for param_name in state_dict.keys():
        if param_name.startswith("blocks."):
            block_names.add(".".join(param_name.split(".")[:2]))
    block_names = sorted(block_names, key=lambda x: int(x.split(".")[-1]))
    print(f"Converting {len(block_names)} transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for block_name in block_names:
        block_converted, block_consumed = convert_to_nunchaku_wan_transformer_block_state_dict(
            state_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            block_name=block_name,
            float_point=float_point,
            skips=skips,
            orig_state_dict=orig_state_dict,
        )
        update_state_dict(converted, block_converted, prefix=block_name)
        consumed.update(block_consumed)
    # rename the converter-style suffixes to the runtime parameter names
    renamed: dict[str, torch.Tensor] = {}
    for key, value in converted.items():
        if key.endswith(".lora_down"):
            key = key[: -len("lora_down")] + "proj_down"
        elif key.endswith(".lora_up"):
            key = key[: -len("lora_up")] + "proj_up"
        elif key.endswith(".smooth_orig"):
            key = key[: -len("smooth_orig")] + "smooth_factor_orig"
        elif key.endswith(".smooth"):
            key = key[: -len("smooth")] + "smooth_factor"
        renamed[key] = value.cpu().contiguous()
    # pass through every unconsumed tensor under its original name
    for param_name, param in state_dict.items():
        if param_name not in consumed:
            assert param_name not in renamed, f"key {param_name} conflicts with a converted tensor"
            renamed[param_name] = param.clone().cpu().contiguous()
    return renamed


def build_wan_metadata(
    converted_state_dict: dict[str, torch.Tensor],
    model_config: dict,
    float_point: bool,
    skips: list[str] | None = None,
) -> dict[str, str]:
    """Build the single-file safetensors metadata consumed by `NunchakuModelLoaderMixin`."""
    rank = 32
    for key, value in converted_state_dict.items():
        if key.endswith(".proj_down"):
            rank = value.shape[1]
            break
    quantization_config = {
        "method": "svdquant",
        "weight": {
            "dtype": "fp4_e2m1_all" if float_point else "int4",
            "scale_dtype": [None, "fp8_e4m3_nan"] if float_point else None,
            "group_size": 16 if float_point else 64,
        },
        "activation": {
            "dtype": "fp4_e2m1_all" if float_point else "int4",
            "scale_dtype": "fp8_e4m3_nan" if float_point else None,
            "group_size": 16 if float_point else 64,
        },
        "rank": rank,
    }
    if skips:
        quantization_config["skips"] = sorted(skips)
    return {
        "config": json.dumps(model_config),
        "model_class": "NunchakuWanTransformer3DModel",
        "quantization_config": json.dumps(quantization_config),
    }


# endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="path to the quantization checkpoint directory.")
    parser.add_argument("--output-root", type=str, default="", help="root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=None, help="name of the model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="diffusers model directory or HuggingFace repo (required for Wan to embed the transformer config).",
    )
    parser.add_argument("--float-point", action="store_true", help="use float-point 4-bit quantization (Flux only).")
    parser.add_argument(
        "--skips",
        type=str,
        nargs="*",
        default=None,
        help="Wan units to keep unquantized, globally (attn2.to_kv) or per block (blocks.5.attn2.to_kv).",
    )
    parser.add_argument("--output-name", type=str, default=None, help="override the output checkpoint filename stem.")
    args = parser.parse_args()
    if not args.output_root:
        args.output_root = args.quant_path
    if args.model_name is None:
        assert args.model_path is not None, "model name or path is required."
        model_name = args.model_path.rstrip(os.sep).split(os.sep)[-1]
        print(f"Model name not provided, using {model_name} as the model name.")
    else:
        model_name = args.model_name
    assert model_name, "Model name must be provided."
    assert "flux" in model_name.lower() or "wan" in model_name.lower(), "Only Flux and Wan models are supported."
    state_dict_path = os.path.join(args.quant_path, "model.pt")
    scale_dict_path = os.path.join(args.quant_path, "scale.pt")
    smooth_dict_path = os.path.join(args.quant_path, "smooth.pt")
    branch_dict_path = os.path.join(args.quant_path, "branch.pt")
    map_location = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    state_dict = torch.load(state_dict_path, map_location=map_location)
    scale_dict = torch.load(scale_dict_path, map_location="cpu")
    smooth_dict = torch.load(smooth_dict_path, map_location=map_location) if os.path.exists(smooth_dict_path) else {}
    branch_dict = torch.load(branch_dict_path, map_location=map_location) if os.path.exists(branch_dict_path) else {}
    if "wan" in model_name.lower():
        # the encoding is derived from the run's quant config, never from --float-point (R1)
        float_point = derive_wan_float_point(args.quant_path, scale_dict)
        assert args.model_path is not None, "--model-path is required for Wan models to embed the transformer config."
        from diffusers import WanTransformer3DModel

        if os.path.exists(os.path.join(args.model_path, "config.json")):
            model_config = WanTransformer3DModel.load_config(args.model_path)
        else:
            model_config = WanTransformer3DModel.load_config(args.model_path, subfolder="transformer")
        model_config = dict(model_config)
        orig_state_dict = None
        if args.skips:
            # skipped units restore their ORIGINAL weights (model.pt holds
            # smoothed, branch-subtracted, fake-quantized ones)
            orig_transformer = WanTransformer3DModel.from_pretrained(
                args.model_path,
                subfolder=None if os.path.exists(os.path.join(args.model_path, "config.json")) else "transformer",
                torch_dtype=torch.bfloat16,
            )
            orig_state_dict = orig_transformer.state_dict()
            del orig_transformer
        converted_state_dict = convert_to_nunchaku_wan_state_dict(
            state_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            float_point=float_point,
            skips=args.skips,
            orig_state_dict=orig_state_dict,
        )
        metadata = build_wan_metadata(converted_state_dict, model_config, float_point, skips=args.skips)
        os.makedirs(args.output_root, exist_ok=True)
        precision_name = "fp4" if float_point else "int4"
        output_stem = args.output_name or f"{model_name}-svdq-{precision_name}"
        output_path = os.path.join(args.output_root, f"{output_stem}.safetensors")
        safetensors.torch.save_file(converted_state_dict, output_path, metadata=metadata)
        print(f"Quantized model saved to {output_path}.")
    else:
        converted_state_dict, other_state_dict = convert_to_nunchaku_flux_state_dicts(
            state_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            float_point=args.float_point,
        )
        output_dirpath = os.path.join(args.output_root, model_name)
        os.makedirs(output_dirpath, exist_ok=True)
        safetensors.torch.save_file(
            converted_state_dict, os.path.join(output_dirpath, "transformer_blocks.safetensors")
        )
        safetensors.torch.save_file(other_state_dict, os.path.join(output_dirpath, "unquantized_layers.safetensors"))
        print(f"Quantized model saved to {output_dirpath}.")
