# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU-optimized operator replacements for HuggingFace transformer models.

When running on Ascend NPUs, this module monkey-patches key operations
(RMSNorm, SwiGLU, RoPE, MoE GMM) with their ``torch_npu`` accelerated
equivalents.  These patches are applied *before* FSDP wrapping so that
the training path uses NPU-native kernels throughout.

Usage:
    from roll.models.transformers import npu_patch  # noqa: F401  (side-effect import)

The import automatically patches all supported model classes.
"""

from importlib import import_module

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import logging

logger = logging.get_logger(__name__)

APPLIED_PATCHES: list[str] = []
SKIPPED_PATCHES: dict[str, str] = {}


def _import_modeling_module(module_path: str):
    try:
        return import_module(module_path)
    except Exception as exc:
        SKIPPED_PATCHES[module_path] = repr(exc)
        logger.debug("Skipping NPU patch for %s: %s", module_path, exc)
        return None


def _apply_patch(patch_name: str, module, patch_fn):
    if module is None:
        return
    try:
        patch_fn(module)
    except Exception as exc:
        SKIPPED_PATCHES[patch_name] = repr(exc)
        logger.warning("Failed to apply NPU patch for %s: %s", patch_name, exc)
        return
    APPLIED_PATCHES.append(patch_name)


modeling_qwen2 = _import_modeling_module("transformers.models.qwen2.modeling_qwen2")
modeling_qwen2_5_vl = _import_modeling_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
modeling_qwen3 = _import_modeling_module("transformers.models.qwen3.modeling_qwen3")
modeling_qwen3_moe = _import_modeling_module("transformers.models.qwen3_moe.modeling_qwen3_moe")
modeling_qwen3_vl = _import_modeling_module("transformers.models.qwen3_vl.modeling_qwen3_vl")
modeling_qwen3_vl_moe = _import_modeling_module("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")
modeling_qwen3_next = _import_modeling_module("transformers.models.qwen3_next.modeling_qwen3_next")
modeling_qwen3_5 = _import_modeling_module("transformers.models.qwen3_5.modeling_qwen3_5")
modeling_qwen3_5_moe = _import_modeling_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")


# ==============================================================================
# Core NPU operator replacements
# ==============================================================================


def rms_norm_forward_npu(self, x):
    """NPU-optimised RMSNorm: delegates to ``torch_npu.npu_rms_norm``."""
    if x.dtype != self.weight.dtype:
        x = x.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def silu_forward_npu(self, hidden_state):
    """NPU-optimised SiLU (SwiGLU) MLP: fuses gate+up projections + activation."""
    gate_up = torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(gate_up, dim=-1))


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU-optimised RoPE: delegates to ``torch_npu.npu_rotary_mul``."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


# Qwen3-Next / Qwen3.5 variant: weight=1+weight, float32 compute
def qwen3_next_rms_norm_forward_npu(self, x):
    return torch_npu.npu_rms_norm(x.float(), 1.0 + self.weight.float(), epsilon=self.eps)[0].type_as(x)


def qwen3_next_rms_norm_forward_gated_npu(self, hidden_states, gate=None):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight.float(), epsilon=self.variance_epsilon)[0]
    hidden_states = hidden_states * F.silu(gate.to(torch.float32))
    return hidden_states.to(input_dtype)


def qwen3_next_apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """RoPE for Qwen3-Next / Qwen3.5: partial rotary (half-dim)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin).to(q.dtype)
    k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin).to(k.dtype)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


# ==============================================================================
# MoE GroupedMatMul (GMM) for NPU
# ==============================================================================


class NPUGmmFunction(torch.autograd.Function):
    """Grouped MatMul (GMM) autograd Function for Ascend NPU.

    Uses ``torch_npu.npu_grouped_matmul`` to compute expert projections in a
    single batched call instead of looping over individual experts.
    """

    @staticmethod
    def forward(ctx, x, weight, group_list, group_list_type=1):
        """
        Args:
            x: (tokens_num * top_k, hidden_size)
            weight: (n_experts, hidden_size, intermediate_size)
            group_list: (n_experts,) token counts per expert.
                type 0 = cumsum, type 1 = direct (default).
            group_list_type: 0 or 1.
        """
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx.group_list_type = group_list_type

        output = torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
        )[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        group_list_type = ctx.group_list_type

        dx = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
        )[0]

        dw = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=group_list_type,
        )[0]

        return dx, dw, None, None


# ==============================================================================
# MoE routed forward (shared by Qwen3Moe / Qwen3Next sparse MoE blocks)
# ==============================================================================


def _qwen3_sparse_moe_uses_legacy_block_api(self) -> bool:
    """Return True for transformers 4.57.x-style Qwen3 MoE blocks."""
    return (
        isinstance(getattr(self, "gate", None), nn.Linear)
        and hasattr(self, "top_k")
        and hasattr(self, "norm_topk_prob")
    )


def _qwen3_sparse_moe_routed_forward_npu(self, hidden_states: torch.Tensor):
    """Shared NPU routed-expert path for Qwen3Moe / Qwen3Next sparse MoE blocks.

    Returns:
        tuple: (flattened_input, routed_hidden_states, router_logits)
    """
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.view(-1, hidden_dim)
    gate_output = self.gate(hidden_states)
    if isinstance(gate_output, tuple) and len(gate_output) == 3:
        router_logits, routing_weights, selected_experts = gate_output
    else:
        router_logits = gate_output
        top_k = getattr(self.gate, "top_k", getattr(self, "top_k", getattr(self, "num_experts_per_tok", 2)))
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        norm_topk_prob = getattr(self.gate, "norm_topk_prob", getattr(self, "norm_topk_prob", True))
        if norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    input_dtype = hidden_states.dtype
    if hasattr(self.experts, "gate_up_proj") and hasattr(self.experts, "down_proj"):
        gate_proj_weight, up_proj_weight = self.experts.gate_up_proj.chunk(2, dim=1)
        w1 = up_proj_weight.transpose(1, 2).to(input_dtype)
        w2 = gate_proj_weight.transpose(1, 2).to(input_dtype)
        w3 = self.experts.down_proj.transpose(1, 2).to(input_dtype)
    else:
        up_weight_list = [e.up_proj.weight for e in self.experts]
        gate_weight_list = [e.gate_proj.weight for e in self.experts]
        down_weight_list = [e.down_proj.weight for e in self.experts]
        w1 = torch.stack(up_weight_list).transpose(1, 2).to(input_dtype)
        w2 = torch.stack(gate_weight_list).transpose(1, 2).to(input_dtype)
        w3 = torch.stack(down_weight_list).transpose(1, 2).to(input_dtype)

    permuted_tokens, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    num_experts = getattr(
        self, "num_experts", getattr(self.experts, "num_experts", getattr(self.gate, "out_features", None))
    )
    if num_experts is None:
        num_experts = self.gate.weight.shape[0]
    tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)

    up_res = NPUGmmFunction.apply(permuted_tokens, w1, tokens_per_expert)
    gate_res = NPUGmmFunction.apply(permuted_tokens, w2, tokens_per_expert)
    act_res = torch_npu.npu_swiglu(torch.cat([gate_res, up_res], dim=-1))
    down_res = NPUGmmFunction.apply(act_res, w3, tokens_per_expert)

    routed_hidden_states = torch_npu.npu_moe_token_unpermute(down_res, row_ids_map, probs=routing_weights)
    return hidden_states, routed_hidden_states, router_logits


# ==============================================================================
# Per-model forward implementations
# ==============================================================================


def qwen3_moe_sparse_moe_block_forward_npu(
    self, hidden_states: torch.Tensor
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    output_shape = hidden_states.shape
    _, routed_hidden_states, router_logits = _qwen3_sparse_moe_routed_forward_npu(self, hidden_states)
    final_hidden_states = routed_hidden_states.reshape(output_shape)
    if _qwen3_sparse_moe_uses_legacy_block_api(self):
        return final_hidden_states, router_logits
    return final_hidden_states


def qwen3_next_sparse_moe_block_forward_npu(
    self, hidden_states: torch.Tensor
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    output_shape = hidden_states.shape
    hidden_states, routed_hidden_states, router_logits = _qwen3_sparse_moe_routed_forward_npu(self, hidden_states)
    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
    final_hidden_states = (routed_hidden_states + shared_expert_output).reshape(output_shape)
    if _qwen3_sparse_moe_uses_legacy_block_api(self):
        return final_hidden_states, router_logits
    return final_hidden_states


class NPUQwen3VLMoeTextExperts(nn.Module):
    """NPU-optimised text experts for Qwen3-VL-MoE."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, routing_weights, router_indices):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        if self.training:
            permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
                hidden_states, router_indices.to(torch.int32)
            )
            tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
            intermediate_hidden_states = NPUGmmFunction.apply(
                permuted_hidden_states, self.gate_up_proj, tokens_per_expert
            )
            intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
            output = NPUGmmFunction.apply(intermediate_activations, self.down_proj, tokens_per_expert)
            num_tokens = hidden_states.shape[0]
            top_k = router_indices.shape[1]
            batch_idx = torch.arange(num_tokens, device=routing_weights.device)
            batch_idx = batch_idx.unsqueeze(1).expand(-1, top_k)
            selected_probs = routing_weights[batch_idx, router_indices]
            next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=selected_probs)
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states


class NPUQwen3VLMoeTextSparseMoeBlock(nn.Module):
    """NPU-optimised Qwen3-VL-MoE sparse MoE block."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = NPUQwen3VLMoeTextExperts(config)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routing_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        return self.experts(hidden_states, routing_weights, router_indices)


def qwen3_5_moe_experts_forward_npu(self, hidden_states, top_k_index, top_k_weights):
    """NPU-optimised Qwen3.5-MoE experts forward."""
    selected_experts = top_k_index
    routing_weights = top_k_weights
    gate_up_proj = self.gate_up_proj.permute(0, 2, 1).contiguous()
    down_proj = self.down_proj.permute(0, 2, 1).contiguous()
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states, selected_experts.to(torch.int32)
    )
    tokens_per_expert = torch.histc(selected_experts, bins=self.num_experts, min=0, max=self.num_experts)
    intermediate_hidden_states = NPUGmmFunction.apply(permuted_hidden_states, gate_up_proj, tokens_per_expert)
    intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
    output = NPUGmmFunction.apply(intermediate_activations, down_proj, tokens_per_expert)
    final_hidden_states = torch_npu.npu_moe_token_unpermute(
        output.to(routing_weights.dtype), row_ids_map, probs=routing_weights
    )
    return final_hidden_states.to(hidden_states.dtype)


# ==============================================================================
# Apply all patches (side-effect on import)
# ==============================================================================

def _patch_qwen2(module):
    module.Qwen2RMSNorm.forward = rms_norm_forward_npu
    module.Qwen2MLP.forward = silu_forward_npu
    module.apply_rotary_pos_emb = apply_rotary_pos_emb_npu


def _patch_qwen2_5_vl(module):
    if hasattr(module, "Qwen2RMSNorm"):
        module.Qwen2RMSNorm.forward = rms_norm_forward_npu
    else:
        module.Qwen2_5_VLRMSNorm.forward = rms_norm_forward_npu
    module.Qwen2_5_VLMLP.forward = silu_forward_npu


def _patch_qwen3(module):
    module.Qwen3RMSNorm.forward = rms_norm_forward_npu
    module.Qwen3MLP.forward = silu_forward_npu
    module.apply_rotary_pos_emb = apply_rotary_pos_emb_npu


def _patch_qwen3_moe(module):
    module.Qwen3MoeRMSNorm.forward = rms_norm_forward_npu
    module.Qwen3MoeSparseMoeBlock.forward = qwen3_moe_sparse_moe_block_forward_npu
    module.apply_rotary_pos_emb = apply_rotary_pos_emb_npu


def _patch_qwen3_vl(module):
    module.Qwen3VLTextRMSNorm.forward = rms_norm_forward_npu
    module.Qwen3VLTextMLP.forward = silu_forward_npu


def _patch_qwen3_vl_moe(module):
    module.Qwen3VLMoeTextSparseMoeBlock = NPUQwen3VLMoeTextSparseMoeBlock
    module.Qwen3VLMoeTextRMSNorm.forward = rms_norm_forward_npu
    module.apply_rotary_pos_emb = apply_rotary_pos_emb_npu


def _patch_qwen3_next(module):
    module.Qwen3NextSparseMoeBlock.forward = qwen3_next_sparse_moe_block_forward_npu
    module.Qwen3NextRMSNormGated.forward = qwen3_next_rms_norm_forward_gated_npu
    module.Qwen3NextRMSNorm.forward = qwen3_next_rms_norm_forward_npu
    module.apply_rotary_pos_emb = qwen3_next_apply_rotary_pos_emb_npu


def _patch_qwen3_5(module):
    module.Qwen3_5RMSNormGated.forward = qwen3_next_rms_norm_forward_gated_npu
    module.Qwen3_5RMSNorm.forward = qwen3_next_rms_norm_forward_npu
    module.apply_rotary_pos_emb = qwen3_next_apply_rotary_pos_emb_npu


def _patch_qwen3_5_moe(module):
    module.Qwen3_5MoeExperts.forward = qwen3_5_moe_experts_forward_npu
    module.Qwen3_5MoeRMSNormGated.forward = qwen3_next_rms_norm_forward_gated_npu
    module.Qwen3_5MoeRMSNorm.forward = qwen3_next_rms_norm_forward_npu
    module.apply_rotary_pos_emb = qwen3_next_apply_rotary_pos_emb_npu


_apply_patch("Qwen2", modeling_qwen2, _patch_qwen2)
_apply_patch("Qwen2.5-VL", modeling_qwen2_5_vl, _patch_qwen2_5_vl)
_apply_patch("Qwen3", modeling_qwen3, _patch_qwen3)
_apply_patch("Qwen3-MoE", modeling_qwen3_moe, _patch_qwen3_moe)
_apply_patch("Qwen3-VL", modeling_qwen3_vl, _patch_qwen3_vl)
_apply_patch("Qwen3-VL-MoE", modeling_qwen3_vl_moe, _patch_qwen3_vl_moe)
_apply_patch("Qwen3-Next", modeling_qwen3_next, _patch_qwen3_next)
_apply_patch("Qwen3.5", modeling_qwen3_5, _patch_qwen3_5)
_apply_patch("Qwen3.5-MoE", modeling_qwen3_5_moe, _patch_qwen3_5_moe)

if APPLIED_PATCHES:
    logger.info("Applied NPU patches for FSDP training backend: %s", ", ".join(APPLIED_PATCHES))
else:
    logger.warning("No NPU patches were applied for FSDP training backend")
