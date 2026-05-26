# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops


def _get_attention_backend():
    """Select the safest attention backend for the current GPU.

    FlashAttention 3 kernels in xFormers are Hopper-specific. Blackwell GPUs
    like RTX 5090 report a higher compute capability than Hopper, but cannot
    execute those Hopper kernels. Prefer PyTorch SDPA on Blackwell and newer.
    """
    if not torch.cuda.is_available():
        return "sdpa", None

    major, minor = torch.cuda.get_device_capability()

    if (major, minor) == (9, 0):
        try:
            return "flash3", (xops.fmha.flash3.FwOp, xops.fmha.flash3.BwOp)
        except AttributeError:
            pass

    if major >= 10:
        return "sdpa", None

    return "flash2", (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias=False,
        fc_bias=False,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_qk_norm = use_qk_norm

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        self.attn_dropout = attn_dropout

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.attn_backend, self.flash_attn_ops = _get_attention_backend()

    def _scaled_dot_product_attention(self, q, k, v):
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        return x.transpose(1, 2)

    def forward(self, q: torch.Tensor, attn_bias, kv=None) -> torch.Tensor:
        # attention block that supports non-query keys and values
        if kv is None:
            kv = q
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q, k, v = (
            einops.rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
            for t in (q, k, v)
        )
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if attn_bias is not None:
            attn_bias = einops.repeat(
                attn_bias,
                "b1 nh1 sq skv -> (b1 b) (nh1 nh) sq skv",
                b=q.shape[0],
                nh=self.num_heads,
            )
            # flash attention does not support custom attention mask
            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_dropout if self.training else 0.0,
            )
        else:
            if self.attn_backend == "sdpa":
                x = self._scaled_dot_product_attention(q, k, v)
            else:
                x = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=attn_bias,
                    p=self.attn_dropout if self.training else 0.0,
                    op=self.flash_attn_ops,
                )

        x = einops.rearrange(x, "b n h d -> b n (h d)")

        x = self.attn_fc_dropout(self.proj(x))
        return x
