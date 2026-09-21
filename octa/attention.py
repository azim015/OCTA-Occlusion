import math

import torch
import torch.nn as nn


class SparseMemoryAttention(nn.Module):
    """Multi-head cross-attention from each object query to its top-k retrieved memories.

    Memory keys/values are enriched with motion-aware geometry (memory position rolled
    forward to now, relative to the current prediction, velocity, age). The relevance
    score is added as an attention bias so the ranking factors receive gradient.
    """

    def __init__(self, dim, num_heads, pos_scale=1.0, vel_scale=1.0):
        super().__init__()
        assert dim % num_heads == 0
        self.h, self.dh = num_heads, dim // num_heads
        self.ps, self.vs = pos_scale, vel_scale
        self.geo = nn.Sequential(nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.bias_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, q_in, sel, score, ref_pos, t):
        """q_in (B,N,C); sel: gathered memory dict (B,N,K,...); score (B,N,K) with -inf for
        ineligible; t scalar or (B,). Returns context (B,N,C), attn (B,N,H,K), has_mem (B,N),
        and the attention-averaged roll-forward position/velocity of the memories."""
        B, N, K = score.shape
        t = torch.as_tensor(t, device=q_in.device, dtype=q_in.dtype)
        t = t.view(-1, 1, 1) if t.dim() else t
        age = (t - sel["time"]).clamp_min(0)
        pos_now = sel["pos"] + sel["vel"] * age[..., None]
        geo = self.geo(torch.cat([(pos_now - ref_pos[:, :, None]) / self.ps, sel["vel"] / self.vs, age[..., None]], -1))
        mem = sel["feat"] + geo

        q = self.q(q_in).view(B, N, self.h, self.dh)
        k = self.k(mem).view(B, N, K, self.h, self.dh)
        v = self.v(mem).view(B, N, K, self.h, self.dh)
        logits = torch.einsum("bnhd,bnkhd->bnhk", q, k) / math.sqrt(self.dh)
        finite = torch.isfinite(score)
        bias = torch.where(finite, score, torch.zeros_like(score)).clamp(-30, 30)
        logits = logits + self.bias_scale * bias[:, :, None]
        logits = logits.masked_fill(~finite[:, :, None], float("-inf"))

        has_mem = finite.any(-1)
        logits = logits.masked_fill(~has_mem[:, :, None, None], 0.0)  # avoid NaN when empty
        attn = torch.softmax(logits, dim=-1) * finite[:, :, None]
        ctx = torch.einsum("bnhk,bnkhd->bnhd", attn, v).reshape(B, N, -1)
        ctx = self.o(ctx) * has_mem[..., None]

        w = attn.mean(2)  # (B,N,K)
        mem_pos = (w[..., None] * pos_now).sum(2)
        mem_vel = (w[..., None] * sel["vel"]).sum(2)
        return ctx, attn, has_mem, mem_pos, mem_vel
