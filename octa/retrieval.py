import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import ObjectMemory


class RelevanceScorer(nn.Module):
    """Ranks each object's memory entries with five factors from the paper:
    feature similarity, spatial consistency, motion compatibility,
    observation quality and memory age. Factors are combined with learned
    positive weights. Entries outside the visibility-dependent window are masked out.
    """

    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))   # cosine temperature
        self.log_sigma_pos = nn.Parameter(torch.tensor(math.log(2.0)))  # metres
        self.log_sigma_vel = nn.Parameter(torch.tensor(math.log(1.0)))  # m/s
        self.log_tau_age = nn.Parameter(torch.tensor(math.log(3.0)))    # seconds
        self.raw_w = nn.Parameter(torch.full((5,), math.log(math.e - 1)))  # softplus -> 1

    def forward(self, query, ref_pos, ref_vel, mem: ObjectMemory, t, window):
        """query (B,N,C); ref_pos/ref_vel (B,N,3); t scalar or (B,); window (B,N) seconds.
        Returns scores (B,N,M) with -inf on ineligible entries, and eligibility mask."""
        t = torch.as_tensor(t, device=query.device, dtype=query.dtype)
        t = t.view(-1, 1, 1) if t.dim() else t
        age = t - mem.time                                          # (B,N,M)
        mem_pos_now = mem.pos + mem.vel * age[..., None]            # constant-velocity roll-forward

        q = F.normalize(self.q_proj(query), dim=-1)
        k = F.normalize(self.k_proj(mem.feat), dim=-1)
        s_feat = self.log_scale.exp() * torch.einsum("bnc,bnmc->bnm", q, k)
        d2 = ((mem_pos_now - ref_pos[:, :, None]) ** 2).sum(-1)
        s_pos = -d2 / (2 * self.log_sigma_pos.exp() ** 2)
        dv2 = ((mem.vel - ref_vel[:, :, None]) ** 2).sum(-1)
        s_vel = -dv2 / (2 * self.log_sigma_vel.exp() ** 2)
        s_qual = torch.log((mem.conf * mem.vis).clamp_min(1e-4))
        s_age = -age.clamp_min(0) / self.log_tau_age.exp()

        terms = torch.stack([s_feat, s_pos, s_vel, s_qual, s_age], dim=-1).clamp_min(-30.0)
        score = (terms * F.softplus(self.raw_w)).sum(-1)

        eligible = mem.valid & (age >= 0) & (age <= window[..., None])
        return score.masked_fill(~eligible, float("-inf")), eligible
