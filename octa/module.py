from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SparseMemoryAttention
from .config import OCTAConfig
from .memory import ObjectMemory
from .retrieval import RelevanceScorer


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


@dataclass
class OCTAOutput:
    query: torch.Tensor        # (B,N,C) memory-fused object query
    pos: torch.Tensor          # (B,N,3) refined position
    vel: torch.Tensor          # (B,N,3) refined velocity
    visibility: torch.Tensor   # (B,N) visibility used for this step
    confidence: torch.Tensor   # (B,N) confidence used for the write gate
    window: torch.Tensor       # (B,N) temporal search range in seconds
    topk_idx: torch.Tensor     # (B,N,k) selected memory slots
    attn: torch.Tensor         # (B,N,H,k)
    has_memory: torch.Tensor   # (B,N) whether any memory was retrievable
    written: torch.Tensor      # (B,N) whether this observation was written to memory


class OCTA(nn.Module):
    """Occlusion-Conditioned Temporal Attention.

    Per step and object:
      1. estimate visibility v (or take it from the caller);
      2. set the temporal search window T(v): short when visible, long when occluded;
      3. rank in-window memory entries and keep the top-k;
      4. sparse-attend to them and fuse with the query and predicted motion state;
      5. write the *observation* to memory only if visibility and confidence are high,
         so predictions made under occlusion never enter long-term memory.
    """

    def __init__(self, cfg: Optional[OCTAConfig] = None):
        super().__init__()
        self.cfg = cfg or OCTAConfig()
        C = self.cfg.dim
        self.vis_head = _mlp(C, C // 2, 1)
        self.conf_head = _mlp(C, C // 2, 1)
        self.motion_embed = _mlp(6, C, C)
        self.scorer = RelevanceScorer(C)
        self.attn = SparseMemoryAttention(C, self.cfg.num_heads, self.cfg.pos_scale, self.cfg.vel_scale)
        self.gate = nn.Sequential(nn.Linear(3 * C + 1, C), nn.Sigmoid())
        self.norm1 = nn.LayerNorm(C)
        self.ffn = _mlp(C, 4 * C, C)
        self.norm2 = nn.LayerNorm(C)
        self.state_head = _mlp(C + 6, C, 6)
        nn.init.zeros_(self.state_head[-1].weight)
        nn.init.zeros_(self.state_head[-1].bias)

    def init_memory(self, batch, num_obj, device=None, dtype=torch.float32) -> ObjectMemory:
        return ObjectMemory(batch, num_obj, self.cfg.mem_size, self.cfg.dim, device, dtype)

    def search_window(self, vis):
        c = self.cfg
        if not c.adaptive_window:
            return torch.full_like(vis, c.t_short)
        return c.t_short + (1 - vis).clamp(0, 1) ** c.gamma * (c.t_long - c.t_short)

    def forward(self, query, ref_pos, ref_vel, t, memory: ObjectMemory,
                confidence=None, visibility=None, write=True, origin=None) -> OCTAOutput:
        """query (B,N,C) current object queries; ref_pos/ref_vel (B,N,3) predicted motion state
        at time t; confidence/visibility (B,N) in [0,1], predicted from the query if omitted;
        origin (B,1,3) optional reference point (e.g. the ego position) so the motion embedding
        sees ego-relative rather than scene-specific absolute coordinates."""
        c = self.cfg
        vis_pred = torch.sigmoid(self.vis_head(query)).squeeze(-1)
        conf_pred = torch.sigmoid(self.conf_head(query)).squeeze(-1)
        vis = vis_pred if visibility is None else visibility
        conf = conf_pred if confidence is None else confidence

        window = self.search_window(vis.detach())
        score, _ = self.scorer(query, ref_pos, ref_vel, memory, t, window)
        k = min(c.top_k, memory.mem_size)
        top_score, idx = score.topk(k, dim=-1)
        sel = memory.gather(idx)

        o = 0.0 if origin is None else origin
        motion = self.motion_embed(torch.cat([(ref_pos - o) / c.pos_scale, ref_vel / c.vel_scale], -1))
        ctx, attn, has_mem, mem_pos, mem_vel = self.attn(query + motion, sel, top_score, ref_pos, t)

        # Visibility-aware gated fusion of memory context with the current query.
        g = self.gate(torch.cat([query, ctx, motion, vis[..., None]], -1))
        x = self.norm1(query + g * ctx)
        x = self.norm2(x + self.ffn(x))

        # State refinement, anchored on the memory-extrapolated state when available.
        m_pos = torch.where(has_mem[..., None], mem_pos, ref_pos)
        m_vel = torch.where(has_mem[..., None], mem_vel, ref_vel)
        delta = self.state_head(torch.cat([x, (m_pos - ref_pos) / c.pos_scale, (m_vel - ref_vel) / c.vel_scale], -1))
        blend = (has_mem * (1 - vis))[..., None]  # trust memory more as visibility drops
        pos = blend * m_pos + (1 - blend) * ref_pos
        pos = pos + delta[..., :3] * c.pos_scale
        vel = ref_vel + delta[..., 3:] * c.vel_scale

        written = (vis >= c.tau_vis) & (conf >= c.tau_conf)
        if write:
            d = (lambda x: x.detach()) if c.detach_memory else (lambda x: x)
            memory.write(d(query), d(ref_pos), d(ref_vel), d(conf), d(vis), t, written.detach())
        return OCTAOutput(x, pos, vel, vis, conf, window, idx, attn, has_mem, written)


def visibility_loss(out: OCTAOutput, gt_visibility):
    """Auxiliary supervision for the visibility estimate (e.g. nuScenes visibility levels
    mapped to [0,1]). Not used when the caller passes `visibility` externally."""
    return F.binary_cross_entropy(out.visibility.clamp(1e-6, 1 - 1e-6), gt_visibility)
