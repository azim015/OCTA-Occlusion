"""Synthetic sanity check for OCTA (NOT a reproduction of the paper's benchmark numbers).

Objects move in BEV with a slowly drifting velocity and have a fixed identity feature.
While occluded, the observed feature is pure noise and the tracker's motion prior is stale
(velocity 0, position frozen). Recovering the object's position and identity is only
possible from memory written before the occlusion. Variants:
  adaptive : OCTA, window 1 s (visible) -> 6 s (occluded)
  fixed    : same model, fixed 1 s window
  nomem    : no memory (window 0)
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from octa import OCTA, OCTAConfig

T, DT, C = 40, 0.5, 64


def make_batch(B, N, dev):
    pos0 = (torch.rand(B, N, 2, device=dev) - 0.5) * 40
    vel = torch.randn(B, N, 2, device=dev) * 1.5
    ident = F.normalize(torch.randn(B, N, C, device=dev), dim=-1)
    pos, v = [], vel
    p = pos0
    for _ in range(T):
        v = v + 0.1 * torch.randn_like(v)
        p = p + v * DT
        pos.append(p); 
    pos = torch.stack(pos)  # (T,B,N,2)
    start = torch.randint(8, 20, (B, N, 1), device=dev)
    dur = torch.randint(2, 11, (B, N, 1), device=dev)  # 1-5 s
    ts = torch.arange(T, device=dev).view(T, 1, 1)
    occ = (ts >= start.squeeze(-1)) & (ts < (start + dur).squeeze(-1))
    return pos, ident, occ.float()


def to3(x):
    return F.pad(x, (0, 1))


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = nn.Linear(C, cfg.dim)
        self.octa = OCTA(cfg)
        self.ident = nn.Linear(cfg.dim, C)

    def run(self, pos, ident, occ):
        B, N = pos.shape[1:3]
        dev = pos.device
        mem = self.octa.init_memory(B, N, dev)
        last_pos, outs = None, []
        for t in range(T):
            vis = 1 - occ[t]
            feat = vis[..., None] * ident + 0.3 * torch.randn_like(ident)
            obs_pos = pos[t] + 0.2 * torch.randn_like(pos[t])
            # stale prior under occlusion: position frozen at last visible estimate
            ref = obs_pos if last_pos is None else torch.where(occ[t][..., None] > 0, last_pos, obs_pos)
            vel = torch.zeros_like(ref) if last_pos is None else (obs_pos - last_pos) / DT * vis[..., None]
            last_pos = ref
            out = self.octa(self.enc(feat), to3(ref) / 10, to3(vel) / 3, t * DT, mem,
                            confidence=vis * 0.9 + 0.05)  # stand-in for a detector score
            outs.append((out, self.ident(out.query), out.pos[..., :2] * 10))
        return outs


def run_variant(name, args, dev):
    torch.manual_seed(0)
    cfg = OCTAConfig(dim=64, num_heads=4, mem_size=16, top_k=4)
    if name == "fixed":
        cfg.adaptive_window = False
    if name == "nomem":
        cfg.adaptive_window, cfg.t_short = False, 0.0
    model = Model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3)
    for it in range(args.iters):
        pos, ident, occ = make_batch(args.batch, args.objs, dev)
        loss = 0
        for t, (out, rec, p) in enumerate(model.run(pos, ident, occ)):
            loss = loss + F.mse_loss(p, pos[t]) / 100 + (1 - F.cosine_similarity(rec, ident, -1)).mean() \
                + F.binary_cross_entropy(out.visibility.clamp(1e-6, 1 - 1e-6), 1 - occ[t])
        opt.zero_grad(); (loss / T).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval(); errs, cos, cnt = 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(10):
            pos, ident, occ = make_batch(args.batch, args.objs, dev)
            for t, (out, rec, p) in enumerate(model.run(pos, ident, occ)):
                m = occ[t] > 0
                errs += (p - pos[t]).norm(dim=-1)[m].sum().item()
                cos += F.cosine_similarity(rec, ident, -1)[m].sum().item()
                cnt += m.sum().item()
    print(f"{name:9s} occluded-frame pos error {errs/cnt:6.2f} m | identity cosine {cos/cnt:5.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--objs", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for v in ("nomem", "fixed", "adaptive"):
        run_variant(v, args, dev)
