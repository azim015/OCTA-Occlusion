"""Train/evaluate OCTA on nuScenes (mini) for persistent object state under occlusion.

Per keyframe and per annotated instance the model sees LiDAR statistics of the points inside
the instance's box (real evidence, empty when occluded/far) and a tracker prior (last observed
position dead-reckoned with a constant-velocity estimate). It must output, for *every* object
including unobserved ones: position, velocity, box size and class. Only memory written when the
object was reliably observed can supply size/class once the points vanish.

Reported: error vs. time since the object was last observed (0-2 s, 2-4 s, 4-6 s, >6 s), the
paper's Table-1 occlusion-duration buckets, using position error / size error / class accuracy
instead of mAP (there is no detector here). Association uses GT boxes (oracle).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from octa import OCTA, OCTAConfig
from octa.nuscenes_data import N_MIN, VIS_VALUE, load_cache, build_cache

VAL_SCENES = ["scene-0103", "scene-0916"]  # official nuScenes mini_val
IN_DIM = 15  # LiDAR stats + range (bearing components are global-frame, so dropped)
KEYS = ["active", "gt_pos", "gt_vel", "gt_size", "vis_level", "n_in", "feat", "ref_pos", "ref_vel",
        "hold_pos", "has_prior", "since", "cls", "time", "ego_pos"]


def to_batch(scenes, dev, crop=None, aug=False, rng=None):
    """Pad scenes to (T, B, N, ...) tensors. `crop`=(min_len) enables a random temporal crop."""
    lens, ns = [], []
    starts = []
    for d in scenes:
        T = d["active"].shape[0]
        L = T if crop is None else int(rng.integers(crop, T + 1))
        s = 0 if crop is None else int(rng.integers(0, T - L + 1))
        starts.append((s, L)); lens.append(L); ns.append(d["active"].shape[1])
    T, N, B = max(lens), max(ns), len(scenes)
    def z(dtype, *shape): return np.zeros(shape, dtype)
    o = dict(active=z(bool, T, B, N), gt_pos=z(np.float32, T, B, N, 3), gt_vel=z(np.float32, T, B, N, 3),
             gt_size=z(np.float32, T, B, N, 3), vis_level=z(np.int64, T, B, N), n_in=z(np.int64, T, B, N),
             feat=z(np.float32, T, B, N, 16), ref_pos=z(np.float32, T, B, N, 3), ref_vel=z(np.float32, T, B, N, 3),
             hold_pos=z(np.float32, T, B, N, 3), has_prior=z(bool, T, B, N), since=np.full((T, B, N), 1e3, np.float32),
             cls=np.full((B, N), -1, np.int64), time=z(np.float32, T, B), ego_pos=z(np.float32, T, B, 3))
    for b, (d, (s, L)) in enumerate(zip(scenes, starts)):
        n = d["active"].shape[1]
        for k, v in o.items():
            if k == "cls": v[b, :n] = d[k]
            elif k in ("time", "ego_pos"): v[:L, b] = d[k][s:s + L]
            else: v[:L, b, :n] = d[k][s:s + L]
        o["time"][L:, b] = o["time"][L - 1, b] + 0.5 * np.arange(1, T - L + 1)  # padded steps stay inactive
    if aug:  # random yaw of the whole scene (LiDAR box-frame features are unaffected)
        for b in range(B):
            a = rng.uniform(0, 2 * math.pi); c, s_ = math.cos(a), math.sin(a)
            R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]], np.float32)
            for k in ("gt_pos", "gt_vel", "ref_pos", "ref_vel", "hold_pos"):
                o[k][:, b] = o[k][:, b] @ R.T
            o["ego_pos"][:, b] = o["ego_pos"][:, b] @ R.T
    return {k: torch.from_numpy(v).to(dev) for k, v in o.items()}


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.dim
        self.enc = nn.Sequential(nn.Linear(IN_DIM, D), nn.LayerNorm(D), nn.GELU(), nn.Linear(D, D))
        self.octa = OCTA(cfg)
        self.cls_head = nn.Linear(D, 10)
        self.size_head = nn.Linear(D, 3)

    def run(self, b):
        T, B, N = b["active"].shape
        mem = self.octa.init_memory(B, N, b["active"].device)
        outs = []
        for t in range(T):
            n_in = b["n_in"][t].float()
            conf = (n_in / 20).clamp(max=1) * b["active"][t]  # stand-in for a detector score
            q = self.enc(b["feat"][t][..., :IN_DIM] * b["active"][t][..., None])
            out = self.octa(q, b["ref_pos"][t], b["ref_vel"][t], b["time"][t], mem,
                            confidence=conf, origin=b["ego_pos"][t][:, None])
            outs.append((out, self.cls_head(out.query), self.size_head(out.query)))
        return outs


def losses(b, outs):
    tot, parts = 0.0, {}
    vis_gt = torch.tensor([0.0] + [VIS_VALUE[str(i)] for i in range(1, 5)], device=b["active"].device)
    for t, (out, cls_logit, size_pred) in enumerate(outs):
        m = b["active"][t] & b["has_prior"][t]
        if not m.any():
            continue
        l_pos = F.huber_loss(out.pos[m], b["gt_pos"][t][m], delta=1.0)
        l_vel = F.huber_loss(out.vel[m], b["gt_vel"][t][m], delta=1.0)
        l_size = F.smooth_l1_loss(size_pred[m], b["gt_size"][t][m].clamp_min(0.05).log())
        cm = m & (b["cls"] >= 0)
        l_cls = F.cross_entropy(cls_logit[cm], b["cls"][cm]) if cm.any() else 0.0
        am = b["active"][t]
        l_vis = F.binary_cross_entropy(out.visibility[am].clamp(1e-6, 1 - 1e-6), vis_gt[b["vis_level"][t][am]])
        tot = tot + l_pos + 0.5 * l_vel + l_size + l_cls + l_vis
    return tot / len(outs)


@torch.no_grad()
def evaluate(model, scenes, dev):
    model.eval()
    rows = []
    for d in scenes:
        b = to_batch([d], dev)
        for t, (out, cls_logit, size_pred) in enumerate(model.run(b)):
            m = (b["active"][t] & b["has_prior"][t])[0]
            if not m.any():
                continue
            g = lambda x: x[t][0][m]
            rows.append(torch.stack([
                (out.pos[0][m] - g(b["gt_pos"])).norm(dim=-1),
                (g(b["ref_pos"]) - g(b["gt_pos"])).norm(dim=-1),
                (g(b["hold_pos"]) - g(b["gt_pos"])).norm(dim=-1),
                (out.vel[0][m] - g(b["gt_vel"])).norm(dim=-1),
                (size_pred[0][m].exp() - g(b["gt_size"])).abs().mean(-1),
                (cls_logit[0][m].argmax(-1) == b["cls"][0][m]).float(),
                (b["cls"][0][m] >= 0).float(),
                g(b["since"]), g(b["n_in"]).float(),
                out.has_memory[0][m].float()], 1))
    r = torch.cat(rows).cpu().numpy()
    pos, cv, hold, vel, size, cor, cvalid, since, n_in, hasmem = r.T
    unobs = n_in < N_MIN
    groups = {"observed": ~unobs, "unobs 0-2s": unobs & (since <= 2), "unobs 2-4s": unobs & (since > 2) & (since <= 4),
              "unobs 4-6s": unobs & (since > 4) & (since <= 6), "unobs >6s": unobs & (since > 6), "unobs all": unobs}
    res = {}
    for name, g in groups.items():
        c = g & (cvalid > 0)
        res[name] = dict(n=int(g.sum()), pos=float(pos[g].mean()), cv_prior=float(cv[g].mean()),
                         hold=float(hold[g].mean()), vel=float(vel[g].mean()), size=float(size[g].mean()),
                         cls_acc=float(cor[c].mean()) if c.any() else float("nan"), mem_used=float(hasmem[g].mean()))
    return res


def run_variant(name, args, data, dev, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    cfg = OCTAConfig(dim=args.dim, num_heads=4, mem_size=32, top_k=4, tau_vis=0.6, tau_conf=0.5,
                     pos_scale=10.0, vel_scale=5.0)
    if name == "fixed": cfg.adaptive_window = False
    if name == "nomem": cfg.adaptive_window, cfg.t_short = False, 0.0
    train = [d for k, d in data.items() if k not in VAL_SCENES]
    val = [data[k] for k in VAL_SCENES]
    model = Model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters)
    t0 = time.time()
    for it in range(args.iters):
        model.train()
        idx = rng.choice(len(train), size=min(args.batch, len(train)), replace=False)
        b = to_batch([train[i] for i in idx], dev, crop=24, aug=True, rng=rng)
        loss = losses(b, model.run(b))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if it % 50 == 0:
            print(f"  [{name} s{seed}] it {it} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    return evaluate(model, val, dev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/datasets/nuscenes"))
    ap.add_argument("--cache", default="cache/nuscenes_mini.npz")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--variants", nargs="+", default=["nomem", "fixed", "adaptive"])
    ap.add_argument("--out", default="results_nuscenes.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.cache):
        build_cache(args.root, out_path=args.cache)
    data = load_cache(args.cache)
    allres = {}
    for v in args.variants:
        for s in args.seeds:
            allres[f"{v}/seed{s}"] = run_variant(v, args, data, dev, s)
    json.dump(allres, open(args.out, "w"), indent=1)
    for k, r in allres.items():
        print(k)
        for g, m in r.items():
            print(f"   {g:11s} n={m['n']:5d} pos {m['pos']:6.2f} (cv {m['cv_prior']:6.2f}, hold {m['hold']:6.2f}) "
                  f"vel {m['vel']:5.2f} size {m['size']:5.2f} cls {m['cls_acc']:.3f} mem {m['mem_used']:.2f}")
