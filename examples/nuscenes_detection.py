"""End-to-end OCTA reproduction of the paper's reported metrics: mAP_obs / mAP_unobs
(Table 2), and mAP_unobs by occlusion duration (Table 1 / Figure 2), on nuScenes-mini LiDAR.

Pipeline (three stages, run in order by __main__):
  1. Train a real single-sweep BEV detector (`octa/detector.py`) jointly with OCTA (adaptive
     variant) end to end: detector proposals feed OCTA's memory/attention as observations, and
     OCTA's fused query drives class/size/existence heads. This is what makes the numbers here
     different from the earlier oracle-association experiment: nothing sees ground-truth boxes
     at inference, only whatever the detector actually proposes from LiDAR.
  2. Freeze that detector and run it once over every scene, caching its proposals per frame
     (matched-to-track-slot observations, plus leftover unmatched proposals that will show up
     as false positives in the mAP). This makes stage 3 fast, since it no longer needs a CNN
     forward pass.
  3. Re-train ONLY the OCTA head from scratch, three times (no-memory / fixed-window /
     adaptive-window), against the frozen cached detections, and evaluate nuScenes-style
     center-distance mAP (obs/unobs split, occlusion-duration buckets) on the frozen
     detector's val-scene output.

Scope (read before trusting the numbers):
  - Track identity is teacher-forced to the GT instance_token: each scene gets one track slot
    per instance that appears in it, "alive" exactly while that instance has a GT box. This
    removes track birth/death as a source of error so the comparison isolates the memory
    mechanism, but it does mean track lifetimes are not themselves inferred by the model.
    Detection quality, position estimation, and false positives ARE fully causal/non-oracle:
    a track slot only gets real information when the frozen detector's own proposals happen to
    land near it (`octa/tracking.match_proposals`), and every unmatched detector proposal is
    scored as a genuine false positive in the mAP.
  - Single LiDAR sweep, yaw-only box orientation, ~320 training frames (nuScenes-mini): all cut
    detector accuracy well below a production 3D detector. Compare shapes of results (does the
    memory help, and where), not absolute numbers against the paper's placeholder OCTA figures.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from octa import OCTA, OCTAConfig
from octa.detect_data import build_index, densify, global_to_ego, load_index, ego_to_global
from octa.detector import Detector, build_targets, detection_loss
from octa.nuscenes_data import CLASSES, VIS_VALUE, add_gt_velocity
from octa.nuscenes_eval import THRESHOLDS, evaluate as map_evaluate
from octa.tracking import TrackState, match_proposals

VAL_SCENES = ["scene-0103", "scene-0916"]
N_MIN = 5
FEAT_DIM = 32  # Detector backbone channel count (must match Detector(base=...))


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def load_scenes(root, cache="cache/nuscenes_detect_index.json"):
    if not os.path.exists(cache):
        build_index(root, out_path=cache)
    idx = load_index(cache)
    scenes = {}
    for name, frames in idx.items():
        d = densify(frames)
        d = add_gt_velocity(d)
        scenes[name] = dict(frames=frames, dense=d)
    return scenes


# --------------------------------------------------------------------------------------- stage 1
class TrackHeads(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg.dim
        self.enc = _mlp(FEAT_DIM, C, C)
        self.octa = OCTA(cfg)
        self.cls_head = nn.Linear(C, len(CLASSES))
        self.size_head = nn.Linear(C, 3)
        self.exist_head = nn.Linear(C, 1)


def run_timestep(detector, heads, dense, frame, points_ego, track_state, t, dev, mem, train, no_detector=False,
                  cached_prop=None):
    """One timestep for one scene. Returns per-track outputs + losses (or None parts if eval).
    If `no_detector` is False, runs the real detector on `points_ego`; else uses `cached_prop`
    (a dict from stage 2) directly, skipping the CNN forward (stage 3)."""
    N = dense["active"].shape[1]
    R_e, t_e = frame["R_e"], frame["t_e"]
    active = dense["active"][t]
    gt_xy_ego = dense["gt_pos_ego"][t][:, :2]
    gt_cls = dense["cls"]

    det_loss, det_parts = torch.tensor(0.0, device=dev), None
    if not no_detector:
        grid = detector.voxelize(points_ego, dev)[None]
        out = detector(grid)
        H, W = detector.grid_size()
        tgt = build_targets([frame["gt"]], detector.x_range, detector.y_range, detector.res, H, W, dev)
        det_loss, det_parts = detection_loss(out, tgt)
        dec = detector.decode(out)[0]
        prop_pos = dec["pos"].detach().cpu().numpy()
        prop_cls = dec["cls"].detach().cpu().numpy()
        prop_score = dec["score"].detach().cpu().numpy()
        prop_feat = dec["feat"]  # keep on device, differentiable
    else:
        prop_pos = cached_prop["pos"]
        prop_cls = cached_prop["cls"]
        prop_score = cached_prop["score"]
        prop_feat = torch.as_tensor(cached_prop["feat"], device=dev)

    matched, unmatched = match_proposals(prop_pos[:, :2], prop_cls, gt_xy_ego, gt_cls, active)
    obs_mask = matched >= 0
    obs_pos_ego = np.where(obs_mask[:, None], prop_pos[np.clip(matched, 0, None)], 0.0)
    obs_pos_global = ego_to_global(obs_pos_ego, R_e, t_e) if obs_mask.any() else np.zeros((N, 3), np.float32)
    obs_score = np.where(obs_mask, prop_score[np.clip(matched, 0, None)], 0.0)

    ref_pos, ref_vel, since, has_prior = track_state.step(
        t, obs_mask[None], obs_pos_global[None])
    ref_pos, ref_vel = torch.as_tensor(ref_pos[0], device=dev), torch.as_tensor(ref_vel[0], device=dev)

    query_feat = torch.zeros(N, FEAT_DIM, device=dev)
    idx_obs = np.where(obs_mask)[0]
    if len(idx_obs):
        query_feat[idx_obs] = prop_feat[matched[idx_obs]]
    query = heads.enc(query_feat)[None]
    origin = torch.as_tensor(t_e, device=dev)[None, None]
    conf = torch.as_tensor(obs_score.astype(np.float32), device=dev)[None]
    octa_out = heads.octa(query, ref_pos[None], ref_vel[None], float(t), mem,
                          confidence=conf, write=train, origin=origin)

    return dict(octa=octa_out, cls_logit=heads.cls_head(octa_out.query), size_pred=heads.size_head(octa_out.query),
                exist_logit=heads.exist_head(octa_out.query).squeeze(-1), active=active, has_prior=has_prior[0],
                since=since[0], obs_mask=obs_mask, det_loss=det_loss, det_parts=det_parts,
                unmatched_pos_ego=prop_pos[unmatched], unmatched_cls=prop_cls[unmatched],
                unmatched_score=prop_score[unmatched], R_e=R_e, t_e=t_e)


def track_losses(dev, dense, t, step_out):
    a, hp = step_out["active"], step_out["has_prior"]
    m = a & hp
    if not m.any():
        return torch.tensor(0.0, device=dev)
    mi = torch.as_tensor(m, device=dev)
    out = step_out["octa"]
    gt_pos = torch.as_tensor(dense["gt_pos"][t], device=dev)
    gt_vel = torch.as_tensor(dense["gt_vel"][t], device=dev)
    gt_size = torch.as_tensor(dense["gt_size"][t], device=dev)
    gt_cls = torch.as_tensor(dense["cls"], device=dev)
    vis_gt = torch.tensor([0.0] + [VIS_VALUE[str(i)] for i in range(1, 5)], device=dev)
    l_pos = F.huber_loss(out.pos[0][mi], gt_pos[mi], delta=1.0)
    l_vel = F.huber_loss(out.vel[0][mi], gt_vel[mi], delta=1.0)
    l_size = F.smooth_l1_loss(step_out["size_pred"][0][mi], gt_size[mi].clamp_min(0.05).log())
    cm = mi & (gt_cls >= 0)
    l_cls = F.cross_entropy(step_out["cls_logit"][0][cm], gt_cls[cm]) if cm.any() else torch.tensor(0.0, device=dev)
    am = torch.as_tensor(a, device=dev)
    vis_level = torch.as_tensor(dense["vis_level"][t], device=dev)
    l_vis = F.binary_cross_entropy(out.visibility[0][am].clamp(1e-6, 1 - 1e-6), vis_gt[vis_level[am]]) if am.any() \
        else torch.tensor(0.0, device=dev)
    obs = torch.as_tensor(step_out["obs_mask"], device=dev)
    exist_tgt = torch.where(obs, torch.ones_like(vis_gt[vis_level]), vis_gt[vis_level])
    l_exist = F.binary_cross_entropy_with_logits(step_out["exist_logit"][0][am], exist_tgt[am]) if am.any() \
        else torch.tensor(0.0, device=dev)
    return l_pos + 0.5 * l_vel + l_size + l_cls + l_vis + l_exist


def train_stage1(root, scenes, args, dev):
    cfg = OCTAConfig(dim=args.dim, num_heads=4, mem_size=32, top_k=4, tau_vis=0.6, tau_conf=0.5,
                     pos_scale=10.0, vel_scale=5.0)
    detector = Detector(base=FEAT_DIM).to(dev)
    heads = TrackHeads(cfg).to(dev)
    opt = torch.optim.AdamW(list(detector.parameters()) + list(heads.parameters()), args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters)
    train_names = [n for n in scenes if n not in VAL_SCENES]
    total_frames = sum(len(scenes[n]["frames"]) for n in train_names)
    from octa.detect_data import load_points_ego
    t0 = time.time()
    for it in range(args.iters):
        detector.train(); heads.train()
        opt.zero_grad()
        loss_sum, det_hm, det_reg = 0.0, 0.0, 0.0
        for name in train_names:  # gradient accumulation, one backward per scene (bounds memory)
            frames, dense = scenes[name]["frames"], scenes[name]["dense"]
            N = dense["active"].shape[1]
            mem = heads.octa.init_memory(1, N, dev)
            ts = TrackState(1, N)
            scene_loss = torch.tensor(0.0, device=dev)
            for t, fr in enumerate(frames):
                pts = load_points_ego(root, fr)
                so = run_timestep(detector, heads, dense, fr, pts, ts, t, dev, mem, train=True)
                scene_loss = scene_loss + so["det_loss"] + track_losses(dev, dense, t, so)
                det_hm += so["det_parts"]["hm"]; det_reg += so["det_parts"]["reg"]
            if scene_loss.requires_grad:
                (scene_loss / total_frames).backward()
            loss_sum += scene_loss.item()
        nn.utils.clip_grad_norm_(list(detector.parameters()) + list(heads.parameters()), 1.0)
        opt.step(); sched.step()
        if it % 20 == 0 or it == args.iters - 1:
            print(f"  [stage1] it {it} loss {loss_sum/total_frames:.3f} hm {det_hm/total_frames:.3f} "
                  f"reg {det_reg/total_frames:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return detector, heads


# --------------------------------------------------------------------------------------- stage 2
def cache_detections(root, detector, scenes, dev, out_path="cache/nuscenes_detections.pkl"):
    detector.eval()
    cache = {}
    with torch.no_grad():
        for name, sc in scenes.items():
            frames, dense = sc["frames"], sc["dense"]
            per_frame = []
            for t, fr in enumerate(frames):
                from octa.detect_data import load_points_ego
                pts = load_points_ego(root, fr)
                grid = detector.voxelize(pts, dev)[None]
                out = detector(grid)
                dec = detector.decode(out)[0]
                per_frame.append(dict(pos=dec["pos"].cpu().numpy(), cls=dec["cls"].cpu().numpy(),
                                      score=dec["score"].cpu().numpy(), feat=dec["feat"].cpu().numpy()))
            cache[name] = per_frame
            print("  [stage2] cached", name, flush=True)
    pickle.dump(cache, open(out_path, "wb"))
    return cache


# --------------------------------------------------------------------------------------- stage 3
def train_stage3(variant, scenes, det_cache, args, dev, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    cfg = OCTAConfig(dim=args.dim, num_heads=4, mem_size=32, top_k=4, tau_vis=0.6, tau_conf=0.5,
                     pos_scale=10.0, vel_scale=5.0)
    if variant == "fixed":
        cfg.adaptive_window = False
    if variant == "nomem":
        cfg.adaptive_window, cfg.t_short = False, 0.0
    heads = TrackHeads(cfg).to(dev)
    opt = torch.optim.AdamW(heads.parameters(), args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters3)
    train_names = [n for n in scenes if n not in VAL_SCENES]
    total_frames = sum(len(scenes[n]["frames"]) for n in train_names)
    t0 = time.time()
    for it in range(args.iters3):
        heads.train(); opt.zero_grad()
        loss_sum = 0.0
        for name in train_names:  # gradient accumulation, one backward per scene
            frames, dense = scenes[name]["frames"], scenes[name]["dense"]
            N = dense["active"].shape[1]
            mem = heads.octa.init_memory(1, N, dev)
            ts = TrackState(1, N)
            scene_loss = torch.tensor(0.0, device=dev)
            for t, fr in enumerate(frames):
                so = run_timestep(None, heads, dense, fr, None, ts, t, dev, mem, train=True,
                                  no_detector=True, cached_prop=det_cache[name][t])
                scene_loss = scene_loss + track_losses(dev, dense, t, so)
            if scene_loss.requires_grad:
                (scene_loss / total_frames).backward()
            loss_sum += scene_loss.item()
        nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 40 == 0:
            print(f"  [{variant} s{seed}] it {it} loss {loss_sum/total_frames:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return evaluate_variant(heads, scenes, det_cache, dev)


@torch.no_grad()
def evaluate_variant(heads, scenes, det_cache, dev):
    heads.eval()
    preds, gts = [], []
    frame_ctr = 0
    for name in VAL_SCENES:
        frames, dense = scenes[name]["frames"], scenes[name]["dense"]
        N = dense["active"].shape[1]
        mem = heads.octa.init_memory(1, N, dev)
        ts = TrackState(1, N)
        for t, fr in enumerate(frames):
            fid = frame_ctr
            so = run_timestep(None, heads, dense, fr, None, ts, t, dev, mem, train=False,
                              no_detector=True, cached_prop=det_cache[name][t])
            a, hp = so["active"], so["has_prior"]
            m = a & hp
            since = so["since"]
            n_in = dense["n_in"][t]
            if m.any():
                pos_ego = global_to_ego(so["octa"].pos[0].cpu().numpy(), fr["R_e"], fr["t_e"])
                score = torch.sigmoid(so["exist_logit"][0]).cpu().numpy()
                cls_pred = so["cls_logit"][0].argmax(-1).cpu().numpy()
                for i in np.where(m)[0]:
                    preds.append(dict(frame=fid, xy=pos_ego[i, :2], cls=int(cls_pred[i]), score=float(score[i])))
            for i in np.where(a)[0]:
                obs = n_in[i] >= N_MIN
                bucket = "obs" if obs else ("0-2" if since[i] <= 2 else "2-4" if since[i] <= 4
                                            else "4-6" if since[i] <= 6 else ">6")
                gts.append(dict(frame=fid, xy=dense["gt_pos_ego"][t, i, :2], cls=int(dense["cls"][i]),
                                obs=bool(obs), bucket=bucket))
            for j in range(len(so["unmatched_score"])):
                preds.append(dict(frame=fid, xy=so["unmatched_pos_ego"][j, :2], cls=int(so["unmatched_cls"][j]),
                                  score=float(so["unmatched_score"][j])))
            frame_ctr += 1

    def run_split(ignore_fn):
        g = [dict(frame=x["frame"], xy=x["xy"], cls=x["cls"], ignore=ignore_fn(x)) for x in gts]
        return map_evaluate(preds, g, len(CLASSES))

    res = dict(
        mAP_obs=run_split(lambda x: not x["obs"]),
        mAP_unobs=run_split(lambda x: x["obs"]),
        mAP_unobs_0_2=run_split(lambda x: x["obs"] or x["bucket"] != "0-2"),
        mAP_unobs_2_4=run_split(lambda x: x["obs"] or x["bucket"] != "2-4"),
        mAP_unobs_4_6=run_split(lambda x: x["obs"] or x["bucket"] != "4-6"),
        mAP_unobs_6p=run_split(lambda x: x["obs"] or x["bucket"] != ">6"),
    )
    return {k: dict(mAP=v["mAP"], n_pred=v["n_pred"], n_gt=v["n_gt_scored"]) for k, v in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/datasets/nuscenes"))
    ap.add_argument("--iters", type=int, default=150, help="stage-1 joint detector+OCTA iterations")
    ap.add_argument("--iters3", type=int, default=250, help="stage-3 OCTA-only iterations per variant/seed")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--variants", nargs="+", default=["nomem", "fixed", "adaptive"])
    ap.add_argument("--out", default="results_nuscenes_detection.json")
    ap.add_argument("--skip_stage1", action="store_true", help="reuse cache/detector.pt + cache/nuscenes_detections.pkl")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scenes = load_scenes(args.root)

    if args.skip_stage1 and os.path.exists("cache/nuscenes_detections.pkl"):
        det_cache = pickle.load(open("cache/nuscenes_detections.pkl", "rb"))
    else:
        detector, heads1 = train_stage1(args.root, scenes, args, dev)
        os.makedirs("cache", exist_ok=True)
        torch.save(detector.state_dict(), "cache/detector.pt")
        det_cache = cache_detections(args.root, detector, scenes, dev)

    allres = {}
    for v in args.variants:
        for s in args.seeds:
            allres[f"{v}/seed{s}"] = train_stage3(v, scenes, det_cache, args, dev, s)
    json.dump(allres, open(args.out, "w"), indent=1)
    for k, r in allres.items():
        print(k)
        for g, m in r.items():
            print(f"   {g:15s} mAP {m['mAP']:.4f}  n_pred={m['n_pred']:5d} n_gt={m['n_gt']:4d}")
