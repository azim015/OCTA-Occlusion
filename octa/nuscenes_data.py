"""nuScenes -> per-scene tensors for OCTA (reads the raw JSON + LiDAR .bin; no devkit needed).

Every annotated instance is a persistent object slot. The *observation* at each keyframe is
computed from the real LiDAR points inside that instance's annotated box, so it degrades
naturally under occlusion (few or zero points). Ground truth (position, velocity, size,
class) is available for every frame, including occluded ones, because nuScenes annotators
track objects through occlusion.

NOTE: object association uses ground-truth boxes (oracle), i.e. this evaluates the
memory/attention stage in isolation, not an end-to-end detector.
"""
import json
import os

import numpy as np

CLASSES = ["car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
           "motorcycle", "bicycle", "traffic_cone", "barrier"]
_CAT2CLS = {
    "vehicle.car": 0, "vehicle.truck": 1, "vehicle.bus.bendy": 2, "vehicle.bus.rigid": 2,
    "vehicle.trailer": 3, "vehicle.construction": 4, "human.pedestrian.adult": 5,
    "human.pedestrian.child": 5, "human.pedestrian.construction_worker": 5,
    "human.pedestrian.police_officer": 5, "vehicle.motorcycle": 6, "vehicle.bicycle": 7,
    "movable_object.trafficcone": 8, "movable_object.barrier": 9,
}
VIS_VALUE = {"1": 0.2, "2": 0.5, "3": 0.7, "4": 0.9}  # midpoints of the visibility bins
FEAT_DIM = 16
N_MIN = 5  # LiDAR points needed for the "tracker" to count an object as observed


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _table(root, version, name):
    with open(os.path.join(root, version, name + ".json")) as f:
        return {r["token"]: r for r in json.load(f)}


def box_features(local, inten, ring):
    """Per-box LiDAR statistics (FEAT_DIM-3; range/bearing are appended by the caller)."""
    n = len(local)
    f = np.zeros(13, np.float32)
    f[0] = np.log1p(n)
    if n:
        f[1:4] = local.mean(0)
        f[4:7] = local.std(0)
        f[7:10] = local.max(0) - local.min(0)
        f[10], f[11] = inten.mean() / 255.0, inten.std() / 255.0
        f[12] = ring.mean() / 32.0
    return f


def preprocess_scene(root, version, scene, tabs):
    sample_t, ann_t, inst_t, cat_t = tabs["sample"], tabs["ann"], tabs["inst"], tabs["cat"]
    sd_t, ego_t, cs_t = tabs["sd"], tabs["ego"], tabs["cs"]
    lidar_by_sample = {r["sample_token"]: r for r in sd_t.values()
                       if r["is_key_frame"] and r["filename"].startswith("samples/LIDAR_TOP")}
    samples, tok = [], scene["first_sample_token"]
    while tok:
        samples.append(sample_t[tok]); tok = sample_t[tok]["next"]
    T = len(samples)
    anns_by_sample = {s["token"]: [] for s in samples}
    for a in ann_t.values():
        if a["sample_token"] in anns_by_sample:
            anns_by_sample[a["sample_token"]].append(a)
    inst_tokens = sorted({a["instance_token"] for v in anns_by_sample.values() for a in v})
    idx = {k: i for i, k in enumerate(inst_tokens)}
    N = len(inst_tokens)

    z = lambda *s: np.zeros(s, np.float32)
    out = dict(active=np.zeros((T, N), bool), gt_pos=z(T, N, 3), gt_size=z(T, N, 3),
               vis_level=np.zeros((T, N), np.int64), ann_pts=np.zeros((T, N), np.int64),
               n_in=np.zeros((T, N), np.int64), feat=z(T, N, FEAT_DIM), obs_pos=z(T, N, 3),
               cls=np.full(N, -1, np.int64), time=z(T), ego_pos=z(T, 3))
    origin = None
    for t, s in enumerate(samples):
        sd = lidar_by_sample[s["token"]]
        ego, cs = ego_t[sd["ego_pose_token"]], cs_t[sd["calibrated_sensor_token"]]
        R_e, t_e = quat_to_mat(ego["rotation"]), np.array(ego["translation"])
        R_c, t_c = quat_to_mat(cs["rotation"]), np.array(cs["translation"])
        if origin is None:
            origin = t_e.copy()
        pts = np.fromfile(os.path.join(root, sd["filename"]), np.float32).reshape(-1, 5)
        pg = (pts[:, :3] @ R_c.T + t_c) @ R_e.T + t_e - origin  # LiDAR -> ego -> global (shifted)
        out["time"][t] = (s["timestamp"] - samples[0]["timestamp"]) * 1e-6
        out["ego_pos"][t] = t_e - origin
        for a in anns_by_sample[s["token"]]:
            i = idx[a["instance_token"]]
            c = np.array(a["translation"]) - origin
            w, l, h = a["size"]
            local = (pg - c) @ quat_to_mat(a["rotation"])  # box frame: x=length, y=width, z=height
            inside = (np.abs(local[:, 0]) <= l / 2) & (np.abs(local[:, 1]) <= w / 2) \
                & (np.abs(local[:, 2]) <= h / 2)
            f = box_features(local[inside], pts[inside, 3], pts[inside, 4])
            rel = c - out["ego_pos"][t]
            rng = np.linalg.norm(rel[:2])
            out["feat"][t, i] = np.concatenate([f, [rng / 50.0, rel[0] / max(rng, 1e-3),
                                                    rel[1] / max(rng, 1e-3)]])
            out["active"][t, i] = True
            out["gt_pos"][t, i] = c
            out["gt_size"][t, i] = [l, w, h]
            out["vis_level"][t, i] = int(a["visibility_token"])
            out["ann_pts"][t, i] = a["num_lidar_pts"]
            out["n_in"][t, i] = int(inside.sum())
            if inside.any():
                out["obs_pos"][t, i] = pg[inside].mean(0)
            out["cls"][i] = _CAT2CLS.get(cat_t[inst_t[a["instance_token"]]["category_token"]]["name"], -1)
    out["origin_note"] = "positions are global minus the ego position at frame 0"
    return out


def add_tracker_prior(d, dt_default=0.5):
    """Model-independent online tracker prior: the last observed state, dead-reckoned with a
    smoothed constant-velocity estimate while the object is unobserved (n_in < N_MIN)."""
    T, N = d["active"].shape
    ref_pos, ref_vel = np.zeros((T, N, 3), np.float32), np.zeros((T, N, 3), np.float32)
    hold_pos = np.zeros((T, N, 3), np.float32)
    has_prior = np.zeros((T, N), bool)
    since = np.full((T, N), 1e3, np.float32)  # seconds since last real observation
    last_obs, last_t, vel = np.zeros((N, 3), np.float32), np.zeros(N), np.zeros((N, 3), np.float32)
    seen = np.zeros(N, bool)
    pos = np.zeros((N, 3), np.float32)
    for t in range(T):
        obs = d["active"][t] & (d["n_in"][t] >= N_MIN)
        dt = d["time"][t] - d["time"][t - 1] if t else dt_default
        pos = np.where((seen & ~obs)[:, None], pos + vel * dt, pos)      # dead reckoning
        upd = obs & seen
        v_new = np.clip((d["obs_pos"][t] - last_obs) / np.maximum(d["time"][t] - last_t, 1e-3)[:, None], -20, 20)
        vel = np.where(upd[:, None], 0.5 * vel + 0.5 * v_new, vel)
        pos = np.where(obs[:, None], d["obs_pos"][t], pos)
        last_obs = np.where(obs[:, None], d["obs_pos"][t], last_obs)
        last_t = np.where(obs, d["time"][t], last_t)
        seen |= obs
        ref_pos[t], ref_vel[t], hold_pos[t] = pos, vel, last_obs
        has_prior[t] = seen & d["active"][t]
        since[t] = np.where(seen, d["time"][t] - last_t, 1e3)
    d.update(ref_pos=ref_pos, ref_vel=ref_vel, hold_pos=hold_pos, has_prior=has_prior, since=since)
    return d


def add_gt_velocity(d):
    T, N = d["active"].shape
    v = np.zeros((T, N, 3), np.float32)
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        both = d["active"][a] & d["active"][b] & (b > a)
        dt = max(d["time"][b] - d["time"][a], 1e-3)
        v[t] = np.where(both[:, None], (d["gt_pos"][b] - d["gt_pos"][a]) / dt, 0)
    d["gt_vel"] = v
    return d


def build_cache(root, version="v1.0-mini", out_path="cache/nuscenes_mini.npz"):
    tabs = dict(sample=_table(root, version, "sample"), ann=_table(root, version, "sample_annotation"),
                inst=_table(root, version, "instance"), cat=_table(root, version, "category"),
                sd=_table(root, version, "sample_data"), ego=_table(root, version, "ego_pose"),
                cs=_table(root, version, "calibrated_sensor"))
    scenes = list(_table(root, version, "scene").values())
    res = {}
    for sc in scenes:
        d = add_gt_velocity(add_tracker_prior(preprocess_scene(root, version, sc, tabs)))
        d.pop("origin_note")
        res[sc["name"]] = d
        print(sc["name"], d["active"].shape, flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **{f"{k}/{kk}": vv for k, v in res.items() for kk, vv in v.items()})
    return res


def load_cache(path="cache/nuscenes_mini.npz"):
    z = np.load(path)
    res = {}
    for key in z.files:
        s, k = key.split("/")
        res.setdefault(s, {})[k] = z[key]
    return res
