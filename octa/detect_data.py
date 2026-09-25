"""Per-frame indexing for the nuScenes detector: raw LiDAR + ego-frame ground truth.

Unlike `nuscenes_data.py` (which aggregates LiDAR statistics per GT box for the oracle
memory experiment), this module keeps enough geometry to build a real BEV detector input
per frame and to place its predictions into a scene-consistent global frame for tracking:

- points are loaded on demand and transformed lidar -> ego (translation-invariant per frame,
  the natural frame for a CNN detector);
- GT boxes are expressed both in ego frame (cx, cy, cz, w, l, h, yaw) for detector supervision
  and via the same global-shift-by-scene-origin convention as `nuscenes_data.py` (so detector
  output can be handed to OCTA's memory, which needs a temporally consistent frame).
"""
import json
import os

import numpy as np

from .nuscenes_data import CLASSES, _CAT2CLS, _table, quat_to_mat

N_MIN = 5  # LiDAR points inside a GT box for it to count as "observed" this frame


def yaw_from_mat(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def index_scene(root, version, scene, tabs):
    """Returns a dict of per-frame numpy arrays/lists (ragged, one entry per keyframe)."""
    sample_t, ann_t, inst_t, cat_t = tabs["sample"], tabs["ann"], tabs["inst"], tabs["cat"]
    sd_t, ego_t, cs_t = tabs["sd"], tabs["ego"], tabs["cs"]
    lidar_by_sample = {r["sample_token"]: r for r in sd_t.values()
                       if r["is_key_frame"] and r["filename"].startswith("samples/LIDAR_TOP")}
    samples, tok = [], scene["first_sample_token"]
    while tok:
        samples.append(sample_t[tok]); tok = sample_t[tok]["next"]
    anns_by_sample = {s["token"]: [] for s in samples}
    for a in ann_t.values():
        if a["sample_token"] in anns_by_sample:
            anns_by_sample[a["sample_token"]].append(a)

    frames = []
    origin = None
    t0 = samples[0]["timestamp"]
    for s in samples:
        sd = lidar_by_sample[s["token"]]
        ego, cs = ego_t[sd["ego_pose_token"]], cs_t[sd["calibrated_sensor_token"]]
        R_e, t_e = quat_to_mat(ego["rotation"]), np.array(ego["translation"], np.float64)
        R_c, t_c = quat_to_mat(cs["rotation"]), np.array(cs["translation"], np.float64)
        if origin is None:
            origin = t_e.copy()
        gt = []
        for a in anns_by_sample[s["token"]]:
            c_g = np.array(a["translation"], np.float64)
            c_e = (c_g - t_e) @ R_e                                  # global -> ego
            R_box = quat_to_mat(a["rotation"])
            yaw_e = yaw_from_mat(R_e.T @ R_box)
            w, l, h = a["size"]
            cls = _CAT2CLS.get(cat_t[inst_t[a["instance_token"]]["category_token"]]["name"], -1)
            gt.append(dict(instance=a["instance_token"], center_ego=c_e.astype(np.float32),
                            wlh=np.array([w, l, h], np.float32), yaw_ego=yaw_e, cls=cls,
                            visibility=int(a["visibility_token"]), num_lidar_pts=a["num_lidar_pts"],
                            center_global_shift=(c_g - origin).astype(np.float32)))
        frames.append(dict(lidar_path=sd["filename"], R_e=R_e.astype(np.float32),
                            t_e=(t_e - origin).astype(np.float32), R_c=R_c.astype(np.float32),
                            t_c=t_c.astype(np.float32), time=(s["timestamp"] - t0) * 1e-6, gt=gt))
    return frames


def load_points_ego(root, frame):
    pts = np.fromfile(os.path.join(root, frame["lidar_path"]), np.float32).reshape(-1, 5)
    xyz = pts[:, :3] @ frame["R_c"].T + frame["t_c"]
    return np.concatenate([xyz, pts[:, 3:5]], 1)  # x,y,z,intensity,ring (ego frame)


def points_in_box_count(points_ego, gt):
    local = (points_ego[:, :3] - gt["center_ego"]) @ _yaw_mat(-gt["yaw_ego"]).T
    w, l, h = gt["wlh"]
    inside = (np.abs(local[:, 0]) <= l / 2) & (np.abs(local[:, 1]) <= w / 2) & (np.abs(local[:, 2]) <= h / 2)
    return int(inside.sum())


def _yaw_mat(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def build_index(root, version="v1.0-mini", out_path="cache/nuscenes_detect_index.json"):
    tabs = dict(sample=_table(root, version, "sample"), ann=_table(root, version, "sample_annotation"),
                inst=_table(root, version, "instance"), cat=_table(root, version, "category"),
                sd=_table(root, version, "sample_data"), ego=_table(root, version, "ego_pose"),
                cs=_table(root, version, "calibrated_sensor"))
    scenes = list(_table(root, version, "scene").values())
    out = {}
    for sc in scenes:
        frames = index_scene(root, version, sc, tabs)
        for fr in frames:
            pts = load_points_ego(root, fr)
            for gt in fr["gt"]:
                gt["n_in"] = points_in_box_count(pts, gt)
        out[sc["name"]] = frames
        print(sc["name"], len(frames), "frames", flush=True)

    def enc(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(o)
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    json.dump(out, open(out_path, "w"), default=enc)
    return out


def densify(frames):
    """Fixed per-scene instance slots (T, N, ...) dense arrays, analogous to
    `nuscenes_data.preprocess_scene` but built from the real ego-frame index above. Track
    identity here is the GT instance_token (teacher-forced during training/eval — see the
    "Scope" note in nuscenes_detection.py for what this does and doesn't assume)."""
    idx = {}
    for fr in frames:
        for gt in fr["gt"]:
            idx.setdefault(gt["instance"], len(idx))
    T, N = len(frames), max(len(idx), 1)
    z = lambda *s: np.zeros(s, np.float32)
    d = dict(active=np.zeros((T, N), bool), gt_pos=z(T, N, 3), gt_pos_ego=z(T, N, 3),
             gt_size=z(T, N, 3), gt_yaw=z(T, N), vis_level=np.zeros((T, N), np.int64),
             n_in=np.zeros((T, N), np.int64), cls=np.full(N, -1, np.int64), time=z(T),
             ego_R=np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)), ego_t=z(T, 3))
    for t, fr in enumerate(frames):
        d["time"][t], d["ego_R"][t], d["ego_t"][t] = fr["time"], fr["R_e"], fr["t_e"]
        for gt in fr["gt"]:
            i = idx[gt["instance"]]
            d["active"][t, i] = True
            d["gt_pos"][t, i] = gt["center_global_shift"]
            d["gt_pos_ego"][t, i] = gt["center_ego"]
            d["gt_size"][t, i] = gt["wlh"]
            d["gt_yaw"][t, i] = gt["yaw_ego"]
            d["vis_level"][t, i] = gt["visibility"]
            d["n_in"][t, i] = gt["n_in"]
            d["cls"][i] = gt["cls"]
    d["instances"] = list(idx)
    return d


def ego_to_global(p_ego, R_e, t_e):
    return p_ego @ R_e.T + t_e


def global_to_ego(p_global, R_e, t_e):
    return (p_global - t_e) @ R_e


def load_index(path="cache/nuscenes_detect_index.json"):
    raw = json.load(open(path))
    out = {}
    for scene, frames in raw.items():
        conv = []
        for fr in frames:
            fr = dict(fr)
            fr["R_e"] = np.array(fr["R_e"], np.float32)
            fr["t_e"] = np.array(fr["t_e"], np.float32)
            fr["R_c"] = np.array(fr["R_c"], np.float32)
            fr["t_c"] = np.array(fr["t_c"], np.float32)
            for gt in fr["gt"]:
                gt["center_ego"] = np.array(gt["center_ego"], np.float32)
                gt["wlh"] = np.array(gt["wlh"], np.float32)
                gt["center_global_shift"] = np.array(gt["center_global_shift"], np.float32)
            conv.append(fr)
        out[scene] = conv
    return out
