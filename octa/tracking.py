"""Glue between a per-frame detector and OCTA's per-object memory: a causal constant-velocity
motion prior, and greedy matching of detector proposals to track slots.

Track identity here is the GT instance_token (teacher-forced): each scene gets one slot per
instance that ever appears in it, active only while that instance exists. This keeps track
birth/death bookkeeping free (no dynamic slot allocation to debug) while keeping detection and
position estimation fully causal and non-oracle — see the "Scope" note in
`examples/nuscenes_detection.py` for exactly what this does and does not assume.
"""
import numpy as np

MATCH_DIST = 2.0  # metres; only gates which proposal feeds which track slot as an "observation"


class TrackState:
    """Per-(scene, slot) constant-velocity prior, updated causally frame by frame."""

    def __init__(self, B, N):
        self.pos = np.zeros((B, N, 3), np.float32)
        self.vel = np.zeros((B, N, 3), np.float32)
        self.last_obs = np.zeros((B, N, 3), np.float32)
        self.last_t = np.zeros((B, N), np.float32)
        self.seen = np.zeros((B, N), bool)

    def step(self, t, obs_mask, obs_pos, dt_default=0.5):
        """obs_mask (B,N) bool: a real (detector- or GT-derived) observation arrived this frame.
        obs_pos (B,N,3) valid where obs_mask. Returns ref_pos, ref_vel, since, has_prior for use
        AT this timestep, then advances the internal state for the next call."""
        dt = np.where(self.seen, np.maximum(t - self.last_t, 1e-2), dt_default).astype(np.float32)
        self.pos = np.where((self.seen & ~obs_mask)[..., None], self.pos + self.vel * dt[..., None], self.pos)
        upd = obs_mask & self.seen
        v_new = np.clip((obs_pos - self.last_obs) / np.maximum(t - self.last_t, 1e-2)[..., None], -20, 20)
        self.vel = np.where(upd[..., None], 0.5 * self.vel + 0.5 * v_new, self.vel)
        self.pos = np.where(obs_mask[..., None], obs_pos, self.pos)
        since = np.where(self.seen, t - self.last_t, 1e3).astype(np.float32)
        self.last_obs = np.where(obs_mask[..., None], obs_pos, self.last_obs)
        self.last_t = np.where(obs_mask, t, self.last_t)
        self.seen = self.seen | obs_mask
        return self.pos.copy(), self.vel.copy(), since, self.seen.copy()


def match_proposals(prop_pos_xy, prop_cls, gt_xy, gt_cls, active, max_dist=MATCH_DIST):
    """Greedy nearest-neighbour, same-class matching for ONE frame of ONE scene.
    prop_pos_xy (P,2), prop_cls (P,), gt_xy (N,2), gt_cls (N,), active (N,) bool.
    Returns matched (N,) int (-1 if unmatched) and unmatched (P,) bool proposal mask."""
    n, p = len(gt_xy), len(prop_pos_xy)
    matched = -np.ones(n, np.int64)
    used = np.zeros(p, bool)
    if p == 0 or not active.any():
        return matched, ~used
    idx_active = np.where(active)[0]
    d = np.linalg.norm(gt_xy[idx_active, None, :] - prop_pos_xy[None, :, :], axis=-1)
    d = np.where(gt_cls[idx_active, None] == prop_cls[None, :], d, 1e9)
    d[d > max_dist] = 1e9
    order = np.argsort(d.ravel())
    W = d.shape[1]
    for k in order:
        if d.ravel()[k] >= 1e9:
            break
        gi, pj = divmod(k, W)
        g = idx_active[gi]
        if used[pj] or matched[g] >= 0:
            continue
        matched[g] = pj
        used[pj] = True
    return matched, ~used
