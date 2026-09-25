"""BEV pillar voxelization: raw LiDAR points -> a per-cell statistics grid.

No learned point encoder (no PointNet) to keep the detector small and fast to train on
nuScenes-mini; each pillar is summarized by simple, informative statistics instead (point
count, local mean/std/extent, intensity, ring). This mirrors the per-box statistics already
used in `nuscenes_data.box_features`.
"""
import torch

PILLAR_DIM = 10  # count, mean(x,y,z) local, std(x,y,z), z-extent, intensity, ring


def pillar_features(points, x_range, y_range, res, device):
    """points: (Ni, 5) x,y,z,intensity,ring in ego frame, one frame, numpy or tensor.
    Returns (PILLAR_DIM, H, W) feature grid, H = W = range*2/res."""
    p = torch.as_tensor(points, dtype=torch.float32, device=device)
    W = int(round((x_range[1] - x_range[0]) / res))
    H = int(round((y_range[1] - y_range[0]) / res))
    gx = ((p[:, 0] - x_range[0]) / res).long()
    gy = ((p[:, 1] - y_range[0]) / res).long()
    keep = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    p, gx, gy = p[keep], gx[keep], gy[keep]
    idx = gy * W + gx
    n_cells = H * W

    cx = x_range[0] + (gx.float() + 0.5) * res
    cy = y_range[0] + (gy.float() + 0.5) * res
    local = torch.stack([p[:, 0] - cx, p[:, 1] - cy, p[:, 2]], 1)

    def scatter_sum(src):
        out = torch.zeros(n_cells, src.shape[-1] if src.dim() > 1 else 1, device=device)
        src = src if src.dim() > 1 else src[:, None]
        out.index_add_(0, idx, src)
        return out

    count = scatter_sum(torch.ones(len(p), device=device)).squeeze(-1)
    safe_n = count.clamp_min(1)
    mean = scatter_sum(local) / safe_n[:, None]
    diff = local - mean[idx]
    var = scatter_sum(diff * diff) / safe_n[:, None]
    std = var.clamp_min(1e-8).sqrt()
    zmax = torch.full((n_cells,), -1e4, device=device).index_reduce_(0, idx, p[:, 2], "amax", include_self=True)
    zmin = torch.full((n_cells,), 1e4, device=device).index_reduce_(0, idx, p[:, 2], "amin", include_self=True)
    extent = (zmax - zmin).clamp_min(0) * (count > 0)
    inten = scatter_sum(p[:, 3:4]).squeeze(-1) / safe_n / 255.0
    ring = scatter_sum(p[:, 4:5]).squeeze(-1) / safe_n / 32.0

    feat = torch.cat([torch.log1p(count)[:, None], mean, std, extent[:, None],
                      inten[:, None], ring[:, None]], 1)
    return feat.t().contiguous().reshape(PILLAR_DIM, H, W)
