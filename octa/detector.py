"""A small CenterPoint-style BEV detector: pillar features -> U-Net -> per-class heatmap +
box regression. This is a real, from-scratch detector (no oracle GT association at inference):
it proposes boxes with confidence scores from LiDAR evidence alone, exactly like the plain
per-frame detectors (SparseDrive, BeyondSight's base detector) the paper compares OCTA against.

Simplifications made for tractability on nuScenes-mini (~320 training frames, no pretraining):
single LiDAR sweep (no multi-sweep accumulation), yaw-only box orientation, a hand-crafted
pillar encoder instead of a learned PointNet (see `voxelize.py`). These trade some accuracy
for a fast, easy-to-verify pipeline; they do not change what is being measured.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nuscenes_data import CLASSES
from .voxelize import PILLAR_DIM, pillar_features

NUM_CLASSES = len(CLASSES)


def _conv(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class BEVBackbone(nn.Module):
    """Two-level U-Net over the pillar grid; returns a full-resolution feature map."""

    def __init__(self, in_dim=PILLAR_DIM, base=32):
        super().__init__()
        self.stem = nn.Sequential(_conv(in_dim, base), _conv(base, base))
        self.down1 = _conv(base, base * 2, s=2)
        self.body1 = nn.Sequential(_conv(base * 2, base * 2), _conv(base * 2, base * 2))
        self.down2 = _conv(base * 2, base * 4, s=2)
        self.body2 = nn.Sequential(_conv(base * 4, base * 4), _conv(base * 4, base * 4))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.merge2 = _conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.merge1 = _conv(base * 2, base)
        self.out_dim = base

    def forward(self, x):
        s = self.stem(x)
        d1 = self.body1(self.down1(s))
        d2 = self.body2(self.down2(d1))
        u2 = self.merge2(torch.cat([self.up2(d2), d1], 1))
        u1 = self.merge1(torch.cat([self.up1(u2), s], 1))
        return u1  # (B, base, H, W)


class CenterHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hm = nn.Conv2d(dim, NUM_CLASSES, 1)
        self.reg = nn.Conv2d(dim, 2, 1)    # dx, dy within cell (metres)
        self.height = nn.Conv2d(dim, 1, 1)  # z (metres, ego frame)
        self.dim_ = nn.Conv2d(dim, 3, 1)    # log(w), log(l), log(h)
        self.rot = nn.Conv2d(dim, 2, 1)    # sin(yaw), cos(yaw)
        nn.init.constant_(self.hm.bias, -2.19)  # focal-loss init (~0.1 prior prob)

    def forward(self, feat):
        return dict(hm=torch.sigmoid(self.hm(feat).clamp(-10, 10)), reg=self.reg(feat),
                    height=self.height(feat), dim=self.dim_(feat), rot=self.rot(feat))


class Detector(nn.Module):
    def __init__(self, x_range=(-51.2, 51.2), y_range=(-51.2, 51.2), res=0.8, base=32):
        super().__init__()
        self.x_range, self.y_range, self.res = x_range, y_range, res
        self.backbone = BEVBackbone(base=base)
        self.head = CenterHead(self.backbone.out_dim)
        self.feat_dim = self.backbone.out_dim

    def grid_size(self):
        W = int(round((self.x_range[1] - self.x_range[0]) / self.res))
        H = int(round((self.y_range[1] - self.y_range[0]) / self.res))
        return H, W

    def voxelize(self, points, device):
        return pillar_features(points, self.x_range, self.y_range, self.res, device)

    def forward(self, grid):
        """grid: (B, PILLAR_DIM, H, W). Returns head outputs + the backbone feature map
        (used as the per-object descriptor OCTA consumes)."""
        feat = self.backbone(grid)
        out = self.head(feat)
        out["feat"] = feat
        return out

    def decode(self, out, score_thresh=0.05, top_k=200):
        """Per-batch-item list of dict(pos[N,3], size[N,3], yaw[N], cls[N], score[N], feat[N,C])."""
        hm = out["hm"]
        B, K, H, W = hm.shape
        pooled = F.max_pool2d(hm, 3, 1, 1)
        peak = (pooled == hm) & (hm > score_thresh)
        results = []
        for b in range(B):
            nz = peak[b].nonzero(as_tuple=False)  # (n, 3) = class, y, x
            if nz.numel() == 0:
                results.append(dict(pos=torch.zeros(0, 3, device=hm.device), size=torch.zeros(0, 3, device=hm.device),
                                    yaw=torch.zeros(0, device=hm.device), cls=torch.zeros(0, dtype=torch.long, device=hm.device),
                                    score=torch.zeros(0, device=hm.device), feat=torch.zeros(0, out["feat"].shape[1], device=hm.device)))
                continue
            cls_i, yy, xx = nz[:, 0], nz[:, 1], nz[:, 2]
            score = hm[b, cls_i, yy, xx]
            if len(score) > top_k:
                score, keep = score.topk(top_k)
                cls_i, yy, xx = cls_i[keep], yy[keep], xx[keep]
            cx = self.x_range[0] + (xx.float() + 0.5) * self.res + out["reg"][b, 0, yy, xx]
            cy = self.y_range[0] + (yy.float() + 0.5) * self.res + out["reg"][b, 1, yy, xx]
            cz = out["height"][b, 0, yy, xx]
            size = out["dim"][b, :, yy, xx].t().exp()
            rot = out["rot"][b, :, yy, xx].t()
            yaw = torch.atan2(rot[:, 0], rot[:, 1])
            feat = out["feat"][b, :, yy, xx].t()
            results.append(dict(pos=torch.stack([cx, cy, cz], 1), size=size, yaw=yaw, cls=cls_i,
                                score=score, feat=feat))
        return results


def gaussian_radius(w_px, l_px, min_overlap=0.7):
    """CornerNet/CenterNet radius so a Gaussian blob at this size gives >= min_overlap IoU."""
    a1, b1, c1 = 1, (w_px + l_px), w_px * l_px * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))) / 2
    a2, b2, c2 = 4, 2 * (w_px + l_px), (1 - min_overlap) * w_px * l_px
    r2 = (b2 + math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))) / 4
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (w_px + l_px), (min_overlap - 1) * w_px * l_px
    r3 = (b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))) / (2 * a3) if a3 > 0 else r1
    return max(min(r1, r2, r3), 1.0)


def _gaussian2d(radius, device):
    r = int(math.ceil(radius))
    y, x = torch.meshgrid(torch.arange(-r, r + 1, device=device), torch.arange(-r, r + 1, device=device), indexing="ij")
    sigma = radius / 3
    g = torch.exp(-(x * x + y * y) / (2 * sigma * sigma + 1e-6))
    g[g < 1e-4] = 0
    return g


def build_targets(gt_list, x_range, y_range, res, H, W, device):
    """gt_list: list (len B) of list-of-dict(center_ego, wlh, yaw_ego, cls). Returns target dict
    with per-cell heatmap, plus per-object (batch, y, x, params) for the regression loss."""
    hm = torch.zeros(len(gt_list), NUM_CLASSES, H, W, device=device)
    obj_b, obj_y, obj_x, obj_reg, obj_h, obj_dim, obj_rot = [], [], [], [], [], [], []
    for b, gts in enumerate(gt_list):
        for gt in gts:
            if gt["cls"] < 0:
                continue
            cx, cy, cz = gt["center_ego"]
            if not (x_range[0] < cx < x_range[1] and y_range[0] < cy < y_range[1]):
                continue
            fx, fy = (cx - x_range[0]) / res, (cy - y_range[0]) / res
            xi, yi = int(fx), int(fy)
            w, l, h = gt["wlh"]
            radius = gaussian_radius(w / res, l / res)
            gauss = _gaussian2d(radius, device)
            r = gauss.shape[0] // 2
            y0, y1 = max(yi - r, 0), min(yi + r + 1, H)
            x0, x1 = max(xi - r, 0), min(xi + r + 1, W)
            gy0, gx0 = y0 - (yi - r), x0 - (xi - r)
            k = int(gt["cls"])
            patch = hm[b, k, y0:y1, x0:x1]
            hm[b, k, y0:y1, x0:x1] = torch.maximum(patch, gauss[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)])
            obj_b.append(b); obj_y.append(yi); obj_x.append(xi)
            obj_reg.append([fx - xi - 0.5, fy - yi - 0.5])
            obj_h.append([cz])
            obj_dim.append([math.log(max(w, 1e-2)), math.log(max(l, 1e-2)), math.log(max(h, 1e-2))])
            obj_rot.append([math.sin(gt["yaw_ego"]), math.cos(gt["yaw_ego"])])
    t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)
    return dict(hm=hm, obj_b=t(obj_b, torch.long), obj_y=t(obj_y, torch.long), obj_x=t(obj_x, torch.long),
                reg=t(obj_reg).view(-1, 2), height=t(obj_h).view(-1, 1), dim=t(obj_dim).view(-1, 3),
                rot=t(obj_rot).view(-1, 2))


def focal_loss(pred, gt):
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    neg_w = (1 - gt).pow(4)
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos_loss = torch.log(pred) * (1 - pred).pow(2) * pos
    neg_loss = torch.log(1 - pred) * pred.pow(2) * neg_w * neg
    n_pos = pos.sum().clamp_min(1)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


def detection_loss(out, target):
    l_hm = focal_loss(out["hm"], target["hm"])
    n = target["obj_b"].numel()
    if n == 0:
        return l_hm, dict(hm=l_hm.item(), reg=0.0, n=0)
    b, y, x = target["obj_b"], target["obj_y"], target["obj_x"]
    l_reg = F.l1_loss(out["reg"][b, :, y, x], target["reg"])
    l_h = F.l1_loss(out["height"][b, :, y, x], target["height"])
    l_dim = F.l1_loss(out["dim"][b, :, y, x], target["dim"])
    l_rot = F.l1_loss(out["rot"][b, :, y, x], target["rot"])
    reg_loss = l_reg + l_h + l_dim + l_rot
    total = l_hm + reg_loss
    return total, dict(hm=l_hm.item(), reg=reg_loss.item(), n=n)
