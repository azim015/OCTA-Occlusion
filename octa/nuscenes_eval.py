"""nuScenes-style detection mAP: greedy matching by BEV center distance (not IoU, following the
official nuScenes detection metric), averaged over distance thresholds {0.5, 1, 2, 4} m and
classes, with 101-point recall interpolation (COCO-style) per class/threshold.

Supports an "ignore" split so a single evaluation run can report `mAP_obs` and `mAP_unobs`
separately (as in the paper's Table 2): GT boxes on the excluded side of the split are neither
scored (they don't count toward recall) nor penalized (a prediction that best-matches one is
simply discarded, not counted as a false positive) — the standard ignore-region trick from
COCO/KITTI-style evaluation, so precision on the scored split isn't distorted by objects the
metric isn't currently judging.
"""
import numpy as np

THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def _ap_101(recall, precision):
    if len(recall) == 0:
        return 0.0
    order = np.argsort(recall)
    recall, precision = recall[order], precision[order]
    for i in range(len(precision) - 2, -1, -1):  # precision envelope (monotone non-increasing)
        precision[i] = max(precision[i], precision[i + 1])
    r_levels = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, r_levels, side="left")
    p = np.array([precision[i] if i < len(precision) else 0.0 for i in idx])
    return float(p.mean())


def _match_class(preds, gts, thresh):
    """preds/gts: list of dict(frame, xy, score[preds only]). Greedy, score-descending, per
    frame. Returns (tp, fp) boolean arrays aligned with score-sorted preds, and n_scored_gt."""
    by_frame_gt = {}
    n_scored = 0
    for g in gts:
        by_frame_gt.setdefault(g["frame"], []).append(g)
        if not g["ignore"]:
            n_scored += 1
    matched = {fid: np.zeros(len(g), bool) for fid, g in by_frame_gt.items()}
    order = np.argsort([-p["score"] for p in preds])
    tp = np.zeros(len(preds), bool)
    fp = np.zeros(len(preds), bool)
    scores = np.zeros(len(preds))
    for rank, i in enumerate(order):
        p = preds[i]
        scores[rank] = p["score"]
        cand = by_frame_gt.get(p["frame"], [])
        best_d, best_j = thresh, -1
        for j, g in enumerate(cand):
            if matched[p["frame"]][j]:
                continue
            d = float(np.linalg.norm(p["xy"] - g["xy"]))
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0:
            fp[rank] = True
        else:
            matched[p["frame"]][best_j] = True
            if not cand[best_j]["ignore"]:
                tp[rank] = True
            # matched an ignored GT: neither TP nor FP
    return tp, fp, n_scored, scores


def average_precision(preds, gts, thresh):
    if sum(not g["ignore"] for g in gts) == 0:
        return None  # nothing to score in this split for this class
    if len(preds) == 0:
        return 0.0
    tp, fp, n_scored, _ = _match_class(preds, gts, thresh)
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / max(n_scored, 1)
    precision = tp_c / np.maximum(tp_c + fp_c, 1)
    return _ap_101(recall, precision)


def evaluate(preds, gts, num_classes, thresholds=THRESHOLDS):
    """preds: list of dict(frame, xy, cls, score). gts: list of dict(frame, xy, cls, ignore).
    Returns dict(mAP=..., per_class={cls: ap or None}, per_threshold=...)."""
    per_class = {}
    grid = np.full((num_classes, len(thresholds)), np.nan)
    for c in range(num_classes):
        pc = [p for p in preds if p["cls"] == c]
        gc = [g for g in gts if g["cls"] == c]
        aps = []
        for ti, th in enumerate(thresholds):
            ap = average_precision(pc, gc, th)
            if ap is not None:
                grid[c, ti] = ap
                aps.append(ap)
        per_class[c] = float(np.mean(aps)) if aps else None
    valid = ~np.isnan(grid)
    mAP = float(np.nanmean(grid)) if valid.any() else float("nan")
    return dict(mAP=mAP, per_class=per_class, n_pred=len(preds),
                n_gt_scored=sum(1 for g in gts if not g["ignore"]))
