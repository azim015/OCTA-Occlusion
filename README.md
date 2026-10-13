# OCTA — Occlusion-Conditioned Temporal Attention

PyTorch implementation of
**OCTA: Occlusion-Conditioned Long-Term Attention for Persistent Object Perception**
Abdolazim Rezaei, Mehdi Sookhak, Danial Zareei — Department of Computer Science, Texas A&M University–Corpus Christi.

OCTA is a drop-in module that sits on top of a detector/tracker that maintains persistent object
queries. It keeps an object-specific episodic memory of *reliable* past observations, widens its
temporal search range as an object's visibility drops, ranks candidate memories, and applies
sparse top-k attention over the best ones. Instead of uniformly propagating or fusing the full
temporal history (as BEVFormer, StreamPETR, RecurrentBEV, LST-BEV or MAD do), it retrieves
historical evidence only when it is needed.

## Method → code

Notation follows the paper. For object $i$ at time $t$: query $q_i^t$, confidence $c_i^t$,
visibility $v_i^t$, motion-predicted state $\hat{s}_i^t$.

| Paper | Equation | Where |
|---|---|---|
| Object-specific episodic memory; each item holds feature, position, velocity, confidence, visibility, timestamp | $\mathcal{M}_i = \{m_i^1, \dots, m_i^K\}$ | `octa/memory.py` |
| Write gate: only confident, visible observations are stored, so occlusion-time predictions never re-enter memory | $w_i^t = [\,c_i^t > \tau_c \wedge v_i^t > \tau_v\,]$ | `OCTA.forward` (`tau_conf` = $\tau_c$, `tau_vis` = $\tau_v$) |
| Adaptive temporal search range: visible objects use recent history, occluded objects search further back | (form not specified in the paper — see below) | `OCTA.search_window` |
| Relevance score: feature similarity, spatial consistency, motion compatibility, observation quality, memory age | $s_{ij} = \dfrac{q_i^t k_j^\top}{\sqrt{d}} + \lambda_p S^{pos}_{ij} + \lambda_m S^{mot}_{ij} + \lambda_q Q_j - \lambda_t \Delta t_j$ | `octa/retrieval.py` |
| Top-k selection | $\mathcal{M}_i^* = \mathrm{TopK}(\mathcal{M}_i, s_{ij})$ | `octa/retrieval.py` (`top_k`) |
| Sparse attention over selected memories | $h_i^t = \mathrm{Attn}(q_i^t, K(\mathcal{M}_i^*), V(\mathcal{M}_i^*))$ | `octa/attention.py` |
| Fusion with current query and predicted motion state | $\tilde{q}_i^t = F(q_i^t, h_i^t, \hat{s}_i^t)$ | `OCTA.forward` (gated fusion + state refinement) |
| Multi-task loss | $\mathcal{L} = \mathcal{L}_{det} + \lambda_{loc}\mathcal{L}_{loc} + \lambda_{vis}\mathcal{L}_{vis} + \lambda_{mem}\mathcal{L}_{mem}$ | $\mathcal{L}_{vis}$: `visibility_loss`; $\mathcal{L}_{det}$, $\mathcal{L}_{loc}$: host detector |

```python
from octa import OCTA, OCTAConfig
model = OCTA(OCTAConfig(dim=256, top_k=4, t_short=1.0, t_long=6.0))
memory = model.init_memory(batch, num_objects, device)
for t, frame in enumerate(sequence):            # query (B,N,C); ref_pos/ref_vel (B,N,3)
    out = model(query, ref_pos, ref_vel, t * dt, memory,
                confidence=det_score, visibility=None)   # visibility is predicted if None
    # out.query -> detection head, out.pos/out.vel -> refined state
```

Use `memory.reset(mask)` when an object slot is reassigned to a new track, and
`visibility_loss` to supervise the visibility head ($\mathcal{L}_{vis}$) with dataset visibility labels.

## Implementation choices not specified in the paper
The paper defines the write gate, relevance score, top-k selection, attention, fusion and loss, but
gives no hyperparameters and no functional form for the adaptive search window. These are my choices
and should be tuned:

- **Search window:** $T(v) = t_{short} + (1-v)^{\gamma}\,(t_{long} - t_{short})$.
- **Defaults:** `t_short=1s`, `t_long=6s` (matching the paper's 6 s maximum memory horizon), `k=4`,
  memory size `M=32`, $\tau_c = \tau_v = 0.5$.
- **Score weights:** $\lambda_p, \lambda_m, \lambda_q, \lambda_t$ are learned and constrained positive.
- **Score usage:** $s_{ij}$ is used both for top-k selection and as an additive attention bias.
- **Positions:** stored memory positions are rolled forward with constant velocity before computing $S^{pos}$.

## Results reported in the paper
The paper evaluates unobservable-object detection ($\mathrm{mAP}_{unobs}$) by occlusion duration
(Table 1, Figure 2):

| Method | 0–2 s | 2–4 s | 4–6 s |
|---|---|---|---|
| BEVFormer | 0.185 | 0.110 | 0.055 |
| StreamPETR | 0.228 | 0.155 | 0.095 |
| RecurrentBEV | 0.242 | 0.170 | 0.110 |
| LST-BEV | 0.255 | 0.185 | 0.130 |
| MAD | 0.275 | 0.205 | 0.165 |
| BeyondSight | 0.307 | 0.230 | 0.219 |
| **OCTA** | **0.335** | **0.285** | **0.260** |

Figure 3 of the paper varies the available memory horizon from 0.5 to 6 s; longer context improves
$\mathrm{mAP}_{unobs}$, with OCTA showing the largest gain.

**Status:** the paper labels OCTA's values as *placeholders* and the comparison as *illustrative*
(starred entries in Figures 2–3). They are not produced by this repository. The experiments below are
what this code actually measures.

## Run
```
python -m tests.test_octa
python -m examples.synthetic_occlusion
```
`synthetic_occlusion.py` is a mechanism check on toy data (300 iterations, one seed, ~4.5 min on GPU).
Result on occluded frames: no memory 4.02 m / identity cosine -0.001; fixed 1 s window 3.97 m / 0.111;
adaptive window 3.83 m / 0.160. The ordering matches the paper's claim, but the gains are small, the
model is undertrained and the run is a single seed, so treat it as a smoke test, not evidence.
Note: the write gate needs a real `confidence` (detector score); with the unsupervised confidence
head it writes almost nothing, which is why the first run showed no difference between variants.

## nuScenes (mini), oracle association
```
python -m examples.nuscenes_occlusion --iters 400 --variants nomem fixed adaptive --seeds 0 1 2
```
`octa/nuscenes_data.py` parses the raw JSON and computes, for every annotated instance and keyframe,
LiDAR statistics of the points inside its box (verified: point counts match the dataset's
`num_lidar_pts` exactly on all 18,538 annotations). `examples/nuscenes_occlusion.py` trains on the 8 mini_train
scenes and evaluates on the 2 mini_val scenes (scene-0103, scene-0916). Object association uses GT boxes
(oracle) and there is no image branch or detector, so this tests the memory/attention stage only.
"Unobserved" means < 5 LiDAR points in the box, which includes distant objects, not only occluded ones.
Buckets are time since the object was last observed, using the paper's 0–2 / 2–4 / 4–6 s split
(plus > 6 s).

Mean over 3 seeds (std <= 0.017 on all numbers below); val set is only ~1000 unobserved object-frames:

| unobserved bucket | n | position err (m): CV prior / no-mem / fixed / **adaptive** | size err (m): no-mem / fixed / **adaptive** | class acc: no-mem / fixed / **adaptive** |
|---|---|---|---|---|
| 0-2 s | 576 | 1.22 / 1.22 / 1.14 / **1.06** | 0.46 / 0.42 / **0.32** | 0.81 / 0.83 / **0.87** |
| 2-4 s | 211 | 1.65 / 1.64 / 1.64 / **1.60** | 0.58 / 0.58 / **0.49** | 0.74 / 0.72 / 0.75 |
| 4-6 s | 109 | 2.21 / 2.18 / 2.19 / 2.19 | 0.54 / 0.55 / 0.54 | 0.71 / 0.69 / 0.65 |
| >6 s | 90 | 4.69 / 4.67 / 4.68 / 4.65 | 0.53 / 0.54 / 0.52 | 0.65 / 0.63 / 0.62 |

Adaptive-window memory helps up to ~4 s and does nothing beyond. In the 4-6 s bucket the memory was
retrievable for only 0.3% of objects (56% at 0-2 s, 22% at 2-4 s). Likely causes (not yet tested): the visibility
head is trained toward 0.2 (not 0) for the lowest annotated bin, capping the window near 5 s, and memory is
written only when $w_i^t = 1$, so the newest stored item is older than "time since last observation".
`t_long`, `gamma`, $\tau_c$ and $\tau_v$ are the knobs; I did not tune them (val is too small to tune on).
This differs from the paper's claim that OCTA's advantage grows at 4–6 s.

## nuScenes with a real detector: mAP_unobs by occlusion duration (the paper's metric)
The experiment above scores position/size/class error under *oracle* GT-box association — useful
for isolating the memory mechanism, but not the metric the paper reports. This one computes the
paper's metric, $\mathrm{mAP}_{unobs}$ by occlusion duration (Table 1 / Figure 2), plus
$\mathrm{mAP}_{obs}$ for reference, with a real, non-oracle, from-scratch detector:

```
python -m examples.nuscenes_detection --iters 200 --iters3 200 --variants nomem fixed adaptive --seeds 0 1
```

**Pipeline** (`examples/nuscenes_detection.py`, `octa/detector.py`, `octa/voxelize.py`, `octa/tracking.py`,
`octa/nuscenes_eval.py`): a small CenterPoint-style single-sweep BEV detector (hand-crafted pillar
statistics, not a learned point encoder, into a 2-level U-Net + per-class heatmap head) is trained
jointly with OCTA end to end (stage 1), then frozen and cached (stage 2), then the OCTA head alone is
retrained three times — no-memory / fixed 1 s window / adaptive window — against the frozen, cached,
*real* detector output (stage 3). Detection mAP uses nuScenes' own metric definition: greedy matching
by BEV center distance at {0.5, 1, 2, 4} m, 101-point interpolated AP, averaged over thresholds and
classes. `mAP_obs`/`mAP_unobs` are computed with an ignore-region split (COCO/KITTI-style): the GT on
the excluded side is neither scored nor penalized, so precision on the scored side isn't distorted by
objects the metric isn't currently judging (`octa/nuscenes_eval.py`, unit-tested against synthetic
perfect/empty/partial predictions).

**Scope — read before trusting these numbers.** Track identity is teacher-forced to the GT
instance_token (one slot per instance, alive exactly while it has a GT box), which removes track
birth/death as a variable so the comparison isolates the memory mechanism. Everything else is real:
a slot only gets an observation when the frozen detector's own proposal lands within 2 m of it
(`octa/tracking.match_proposals`), and every detector proposal that matches no track is scored as a
genuine false positive. Single LiDAR sweep, yaw-only boxes, ~320 training frames — this is well below
a production detector, and well below the camera/BEV baselines in the paper's Table 1.

Mean over 2 seeds (differences below are all within 1–2 std devs — no variant separates from the others):

| Metric | no-mem | fixed 1s | adaptive | n (scored GT) | paper (OCTA, placeholder) |
|---|---|---|---|---|---|
| mAP_obs | 0.0888 ± 0.0004 | 0.0884 ± 0.0002 | 0.0884 ± 0.0004 | 2441 | — |
| mAP_unobs | 0.0424 ± 0.0006 | 0.0422 ± 0.0002 | 0.0427 ± 0.0002 | 2073 | — |
| mAP_unobs, 0–2 s | 0.1418 ± 0.0117 | 0.1304 ± 0.0011 | 0.1308 ± 0.0006 | 513 | 0.335 |
| mAP_unobs, 2–4 s | 0.0351 ± 0.0002 | 0.0349 ± 0.0001 | 0.0351 ± 0.0001 | 208 | 0.285 |
| mAP_unobs, 4–6 s | 0.0105 ± 0.0002 | 0.0121 ± 0.0014 | 0.0115 ± 0.0008 | 126 | 0.260 |
| mAP_unobs, >6 s | 0.0196 ± 0.0002 | 0.0192 ± 0.0001 | 0.0196 ± 0.0002 | 1226 | — |

**Unlike the oracle-association experiment above, memory shows no measurable effect on mAP here,**
and absolute values are far below the paper's. Diagnosed directly on the frozen detector's cached
output (no retraining needed to check):
- The detector itself is the bottleneck: it matches only 42% of *visible* GT objects (recall, not
  memory, caps mAP_obs) and 17% of *occluded* ones — expected given ~320 training frames and a single
  sweep, but it means detection quality dominates the metric far more than in the paper's
  well-trained baselines.
- 92% of all scored candidates (16,600 of 17,966) are raw unmatched detector proposals that never
  touch OCTA at all (`top_k=200` peaks/frame, most far below real-detection quality) — they dilute
  whatever the ~1,365 track-slot predictions that do run through OCTA can contribute to the PR curve.
- The 17% match rate on occluded objects, itself unaffected by the OCTA variant (same frozen
  detector, same matching), already caps how often deep, memory-dependent recovery is even the
  deciding factor for a given object-frame — consistent with the oracle experiment's finding that
  the effect is concentrated at 0–4 s and fades by 4–6 s.

In short: at this training scale, detector quality — not the memory mechanism — limits these
numbers, and a comparison at this scale can't distinguish the three variants on mAP. The oracle
experiment above remains the more informative test of OCTA itself; reproducing the paper's Table 1
would require a full-scale detector on the complete nuScenes dataset.

## Citation
```bibtex
@inproceedings{rezaei2027octa,
  title     = {OCTA: Occlusion-Conditioned Long-Term Attention for Persistent Object Perception},
  author    = {Rezaei, Abdolazim and Sookhak, Mehdi and Zareei, Danial},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027}
}
```

Code: https://github.com/azim015/OCTA
