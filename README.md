# OCTA — Occlusion-Conditioned Temporal Attention

PyTorch implementation of the method in *OCTA: Occlusion-Conditioned Long-Term Attention for
Persistent Object Perception in Autonomous Driving*. It is a drop-in module that sits on top of a
detector/tracker that maintains persistent object queries.

| Paper idea | Where |
|---|---|
| Object-specific episodic memory (feature, position, velocity, confidence, visibility, timestamp) | `octa/memory.py` |
| Only high-confidence observations are written; occlusion-time predictions are not | write gate in `OCTA.forward` (`tau_vis`, `tau_conf`) |
| Search range expands as visibility drops | `OCTA.search_window`: `T(v) = t_short + (1-v)^γ (t_long - t_short)` |
| Ranking by feature similarity, spatial consistency, motion compatibility, observation quality, age; keep top-k | `octa/retrieval.py` |
| Sparse attention over the selected memories | `octa/attention.py` |
| Fusion with the current query and predicted motion state | `OCTA.forward` (gated fusion + state refinement) |

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
`visibility_loss` to supervise the visibility head with dataset visibility labels.

## Not in the paper (choices I made)
The paper gives no equations or hyperparameters. The defaults (`t_short=1s`, `t_long=6s`, `k=4`,
`M=32`, thresholds 0.5), the constant-velocity roll-forward of stored positions, the learned
positive weights combining the five ranking factors, and the relevance-as-attention-bias trick
are my own and should be tuned.

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
head it writes almost nothing, which is why the first run showed no difference between variants. The paper's mAP tables use
nuScenes-Permanence with a full detector; they are not reproduced here (the paper's OCTA numbers are
themselves marked as placeholders).

## nuScenes (mini)
```
python -m examples.nuscenes_occlusion --iters 400 --variants nomem fixed adaptive --seeds 0 1 2
```
`octa/nuscenes_data.py` parses the raw JSON and computes, for every annotated instance and keyframe,
LiDAR statistics of the points inside its box (verified: point counts match the dataset's
`num_lidar_pts` exactly on all 18,538 annotations). `examples/nuscenes_occlusion.py` trains on the 8 mini_train
scenes and evaluates on the 2 mini_val scenes (scene-0103, scene-0916). Object association uses GT boxes
(oracle) and there is no image branch or detector, so this tests the memory/attention stage only.
"Unobserved" means < 5 LiDAR points in the box, which includes distant objects, not only occluded ones.
Buckets are time since the object was last observed (the paper's 0-2 / 2-4 / 4-6 s split).

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
written only for reliable observations, which are older than "time since last observation".
`t_long`, `gamma` and the write thresholds are the knobs; I did not tune them (val is too small to tune on).
