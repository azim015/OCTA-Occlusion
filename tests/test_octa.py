"""Run with `python -m tests.test_octa` (or pytest)."""
import torch

from octa import OCTA, OCTAConfig

B, N, C = 2, 3, 32


def _model(**kw):
    cfg = OCTAConfig(dim=C, num_heads=4, mem_size=8, top_k=3, **kw)
    return OCTA(cfg).eval(), cfg


def _step(m, mem, t, vis, conf=0.9, query=None):
    q = torch.randn(B, N, C) if query is None else query
    pos, vel = torch.zeros(B, N, 3), torch.zeros(B, N, 3)
    return m(q, pos, vel, t, mem, confidence=torch.full((B, N), conf),
             visibility=torch.full((B, N), vis))


def test_write_gate_blocks_occluded_and_low_conf():
    m, _ = _model()
    mem = m.init_memory(B, N)
    assert _step(m, mem, 0.0, vis=0.1).written.sum() == 0      # occluded
    assert _step(m, mem, 0.5, vis=0.9, conf=0.2).written.sum() == 0  # unreliable
    assert not mem.valid.any()
    assert _step(m, mem, 1.0, vis=0.9).written.all()
    assert mem.valid.sum() == B * N


def test_window_grows_as_visibility_drops():
    m, cfg = _model()
    v = torch.tensor([1.0, 0.5, 0.0])
    w = m.search_window(v)
    assert torch.isclose(w[0], torch.tensor(cfg.t_short))
    assert torch.isclose(w[2], torch.tensor(cfg.t_long))
    assert w[0] < w[1] < w[2]
    m2, cfg2 = _model(adaptive_window=False)
    assert (m2.search_window(v) == cfg2.t_short).all()


def test_long_memory_only_reachable_when_occluded():
    m, _ = _model()  # t_short=1s, t_long=6s
    mem = m.init_memory(B, N)
    with torch.no_grad():
        _step(m, mem, 0.0, vis=1.0)                 # reliable observation at t=0
        out_vis = _step(m, mem, 4.0, vis=0.95, conf=0.0)  # visible, 4 s later: too old
        out_occ = _step(m, mem, 4.0, vis=0.0, conf=0.0)   # occluded: window = 6 s
    assert not out_vis.has_memory.any()
    assert out_occ.has_memory.all()
    assert not out_occ.written.any()                # occlusion-time predictions not stored


def test_topk_prefers_relevant_memory():
    m, _ = _model()
    mem = m.init_memory(B, N)
    with torch.no_grad():
        for i in range(6):
            _step(m, mem, float(i) * 0.1, vis=1.0)
        out = _step(m, mem, 0.7, vis=0.0, conf=0.0)
    assert out.topk_idx.shape == (B, N, 3)
    assert torch.allclose(out.attn.sum(-1), torch.ones(B, N, 4), atol=1e-5)
    # empty slots (6, 7) are never selected with positive weight
    picked = out.topk_idx.gather(-1, out.attn.mean(2).argmax(-1, keepdim=True))
    assert (picked < 6).all()


def test_ring_buffer_overwrites_oldest():
    m, cfg = _model()
    mem = m.init_memory(B, N)
    with torch.no_grad():
        for i in range(cfg.mem_size + 3):
            _step(m, mem, float(i), vis=1.0)
    assert mem.valid.all()
    assert mem.time[0, 0].min() == 3.0 and mem.time[0, 0].max() == cfg.mem_size + 2


def test_reset_clears_reassigned_tracks():
    m, _ = _model()
    mem = m.init_memory(B, N)
    with torch.no_grad():
        _step(m, mem, 0.0, vis=1.0)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[0, 1] = True
    mem.reset(mask)
    assert not mem.valid[0, 1].any() and mem.valid[0, 0].any()


def test_gradients_and_no_nan():
    m, _ = _model()
    m.train()
    mem = m.init_memory(B, N)
    total = 0
    for i in range(5):
        q = torch.randn(B, N, C, requires_grad=True)
        out = m(q, torch.zeros(B, N, 3), torch.zeros(B, N, 3), float(i) * 0.5, mem)
        total = total + out.query.sum() + out.pos.sum()
    total.backward()
    assert not any(torch.isnan(p.grad).any() for p in m.parameters() if p.grad is not None)
    assert m.scorer.raw_w.grad is not None and m.attn.q.weight.grad is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
