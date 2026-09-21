import torch
import torch.nn.functional as F


class ObjectMemory:
    """Per-object episodic memory: a ring buffer of M entries for each of N object slots.

    Each entry stores feature, position, velocity, confidence, visibility and timestamp.
    Writes are gated per object, so an object that is occluded does not overwrite its
    reliable history. Tensors are replaced (never modified in place) so autograd stays valid.
    """

    def __init__(self, batch, num_obj, mem_size, dim, device=None, dtype=torch.float32):
        B, N, M = batch, num_obj, mem_size
        kw = dict(device=device, dtype=dtype)
        self.mem_size = M
        self.feat = torch.zeros(B, N, M, dim, **kw)
        self.pos = torch.zeros(B, N, M, 3, **kw)
        self.vel = torch.zeros(B, N, M, 3, **kw)
        self.conf = torch.zeros(B, N, M, **kw)
        self.vis = torch.zeros(B, N, M, **kw)
        self.time = torch.zeros(B, N, M, **kw)
        self.valid = torch.zeros(B, N, M, dtype=torch.bool, device=device)
        self.ptr = torch.zeros(B, N, dtype=torch.long, device=device)

    def write(self, feat, pos, vel, conf, vis, t, mask):
        """Write one observation per object where `mask` (B, N) is True."""
        M = self.mem_size
        slot = F.one_hot(self.ptr, M).bool() & mask[..., None]  # (B, N, M)
        t = torch.as_tensor(t, device=feat.device, dtype=feat.dtype)
        t = t.expand(feat.shape[0]) if t.dim() == 0 else t

        def put(old, new):
            s = slot.reshape(*slot.shape, *([1] * (old.dim() - 3)))
            return torch.where(s, new.unsqueeze(2).to(old.dtype), old)

        self.feat = put(self.feat, feat)
        self.pos = put(self.pos, pos)
        self.vel = put(self.vel, vel)
        self.conf = put(self.conf, conf)
        self.vis = put(self.vis, vis)
        self.time = put(self.time, t[:, None].expand_as(conf))
        self.valid = self.valid | slot
        self.ptr = torch.where(mask, (self.ptr + 1) % M, self.ptr)

    def reset(self, obj_mask):
        """Clear memory of object slots (B, N) that were reassigned to a new track."""
        keep = ~obj_mask[..., None]
        self.valid = self.valid & keep
        self.ptr = torch.where(obj_mask, torch.zeros_like(self.ptr), self.ptr)

    def gather(self, idx):
        """Select entries by index (B, N, K) -> dict of (B, N, K, ...) tensors."""
        def g(x):
            i = idx.reshape(*idx.shape, *([1] * (x.dim() - 3))).expand(*idx.shape, *x.shape[3:])
            return torch.gather(x, 2, i)
        return {k: g(getattr(self, k)) for k in
                ("feat", "pos", "vel", "conf", "vis", "time", "valid")}
