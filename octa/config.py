from dataclasses import dataclass


@dataclass
class OCTAConfig:
    dim: int = 256
    num_heads: int = 8
    mem_size: int = 32          # M: episodic entries kept per object
    top_k: int = 4              # k: memories attended to per object
    # Visibility-conditioned temporal search range (seconds):
    #   T(v) = t_short + (1 - v) ** gamma * (t_long - t_short)
    t_short: float = 1.0
    t_long: float = 6.0
    gamma: float = 1.0
    adaptive_window: bool = True  # False -> fixed window of t_short (ablation)
    # Memory write gate: only reliable observations enter long-term memory.
    tau_vis: float = 0.5
    tau_conf: float = 0.5
    detach_memory: bool = True  # stop gradients through stored entries (truncated BPTT)
    # Input scaling so metric positions/velocities are O(1) inside the networks
    # (outputs of the state head are scaled back to metres and m/s).
    pos_scale: float = 1.0
    vel_scale: float = 1.0
