from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    n_odors: int = 7
    block_len: int = 10
    dwell_s: float = 2.0  # paid only when staying
    travel_s: float = 5.0  # paid after either action
    reward: float = 1.0
    # "full": (odor, prev odor, prev outcome R/U/S); "binary": outcome is R / not R;
    # "relational": (prev odor == odor, prev rewarded), no odor identity at all.
    state_mode: str = "full"
