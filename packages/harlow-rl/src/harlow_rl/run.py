import numpy as np

from .env import HarlowEnv


def simulate(env: HarlowEnv, agent, n_steps: int, learn: bool = True) -> dict:
    """Run ``n_steps`` trials; returns arrays of shape ``(n_steps, n_envs)``."""
    keys = ("duration", "outcome", "odor", "rewarded_odor", "block", "trial_in_block")
    out = {k: np.empty((n_steps, env.n_envs), dtype=np.int64) for k in keys}
    out["duration"] = out["duration"].astype(np.float64)
    out["action"] = np.empty((n_steps, env.n_envs), dtype=np.int64)
    out["reward"] = np.empty((n_steps, env.n_envs), dtype=np.float64)

    s, odor = env.observation, env.odor
    for t in range(n_steps):
        a = agent.act(s, odor)
        s2, r, _, _, info = env.step(a)
        if learn:
            agent.update(s, odor, a, r, s2, env.odor, info["duration"])
        out["action"][t], out["reward"][t] = a, r
        for k in keys:
            out[k][t] = info[k]
        s, odor = s2, env.odor
    return out
