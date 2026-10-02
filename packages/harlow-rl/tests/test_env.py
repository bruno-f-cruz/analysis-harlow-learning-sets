from itertools import pairwise

import numpy as np
from harlow_rl import (
    STAY,
    HarlowEnv,
    MixtureAgent,
    Outcome,
    QAgent,
    TaskConfig,
    decode,
    simulate,
)


def test_rewards_durations_outcomes():
    env = HarlowEnv(n_envs=64, seed=0)
    for _ in range(50):
        odor, rewarded = env.odor.copy(), env.pair[:, 0].copy()
        a = env.rng.integers(0, 2, 64)
        _, r, _, _, info = env.step(a)
        stay = a == STAY
        assert np.array_equal(r, (stay & (odor == rewarded)).astype(float))
        assert np.array_equal(info["duration"], np.where(stay, 7.0, 5.0))
        assert np.all(info["outcome"][~stay] == Outcome.SKIPPED)
        assert np.all(info["outcome"][stay & (odor == rewarded)] == Outcome.REWARDED)
        assert np.all(info["outcome"][stay & (odor != rewarded)] == Outcome.UNREWARDED)


def test_state_carries_previous_trial_and_resets_at_block():
    cfg = TaskConfig()
    env = HarlowEnv(cfg, n_envs=8, seed=1)
    for t in range(3 * cfg.block_len):
        odor = env.odor.copy()
        a = env.rng.integers(0, 2, 8)
        obs, _, _, _, info = env.step(a)
        cur, prev_odor, prev_out = decode(obs, cfg.n_odors)
        first = env.trial == 0
        assert np.all(prev_odor[first] == -1) and np.all(prev_out[first] == -1)
        assert np.array_equal(prev_odor[~first], odor[~first])
        assert np.array_equal(prev_out[~first], info["outcome"][~first])
        assert np.array_equal(cur, env.odor)


def test_blocks():
    cfg = TaskConfig()
    env = HarlowEnv(cfg, n_envs=16, seed=2)
    history = [env.pair.copy()]
    for t in range(5 * cfg.block_len):
        assert np.all(np.isin(env.odor, env.pair))
        _, _, _, _, info = env.step(np.zeros(16, dtype=int))
        assert np.array_equal(
            info["new_block"], info["trial_in_block"] == cfg.block_len - 1
        )
        if t % cfg.block_len == cfg.block_len - 1:
            history.append(env.pair.copy())
    for old, new in pairwise(history):
        assert not (
            new[:, :, None] == old[:, None, :]
        ).any()  # drawn from the remaining 5
    assert np.all(env.block == 5)


def test_reproducible():
    def roll(seed):
        env = HarlowEnv(n_envs=4, seed=seed)
        return [env.step(np.ones(4, dtype=int))[0] for _ in range(30)]

    assert np.array_equal(roll(3), roll(3))
    assert not np.array_equal(roll(3), roll(4))


def _precision(out, sl):
    """P(rewarded | stay) on trials after the first of each block."""
    m = (out["action"][sl] == STAY) & (out["trial_in_block"][sl] > 0)
    return out["reward"][sl][m].mean()


def test_q_agent_learns_the_rule():
    env = HarlowEnv(n_envs=16, seed=0)
    agent = QAgent(16, env.n_states, epsilon=0.1, q_init=1.0, seed=0)
    out = simulate(env, agent, 60_000)
    early, late = _precision(out, slice(0, 50)), _precision(out, slice(-5_000, None))
    assert early < 0.6  # near chance (0.5) before any learning
    # Ceiling is ~0.8, not 1: after a skipped trial the previous odor says nothing
    # about the pair, so those states stay a guess.
    assert late > 0.75


def test_speed():
    import time

    env = HarlowEnv(n_envs=32, seed=0)
    agent = QAgent(32, env.n_states, seed=0)
    t0 = time.perf_counter()
    simulate(env, agent, 10_000)
    print("steps/s (x32 agents):", 10_000 / (time.perf_counter() - t0))


def test_binary_outcome_state():
    cfg = TaskConfig(state_mode="binary")
    env = HarlowEnv(cfg, n_envs=32, seed=5)
    assert env.n_states == 7 * (1 + 7 * 2)
    for _ in range(40):
        odor = env.odor.copy()
        obs, _, _, _, info = env.step(env.rng.integers(0, 2, 32))
        _, prev_odor, prev_out = decode(obs, cfg.n_odors, env.n_outcomes)
        first = env.trial == 0
        assert np.all(prev_out[first] == -1)
        assert np.array_equal(prev_odor[~first], odor[~first])
        # skipped and unrewarded collapse onto the same "not rewarded" code
        assert np.array_equal(
            prev_out[~first], (info["outcome"][~first] != Outcome.REWARDED).astype(int)
        )
        assert obs.max() < env.n_states


def test_relational_state():
    cfg = TaskConfig(state_mode="relational")
    env = HarlowEnv(cfg, n_envs=32, seed=6)
    assert env.n_states == 5
    for _ in range(40):
        odor = env.odor.copy()
        obs, _, _, _, info = env.step(env.rng.integers(0, 2, 32))
        first = env.trial == 0
        assert np.all(obs[first] == 0)
        same = env.odor == odor
        expected = 1 + 2 * same + (info["outcome"] != Outcome.REWARDED)
        assert np.array_equal(obs[~first], expected[~first])


def test_relational_agent_learns():
    env = HarlowEnv(TaskConfig(state_mode="relational"), n_envs=16, seed=0)
    agent = QAgent(16, env.n_states, epsilon=0.1, q_init=1.0, seed=0)
    out = simulate(env, agent, 20_000)
    assert _precision(out, slice(-5_000, None)) > 0.75


def _mixture_run(w, n_steps=60_000, seed=0):
    env = HarlowEnv(TaskConfig(state_mode="binary"), n_envs=16, seed=seed)
    agent = MixtureAgent(
        16, env.n_states, env.config.n_odors, w=w, q_init=1.0, seed=seed
    )
    return simulate(env, agent, n_steps)


def test_mixture_components_alone():
    # WM alone solves the task; the odor-only agent cannot track per-block contingencies.
    wm = _precision(_mixture_run(1.0), slice(-5_000, None))
    sens = _precision(_mixture_run(0.0), slice(-5_000, None))
    assert wm > 0.75
    assert sens < 0.65


def test_mixture_bias_across_blocks():
    """First-trial P(stay) is higher for odors that paid the last time they were seen."""
    out = _mixture_run(0.5, n_steps=80_000)
    n_steps, n_animals = out["action"].shape
    last = np.full(
        (n_animals, 7), -1
    )  # last stay outcome per (animal, odor); -1 = never
    rows = np.arange(n_animals)
    stays = {Outcome.REWARDED: [], Outcome.UNREWARDED: []}
    for t in range(n_steps):
        odor, first = out["odor"][t], out["trial_in_block"][t] == 0
        if t >= n_steps // 2:  # first half warms up the odor values
            for oc, acts in stays.items():
                acts.extend(out["action"][t][first & (last[rows, odor] == oc)])
        stayed = out["action"][t] == 1
        last[rows[stayed], odor[stayed]] = out["outcome"][t][stayed]
    assert np.mean(stays[Outcome.REWARDED]) > np.mean(stays[Outcome.UNREWARDED]) + 0.01
