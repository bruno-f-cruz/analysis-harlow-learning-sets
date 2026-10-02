import numpy as np

from . import state as st
from .config import TaskConfig

LEAVE, STAY = 0, 1


class HarlowEnv:
    """Stay/leave Harlow learning-set task, vectorised over ``n_envs`` independent animals.

    Every array has a leading ``n_envs`` axis; one ``step`` is one trial. Time is
    reported in ``info["duration"]`` (semi-MDP), and the task is continuing (never
    terminates). Observations are integer state indices (see ``state``).
    """

    n_actions = 2

    def __init__(self, config: TaskConfig | None = None, n_envs: int = 1, seed=None):
        self.config = config = config or TaskConfig()
        self.n_envs = n_envs
        if config.state_mode not in st.STATE_MODES:
            raise ValueError(f"state_mode must be one of {st.STATE_MODES}")
        self.n_outcomes = st.N_OUTCOMES if config.state_mode == "full" else 2
        self.n_states = (
            st.N_RELATIONAL_STATES
            if config.state_mode == "relational"
            else st.n_states(config.n_odors, self.n_outcomes)
        )
        self._seed = seed
        self._rows = np.arange(n_envs)
        self.reset()

    def _draw_pairs(self, n, exclude=None):
        """Random (rewarded, other) odor pairs, avoiding the ``exclude`` pair per row."""
        r = self.rng.random((n, self.config.n_odors))
        if exclude is not None:
            np.put_along_axis(r, exclude, 2.0, axis=1)
        return np.argsort(r, axis=1)[:, :2]

    def _draw_odor(self, rows):
        return self.pair[rows, self.rng.integers(0, 2, len(rows))]

    def reset(self, seed=None):
        self.rng = np.random.default_rng(self._seed if seed is None else seed)
        n = self.n_envs
        self.pair = self._draw_pairs(n)  # column 0 is the rewarded odor
        self.trial = np.zeros(n, dtype=np.int64)
        self.block = np.zeros(n, dtype=np.int64)
        self.prev_odor = np.full(n, -1, dtype=np.int64)  # -1: first trial of a block
        self.prev_outcome = np.full(n, -1, dtype=np.int64)
        self.odor = self._draw_odor(self._rows)
        return self.observation

    @property
    def observation(self):
        if self.config.state_mode == "relational":
            return st.relational(self.odor, self.prev_odor, self.prev_outcome)
        code = np.where(
            self.prev_odor < 0,
            0,
            st.prev_code(self.prev_odor, self.prev_outcome, self.n_outcomes),
        )
        return st.encode(self.odor, code, self.config.n_odors, self.n_outcomes)

    def step(self, actions):
        cfg = self.config
        stay = np.asarray(actions) == STAY
        hit = stay & (self.odor == self.pair[:, 0])
        outcome = np.where(
            stay,
            np.where(hit, st.Outcome.REWARDED, st.Outcome.UNREWARDED),
            st.Outcome.SKIPPED,
        )
        info = {
            "duration": cfg.travel_s + cfg.dwell_s * stay,
            "outcome": outcome,
            "odor": self.odor,
            "rewarded_odor": self.pair[:, 0].copy(),
            "block": self.block.copy(),
            "trial_in_block": self.trial.copy(),
        }

        self.trial = self.trial + 1
        self.prev_odor, self.prev_outcome = self.odor.copy(), outcome.copy()
        done = self.trial == cfg.block_len
        info["new_block"] = done
        if done.any():
            rows = np.flatnonzero(done)
            self.pair[rows] = self._draw_pairs(len(rows), exclude=self.pair[rows])
            self.trial[rows] = 0
            self.block[rows] += 1
            self.prev_odor[rows] = self.prev_outcome[rows] = -1
        self.odor = self._draw_odor(self._rows)

        false = np.zeros(self.n_envs, dtype=bool)
        return self.observation, hit * cfg.reward, false, false, info
