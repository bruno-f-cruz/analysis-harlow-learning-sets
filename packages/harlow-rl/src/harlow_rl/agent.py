import numpy as np

# Agent protocol (what ``simulate`` calls):
#   act(state, odor) -> actions
#   update(state, odor, action, reward, next_state, next_odor, duration)
# ``state`` is the environment's observation (its ``state_mode``); ``odor`` is the
# current odor alone, for agents whose state is "only the now".


class QAgent:
    """Tabular semi-MDP Q-learning, ``n_agents`` independent learners in parallel.

    Bootstraps with ``gamma ** duration`` (gamma is per second), so the 7 s stay
    trials and 5 s leave trials are discounted differently.
    """

    def __init__(
        self,
        n_agents: int,
        n_states: int,
        n_actions: int = 2,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        q_init: float = 0.0,
        seed=None,
    ):
        self.n_agents = n_agents
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = np.full((n_agents, n_states, n_actions), q_init, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self._rows = np.arange(n_agents)

    def act(self, states, odors=None):
        q = self.q[self._rows, states]
        tie = q[:, 1] == q[:, 0]
        greedy = np.where(
            tie, self.rng.integers(0, 2, self.n_agents), q[:, 1] > q[:, 0]
        )
        explore = self.rng.random(self.n_agents) < self.epsilon
        return np.where(explore, self.rng.integers(0, 2, self.n_agents), greedy)

    def update(
        self, states, odors, actions, rewards, next_states, next_odors, durations
    ):
        rows = self._rows
        target = rewards + self.gamma**durations * self.q[rows, next_states].max(axis=1)
        self.q[rows, states, actions] += self.alpha * (
            target - self.q[rows, states, actions]
        )


class MixtureAgent:
    """Working-memory agent + sensory agent, blended at decision time.

    - WM agent: Q-table over the environment's state (e.g. ``state_mode="binary"``:
      odor, previous odor, rewarded / not). Its state forgets at block boundaries, and
      it learns fast.
    - Sensory agent: Q-table over the current odor only. Nothing resets it at block
      boundaries, so what it learned about an odor carries into later blocks (where
      the odor's rewarded status has been re-randomised: a bias, not a help).

    ``Q_mix = w * Q_wm + (1 - w) * Q_sens``; the action is a softmax on ``Q_mix``
    (inverse temperature ``beta``). Each table is trained on its own TD error for the
    action actually taken (with its own ``gamma``, per second); training the sum jointly would let the sensory table
    cancel its own bias.
    """

    def __init__(
        self,
        n_agents: int,
        n_states: int,
        n_odors: int,
        w: float = 0.5,
        beta: float = 10.0,
        alpha_wm: float = 0.1,
        alpha_sens: float = 0.1,
        gamma_wm: float = 0.95,
        gamma_sens: float = 0.95,
        q_init: float = 0.0,
        seed=None,
    ):
        self.n_agents = n_agents
        self.w, self.beta = w, beta
        self.gamma_wm, self.gamma_sens = gamma_wm, gamma_sens
        self.alpha_wm, self.alpha_sens = alpha_wm, alpha_sens
        self.q_wm = np.full((n_agents, n_states, 2), q_init, dtype=np.float64)
        self.q_sens = np.full((n_agents, n_odors, 2), q_init, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self._rows = np.arange(n_agents)

    def q_mix(self, states, odors):
        r = self._rows
        return self.w * self.q_wm[r, states] + (1 - self.w) * self.q_sens[r, odors]

    def p_stay(self, states, odors):
        q = self.q_mix(states, odors)
        return 1.0 / (1.0 + np.exp(-self.beta * (q[:, 1] - q[:, 0])))

    def act(self, states, odors):
        return (self.rng.random(self.n_agents) < self.p_stay(states, odors)).astype(
            np.int64
        )

    def update(
        self, states, odors, actions, rewards, next_states, next_odors, durations
    ):
        r = self._rows
        for q, s, s2, alpha, gamma in (
            (self.q_wm, states, next_states, self.alpha_wm, self.gamma_wm),
            (self.q_sens, odors, next_odors, self.alpha_sens, self.gamma_sens),
        ):
            target = rewards + gamma**durations * q[r, s2].max(axis=1)
            q[r, s, actions] += alpha * (target - q[r, s, actions])
