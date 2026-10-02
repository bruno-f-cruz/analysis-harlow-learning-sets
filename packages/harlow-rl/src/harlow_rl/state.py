"""State = (current odor, previous odor, previous outcome), as one integer.

The "previous" part is a single code: 0 means no previous trial (first trial of a
block), otherwise ``1 + prev_odor * n_outcomes + outcome``. ``n_outcomes`` is 3
(rewarded / unrewarded / skipped) or 2 (rewarded / not rewarded, with unrewarded
and skipped merged; see ``TaskConfig.binary_outcome``).
"""

from enum import IntEnum

import numpy as np

N_OUTCOMES = 3


class Outcome(IntEnum):
    REWARDED = 0
    UNREWARDED = 1
    SKIPPED = 2


def n_prev_codes(n_odors: int, n_outcomes: int = N_OUTCOMES) -> int:
    return 1 + n_odors * n_outcomes


def n_states(n_odors: int, n_outcomes: int = N_OUTCOMES) -> int:
    return n_odors * n_prev_codes(n_odors, n_outcomes)


def prev_code(prev_odor, outcome, n_outcomes: int = N_OUTCOMES):
    """``outcome`` is clipped to ``n_outcomes - 1``, which merges the non-rewarded ones."""
    return 1 + prev_odor * n_outcomes + np.minimum(outcome, n_outcomes - 1)


def encode(odor, code, n_odors: int, n_outcomes: int = N_OUTCOMES):
    return odor * n_prev_codes(n_odors, n_outcomes) + code


def decode(state, n_odors: int, n_outcomes: int = N_OUTCOMES):
    """Return (odor, prev_odor, outcome); prev_odor and outcome are -1 when absent."""
    state = np.asarray(state)
    odor, code = np.divmod(state, n_prev_codes(n_odors, n_outcomes))
    has_prev = code > 0
    prev_odor, outcome = np.divmod(np.where(has_prev, code - 1, 0), n_outcomes)
    return odor, np.where(has_prev, prev_odor, -1), np.where(has_prev, outcome, -1)


STATE_MODES = ("full", "binary", "relational")
N_RELATIONAL_STATES = (
    5  # first trial, or (same odor as previous?) x (previous rewarded?)
)


def relational(odor, prev_odor, prev_outcome):
    """0 on the first trial of a block, else ``1 + 2 * same_odor + (not rewarded)``."""
    rel = 1 + 2 * (odor == prev_odor) + (prev_outcome != Outcome.REWARDED)
    return np.where(prev_odor < 0, 0, rel)


def state_features(mode: str, n_odors: int):
    """Per state index: (odor, same_as_prev, prev_outcome).

    ``odor`` is -1 for ``relational`` (identity is not in the state); ``same_as_prev``
    and ``prev_outcome`` are -1 where they do not exist (first trial of a block).
    Outcome indices follow ``Outcome``; the non-rewarded ones are merged in
    ``binary`` / ``relational``.
    """
    if mode == "relational":
        k = np.maximum(np.arange(N_RELATIONAL_STATES) - 1, 0)
        first = np.arange(N_RELATIONAL_STATES) == 0
        return (
            np.full(N_RELATIONAL_STATES, -1),
            np.where(first, -1, k // 2),
            np.where(first, -1, k % 2),
        )
    n_out = N_OUTCOMES if mode == "full" else 2
    odor, prev_odor, outcome = decode(
        np.arange(n_states(n_odors, n_out)), n_odors, n_out
    )
    return odor, np.where(prev_odor < 0, -1, odor == prev_odor), outcome
