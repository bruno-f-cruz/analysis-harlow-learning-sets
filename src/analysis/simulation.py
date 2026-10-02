"""Adapters that let simulated rollouts (``harlow_rl``) run through the real-data plots."""

import pandas as pd


def trials_from_rollout(df: pd.DataFrame) -> pd.DataFrame:
    """Map a rollout frame onto the dataset's trial schema.

    ``df`` has one row per (animal, trial) with ``subject_id``, ``block``, ``trial``,
    ``stay``, ``reward`` and ``odor_rewarded``. The result carries the columns that
    :func:`analysis.features.appearance_table` and
    :func:`analysis.plotting.plot_choice_by_odor_appearance` read, plus
    ``first_stop_rewarded`` (same meaning as :func:`analysis.features.label_first_stop`,
    vectorised because ``label_first_stop`` loops block by block in Python, which is
    too slow for simulated animals with thousands of blocks each). ``session_id`` is the
    animal, so every animal is one long session.
    """
    animal = df["subject_id"].astype(str)
    trials = pd.DataFrame(
        {
            "subject_id": animal,
            "session_id": animal,
            "block": df["block"].astype(float),
            "start_time": df["trial"].astype(float),
            "site_label": "RewardSite",
            "has_choice": df["stay"].astype(bool),
            "has_reward": df["reward"] > 0,
            "is_rewarded_odor": df["odor_rewarded"].astype(bool),
        },
        index=df.index,
    )
    keys = ["session_id", "block"]
    first_stop = (
        trials[trials["has_choice"]]
        .sort_values("start_time")
        .groupby(keys)["has_reward"]
        .first()
        .rename("first_stop_rewarded")
    )
    trials = trials.join(first_stop, on=keys)
    trials["first_stop_rewarded"] = trials["first_stop_rewarded"].astype("boolean")
    return trials
