"""Curriculum-stage dataset selection shared by the analysis notebooks.

Both `workflows/exploratory.py` and `workflows/pipeline.py` split sessions into
a "Full" and an "ABReversal" curriculum-stage dataset, let the user toggle
between them, and then summarize whichever one is selected. This module holds
that shared logic so the two notebooks stay in sync.
"""

import pandas as pd

FULL_STAGES = [
    "LearningSets",
    "manual_LearningSets_v1.2.0_2Contrasts_5Rew_5NonRew",
    "manual_LearningSets_v1.2.0_2Contrasts_8Rew_10NonRew",
]
REVERSAL_STAGES = [
    "manual_LearningSets_v1.2.0_ABReversal_odor0v1_5Rew_5NonRew",
]

DATASET_OPTIONS = ["Full", "ABReversal"]


def curriculum_stage_session_ids(sessions: pd.DataFrame) -> dict[str, pd.Series]:
    """Session ids for each curriculum-stage dataset, keyed by `DATASET_OPTIONS`."""
    return {
        "Full": sessions.loc[
            sessions["curriculum_stage_name"].isin(FULL_STAGES), "session_id"
        ],
        "ABReversal": sessions.loc[
            sessions["curriculum_stage_name"].isin(REVERSAL_STAGES), "session_id"
        ],
    }


def select_trials_by_session(trials: pd.DataFrame, session_ids: pd.Series) -> pd.DataFrame:
    """Rows of `trials` whose `session_id` is in `session_ids`."""
    return trials[trials["session_id"].isin(session_ids)]


def session_day(session_ids: pd.Series) -> pd.Series:
    """The day/date token embedded in a `{subject}_{day}_...` session id."""
    return session_ids.str.split("_").str[1]


def summarize_dataset(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-animal days and reward-block transition counts for a trial set."""
    session_days = trials[["subject_id", "session_id"]].drop_duplicates()
    session_days["day"] = session_day(session_days["session_id"])
    days = session_days.groupby("subject_id")["day"].nunique().rename("days")

    reward_site_blocks = trials[
        (trials["site_label"] == "RewardSite") & trials["block"].notna()
    ]
    blocks_per_session = reward_site_blocks.groupby(["subject_id", "session_id"])[
        "block"
    ].nunique()
    block_summary = blocks_per_session.groupby(level="subject_id").agg(
        blocks="sum",
        block_transitions=lambda counts: counts.sub(1).clip(lower=0).sum(),
    )
    summary = days.to_frame().join(block_summary).reset_index()
    return summary.rename(columns={"subject_id": "animal"})
