"""Does within-block performance decay over the course of a session?

Blocks are binned by how long into their session they started (15-minute
bins by default), and each block's departure from optimal performance --
stopping at every rewarded-odor site, skipping every non-rewarded one -- is
scored per bin. Uses trials *before* the degenerate-block filter: a block
where the animal stops at everything late in a session (e.g. satiation) is
exactly the kind of decay this is looking for, not noise to discard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.plotting import bootstrap_group_stats


def within_session_block_bins(trials: pd.DataFrame, bin_minutes: float = 15.0) -> pd.DataFrame:
    """Tag every kept block with the time bin its start falls into, relative
    to its own session's start -- so blocks from different sessions align on
    a shared "time since session start" axis. Also adds ``is_correct``: did
    the trial match optimal behaviour (stop at the rewarded odor, skip the
    non-rewarded one)?
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    rs = rs.sort_values(["session_id", "block", "start_time"])

    session_start = rs.groupby("session_id")["start_time"].transform("min")
    block_start = rs.groupby(["session_id", "block"])["start_time"].transform("min")
    elapsed = block_start - session_start
    elapsed_minutes = (
        elapsed.dt.total_seconds() / 60.0
        if pd.api.types.is_timedelta64_dtype(elapsed)
        else elapsed / 60.0
    )
    rs["time_bin"] = (elapsed_minutes // bin_minutes).astype(int)
    rs["is_correct"] = (rs["is_rewarded_odor"] & rs["has_choice"]) | (
        ~rs["is_rewarded_odor"] & ~rs["has_choice"]
    )
    return rs


def within_session_performance_gap(trials: pd.DataFrame, bin_minutes: float = 15.0) -> pd.DataFrame:
    """Per (subject, session, time_bin) accuracy and gap from optimal.

    Optimal is P(correct) = 1 in every bin; ``gap_from_optimal`` = 1 -
    accuracy, 0 meaning perfect. Computed per block first, then averaged
    within a (session, time_bin) -- the block, not the raw trial, is the
    unit throughout this codebase's other block-based analyses.
    """
    tagged = within_session_block_bins(trials, bin_minutes=bin_minutes)
    per_block = (
        tagged.groupby(["subject_id", "session_id", "block", "time_bin"])["is_correct"]
        .mean()
        .reset_index()
    )
    per_session_bin = (
        per_block.groupby(["subject_id", "session_id", "time_bin"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
    )
    per_session_bin["gap_from_optimal"] = 1.0 - per_session_bin["accuracy"]
    return per_session_bin


def plot_within_session_performance(per_session_bin: pd.DataFrame, bin_minutes: float = 15.0):
    """Per-animal gap-from-optimal across time-since-session-start bins:
    individual sessions faint, cross-session mean +/- bootstrapped 95% CI
    shaded -- same treatment as the other cohort plots in this codebase.
    """
    rng = np.random.default_rng(0)
    cohort = bootstrap_group_stats(
        per_session_bin["gap_from_optimal"],
        [per_session_bin["subject_id"], per_session_bin["time_bin"]],
        rng,
    ).reset_index()
    subjects = sorted(per_session_bin["subject_id"].unique())

    fig, axes = plt.subplots(
        1, len(subjects), figsize=(4.5 * len(subjects), 4.5), sharey=True, sharex=True,
        squeeze=False, constrained_layout=True,
    )
    for ax, subject in zip(axes[0], subjects):
        sub = per_session_bin[per_session_bin["subject_id"] == subject]
        for _, session_grp in sub.groupby("session_id"):
            session_grp = session_grp.sort_values("time_bin")
            ax.plot(
                session_grp["time_bin"],
                session_grp["gap_from_optimal"],
                color="#c0392b",
                linewidth=1,
                alpha=0.25,
            )

        sub_cohort = cohort[cohort["subject_id"] == subject].sort_values("time_bin")
        x = sub_cohort["time_bin"].to_numpy(dtype=float)
        mean = sub_cohort["mean"].to_numpy()
        ci_lo = sub_cohort["ci_lo"].to_numpy()
        ci_hi = sub_cohort["ci_hi"].to_numpy()
        ax.plot(x, mean, color="#c0392b", linewidth=2.2, label="Cross-session mean ± 95% CI")
        ax.fill_between(x, ci_lo, ci_hi, color="#c0392b", alpha=0.25, linewidth=0)

        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{int(b * bin_minutes)}-{int((b + 1) * bin_minutes)}" for b in x],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_xlabel("Time since session start (min)")
        ax.set_title(f"Subject {subject}")

    axes[0][0].set_ylabel("Gap from optimal performance\n(1 - P(correct); 0 = optimal)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Within-session performance decay ({int(bin_minutes)}-min bins)\n"
        "(faint = individual sessions; bold line + shaded band = cross-session "
        "mean & 95% CI; computed before the degenerate-block filter)"
    )
    return fig
