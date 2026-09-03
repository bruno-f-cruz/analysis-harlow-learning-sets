"""Does an odor's own reward-status flip predict the animal's stop rate for it
next time, independent of which physical odor pair is in play?

Unlike the counterfactual analysis (first-stop-of-block, both odors), this
tracks each odor *identity*'s own timeline of consecutive occurrences across
blocks/sessions, regardless of how many blocks separate two sightings of it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.plotting import TWO_BY_TWO_COLORS, bootstrap_mean_ci

#: (prev_rewarded, curr_rewarded, label, color)
CONDITIONS = [
    (True, True, "Prev Rew\n-> Curr Rew", TWO_BY_TWO_COLORS[0]),
    (True, False, "Prev Rew\n-> Curr NoRew", TWO_BY_TWO_COLORS[1]),
    (False, True, "Prev NoRew\n-> Curr Rew", TWO_BY_TWO_COLORS[2]),
    (False, False, "Prev NoRew\n-> Curr NoRew", TWO_BY_TWO_COLORS[3]),
]


def odor_identity_bias_pairs(trials: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-occurrence pairs of the same odor, per animal.

    Collapses each block to one row per odor identity actually encountered
    (its stop rate within that block), then walks each odor's own occurrences
    in chronological order and pairs up consecutive ones: `prev_rewarded` /
    `curr_rewarded` are that odor's reward status at the earlier and later
    occurrence, and `p_stop_curr` is the animal's stop rate for it *this* time.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    odor_block = (
        rs.groupby(["subject_id", "session_id", "block", "odor_index"])
        .agg(
            p_stop=("has_choice", "mean"),
            n_trials=("has_choice", "count"),
            is_rewarded_odor=("is_rewarded_odor", "first"),
        )
        .reset_index()
        .sort_values(["subject_id", "odor_index", "session_id", "block"])
    )

    records = []
    for (subject, odor), grp in odor_block.groupby(["subject_id", "odor_index"]):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp) - 1):
            prev, curr = grp.iloc[i], grp.iloc[i + 1]
            records.append(
                {
                    "subject_id": subject,
                    "odor_index": int(odor),
                    "prev_rewarded": bool(prev["is_rewarded_odor"]),
                    "curr_rewarded": bool(curr["is_rewarded_odor"]),
                    "p_stop_curr": curr["p_stop"],
                    "n_trials_curr": int(curr["n_trials"]),
                }
            )
    return pd.DataFrame(records)


def plot_bias_by_odor_identity(pairs_df: pd.DataFrame):
    """Per-subject bar chart of mean p_stop_curr across the 4 CONDITIONS.

    The y-axis is sized to the bars actually drawn (not a fixed range), and a
    condition backed by a single pair (n=1, no bootstrapped CI) is skipped
    rather than drawn as a bar that looks as certain as a well-sampled one.
    """
    subjects = sorted(pairs_df["subject_id"].unique())
    rng = np.random.default_rng(0)

    bar_stats = {}  # (subject, xi) -> (mean, ci_lo, ci_hi, n, label_y)
    for subject in subjects:
        sub = pairs_df[pairs_df["subject_id"] == subject]
        for xi, (prev_rew, curr_rew, _label, _color) in enumerate(CONDITIONS):
            grp = sub[(sub["prev_rewarded"] == prev_rew) & (sub["curr_rewarded"] == curr_rew)][
                "p_stop_curr"
            ]
            n = len(grp)
            mean, ci_lo, ci_hi = bootstrap_mean_ci(grp.to_numpy(dtype=float), rng)
            if np.isnan(mean):
                continue
            bar_stats[(subject, xi)] = (mean, ci_lo, ci_hi, n, ci_hi + 0.04)

    y_hi = max((v[4] for v in bar_stats.values()), default=1.0)
    y_lo = min(0.0, min((v[1] for v in bar_stats.values()), default=0.0))
    pad = 0.08 * (y_hi - y_lo)
    y_top, y_bot = y_hi + pad, y_lo - pad

    fig, axes = plt.subplots(1, len(subjects), figsize=(5 * len(subjects), 5), sharey=True, squeeze=False)
    fig.suptitle(
        "P(stop) at next odor encounter — 4 conditions\n(prev block rewarded / not) x (curr block rewarded / not)",
        fontsize=10,
    )
    for ai, subject in enumerate(subjects):
        ax = axes[0][ai]
        ax.axvspan(-0.5, 1.5, color="#fff0eb", zorder=0)
        ax.axvspan(1.5, 3.5, color="#eaf3fb", zorder=0)
        for xi, (_, _, _label, color) in enumerate(CONDITIONS):
            stats = bar_stats.get((subject, xi))
            if stats is None:
                continue
            m, ci_lo, ci_hi, n, label_y = stats
            yerr = [[max(m - ci_lo, 0)], [max(ci_hi - m, 0)]]
            ax.bar(xi, m, color=color, alpha=0.85, width=0.65, zorder=2)
            ax.errorbar(xi, m, yerr=yerr, fmt="none", color="black", capsize=5, lw=1.5, zorder=3)
            ax.text(xi, label_y, f"n={n}", ha="center", va="bottom", fontsize=7, zorder=4)
        ax.axvline(1.5, color="black", lw=1.0, alpha=0.3, zorder=1)
        ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([c[2] for c in CONDITIONS], fontsize=8)
        ax.set_ylim(y_bot, y_top)
        ax.set_title(f"Subject {subject}", fontsize=10)
        if ai == 0:
            ax.set_ylabel("Average P(stop) in next block encounter")
        ax.text(
            0.5, 1.1, "Prev: Rewarded", ha="center", va="bottom", fontsize=7.5, color="#8b0000",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            2.5, 1.1, "Prev: Not Rewarded", ha="center", va="bottom", fontsize=7.5, color="#1a5276",
            transform=ax.get_xaxis_transform(),
        )
    fig.tight_layout()
    return fig
