"""Block transitions: does what happened at the end of one block change how
willing the animal is to stop at the very first site of the next?

At a block's first site the animal has no evidence yet about the new odor
mapping, so P(Stop) there is its "naive" willingness to sample. Each
within-session transition (block ``b - 1`` -> block ``b``) is labelled by the
end of the previous block in two independent ways:

* outcome at the previous block's **last site**: stopped & rewarded, stopped
  & unrewarded, or skipped;
* outcome at the previous block's **last stop** (skipped sites after it are
  ignored): rewarded or unrewarded. Blocks with no stop at all drop out here.

Odor identity at the next block's first site is deliberately not split on --
the animal cannot know yet which odor is rewarded.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.features import block_window_index
from analysis.plotting import bootstrap_across_animals, plot_mean_ci_band
from analysis.plotting_style import INK_MUTED, animal_palette, categorical

_BLUE, _ORANGE, _AQUA = categorical(3)

#: (condition key, label, color, split) -- in plotting order. ``split`` groups
#: the two independent labellings of a transition (last site vs last stop).
TRANSITION_CONDITIONS = [
    ("stop_rew", "Stop, rewarded", _ORANGE, "last_site"),
    ("stop_norew", "Stop, unrewarded", _BLUE, "last_site"),
    ("skip", "Skip", _AQUA, "last_site"),
    ("last_stop_rew", "Last stop rewarded", _ORANGE, "last_stop"),
    ("last_stop_norew", "Last stop unrewarded", _BLUE, "last_stop"),
]
_SPLIT_LABELS = {"last_site": "Last site of block", "last_stop": "Last stop of block"}
#: Shorter x tick labels for the summary plot, where the split header above
#: each group already says "last site" / "last stop".
_TICK_LABELS = {
    "stop_rew": "Stop,\nrewarded",
    "stop_norew": "Stop,\nunrewarded",
    "skip": "Skip",
    "last_stop_rew": "Rewarded",
    "last_stop_norew": "Unrewarded",
}


def block_transition_table(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per within-session block transition ``b - 1 -> b``.

    A transition is only kept when *both* blocks are present in ``trials``
    and consecutive, so passing the degenerate-filtered frame drops every
    transition that touches an excluded block, while the unfiltered frame
    keeps them all.

    Returns ``subject_id, session_id, block`` (the *next* block) plus
    ``last_stopped``, ``last_rewarded`` (previous block's last site),
    ``last_stop_rewarded`` (previous block's last stop; ``<NA>`` if it never
    stopped) and ``next_first_stopped`` (stop at the next block's first site).
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    rs = rs.sort_values(["session_id", "block", "start_time"])
    by_block = rs.groupby(["subject_id", "session_id", "block"])

    blocks = pd.DataFrame(
        {
            "first_stopped": by_block["has_choice"].first().astype(bool),
            "last_stopped": by_block["has_choice"].last().astype(bool),
            "last_rewarded": by_block["has_reward"].last().astype(bool),
        }
    )
    stops = rs[rs["has_choice"].astype(bool)]
    blocks["last_stop_rewarded"] = (
        stops.groupby(["subject_id", "session_id", "block"])["has_reward"]
        .last()
        .astype("boolean")
    )
    blocks = blocks.reset_index()

    prev = blocks.drop(columns="first_stopped").assign(block=lambda d: d["block"] + 1)
    nxt = blocks[["subject_id", "session_id", "block", "first_stopped"]].rename(
        columns={"first_stopped": "next_first_stopped"}
    )
    return prev.merge(nxt, on=["subject_id", "session_id", "block"], how="inner")


def transition_conditions_long(table: pd.DataFrame) -> pd.DataFrame:
    """:func:`block_transition_table` reshaped to one row per (transition,
    condition), with a ``condition`` key from :data:`TRANSITION_CONDITIONS`.

    Every transition appears once under a last-site condition and, if its
    previous block had any stop, once more under a last-stop condition --
    group by ``condition`` rather than summing across conditions.
    """
    last_site = np.select(
        [
            table["last_stopped"] & table["last_rewarded"],
            table["last_stopped"] & ~table["last_rewarded"],
        ],
        ["stop_rew", "stop_norew"],
        default="skip",
    )
    has_stop = table["last_stop_rewarded"].notna()
    last_stop = table.loc[has_stop, "last_stop_rewarded"].map(
        {True: "last_stop_rew", False: "last_stop_norew"}
    )
    return pd.concat(
        [
            table.assign(condition=last_site),
            table[has_stop].assign(condition=last_stop.to_numpy()),
        ],
        ignore_index=True,
    ).assign(next_first_stopped=lambda d: d["next_first_stopped"].astype(float))


def _set_p_stop_axis(ax):
    ax.set_ylabel("P(Stop) at next block's first site")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])


def plot_first_site_p_stop_by_transition(
    trials: pd.DataFrame, ax=None, animal_colors: dict | None = None
):
    """P(Stop) at the next block's first site, per transition condition.

    One dot per animal (its mean over that condition's transitions), joined
    within each split; the cohort mean is a short bar with its bootstrapped
    95% CI across animals as a translucent box
    (:func:`analysis.plotting.bootstrap_across_animals`). The dotted line is
    P(Stop) at the first site over all transitions, pooled the same way.
    """
    table = block_transition_table(trials)
    long = transition_conditions_long(table)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    keys = [c[0] for c in TRANSITION_CONDITIONS]
    # leave a gap between the last-site and last-stop splits
    xpos = dict(zip(keys, [0, 1, 2, 3.5, 4.5]))

    rng = np.random.default_rng(0)
    stats = bootstrap_across_animals(long, "next_first_stopped", ["condition"], rng)
    baseline = bootstrap_across_animals(
        table.assign(all=0, v=table["next_first_stopped"].astype(float)), "v", ["all"], rng
    )
    ax.axhline(baseline["mean"].iloc[0], color=INK_MUTED, ls=":", lw=1, label="All transitions")

    per_animal = (
        long.groupby(["subject_id", "condition"])["next_first_stopped"].mean().unstack()
    )
    subjects = [str(s) for s in per_animal.index]
    animal_colors = animal_colors if animal_colors is not None else animal_palette(subjects)
    for subject_id, row in per_animal.iterrows():
        for split in _SPLIT_LABELS:
            split_keys = [c[0] for c in TRANSITION_CONDITIONS if c[3] == split]
            vals = row.reindex(split_keys)
            ax.plot(
                [xpos[k] for k in split_keys], vals.to_numpy(),
                color=animal_colors[str(subject_id)], lw=1, alpha=0.6, marker="o", ms=4,
            )

    half = 0.3
    for key, label, color, _ in TRANSITION_CONDITIONS:
        if key not in stats.index:
            continue
        mean, lo, hi = stats.loc[key, ["mean", "ci_lo", "ci_hi"]]
        x = xpos[key]
        ax.fill_between([x - half, x + half], lo, hi, color=color, alpha=0.35, linewidth=0)
        ax.plot([x - half, x + half], [mean, mean], color=color, lw=3)

    ax.set_xticks([xpos[k] for k in keys])
    ax.set_xticklabels([_TICK_LABELS[k] for k in keys])
    for split, label in _SPLIT_LABELS.items():
        split_x = [xpos[c[0]] for c in TRANSITION_CONDITIONS if c[3] == split]
        ax.text(np.mean(split_x), 1.08, label, ha="center", va="bottom", color=INK_MUTED,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.6, 5.1)
    _set_p_stop_axis(ax)
    ax.legend(loc="lower right")
    return ax


def first_site_p_stop_by_transition_window(
    trials: pd.DataFrame, window_blocks: int = 100, skip_blocks: int = 20
) -> pd.DataFrame:
    """Transition-level rows (:func:`transition_conditions_long`) tagged with
    the block ``window`` (:func:`analysis.features.block_window_index`) the
    *next* block falls in -- duplicated once per overlapping window -- plus
    ``x``, the cumulative block count at that window's end (same x-axis as
    the counterfactual by-window plots).
    """
    long = transition_conditions_long(block_transition_table(trials))
    windows = block_window_index(trials, window_blocks, skip_blocks)
    long = long.merge(
        windows[["subject_id", "session_id", "block", "window"]],
        on=["subject_id", "session_id", "block"],
        how="inner",
    )
    long["x"] = long["window"] * skip_blocks + window_blocks
    return long


def plot_first_site_p_stop_by_transition_window(
    trials: pd.DataFrame,
    window_blocks: int = 100,
    skip_blocks: int = 20,
    axes=None,
):
    """:func:`plot_first_site_p_stop_by_transition` as a timecourse over
    block windows: one panel per split (last site / last stop), one line per
    condition with its bootstrapped 95% CI across animals as a band.
    """
    from matplotlib.ticker import MaxNLocator

    long = first_site_p_stop_by_transition_window(trials, window_blocks, skip_blocks)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    rng = np.random.default_rng(0)
    stats = bootstrap_across_animals(long, "next_first_stopped", ["condition", "x"], rng)
    for ax, (split, split_label) in zip(axes, _SPLIT_LABELS.items()):
        for key, label, color, cond_split in TRANSITION_CONDITIONS:
            if cond_split != split or key not in stats.index.get_level_values("condition"):
                continue
            plot_mean_ci_band(ax, stats.loc[key], color, label=label, marker="o")
        ax.text(0.5, 1.0, split_label, transform=ax.transAxes, ha="center", va="bottom",
                color=INK_MUTED)
        ax.set_xlabel(f"Block number (window {window_blocks}, stride {skip_blocks})")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        _set_p_stop_axis(ax)
        ax.legend(loc="lower right")
    axes[-1].set_ylabel("")
    return axes
