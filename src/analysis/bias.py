"""Does an odor's own reward-status flip predict the animal's stop rate for it
next time, independent of which physical odor pair is in play?

Unlike the counterfactual analysis (first-stop-of-block, both odors), this
tracks each odor *identity*'s own timeline of consecutive occurrences across
blocks/sessions, regardless of how many blocks separate two sightings of it.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plotting import TWO_BY_TWO_COLORS, bootstrap_mean_ci
from analysis.plotting_style import animal_palette

#: (prev_rewarded, curr_rewarded, label, color)
CONDITIONS = [
    (True, True, "Prev Rew\n-> Curr Rew", TWO_BY_TWO_COLORS[0]),
    (True, False, "Prev Rew\n-> Curr NoRew", TWO_BY_TWO_COLORS[1]),
    (False, True, "Prev NoRew\n-> Curr Rew", TWO_BY_TWO_COLORS[2]),
    (False, False, "Prev NoRew\n-> Curr NoRew", TWO_BY_TWO_COLORS[3]),
]


def odor_identity_bias_pairs(trials: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-occurrence pairs of the same odor, per animal.

    Collapses each block to one row per odor identity actually encountered,
    using only that odor's *first* trial in the block (not an average over
    every occurrence) -- the animal's reaction the moment it re-meets this
    odor in a fresh block, before any within-block learning about *this*
    block's contingency can contaminate it. Then walks each odor's own
    occurrences in chronological order and pairs up consecutive ones:
    `prev_rewarded` / `curr_rewarded` are that odor's reward status at the
    earlier and later occurrence, and `p_stop_curr` is whether the animal
    stopped at its first trial *this* time (0/1; the mean over many pairs is
    what the plots report as a probability).
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    rs = rs.sort_values(["session_id", "block", "start_time"])
    odor_block = (
        rs.groupby(["subject_id", "session_id", "block", "odor_index"])
        .agg(
            p_stop=("has_choice", "first"),
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
            grp = sub[
                (sub["prev_rewarded"] == prev_rew) & (sub["curr_rewarded"] == curr_rew)
            ]["p_stop_curr"]
            n = len(grp)
            mean, ci_lo, ci_hi = bootstrap_mean_ci(grp.to_numpy(dtype=float), rng)
            if np.isnan(mean):
                continue
            bar_stats[(subject, xi)] = (mean, ci_lo, ci_hi, n, ci_hi + 0.04)

    y_hi = max((v[4] for v in bar_stats.values()), default=1.0)
    y_lo = min(0.0, min((v[1] for v in bar_stats.values()), default=0.0))
    pad = 0.08 * (y_hi - y_lo)
    y_top, y_bot = y_hi + pad, y_lo - pad

    fig, axes = plt.subplots(
        1, len(subjects), figsize=(5 * len(subjects), 5), sharey=True, squeeze=False
    )
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
            ax.errorbar(
                xi, m, yerr=yerr, fmt="none", color="black", capsize=5, lw=1.5, zorder=3
            )
            ax.text(
                xi, label_y, f"n={n}", ha="center", va="bottom", fontsize=7, zorder=4
            )
        ax.axvline(1.5, color="black", lw=1.0, alpha=0.3, zorder=1)
        ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([c[2] for c in CONDITIONS], fontsize=8)
        ax.set_ylim(y_bot, y_top)
        ax.set_title(f"Subject {subject}", fontsize=10)
        if ai == 0:
            ax.set_ylabel("Average P(stop) in next block encounter")
        ax.text(
            0.5,
            1.1,
            "Prev: Rewarded",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#8b0000",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            2.5,
            1.1,
            "Prev: Not Rewarded",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#1a5276",
            transform=ax.get_xaxis_transform(),
        )
    fig.tight_layout()
    return fig


def bias_first_site_pairs(trials: pd.DataFrame) -> pd.DataFrame:
    """For each block, the animal's stop/leave decision at the block's very
    first RewardSite trial (whichever odor identity that happens to be),
    paired with whether *that same odor identity* was rewarded the last
    time this animal saw it, in an earlier block.

    Deliberately does **not** condition on the current block's reward
    status for that odor -- at this first, uninformed encounter the animal
    has no way to know it yet, so splitting by it would credit the model
    with information the animal doesn't have. Unlike
    :func:`odor_identity_bias_pairs` (which uses every block where an odor
    appears, anywhere in the block), only occurrences that land on the
    block's first site are kept as the "current" event; a non-first-site
    occurrence can still supply a "previous" reward status for a later
    first-site encounter of that odor.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    rs = rs.sort_values(["session_id", "block", "start_time"])
    rs["_block_pos"] = rs.groupby(["session_id", "block"]).cumcount()

    odor_block = (
        rs.groupby(["subject_id", "session_id", "block", "odor_index"])
        .agg(
            stopped=("has_choice", "first"),
            is_rewarded_odor=("is_rewarded_odor", "first"),
            block_pos=("_block_pos", "first"),
        )
        .reset_index()
        .sort_values(["subject_id", "odor_index", "session_id", "block"])
    )

    records = []
    for (subject, odor), grp in odor_block.groupby(["subject_id", "odor_index"]):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp) - 1):
            prev, curr = grp.iloc[i], grp.iloc[i + 1]
            if curr["block_pos"] != 0:
                continue  # only keep it when this occurrence is the block's first site
            records.append(
                {
                    "subject_id": subject,
                    "odor_index": int(odor),
                    "prev_rewarded": bool(prev["is_rewarded_odor"]),
                    "stopped": bool(curr["stopped"]),
                }
            )
    return pd.DataFrame(records)


def plot_bias_first_site(
    pairs_df: pd.DataFrame,
    ax=None,
    reward_colors: dict | None = None,
    animal_colors: dict | None = None,
):
    """2-bar cohort P(Stop) at a block's first site (any odor identity),
    split only by whether *that* odor was rewarded the last time this
    animal saw it (:func:`bias_first_site_pairs`) -- not by the current
    block's reward status, which the animal can't know yet at this first,
    uninformed encounter.

    Each bar is the cohort mean +/- bootstrapped 95% CI across animals
    (:func:`analysis.plotting.bootstrap_mean_ci` on one row per animal per
    condition -- see AGENTS.md). Every animal's own pair of points is drawn
    on the bars and connected by a thin line, colored by
    :func:`analysis.plotting_style.animal_palette` (override via
    `animal_colors`), so the bars show the cohort effect and the lines show
    whether it holds animal-by-animal. Bar fill is `reward_colors` (default
    matches :func:`analysis.plotting.plot_choice_by_odor_appearance`'s
    ``{True: "tab:orange", False: "tab:blue"}``), now keyed to the odor's
    *previous* reward status since that's the only condition left.
    """
    reward_colors = (
        reward_colors
        if reward_colors is not None
        else {True: "tab:orange", False: "tab:blue"}
    )
    per_animal = (
        pairs_df.groupby(["subject_id", "prev_rewarded"])["stopped"]
        .mean()
        .reset_index()
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))

    subjects = sorted(per_animal["subject_id"].unique())
    animal_colors = (
        animal_colors if animal_colors is not None else animal_palette(subjects)
    )
    rng = np.random.default_rng(0)

    for x, prev_rewarded in [(0, True), (1, False)]:
        grp = per_animal[per_animal["prev_rewarded"] == prev_rewarded]["stopped"]
        mean, ci_lo, ci_hi = bootstrap_mean_ci(grp.to_numpy(dtype=float), rng)
        if np.isnan(mean):
            continue
        yerr = [[max(mean - ci_lo, 0)], [max(ci_hi - mean, 0)]]
        ax.bar(
            x,
            mean,
            color=reward_colors[prev_rewarded],
            alpha=0.85,
            width=0.65,
            zorder=2,
        )
        ax.errorbar(
            x, mean, yerr=yerr, fmt="none", color="black", capsize=5, lw=1.5, zorder=3
        )

    wide = per_animal.pivot(
        index="subject_id", columns="prev_rewarded", values="stopped"
    )
    for subject_id, row in wide.iterrows():
        if True not in row.index or False not in row.index:
            continue
        if pd.isna(row[True]) or pd.isna(row[False]):
            continue
        color = animal_colors.get(str(subject_id), "gray")
        ax.plot(
            [0, 1],
            [row[True], row[False]],
            color=color,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=4,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Prev Rewarded", "Prev Non-rewarded"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("P(Stop) at block's first site")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    return ax


def block_identity_table(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per block: (rewarded odor, non-rewarded odor) identity.

    Only blocks with exactly one rewarded and one non-rewarded odor are kept,
    so each block maps to a single ordered pair.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    rew = rs[rs["is_rewarded_odor"]].groupby(["subject_id", "session_id", "block"])[
        "odor_index"
    ]
    non = rs[~rs["is_rewarded_odor"]].groupby(["subject_id", "session_id", "block"])[
        "odor_index"
    ]
    out = pd.concat(
        [
            rew.agg(["nunique", "first"]).add_prefix("rew_"),
            non.agg(["nunique", "first"]).add_prefix("non_"),
        ],
        axis=1,
    ).dropna()
    out = out[(out["rew_nunique"] == 1) & (out["non_nunique"] == 1)]
    return (
        out.rename(columns={"rew_first": "rewarded", "non_first": "nonrewarded"})[
            ["rewarded", "nonrewarded"]
        ]
        .astype(int)
        .reset_index()
    )


def plot_block_identity_counts(blocks: pd.DataFrame, n_odors: int = 7, axes=None):
    """Per-animal heatmap of how often each (rewarded, non-rewarded) odor pair
    occurs as a block. Rows = rewarded odor, columns = non-rewarded odor."""
    subjects = sorted(blocks["subject_id"].unique())
    if axes is None:
        _, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = np.ravel(axes)
    letters = [chr(ord("A") + i) for i in range(n_odors)]
    counts = {
        s: blocks[blocks["subject_id"] == s]
        .groupby(["rewarded", "nonrewarded"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(n_odors), columns=range(n_odors), fill_value=0)
        for s in subjects
    }
    vmax = max(c.to_numpy().max() for c in counts.values())
    for ax, s in zip(axes, subjects):
        c = counts[s].to_numpy()
        im = ax.imshow(c, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(n_odors), letters)
        ax.set_yticks(range(n_odors), letters)
        ax.set_xlabel("Non-rewarded odor")
        ax.set_ylabel("Rewarded odor")
        ax.text(0.5, 1.02, str(s), transform=ax.transAxes, ha="center", va="bottom")
        for i in range(n_odors):
            for j in range(n_odors):
                if i != j:
                    ax.text(
                        j,
                        i,
                        c[i, j],
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if c[i, j] < 0.6 * vmax else "black",
                    )
    for ax in axes[len(subjects) : -1]:
        ax.set_visible(False)
    if len(axes) > len(subjects):
        pdf_ax = axes[len(subjects)]
        pairs = [(i, j) for i in range(n_odors) for j in range(n_odors) if i != j]
        colors = animal_palette([str(s) for s in subjects])
        probs = {}
        for s in subjects:
            c = counts[s].to_numpy()
            v = np.array([c[i, j] for i, j in pairs], dtype=float)
            probs[s] = v / v.sum()
        edges = np.linspace(0, max(v.max() for v in probs.values()) * 1.05, 31)
        pdf_ax.axis("off")
        n = len(subjects)
        gap = 0.04
        h_ax = (1 - gap * (n - 1)) / n
        ymax = 0
        hists = {}
        for s in subjects:
            h, _ = np.histogram(probs[s], bins=edges)
            hists[s] = h / h.sum()
            ymax = max(ymax, hists[s].max())
        for k, s in enumerate(subjects):
            sub = pdf_ax.inset_axes([0, 1 - (k + 1) * h_ax - k * gap, 1, h_ax])
            col = colors.get(str(s), "gray")
            sub.stairs(hists[s], edges, color=col, lw=1.6, fill=False)
            sub.stairs(hists[s], edges, color=col, alpha=0.3, fill=True)
            sub.axvline(1 / len(pairs), color="white", ls="--", lw=1.2)
            sub.set_xlim(edges[0], edges[-1])
            sub.set_ylim(0, ymax * 1.05)
            sub.set_yticks([])
            sub.text(
                0.98,
                0.85,
                str(s),
                transform=sub.transAxes,
                ha="right",
                va="top",
                color=col,
            )
            if k < n - 1:
                sub.set_xticks([])
                sub.spines["bottom"].set_visible(False)
            else:
                sub.set_xlabel(
                    "P(block identity)  (dashed = chance, 1/%d)" % len(pairs)
                )
            if k == n // 2:
                sub.set_ylabel("Fraction of block identities")
    cax = axes[2].inset_axes([1.04, 0, 0.04, 1])
    plt.colorbar(im, cax=cax, label="Number of blocks")
    return axes


def block_identity_second_trial_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Per (animal, block identity, first-stop outcome): occurrence probability
    of the identity and the mean counterfactual P(Stop) on the *other* odor.

    ``first_stop_rewarded=True`` -> P(Stop) at the next non-rewarded odor
    (``stop_next_bad``); ``False`` -> P(Stop) at the next rewarded odor
    (``stop_next_good``). ``p_occurrence`` is the identity's share of that
    animal's blocks (:func:`block_identity_table`); ``n`` is the number of
    blocks behind ``p_stop``.
    """
    from analysis.counterfactual import counterfactual_block_table

    ident = block_identity_table(trials)
    ident["p_occurrence"] = ident.groupby(["subject_id", "rewarded", "nonrewarded"])[
        "block"
    ].transform("size")
    ident["p_occurrence"] /= ident.groupby("subject_id")["block"].transform("size")
    cf = counterfactual_block_table(trials).merge(
        ident, on=["subject_id", "session_id", "block"]
    )
    cf["p_stop"] = np.where(
        cf["first_stop_rewarded"], cf["stop_next_bad"], cf["stop_next_good"]
    )
    cf["p_stop"] = pd.to_numeric(cf["p_stop"], errors="coerce")
    return (
        cf.dropna(subset=["p_stop"])
        .groupby(["subject_id", "rewarded", "nonrewarded", "first_stop_rewarded"])
        .agg(
            p_occurrence=("p_occurrence", "first"),
            p_stop=("p_stop", "mean"),
            n=("p_stop", "size"),
        )
        .reset_index()
    )


def plot_occurrence_vs_second_trial(table: pd.DataFrame, colors: tuple, axes=None):
    """5x2 scatter (rows = first stop rewarded / non-rewarded, cols = animals):
    identity occurrence probability vs counterfactual P(Stop), with an OLS fit."""
    from scipy import stats

    subjects = sorted(table["subject_id"].unique())
    if axes is None:
        _, axes = plt.subplots(
            2, len(subjects), figsize=(4 * len(subjects), 8), sharex=True, sharey=True
        )
    rows = [
        (True, "P(Stop) non-rewarded odor\n(first stop rewarded)", colors[0]),
        (False, "P(Stop) rewarded odor\n(first stop non-rewarded)", colors[1]),
    ]
    for r, (first_rew, ylabel, color) in enumerate(rows):
        for c, s in enumerate(subjects):
            ax = axes[r][c]
            d = table[
                (table["subject_id"] == s) & (table["first_stop_rewarded"] == first_rew)
            ]
            ax.scatter(
                d["p_occurrence"],
                d["p_stop"],
                s=25 + 4 * d["n"],
                color=color,
                alpha=0.6,
                edgecolor="white",
                linewidth=0.5,
            )
            if len(d) > 2 and d["p_occurrence"].nunique() > 1:
                lr = stats.linregress(d["p_occurrence"], d["p_stop"])
                xs = np.array([d["p_occurrence"].min(), d["p_occurrence"].max()])
                ax.plot(xs, lr.intercept + lr.slope * xs, color="white", lw=1.5)
                ax.text(
                    0.03,
                    0.05,
                    f"r = {lr.rvalue:.2f}, p = {lr.pvalue:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.text(
                    0.5, 1.02, str(s), transform=ax.transAxes, ha="center", va="bottom"
                )
            if r == 1:
                ax.set_xlabel("P(block identity)")
    return axes
