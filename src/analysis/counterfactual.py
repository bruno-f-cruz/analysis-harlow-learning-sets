"""Counterfactual learning: does the animal update the odor it *didn't* just
sample, not just the one it did?

After a block's first stop the animal holds exactly one piece of evidence
about the current odor mapping. A purely factual learner only updates the
odor it just sampled; a counterfactual learner also updates the odor it did
not sample (rewarded here implies not rewarded there, and vice versa). Each
block is split by whether its first stop was rewarded, and for each split we
score the decision at the *next* encounter of each odor type.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.features import block_window_index, blocks_first_last_tags, expand_to_block_windows
from analysis.plotting import (
    TWO_BY_TWO_COLORS,
    bootstrap_diff_ci,
    bootstrap_group_stats,
    bootstrap_mean_ci,
    ci_errorbar,
    plot_mean_ci_band,
)
from analysis.plotting_style import INK_MUTED, INK_PRIMARY, animal_palette, diverging_cmap

#: (first_stop_rewarded, next_odor_is_rewarded, short_label, ideal_p_stop)
COUNTERFACTUAL_CELLS = [
    (True, True, "1st stop REW\n→ next REW", 1.0),
    (True, False, "1st stop REW\n→ next NOREW", 0.0),
    (False, True, "1st stop NOREW\n→ next REW", 1.0),
    (False, False, "1st stop NOREW\n→ next NOREW", 0.0),
]
COUNTERFACTUAL_CELL_KEYS = [(a, b) for a, b, _, _ in COUNTERFACTUAL_CELLS]
COUNTERFACTUAL_COLORS = TWO_BY_TWO_COLORS

#: How each plottable value is rendered. `ideal` is the target value in each
#: of the four COUNTERFACTUAL_CELLS columns; `cmap` is sequential for the
#: polarity-corrected `accuracy` and diverging (red = stop-ish, blue =
#: leave-ish) for the raw probabilities, whose targets flip per column.
_VALUE_STYLES = {
    "accuracy": {
        "cmap": "viridis",
        "label": "accuracy (higher = better in every row)",
        "ideal": (1.0, 1.0, 1.0, 1.0),
    },
    "p_stop": {
        "cmap": "coolwarm",
        "label": "P(stop) at next site",
        "ideal": tuple(ideal for _, _, _, ideal in COUNTERFACTUAL_CELLS),
    },
    "p_leave": {
        "cmap": "coolwarm",
        "label": "P(leave) at next site",
        "ideal": tuple(1.0 - ideal for _, _, _, ideal in COUNTERFACTUAL_CELLS),
    },
}


def _style(value):
    try:
        return _VALUE_STYLES[value]
    except KeyError:
        raise ValueError(
            f"value must be one of {sorted(_VALUE_STYLES)}, got {value!r}"
        ) from None


def _require_value(matrix, value):
    if value not in matrix.columns:
        available = sorted(c for c in matrix.columns if c in _VALUE_STYLES)
        raise KeyError(
            f"{value!r} is not a column of the supplied matrix (it has {available}). "
            "Rebuild it with counterfactual_session_matrix(trials)."
        )


def _text_color(v, cmap):
    """Black on the light part of `cmap`, white elsewhere, for cell annotations."""
    light = v > 0.75 if cmap == "viridis" else 0.35 < v < 0.65
    return "black" if light else "white"


def counterfactual_block_table(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per block with the animal's decision at the next odor of each type.

    For every (session_id, block) the RewardSite trials are walked in
    temporal order and the first stop (has_choice) is located; that stop's
    has_reward defines first_stop_rewarded. Strictly *after* that trial, the
    first rewarded-odor site and the first non-rewarded-odor site are found
    and whether the animal stopped there is recorded. Blocks where the
    animal never stopped are omitted (no split can be assigned).
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    rs = rs.sort_values(["session_id", "block", "start_time"])
    subject_map = (
        rs.drop_duplicates("session_id").set_index("session_id")["subject_id"].to_dict()
    )

    records = []
    for (session_id, block), grp in rs.groupby(["session_id", "block"], sort=False):
        choice = grp["has_choice"].to_numpy(dtype=bool)
        if not choice.any():
            continue  # never stopped -> block is unlabelled
        first = int(np.argmax(choice))

        rewarded_odor = grp["is_rewarded_odor"].to_numpy(dtype=bool)
        post_choice = choice[first + 1 :]
        post_type = rewarded_odor[first + 1 :]

        def _first_stop_of_type(is_rewarded):
            hits = np.flatnonzero(post_type == is_rewarded)
            if hits.size == 0:
                return pd.NA
            return bool(post_choice[hits[0]])

        records.append(
            {
                "subject_id": subject_map[session_id],
                "session_id": session_id,
                "block": int(block),
                "first_stop_rewarded": bool(grp["has_reward"].to_numpy()[first]),
                "first_stop_pos": first,
                "stop_next_good": _first_stop_of_type(True),
                "stop_next_bad": _first_stop_of_type(False),
            }
        )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    for col in ("stop_next_good", "stop_next_bad"):
        out[col] = out[col].astype("boolean")
    return out


def plot_counterfactual_p_stop_by_window(
    trials: pd.DataFrame,
    first_stop_rewarded: bool,
    window_blocks: int = 30,
    skip_blocks: int | None = None,
    subtract_chance: bool = False,
    ax=None,
    animal_colors: dict | None = None,
    reward_colors: dict | None = None,
):
    """P(Stop) at the next occurrence of the *other* odor type after a
    block's first stop (:func:`counterfactual_block_table`'s
    ``stop_next_good``/``stop_next_bad``), aggregated into `window_blocks`-block
    windows (:func:`analysis.features.block_window_index`). ``skip_blocks``
    (default: ``window_blocks``, i.e. non-overlapping) is the stride between
    window starts; pass e.g. ``skip_blocks=15`` with ``window_blocks=30`` for
    50%-overlapping windows.

    Windows are computed over each animal's *entire* block stream, not
    restricted to blocks matching `first_stop_rewarded` -- so window
    boundaries stay calendar-aligned across animals and across this
    function's two mirror-image calls; the condition is applied only when
    aggregating each window's P(Stop). The x-axis is the literal cumulative
    block count at each window's end (``window_blocks``, ``window_blocks +
    skip_blocks``, ``window_blocks + 2 * skip_blocks``, ...) -- overlapping
    windows just make consecutive points share more of their blocks.

    ``first_stop_rewarded=False`` plots ``stop_next_good``: after a
    *non-rewarded* first stop, does the animal now stop for the rewarded
    odor the next time it appears (ideally rises with training).
    ``first_stop_rewarded=True`` plots ``stop_next_bad``: after a *rewarded*
    first stop, does the animal still stop for the non-rewarded odor the
    next time it appears (ideally falls with training). Blocks where that
    odor never recurred before the block ended contribute no data point
    (``stop_next_good``/``stop_next_bad`` is ``<NA>`` there, dropped).

    ``subtract_chance=True`` subtracts, per animal per window,
    :func:`first_site_chance_by_window`'s ``stopped_first_site`` -- P(Stop)
    at the block's very first RewardSite, i.e. before the animal has any
    evidence about that block's odor mapping, the same "no information"
    chance level `pipeline.py` overlays on the counterfactual cohort plots.
    The y-axis becomes a signed P(Stop) - chance difference (no longer
    bounded to ``[0, 1]``) with a dashed zero line marking "no better than
    chance"; the plotted mean/CI is bootstrapped on the already-subtracted
    per-animal-per-window values, not on the two probabilities separately.

    Individual animals are drawn as thin, semi-transparent lines (colored by
    :func:`analysis.plotting_style.animal_palette`, override via
    `animal_colors`); the bold line + shaded band is the cohort mean and
    bootstrapped 95% CI across animals (:func:`analysis.plotting.bootstrap_group_stats`,
    since the input is already one row per animal per window -- see
    AGENTS.md), colored by `reward_colors` (default ``{True: "tab:orange",
    False: "tab:blue"}``, same convention as :func:`analysis.plotting.plot_choice_by_odor_appearance`)
    according to which odor is actually being tracked -- the rewarded one
    when `first_stop_rewarded` is False, the non-rewarded one when True.
    """
    from matplotlib.ticker import MaxNLocator

    skip_blocks = window_blocks if skip_blocks is None else skip_blocks

    block_table = counterfactual_block_table(trials)
    windows = block_window_index(trials, window_blocks=window_blocks, skip_blocks=skip_blocks)
    block_table = block_table.merge(
        windows[["subject_id", "session_id", "block", "window"]],
        on=["subject_id", "session_id", "block"],
        how="inner",  # drop the trailing partial window's blocks; duplicates
        # a block onto every overlapping window it belongs to
    )

    value_col = "stop_next_bad" if first_stop_rewarded else "stop_next_good"
    cond = block_table[block_table["first_stop_rewarded"] == first_stop_rewarded]
    cond = cond.dropna(subset=[value_col])

    per_animal = (
        cond.groupby(["subject_id", "window"])[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: "p_stop"})
    )

    if subtract_chance:
        chance = first_site_chance_by_window(trials, window_blocks, skip_blocks)
        per_animal = per_animal.merge(chance, on=["subject_id", "window"], how="left")
        per_animal["p_stop"] = per_animal["p_stop"] - per_animal["stopped_first_site"]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    subjects = sorted(per_animal["subject_id"].unique())
    animal_colors = animal_colors if animal_colors is not None else animal_palette(subjects)
    reward_colors = (
        reward_colors if reward_colors is not None else {True: "tab:orange", False: "tab:blue"}
    )
    tracked_odor_is_rewarded = not first_stop_rewarded

    for subject_id, grp in per_animal.groupby("subject_id"):
        grp = grp.sort_values("window")
        x = grp["window"].to_numpy() * skip_blocks + window_blocks
        ax.plot(
            x, grp["p_stop"], color=animal_colors[str(subject_id)], linewidth=1, alpha=0.5
        )

    rng = np.random.default_rng(0)
    stats = bootstrap_group_stats(per_animal["p_stop"], per_animal["window"], rng)
    stats.index = stats.index * skip_blocks + window_blocks
    plot_mean_ci_band(ax, stats, color=reward_colors[tracked_odor_is_rewarded], marker="o")

    ax.set_xlabel("Block number")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 3, 6, 9, 10], integer=True))
    if subtract_chance:
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_ylabel("P(Stop) − chance (P(Stop) @ site 1)")
    else:
        ax.set_ylabel("P(Stop)")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    return ax


def counterfactual_session_matrix(
    trials: pd.DataFrame, min_blocks: int = 3
) -> pd.DataFrame:
    """Per-session P(stop) for the four counterfactual conditions.

    Aggregates counterfactual_block_table over blocks within a session.
    `accuracy` is p_stop for rewarded odors and 1 - p_stop for non-rewarded
    ones, so all four conditions share a higher-is-better polarity. Cells
    backed by fewer than `min_blocks` blocks get p_stop = NaN (kept as rows
    so the heatmap keeps a stable 4-column grid).
    """
    blocks = counterfactual_block_table(trials)
    if blocks.empty:
        return blocks

    long = blocks.melt(
        id_vars=["subject_id", "session_id", "block", "first_stop_rewarded"],
        value_vars=["stop_next_good", "stop_next_bad"],
        var_name="_next",
        value_name="stopped",
    )
    long["next_rewarded"] = long["_next"] == "stop_next_good"
    long = long.dropna(subset=["stopped"])

    agg = (
        long.groupby(
            ["subject_id", "session_id", "first_stop_rewarded", "next_rewarded"]
        )["stopped"]
        .agg(p_stop="mean", n_blocks="count")
        .reset_index()
    )

    # Reindex onto the full (session x 4 conditions) grid so missing cells
    # show up as gaps in the heatmap instead of silently shifting columns.
    sessions = agg[["subject_id", "session_id"]].drop_duplicates()
    grid = sessions.merge(
        pd.DataFrame(
            COUNTERFACTUAL_CELL_KEYS, columns=["first_stop_rewarded", "next_rewarded"]
        ),
        how="cross",
    )
    agg = grid.merge(
        agg,
        on=["subject_id", "session_id", "first_stop_rewarded", "next_rewarded"],
        how="left",
    )
    agg["n_blocks"] = agg["n_blocks"].fillna(0).astype(int)
    # cast off the nullable dtype inherited from the boolean mean so
    # downstream numpy/matplotlib code sees plain float NaN, not pd.NA
    agg["p_stop"] = agg["p_stop"].astype(float)
    agg.loc[agg["n_blocks"] < min_blocks, "p_stop"] = np.nan

    agg["p_leave"] = 1.0 - agg["p_stop"]
    agg["accuracy"] = np.where(agg["next_rewarded"], agg["p_stop"], 1.0 - agg["p_stop"])

    # session_id encodes the datetime -> lexicographic sort is chronological
    agg["session_date"] = agg["session_id"].str.split("_").str[1]
    agg = agg.sort_values(["subject_id", "session_id"])
    agg["session_index"] = agg.groupby("subject_id")["session_id"].transform(
        lambda s: s.rank(method="dense").astype(int) - 1
    )
    return agg


def counterfactual_window_matrix(
    trials: pd.DataFrame, window_blocks: int, skip_blocks: int, min_blocks: int = 3
) -> pd.DataFrame:
    """Per-block-window analogue of counterfactual_session_matrix.

    Re-keys the trials rather than reimplementing the aggregation:
    session_id becomes a synthetic "{subject}_w{window:04d}" window key and
    block becomes the animal's global block ordinal, so the result has
    exactly counterfactual_session_matrix's columns with session_id holding
    the window key.
    """
    expanded = expand_to_block_windows(trials, window_blocks, skip_blocks)
    rekeyed = expanded.assign(
        session_id=expanded["subject_id"].astype(str)
        + "_w"
        + expanded["window"].map("{:04d}".format),
        block=expanded["block_ordinal"],
    )
    matrix = counterfactual_session_matrix(rekeyed, min_blocks=min_blocks)
    matrix["window"] = matrix["session_id"].str.rsplit("_w", n=1).str[1].astype(int)
    bounds = expanded.drop_duplicates("window")[
        ["window", "window_start", "window_end"]
    ]
    return matrix.merge(bounds, on="window", how="left")


def _pivot(matrix, subject, value):
    """(n_sessions, 4) arrays of `value` and block counts for one subject."""
    sub = matrix[matrix["subject_id"] == subject]
    sessions = sorted(sub["session_id"].unique())
    keys = ["session_id", "first_stop_rewarded", "next_rewarded"]
    values = sub.set_index(keys)[value]
    n_blocks = sub.set_index(keys)["n_blocks"]

    grid = np.full((len(sessions), len(COUNTERFACTUAL_CELL_KEYS)), np.nan)
    counts = np.zeros_like(grid, dtype=int)
    for r, session_id in enumerate(sessions):
        for c, key in enumerate(COUNTERFACTUAL_CELL_KEYS):
            idx = (session_id, *key)
            if idx in values.index:
                grid[r, c] = values.loc[idx]
                counts[r, c] = n_blocks.loc[idx]
    return grid, sessions, counts


def plot_counterfactual_heatmap(
    trials,
    value="p_leave",
    min_blocks=3,
    annotate=True,
    align_rows=True,
    matrix=None,
    ylabel="Session (chronological)",
    row_label_fn=lambda s: s.split("_")[1],
):
    """Row x 4-condition heatmap of counterfactual behaviour, one panel per animal.

    Rows are whatever `matrix`'s `session_id` column holds (raw sessions by
    default, or a rekeyed unit like a block-window -- see
    `counterfactual_window_matrix`); `row_label_fn` turns a row key into a
    tick label and `ylabel` sets the axis title.
    """
    style = _style(value)
    if matrix is None:
        matrix = counterfactual_session_matrix(trials, min_blocks=min_blocks)
    _require_value(matrix, value)

    subjects = sorted(matrix["subject_id"].unique())
    max_rows = max(
        matrix[matrix["subject_id"] == s]["session_id"].nunique() for s in subjects
    )
    cmap = style["cmap"]

    fig, axes = plt.subplots(
        1,
        len(subjects),
        figsize=(3.2 * len(subjects), 0.42 * max_rows + 3.2),
        squeeze=False,
        layout="constrained",
    )
    images = []
    for ax, subject in zip(axes[0], subjects):
        grid, sessions, counts = _pivot(matrix, subject, value)
        images.append(
            ax.imshow(
                grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest"
            )
        )
        ax.grid(False)
        ax.set_xticks(range(len(COUNTERFACTUAL_CELLS)))
        ax.set_xticklabels(
            [label for _, _, label, _ in COUNTERFACTUAL_CELLS],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_yticks(range(len(sessions)))
        ax.set_yticklabels([row_label_fn(s) for s in sessions], fontsize=6)
        if align_rows:
            # pad every panel to the longest animal so one row = one session
            # at the same height everywhere, making panels comparable by eye
            ax.set_ylim(max_rows - 0.5, -0.5)
        ax.axvline(1.5, color="white", lw=2.5)  # separate the two first-stop splits
        ax.set_title(f"Subject {subject}", fontsize=10)

        if annotate:
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    v = grid[r, c]
                    if np.isnan(v):
                        ax.text(
                            c,
                            r,
                            "·",
                            ha="center",
                            va="center",
                            color="gray",
                            fontsize=8,
                        )
                        continue
                    ax.text(
                        c,
                        r,
                        f"{v:.2f}\nn{counts[r, c]}",
                        ha="center",
                        va="center",
                        fontsize=4.5,
                        color=_text_color(v, cmap),
                    )

    axes[0][0].set_ylabel(ylabel)
    cb = fig.colorbar(images[0], ax=axes[0], fraction=0.02, pad=0.02)
    cb.set_label(style["label"])
    ideal = "  —  ideal: " + " / ".join(f"{i:g}" for i in style["ideal"])
    fig.suptitle(
        "Counterfactual learning: decision at the next odor of each type,\n"
        "split by whether the block's first stop was rewarded" + ideal,
        fontsize=11,
    )
    return fig, matrix


def plot_counterfactual_heatmap_by_window(
    trials: pd.DataFrame,
    window_blocks: int = 50,
    skip_blocks: int | None = None,
    min_blocks: int = 3,
    annotate: bool = True,
    subtract_chance: bool = True,
):
    """Row x 4-condition heatmap of P(Stop), one panel per animal.

    A presentation-styled ("pretty") counterpart of
    :func:`plot_counterfactual_heatmap` (which `pipeline.py` uses with raw
    ``p_stop``/``p_leave``/``accuracy``): rows are non-overlapping
    `window_blocks`-block chunks (`skip_blocks` defaults to `window_blocks`,
    i.e. non-overlapping -- pass a smaller value for overlapping windows)
    instead of raw sessions.

    ``subtract_chance=True`` (default) plots P(Stop) minus that (subject,
    window)'s chance level (:func:`first_site_chance_by_window`'s P(Stop) at
    the block's very first site) instead of the raw probability -- so a
    cell reads "how much better/worse than no-information guessing", not an
    absolute rate, and 0 (at chance) sits at the diverging colormap's
    neutral midpoint (:func:`analysis.plotting_style.diverging_cmap`, the
    dark-surface-validated diverging map -- never a hue at the midpoint),
    symmetric around 0 at the largest |value| seen anywhere in the matrix.
    ``subtract_chance=False`` plots the raw P(Stop) instead (0-1, "coolwarm"
    -- same colormap `pipeline.py` uses for its own raw ``p_stop`` heatmap).
    Color scale is shared across all animals' panels either way.

    Meant to be called inside :func:`analysis.plotting_style.presentation_style`,
    matching every other plot in this notebook -- unlike
    :func:`plot_counterfactual_heatmap`, it doesn't set its own light-background
    styling, colorbar, or figure title (presentation figures don't get one;
    the slide/caption carries that context).
    """
    skip_blocks = window_blocks if skip_blocks is None else skip_blocks

    matrix = counterfactual_window_matrix(
        trials, window_blocks, skip_blocks, min_blocks=min_blocks
    )

    if subtract_chance:
        chance = first_site_chance_by_window(trials, window_blocks, skip_blocks)
        matrix = matrix.merge(chance, on=["subject_id", "window"], how="left")
        matrix["value"] = matrix["p_stop"] - matrix["stopped_first_site"]
        cmap = diverging_cmap()
        vlim = np.nanmax(np.abs(matrix["value"].to_numpy(dtype=float)))
        vmin, vmax = (-vlim, vlim) if np.isfinite(vlim) and vlim > 0 else (-1.0, 1.0)
        cell_fmt = "{:+.2f}\nn{}"
    else:
        matrix["value"] = matrix["p_stop"]
        cmap = "coolwarm"
        vmin, vmax = 0.0, 1.0
        cell_fmt = "{:.2f}\nn{}"

    subjects = sorted(matrix["subject_id"].unique())
    max_rows = max(
        matrix[matrix["subject_id"] == s]["session_id"].nunique() for s in subjects
    )

    fig, axes = plt.subplots(
        1,
        len(subjects),
        figsize=(3.2 * len(subjects), 0.42 * max_rows + 3.2),
        squeeze=False,
        layout="constrained",
    )
    for ax, subject in zip(axes[0], subjects):
        grid, sessions, counts = _pivot(matrix, subject, "value")
        bounds = (
            matrix[matrix["subject_id"] == subject]
            .drop_duplicates("session_id")
            .set_index("session_id")
        )
        row_labels = [
            f"{int(bounds.loc[s, 'window_start']) + 1}-{int(bounds.loc[s, 'window_end']) + 1}"
            for s in sessions
        ]

        ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.grid(False)
        ax.set_xticks(range(len(COUNTERFACTUAL_CELLS)))
        ax.set_xticklabels(
            [label for _, _, label, _ in COUNTERFACTUAL_CELLS],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_yticks(range(len(sessions)))
        ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_ylim(max_rows - 0.5, -0.5)
        ax.axvline(1.5, color=INK_PRIMARY, lw=2.5)
        ax.set_title(f"Subject {subject}", fontsize=10)

        if annotate:
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    v = grid[r, c]
                    if np.isnan(v):
                        ax.text(
                            c, r, "·", ha="center", va="center", color=INK_MUTED, fontsize=8
                        )
                        continue
                    ax.text(
                        c,
                        r,
                        cell_fmt.format(v, counts[r, c]),
                        ha="center",
                        va="center",
                        fontsize=4.5,
                        color=INK_PRIMARY,
                    )

    axes[0][0].set_ylabel(f"Block window ({window_blocks} blocks)")
    return fig, axes[0]


def counterfactual_cohort_average(
    matrix, value="accuracy", min_animals=1, rng=None, n_boot=2000
):
    """Average counterfactual_session_matrix across mice per session number.

    Sessions are aligned by session_index -- each animal's own 0-based
    chronological session number -- and averaged across animals, with a
    percentile-bootstrap 95% CI across animals (``mean``/``ci_lo``/``ci_hi``
    all NaN below 2 contributing animals -- a single animal is a gap, not a
    confident point). Pass a shared ``rng`` across a sweep of calls for a
    reproducible run.
    """
    _require_value(matrix, value)
    rng = rng if rng is not None else np.random.default_rng(0)
    records = []
    for (session_index, fsr, nr), grp in matrix.dropna(subset=[value]).groupby(
        ["session_index", "first_stop_rewarded", "next_rewarded"]
    ):
        vals = grp[value].to_numpy(dtype=float)
        mean, ci_lo, ci_hi = bootstrap_mean_ci(vals, rng, min_n=2, n_boot=n_boot)
        records.append(
            {
                "session_index": session_index,
                "first_stop_rewarded": fsr,
                "next_rewarded": nr,
                "mean": mean,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_animals": len(vals),
                "n_blocks": grp["n_blocks"].sum(),
            }
        )
    agg = pd.DataFrame.from_records(
        records,
        columns=[
            "session_index",
            "first_stop_rewarded",
            "next_rewarded",
            "mean",
            "ci_lo",
            "ci_hi",
            "n_animals",
            "n_blocks",
        ],
    )
    return agg[agg["n_animals"] >= min_animals].reset_index(drop=True)


def plot_counterfactual_cohort_average(
    matrix,
    value="p_leave",
    min_animals=1,
    annotate=True,
    x_label="Session number (aligned across mice)",
    title=(
        "Counterfactual learning, averaged across mice at the same session number\n"
        "(error bars = bootstrapped 95% CI across animals; n per session falls off as animals run out)"
    ),
):
    """Cross-mouse counterfactual matrix, laid out with session number on the x axis.

    Returns ``(fig, cohort, ax_ln)`` -- ``ax_ln`` is the bottom line-plot axes,
    handed back so a caller can overlay an extra series on the same axis.
    """
    style = _style(value)
    cohort = counterfactual_cohort_average(matrix, value=value, min_animals=min_animals)
    labels = [label for _, _, label, _ in COUNTERFACTUAL_CELLS]

    # dense contiguous session axis so the heatmap's extent lines up exactly
    # with the line plot below it (a skipped index would shift the strips)
    lo = int(cohort["session_index"].min())
    hi = int(cohort["session_index"].max())
    session_indices = list(range(lo, hi + 1))
    y = np.arange(len(COUNTERFACTUAL_CELL_KEYS))

    keys = ["session_index", "first_stop_rewarded", "next_rewarded"]
    means = cohort.set_index(keys)["mean"]
    ci_los = cohort.set_index(keys)["ci_lo"]
    ci_his = cohort.set_index(keys)["ci_hi"]
    n_animals = cohort.set_index(keys)["n_animals"]

    def _strip(series):
        """(4, n_sessions) array: one row per condition, one column per session."""
        return np.array(
            [
                [series.get((s, *key), np.nan) for s in session_indices]
                for key in COUNTERFACTUAL_CELL_KEYS
            ],
            dtype=float,
        )

    grid = _strip(means)
    grid_ci_lo = _strip(ci_los)
    grid_ci_hi = _strip(ci_his)
    grid_n = _strip(n_animals)

    fig, (ax_hm, ax_ln) = plt.subplots(
        2,
        1,
        figsize=(0.46 * len(session_indices) + 4.5, 8.5),
        gridspec_kw={"height_ratios": [1, 1.6]},
        sharex=True,
        layout="constrained",
    )

    # ── top: conditions x session number heatmap ──────────────────────────
    im = ax_hm.imshow(
        grid,
        cmap=style["cmap"],
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=(lo - 0.5, hi + 0.5, len(y) - 0.5, -0.5),
    )
    ax_hm.grid(False)
    ax_hm.set_yticks(y)
    ax_hm.set_yticklabels([label.replace("\n", " ") for label in labels], fontsize=8)
    ax_hm.axhline(1.5, color="white", lw=2.5)  # separate the two first-stop splits
    for tick, color in zip(ax_hm.get_yticklabels(), COUNTERFACTUAL_COLORS):
        tick.set_color(color)
    if annotate:
        for r in range(grid.shape[0]):
            for c, s in enumerate(session_indices):
                v = grid[r, c]
                if np.isnan(v):
                    continue
                ax_hm.text(
                    s,
                    r,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=_text_color(v, style["cmap"]),
                )
    fig.colorbar(im, ax=(ax_hm, ax_ln), fraction=0.02, pad=0.01).set_label(
        style["label"], fontsize=9
    )

    # ── bottom: the same four rows as cohort timecourses ───────────────────
    xs = np.asarray(session_indices, dtype=float)
    for r, ((_, _, label, _), color, ideal) in enumerate(
        zip(COUNTERFACTUAL_CELLS, COUNTERFACTUAL_COLORS, style["ideal"])
    ):
        ok = ~np.isnan(grid[r])
        yerr = np.vstack(
            [
                (grid[r][ok] - grid_ci_lo[r][ok]).clip(min=0),
                (grid_ci_hi[r][ok] - grid[r][ok]).clip(min=0),
            ]
        )
        ax_ln.errorbar(
            xs[ok],
            grid[r][ok],
            yerr=yerr,
            marker="o",
            ms=5,
            lw=1.8,
            capsize=3,
            color=color,
            label=f"{label.replace(chr(10), ' ')}  (ideal {ideal:g})",
        )

    # shade where fewer than half the animals still contribute, on both panels
    n_per_session = np.full(grid_n.shape[1], np.nan)
    populated = ~np.all(np.isnan(grid_n), axis=0)
    n_per_session[populated] = np.nanmax(grid_n[:, populated], axis=0)

    thin = ~(n_per_session >= np.nanmax(n_per_session) / 2)  # NaN column -> thin
    tail = len(thin)
    while tail > 0 and thin[tail - 1]:
        tail -= 1
    spans = [(xs[tail] - 0.5, hi + 0.5)] if tail < len(xs) else []
    spans += [(x - 0.5, x + 0.5) for x in xs[:tail][thin[:tail]]]
    for ax, legend_label in ((ax_hm, None), (ax_ln, "< half the cohort")):
        for i, (x0, x1) in enumerate(spans):
            ax.axvspan(
                x0,
                x1,
                color="gray",
                alpha=0.12,
                zorder=0,
                label=legend_label if i == 0 else None,
            )
    if tail < len(xs):
        ax_hm.axvline(xs[tail] - 0.5, color="black", lw=1.2, alpha=0.6)
        ax_ln.axvline(xs[tail] - 0.5, color="black", lw=1.2, alpha=0.6)

    ax_ln.axhline(0.5, color="gray", ls=":", lw=1)
    ax_ln.set_ylim(-0.03, 1.05)
    ax_ln.set_xlim(lo - 0.5, hi + 0.5)
    ax_ln.xaxis.get_major_locator().set_params(integer=True)
    ax_ln.set_xlabel(x_label)
    ax_ln.set_ylabel(style["label"])
    ax_ln.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)

    ax_top = ax_hm.secondary_xaxis("top")
    ax_top.set_xticks(session_indices)
    ax_top.set_xticklabels(
        [str(int(n)) if np.isfinite(n) else "" for n in n_per_session], fontsize=5.5
    )
    ax_top.set_xlabel(
        "animals contributing (max over the 4 conditions; a cell below "
        "min_blocks drops out, so this can dip and recover)",
        fontsize=8,
    )

    fig.suptitle(title, fontsize=11)
    return fig, cohort, ax_ln


def plot_counterfactual_cohort_minus_chance(
    trials: pd.DataFrame,
    window_blocks: int = 100,
    skip_blocks: int = 20,
    min_blocks: int = 3,
    min_animals: int = 2,
    exclude_subjects=(),
    subtract_chance: bool = True,
    ax=None,
):
    """Chance-normalized cohort timecourses of the four counterfactual
    conditions (the line panel of `pipeline.py`'s cohort figure): mean +/-
    bootstrapped 95% CI across animals per block window, in the pipeline's
    condition colors (:data:`COUNTERFACTUAL_COLORS`).

    Values are P(stop) at the next site minus each animal's chance level in
    that window (:func:`first_site_chance_by_window`), subtracted per animal
    per window before the cohort bootstrap, so 0 = "no better than
    no-information guessing". Windows with fewer than `min_animals` animals
    (default 2, the minimum a CI needs) are not drawn, and the x axis stops
    at the last window that is. The x axis is the block count at each
    window's end; the number of contributing animals is annotated only where
    it changes (max over the four conditions), as small ``n=`` labels along
    the top with a faint guide line, rather than at every window.

    `exclude_subjects` drops those animals from `trials` up front.
    ``subtract_chance=False`` plots the raw P(Stop) instead (probability
    axis, no chance subtraction). Call
    inside :func:`analysis.plotting_style.presentation_style`; no title.
    Returns ``(ax, cohort)``.
    """
    from matplotlib.ticker import MaxNLocator

    if len(exclude_subjects):
        drop = {str(s) for s in exclude_subjects}
        trials = trials[~trials["subject_id"].astype(str).isin(drop)]

    matrix = counterfactual_window_matrix(
        trials, window_blocks, skip_blocks, min_blocks=min_blocks
    )
    chance = first_site_chance_by_window(trials, window_blocks, skip_blocks)
    matrix = matrix.merge(chance, on=["subject_id", "window"], how="left")
    matrix["p_stop_minus_chance"] = matrix["p_stop"] - matrix["stopped_first_site"]
    value = "p_stop_minus_chance" if subtract_chance else "p_stop"

    cohort = counterfactual_cohort_average(
        matrix, value=value, min_animals=min_animals
    )
    cohort = cohort.dropna(subset=["mean"])
    cohort["x"] = cohort["session_index"] * skip_blocks + window_blocks

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    for (fsr, nr), (_, _, label, _), color in zip(
        COUNTERFACTUAL_CELL_KEYS, COUNTERFACTUAL_CELLS, COUNTERFACTUAL_COLORS
    ):
        sub = cohort[
            (cohort["first_stop_rewarded"] == fsr) & (cohort["next_rewarded"] == nr)
        ].sort_values("x")
        yerr = np.vstack(
            [
                (sub["mean"] - sub["ci_lo"]).clip(lower=0).fillna(0),
                (sub["ci_hi"] - sub["mean"]).clip(lower=0).fillna(0),
            ]
        )
        ax.errorbar(
            sub["x"], sub["mean"], yerr=yerr, marker="o", ms=5, lw=1.8,
            capsize=3, color=color, label=label.replace("\n", " "),
        )

    if subtract_chance:
        ax.axhline(0, color="gray", ls=":", lw=1)
    else:
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlim(window_blocks - skip_blocks, cohort["x"].max() + skip_blocks)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    n_by_x = cohort.groupby("x")["n_animals"].max().sort_index()
    prev = None
    for x, n in n_by_x.items():
        if n != prev:
            ax.axvline(x, color="gray", lw=0.6, ls=":", alpha=0.6, zorder=0)
            ax.text(x, 1.01, f"n={int(n)}", transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=8, color=INK_MUTED)
            prev = n
    ax.set_xlabel("Number of blocks")
    ax.set_ylabel("P(Stop) − chance" if subtract_chance else "P(Stop)")
    ax.legend(frameon=False, fontsize=8, loc="best", ncol=2)
    return ax, cohort


def plot_counterfactual_cohort_by_condition(
    matrix: pd.DataFrame,
    value: str = "p_leave",
    x_label: str = "Block window (aligned across mice)",
    min_animals: int = 1,
    title: str = (
        "Counterfactual learning, averaged across mice\n"
        "(faint = individual animals; bold line + shaded band = cohort mean "
        "and 95% CI across animals)"
    ),
):
    """One subplot per counterfactual condition, individual animals faint,
    cohort mean +/- bootstrapped 95% CI shaded -- same treatment as
    :func:`analysis.glm.plot_history_glm_reward_cells_by_window`.

    Aligns on ``session_index`` (equals the window number for a window-keyed
    matrix, see ``counterfactual_window_matrix``), so this works for both a
    per-session and a per-window matrix.
    """
    style = _style(value)
    subjects = sorted(matrix["subject_id"].unique())
    subject_colors = dict(zip(subjects, plt.cm.tab10.colors))
    cohort = counterfactual_cohort_average(matrix, value=value, min_animals=min_animals)

    fig, axes = plt.subplots(
        1,
        len(COUNTERFACTUAL_CELLS),
        figsize=(6 * len(COUNTERFACTUAL_CELLS), 4),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    for ax, (fsr, nr, label, _), color, ideal in zip(
        axes[0], COUNTERFACTUAL_CELLS, COUNTERFACTUAL_COLORS, style["ideal"]
    ):
        cond = matrix[
            (matrix["first_stop_rewarded"] == fsr) & (matrix["next_rewarded"] == nr)
        ]
        for subject in subjects:
            sub = (
                cond[cond["subject_id"] == subject]
                .dropna(subset=[value])
                .sort_values("session_index")
            )
            ax.plot(
                sub["session_index"],
                sub[value],
                color=subject_colors[subject],
                linewidth=1,
                alpha=0.4,
                label=f"Subject {subject}",
            )

        cohort_cond = cohort[
            (cohort["first_stop_rewarded"] == fsr) & (cohort["next_rewarded"] == nr)
        ].sort_values("session_index")
        x = cohort_cond["session_index"].to_numpy(dtype=float)
        mean = cohort_cond["mean"].to_numpy()
        ci_lo = cohort_cond["ci_lo"].to_numpy()
        ci_hi = cohort_cond["ci_hi"].to_numpy()
        ax.plot(x, mean, color=color, linewidth=2.2, label="Cohort mean ± 95% CI")
        ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.25, linewidth=0)

        ax.axhline(ideal, color="gray", ls=":", lw=1)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(x_label)
        ax.set_title(label.replace("\n", " "))

    axes[0][0].set_ylabel(style["label"])
    axes[0][-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig, axes[0]


def counterfactual_session_trends(matrix, value="accuracy"):
    """Per-subject, per-condition OLS slope of `value` against session index."""
    rows = []
    for (subject, fsr, nr), grp in matrix.groupby(
        ["subject_id", "first_stop_rewarded", "next_rewarded"]
    ):
        g = grp.dropna(subset=[value]).sort_values("session_index")
        if len(g) < 3:
            continue
        x = g["session_index"].to_numpy(dtype=float)
        y = g[value].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        rows.append(
            {
                "subject_id": subject,
                "first_stop_rewarded": fsr,
                "next_rewarded": nr,
                "slope": slope,
                "intercept": intercept,
                "r": np.corrcoef(x, y)[0, 1],
                "n_sessions": len(g),
            }
        )
    return pd.DataFrame(rows)


def plot_counterfactual_trends_per_animal(
    cf_window: pd.DataFrame, window_blocks: int, skip_blocks: int
):
    """Per-subject accuracy timecourse across block windows, with an OLS trend line.

    ``cf_window``'s ``session_index`` already equals its window number (see
    ``counterfactual_window_matrix``), but ``window`` is used directly here
    for clarity.
    """
    from matplotlib.ticker import MaxNLocator

    subjects = sorted(cf_window["subject_id"].unique())
    fig, axes = plt.subplots(
        1, len(subjects), figsize=(4.2 * len(subjects), 4), sharey=True, squeeze=False
    )
    for ax, subject in zip(axes[0], subjects):
        sub = cf_window[cf_window["subject_id"] == subject]
        for (fsr, nr, label, _), color in zip(
            COUNTERFACTUAL_CELLS, COUNTERFACTUAL_COLORS
        ):
            g = (
                sub[(sub["first_stop_rewarded"] == fsr) & (sub["next_rewarded"] == nr)]
                .dropna(subset=["accuracy"])
                .sort_values("window")
            )
            ax.plot(
                g["window"],
                g["accuracy"],
                marker="o",
                ms=4,
                color=color,
                label=label.replace("\n", " "),
            )
            if len(g) >= 3:
                x = g["window"].to_numpy(dtype=float)
                m, b = np.polyfit(x, g["accuracy"].to_numpy(dtype=float), 1)
                ax.plot(x, m * x + b, color=color, ls="--", lw=1.2, alpha=0.7)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(
            f"Block-window number ({window_blocks} blocks, stride {skip_blocks})"
        )
        ax.set_title(f"Subject {subject}", fontsize=10)
    axes[0][0].set_ylabel("accuracy (higher = better)")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(
        "Counterfactual accuracy across block windows (dashed = OLS fit)", fontsize=11
    )
    fig.tight_layout()
    return fig


def first_site_chance_by_window(
    trials: pd.DataFrame, window_blocks: int, skip_blocks: int
) -> pd.DataFrame:
    """P(stop) at each block's very first RewardSite encounter, per animal per window.

    By that first site the animal has zero evidence yet about *this* block's
    odor-reward mapping, so this is what "no information" stopping looks
    like -- a chance-level baseline for the counterfactual cohort plots.
    Computed at the block level, then folded into the same block-windows the
    counterfactual matrix uses, so it lands on the same x-axis.

    Returns one row per (subject_id, window) with a ``stopped_first_site`` mean.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    rs = rs.sort_values(["session_id", "block", "start_time"])
    rs = rs.assign(_block_pos=rs.groupby(["session_id", "block"]).cumcount())
    first_site = rs[rs["_block_pos"] == 0][
        ["session_id", "block", "has_choice"]
    ].rename(columns={"has_choice": "stopped_first_site"})

    windows_map = block_window_index(trials, window_blocks, skip_blocks)
    blocks = windows_map.merge(first_site, on=["session_id", "block"], how="inner")
    return (
        blocks.groupby(["subject_id", "window"])["stopped_first_site"]
        .mean()
        .reset_index()
    )


def p_stop_hazard_by_session(trials: pd.DataFrame, position: int) -> pd.DataFrame:
    """P(stop at raw within-block site `position` | the animal hasn't already
    stopped earlier in this block), per animal per session.

    `position` is 0-based and pools both odor types (like
    :func:`first_site_chance_by_window`'s ``_block_pos``, not
    :func:`analysis.features.appearance_table`'s per-odor ``appearance``).
    A block only contributes at `position` if it reached that many
    RewardSite trials *and* the animal had not yet stopped at an earlier one
    -- blocks that already stopped are excluded rather than counted as a
    miss, and blocks that never stop are still "at risk" at every position
    they reach. At ``position=0`` nothing could have happened yet, so every
    block is at risk and this is just plain P(stop) at the block's first
    site; at later positions it's the discrete hazard rate.

    Returns one row per (subject_id, session_id) with a 0-based
    ``session_index`` (that animal's own chronological session rank, same
    convention as :func:`counterfactual_session_matrix`) and a ``p_stop`` mean.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    rs = rs.sort_values(["session_id", "block", "start_time"])
    rs["_block_pos"] = rs.groupby(["session_id", "block"]).cumcount()

    first_stop_pos = (
        rs[rs["has_choice"]].groupby(["session_id", "block"])["_block_pos"].min()
    )
    target = rs[rs["_block_pos"] == position].join(
        first_stop_pos.rename("_first_stop_pos"), on=["session_id", "block"]
    )
    at_risk = target["_first_stop_pos"].isna() | (target["_first_stop_pos"] >= position)
    target = target[at_risk]

    per_session = (
        target.groupby(["subject_id", "session_id"])["has_choice"]
        .mean()
        .reset_index()
        .rename(columns={"has_choice": "p_stop"})
    )
    per_session = per_session.sort_values(["subject_id", "session_id"])
    per_session["session_index"] = per_session.groupby("subject_id")[
        "session_id"
    ].transform(lambda s: s.rank(method="dense").astype(int) - 1)
    return per_session


def plot_p_stop_hazard_by_session(
    trials: pd.DataFrame,
    position: int,
    ax=None,
    animal_colors: dict | None = None,
    color: str = "white",
):
    """:func:`p_stop_hazard_by_session` plotted across sessions: thin,
    semi-transparent per-animal lines (colored by
    :func:`analysis.plotting_style.animal_palette`, override via
    `animal_colors`) plus the bold `color` line + shaded band for the
    cohort mean and bootstrapped 95% CI across animals
    (:func:`analysis.plotting.bootstrap_group_stats`, since the input is
    already one row per animal per session -- see AGENTS.md).

    The x-axis is each animal's own 0-based chronological session number
    (``session_index`` + 1), not a calendar date, so animals that started
    training on different days still line up on "days of training".
    """
    from matplotlib.ticker import MaxNLocator

    per_session = p_stop_hazard_by_session(trials, position)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    subjects = sorted(per_session["subject_id"].unique())
    animal_colors = animal_colors if animal_colors is not None else animal_palette(subjects)

    for subject_id, grp in per_session.groupby("subject_id"):
        grp = grp.sort_values("session_index")
        ax.plot(
            grp["session_index"] + 1,
            grp["p_stop"],
            color=animal_colors[str(subject_id)],
            linewidth=1,
            alpha=0.5,
        )

    rng = np.random.default_rng(0)
    stats = bootstrap_group_stats(per_session["p_stop"], per_session["session_index"], rng)
    stats.index = stats.index + 1
    plot_mean_ci_band(ax, stats, color=color, marker="o")

    ax.set_xlabel("Session number")
    ax.set_ylabel("P(Stop)" if position == 0 else "P(Stop | hasn't stopped yet)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    return ax


def counterfactual_p_stop_first_last_blocks(
    trials: pd.DataFrame,
    first_stop_rewarded: bool,
    n_blocks: int = 30,
    last_n_blocks: int | None = None,
    rng=None,
) -> pd.DataFrame:
    """Chance-subtracted P(Stop) (see :func:`plot_counterfactual_p_stop_by_window`'s
    ``subtract_chance``) for each animal's first `n_blocks` vs last
    `last_n_blocks` (default: same as `n_blocks`) blocks
    (:func:`analysis.features.blocks_first_last_tags`), computed over the
    animal's *entire* block stream -- not restricted to blocks matching
    `first_stop_rewarded` -- so "first"/"last" means the same span of
    training regardless of which condition is requested.

    Returns one row per (subject_id, block_range) with ``p_stop_minus_chance``
    and its within-animal bootstrapped 95% CI (``ci_lo``, ``ci_hi``) --
    :func:`analysis.plotting.bootstrap_diff_ci` on that (subject_id,
    block_range)'s block-level P(Stop) outcomes
    (``stop_next_good``/``stop_next_bad``) versus its block-level chance
    outcomes (``stopped_first_site``). The two are independent, not paired
    samples -- they use different, only partially-overlapping sets of
    blocks (the P(Stop) blocks are also restricted to `first_stop_rewarded`
    and to blocks where the tracked odor recurred) -- so the CI is bootstrapped
    as a difference of two independent means, not a paired difference.

    Feeds :func:`plot_counterfactual_p_stop_first_last_scatter`.
    """
    value_col = "stop_next_bad" if first_stop_rewarded else "stop_next_good"
    rng = rng if rng is not None else np.random.default_rng(0)

    ranges = blocks_first_last_tags(trials, n_blocks=n_blocks, last_n_blocks=last_n_blocks)[
        ["subject_id", "session_id", "block", "block_range"]
    ].drop_duplicates()

    block_table = counterfactual_block_table(trials)
    block_table = block_table.merge(
        ranges, on=["subject_id", "session_id", "block"], how="inner"
    )
    cond = block_table[block_table["first_stop_rewarded"] == first_stop_rewarded]
    cond = cond.dropna(subset=[value_col])

    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    rs = rs.sort_values(["session_id", "block", "start_time"])
    rs = rs.assign(_block_pos=rs.groupby(["session_id", "block"]).cumcount())
    first_site = rs[rs["_block_pos"] == 0][
        ["subject_id", "session_id", "block", "has_choice"]
    ].rename(columns={"has_choice": "stopped_first_site"})
    first_site = first_site.merge(
        ranges, on=["subject_id", "session_id", "block"], how="inner"
    )

    p_stop_groups = cond.groupby(["subject_id", "block_range"])[value_col]
    chance_groups = first_site.groupby(["subject_id", "block_range"])["stopped_first_site"]
    keys = sorted(set(p_stop_groups.groups) & set(chance_groups.groups))

    records = []
    for subject_id, block_range in keys:
        a = p_stop_groups.get_group((subject_id, block_range)).to_numpy(dtype=float)
        b = chance_groups.get_group((subject_id, block_range)).to_numpy(dtype=float)
        mean_diff, ci_lo, ci_hi = bootstrap_diff_ci(a, b, rng)
        records.append(
            {
                "subject_id": subject_id,
                "block_range": block_range,
                "p_stop_minus_chance": mean_diff,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )
    return pd.DataFrame.from_records(records)


def plot_counterfactual_p_stop_first_last_scatter(
    trials: pd.DataFrame,
    n_blocks: int = 30,
    last_n_blocks: int | None = None,
    ax=None,
    reward_colors: dict | None = None,
    animal_colors: dict | None = None,
):
    """Paired first-`n_blocks`-vs-last-`last_n_blocks`-blocks scatter of the
    chance-subtracted P(Stop) values (:func:`counterfactual_p_stop_first_last_blocks`),
    one point per animal, both mirror-image conditions overlaid on the same
    axes. Fill color encodes the condition (`reward_colors`, same convention
    as :func:`plot_counterfactual_p_stop_by_window` -- orange: after a
    non-rewarded first stop, does the animal now stop for the rewarded odor;
    blue: after a rewarded first stop, does it still stop for the
    non-rewarded odor); the outer ring encodes animal identity
    (:func:`analysis.plotting_style.animal_palette`, override via
    `animal_colors`).

    `last_n_blocks` defaults to `n_blocks` (a symmetric first-vs-last
    comparison); pass a different value (e.g. ``n_blocks=30,
    last_n_blocks=100``) for an asymmetric one.

    A gray dashed y = x line marks "no change from first to last blocks";
    points above it improved. Dotted lines at x = 0 / y = 0 mark chance.
    Axes are equal and shared so the diagonal is a true 45 degrees. Each
    point also gets bootstrapped 95% CI error bars in both directions (x
    from the "first" blocks' CI, y from the "last" blocks' CI --
    :func:`counterfactual_p_stop_first_last_blocks`), colored to match that
    point's fill.
    """
    reward_colors = reward_colors if reward_colors is not None else {True: "tab:orange", False: "tab:blue"}
    last_n_blocks = n_blocks if last_n_blocks is None else last_n_blocks

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    for first_stop_rewarded in [False, True]:
        data = counterfactual_p_stop_first_last_blocks(
            trials, first_stop_rewarded, n_blocks=n_blocks, last_n_blocks=last_n_blocks
        )
        wide = data.pivot(
            index="subject_id",
            columns="block_range",
            values=["p_stop_minus_chance", "ci_lo", "ci_hi"],
        ).dropna()

        first_stats = wide.xs("first", axis=1, level=1).rename(
            columns={"p_stop_minus_chance": "mean"}
        )
        last_stats = wide.xs("last", axis=1, level=1).rename(
            columns={"p_stop_minus_chance": "mean"}
        )
        x, y = first_stats["mean"], last_stats["mean"]
        color = reward_colors[not first_stop_rewarded]

        ax.errorbar(
            x, y,
            xerr=ci_errorbar(first_stats), yerr=ci_errorbar(last_stats),
            fmt="none", ecolor=color, elinewidth=1.2, capsize=3, alpha=0.7, zorder=2,
        )

        subjects = x.index.tolist()
        subject_colors = animal_colors if animal_colors is not None else animal_palette(subjects)
        edge_colors = [subject_colors[str(s)] for s in subjects]
        ax.scatter(
            x, y,
            color=color,
            edgecolors=edge_colors,
            linewidths=1.8,
            s=70,
            zorder=3,
        )

    # bounds from the actual drawn extent (points + error bars), not just
    # the point values, so error bars never get clipped at the axis edge
    ax.relim()
    ax.autoscale_view()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    lo = min(x0, y0, 0.0) - 0.05
    hi = max(x1, y1, 0.0) + 0.05
    ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, zorder=1)
    ax.axhline(0, color="gray", ls=":", lw=0.8, zorder=0)
    ax.axvline(0, color="gray", ls=":", lw=0.8, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"First {n_blocks} blocks\n(P(Stop) − chance)")
    ax.set_ylabel(f"Last {last_n_blocks} blocks\n(P(Stop) − chance)")
    return ax
