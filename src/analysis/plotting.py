"""Matplotlib plot-drawing / styling helpers shared across the notebook's cells.

Holds ``a_lot_of_style`` plus the generic "choice by block position" plotting
family, reused across several notebook cells with no research-question-
specific interpretation of its own. Plots specific to one analysis (GLM,
bias, counterfactual) live alongside that analysis's data prep instead --
see :mod:`analysis.glm`, :mod:`analysis.bias`, :mod:`analysis.counterfactual`.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.features import (
    appearance_table,
    assign_odor_blocks,
    block_tercile_tags,
    blocks_first_last_tags,
    label_first_stop,
)

#: Shared 2x2 condition palette (dark red, orange, dark blue, light blue/cyan)
#: -- reused verbatim by every plot that splits on two reward-related
#: booleans (:mod:`analysis.counterfactual`, :mod:`analysis.bias`,
#: :mod:`analysis.glm`) so the same color always means the same condition
#: across figures. Order is (True, True), (True, False), (False, True),
#: (False, False) for whatever two booleans a given plot uses.
TWO_BY_TWO_COLORS = ["#c0392b", "#e07b39", "#1a5276", "#4f8fc0"]


def bootstrap_mean_ci(values, rng, min_n=2, n_boot=2000):
    """Percentile-bootstrap ``(mean, ci_lo, ci_hi)`` across the entries of ``values``.

    NaN below ``min_n`` non-NaN values -- too few to bootstrap a spread from
    (n=1 would just resample itself every draw) -- so the caller should treat
    NaN as "leave a gap", not a zero-width CI. Pass one shared ``rng`` across
    a whole sweep of calls for a reproducible run. This is the codebase's one
    CI statistic -- no plot here uses a SEM/normal-approximation band.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size < min_n:
        return np.nan, np.nan, np.nan
    boot_means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(
        axis=1
    )
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return values.mean(), lo, hi


def bootstrap_diff_ci(values_a, values_b, rng, min_n=2, n_boot=2000):
    """Percentile-bootstrap ``(mean_a - mean_b, ci_lo, ci_hi)`` for the
    difference of two *independent* samples' means.

    Unlike :func:`bootstrap_mean_ci`, each bootstrap draw resamples ``values_a``
    and ``values_b`` separately (they aren't paired observations) and takes
    their difference, so the resulting CI on the difference properly
    combines both samples' uncertainty rather than assuming one is exact.
    NaN below ``min_n`` non-NaN values in *either* sample, same rationale as
    :func:`bootstrap_mean_ci`.
    """
    a = np.asarray(values_a, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(values_b, dtype=float)
    b = b[~np.isnan(b)]
    if a.size < min_n or b.size < min_n:
        return np.nan, np.nan, np.nan
    boot_a = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    boot_b = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    diff = boot_a - boot_b
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return a.mean() - b.mean(), lo, hi


def bootstrap_group_stats(
    values: pd.Series, keys, rng, min_n=2, n_boot=2000
) -> pd.DataFrame:
    """Bootstrapped 95% CI of ``values`` grouped by ``keys`` (a column/Series,
    or a list of them for a multi-key grouping).

    Returns a frame indexed like ``values.groupby(keys)`` -- a plain Index for
    a single key, a named ``MultiIndex`` for several -- with ``mean``,
    ``ci_lo``, ``ci_hi`` columns. Call ``.reset_index()`` to get the group
    key(s) back as columns.
    """
    index, rows = [], []
    for key, grp in values.groupby(keys):
        index.append(key)
        rows.append(
            bootstrap_mean_ci(
                grp.to_numpy(dtype=float), rng, min_n=min_n, n_boot=n_boot
            )
        )

    if isinstance(keys, (list, tuple)):
        names = [k.name if isinstance(k, pd.Series) else k for k in keys]
        idx = pd.MultiIndex.from_tuples(index, names=names)
    else:
        idx = pd.Index(index, name=keys.name if isinstance(keys, pd.Series) else keys)
    return pd.DataFrame(rows, index=idx, columns=["mean", "ci_lo", "ci_hi"])


def bootstrap_across_animals(
    trials: pd.DataFrame,
    value_col: str,
    group_keys: list,
    rng,
    min_n: int = 2,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Mean + 95% CI of ``value_col``, grouped by ``group_keys``, bootstrapped
    across animals rather than pooled trials.

    This codebase's CI convention (see AGENTS.md) is that every aggregation
    is mean + 95% CI bootstrapped *across animals*, never pooled raw trials --
    pooling would resample thousands of same-animal trials and understate
    animal-to-animal variability. This collapses to one row per
    (``group_keys``..., ``subject_id``) mean first, then bootstraps those
    per-animal means within each ``group_keys`` group (N = number of
    contributing animals). ``min_n=2`` means a group with only one
    contributing animal gets NaN, not a confident-looking zero-width CI --
    same rule :func:`analysis.counterfactual.counterfactual_cohort_average`
    uses.
    """
    group_keys = list(group_keys)
    per_animal = (
        trials.groupby(group_keys + ["subject_id"])[value_col].mean().reset_index()
    )
    keys = (
        per_animal[group_keys[0]]
        if len(group_keys) == 1
        else [per_animal[k] for k in group_keys]
    )
    return bootstrap_group_stats(
        per_animal[value_col], keys, rng, min_n=min_n, n_boot=n_boot
    )


def ci_errorbar(stats: pd.DataFrame) -> np.ndarray:
    """(2, n) ``yerr`` array from a ``mean``/``ci_lo``/``ci_hi`` frame, floored at 0."""
    lower = (stats["mean"] - stats["ci_lo"]).clip(lower=0).fillna(0)
    upper = (stats["ci_hi"] - stats["mean"]).clip(lower=0).fillna(0)
    return np.vstack([lower, upper])


def plot_mean_ci_band(
    ax,
    stats: pd.DataFrame,
    color,
    label: str | None = None,
    alpha: float = 0.25,
    marker: str | None = None,
    linestyle: str = "-",
) -> None:
    """Draw `stats`'s mean as a line with its CI as a translucent shaded band.

    `stats` is a mean/ci_lo/ci_hi frame as returned by `bootstrap_group_stats`
    / `bootstrap_across_animals`, indexed by the x variable. Preferred over
    `ax.errorbar` for presentation plots (see AGENTS.md) -- a translucent
    band reads better than error-bar caps against a dark slide background,
    and stays legible however the deck's background ends up colored.
    """
    ax.plot(
        stats.index, stats["mean"], color=color, label=label, marker=marker,
        linestyle=linestyle,
    )
    ax.fill_between(
        stats.index, stats["ci_lo"], stats["ci_hi"], color=color, alpha=alpha, linewidth=0
    )


#: Linestyle for each half of a first-vs-last-`n_blocks` split (see
#: :func:`_first_last_blocks_table` / :func:`_first_last_legend_handles`) --
#: dashed for an animal's first `n_blocks` blocks, solid for its last,
#: shared by every plot that draws this split on one axis instead of as
#: separate side-by-side panels (unlike :func:`plot_naive_p_stop_first_last`).
BLOCK_RANGE_LINESTYLES = {"first": "--", "last": "-"}


def _first_last_blocks_table(
    trials: pd.DataFrame, n_blocks: int = 100, from_first_stop: bool = False
) -> pd.DataFrame:
    """:func:`analysis.features.appearance_table` restricted to each animal's
    first and last `n_blocks` chronological blocks (pooled across sessions;
    see :func:`analysis.features.blocks_first_last_tags`), tagged with
    ``block_range`` (``"first"``/``"last"``).

    Shared by :func:`plot_choice_by_odor_appearance` and
    :func:`plot_choice_by_odor_appearance_by_animal` so the two stay in sync.
    """
    tagged = blocks_first_last_tags(trials, n_blocks=n_blocks)
    return appearance_table(tagged, from_first_stop=from_first_stop)


def _first_last_legend_handles(colors: dict, n_blocks: int) -> list:
    """Custom legend handles separating the two encodings of a first/last-
    `n_blocks`-split plot: color for reward status, linestyle for
    first-vs-last blocks -- rather than four combined labels
    ("Rewarded odor (first 100 blocks)", ...).
    """
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], color=colors[True], marker="o", label="Rewarded odor"),
        Line2D([0], [0], color=colors[False], marker="o", label="Non-rewarded odor"),
        Line2D(
            [0], [0], color="gray", linestyle="--", label=f"First {n_blocks} blocks"
        ),
        Line2D([0], [0], color="gray", linestyle="-", label=f"Last {n_blocks} blocks"),
    ]


@contextmanager
def a_lot_of_style(
    font_scale=1.2,
    line_width=2,
    grid=True,
    despine=True,
    ticks_out=True,
):
    old_params = plt.rcParams.copy()

    plt.style.use("default")
    plt.rcParams.update(
        {
            # Fonts
            "font.size": 10 * font_scale,
            "axes.titlesize": 12 * font_scale,
            "axes.labelsize": 11 * font_scale,
            "xtick.labelsize": 9 * font_scale,
            "ytick.labelsize": 9 * font_scale,
            "legend.fontsize": 9 * font_scale,
            "font.family": "DejaVu Sans",
            # Lines and markers
            "lines.linewidth": line_width,
            "lines.markersize": 6 * font_scale,
            # Axes and grid
            "axes.spines.top": not despine,
            "axes.spines.right": not despine,
            "axes.grid": grid,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            # Ticks
            "xtick.direction": "out" if ticks_out else "in",
            "ytick.direction": "out" if ticks_out else "in",
            "xtick.major.size": 4 * font_scale,
            "ytick.major.size": 4 * font_scale,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    try:
        yield
    finally:
        plt.rcParams.update(old_params)


def plot_choice_by_block_position(
    trials: pd.DataFrame,
    ax=None,
    legend: bool = True,
    from_first_stop: bool = False,
    colors: dict | None = None,
):
    """Plot ``P(Choice)`` against within-block appearance index, per odor.

    Restricting to RewardSite trials that belong to a kept block, the appearance
    index within each block is computed *separately* for the rewarded and the
    non-rewarded odor (0-4, since each odor appears 5 times per block of 10).
    The mean ``has_choice`` is then plotted against that index, giving two series
    -- one for ``is_rewarded_odor`` True and one for False -- with bootstrapped
    95% CI error bars (across trials at that appearance index).

    When ``from_first_stop`` is True, the index origin is the block's first stop
    instead of the block start: trials before the first stop (first ``has_choice``)
    are dropped and the remaining trials are re-numbered from 0, still counted
    separately per odor. The odor the first stop landed on therefore has
    ``has_choice == 1`` at index 0 by construction.

    Requires the ``block`` (see :func:`analysis.features.assign_blocks`) and
    ``is_rewarded_odor`` columns to already be present on *trials*.
    """
    rs = appearance_table(trials, from_first_stop=from_first_stop)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    rng = np.random.default_rng(0)
    colors = colors if colors is not None else {True: "tab:orange", False: "tab:blue"}
    for is_rewarded, grp in rs.groupby("is_rewarded_odor"):
        stats = bootstrap_group_stats(grp["has_choice"], grp["appearance"], rng)
        label = "Rewarded odor" if is_rewarded else "Non-rewarded odor"
        plot_mean_ci_band(
            ax, stats, color=colors[bool(is_rewarded)], label=label, marker="o"
        )

    ax.set_xlabel(
        "Appearance from first stop" if from_first_stop else "Appearance within block"
    )
    ax.set_ylabel("P(Choice)")
    ax.set_xticks(range(5))
    ax.set_ylim(0, 1.05)
    if legend:
        ax.legend()
    return ax


def plot_choice_by_odor_appearance(
    trials: pd.DataFrame,
    ax=None,
    legend: bool = True,
    from_first_stop: bool = False,
    colors: dict | None = None,
    n_blocks: int | None = 100,
):
    """Presentation twin of :func:`plot_choice_by_block_position`.

    Same x-axis -- the per-odor within-block occurrence index (0-4, computed
    separately for the rewarded and non-rewarded odor) -- but bootstrapped
    *across animals* rather than pooled across trials, and styled to the
    probability-axis convention (`(-0.1, 1.1)` limits, quartile ticks) instead
    of the pipeline's `(0, 1.05)`. :func:`plot_choice_by_block_position` stays
    exactly as-is (trial-level bootstrap, single-animal-only calls in
    ``pipeline.py``) -- this is a separate function, not a parameterization of
    it, so that one's output can't drift.

    ``n_blocks`` (default 100) splits each animal's first vs. last `n_blocks`
    chronological blocks (:func:`_first_last_blocks_table`, pooled across
    sessions) -- first `n_blocks` dashed, last `n_blocks` solid, color still
    carrying reward status -- so the plot itself shows whether the curve
    changes over training. Pass ``n_blocks=None`` to pool every block into a
    single line per reward status instead (no split), e.g. when the caller
    is already conditioning on something else (like
    :func:`analysis.features.label_first_stop`'s outcome) and a first/last
    split would just thin the data further.

    Requires the ``block`` and ``is_rewarded_odor`` columns to already be
    present on *trials*.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    rng = np.random.default_rng(0)
    colors = colors if colors is not None else {True: "tab:orange", False: "tab:blue"}

    if n_blocks is None:
        rs = appearance_table(trials, from_first_stop=from_first_stop)
        for is_rewarded, grp in rs.groupby("is_rewarded_odor"):
            stats = bootstrap_across_animals(grp, "has_choice", ["appearance"], rng)
            label = "Rewarded odor" if is_rewarded else "Non-rewarded odor"
            plot_mean_ci_band(
                ax, stats, color=colors[bool(is_rewarded)], label=label, marker="o"
            )
    else:
        rs = _first_last_blocks_table(
            trials, n_blocks=n_blocks, from_first_stop=from_first_stop
        )
        for (is_rewarded, block_range), grp in rs.groupby(
            ["is_rewarded_odor", "block_range"]
        ):
            stats = bootstrap_across_animals(grp, "has_choice", ["appearance"], rng)
            plot_mean_ci_band(
                ax,
                stats,
                color=colors[bool(is_rewarded)],
                marker="o",
                linestyle=BLOCK_RANGE_LINESTYLES[block_range],
            )

    ax.set_xlabel(
        "Appearance from first stop" if from_first_stop else "Appearance within block"
    )
    ax.set_ylabel("P(Choice)")
    ax.set_xticks(range(5))
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    if legend:
        if n_blocks is None:
            ax.legend()
        else:
            ax.legend(handles=_first_last_legend_handles(colors, n_blocks))
    return ax


def plot_choice_by_odor_appearance_by_animal(
    trials: pd.DataFrame,
    axes,
    ncols: int = 3,
    from_first_stop: bool = False,
    colors: dict | None = None,
    legend: bool = True,
    n_blocks: int = 100,
) -> None:
    """Per-animal small multiples of :func:`plot_choice_by_odor_appearance`.

    One subplot per `subject_id` (sorted), bootstrapped across that animal's
    own trials rather than across animals -- with only one animal per
    subplot, :func:`bootstrap_across_animals`'s across-animal convention
    would just return NaN (its `min_n=2` guard). Same probability-axis
    convention as the pooled plot (`(-0.1, 1.1)` limits, quartile ticks)
    since this is still a presentation panel, just faceted by animal. Same
    first-vs-last `n_blocks` split too (:func:`_first_last_blocks_table`,
    dashed/solid) -- the block-range boundaries are per-animal already, so
    restricting the input to one subject afterward doesn't change them.

    `axes` is a flat sequence of matplotlib axes arranged in a grid with
    `ncols` columns (e.g. a 2x3 grid passed as 6 axes for 5 animals); axes
    past the number of animals are hidden rather than left empty. Tick
    labels are kept only on the left column and the bottom-most populated
    axis of each column, so a jagged last row (as with 5 animals in a 3-wide
    grid) doesn't leave orphaned labels above a hidden axis. If `legend` is
    True, the reward-status/block-range legend is drawn in the first
    leftover axis (there is one whenever animals < len(axes)) instead of
    repeating it on every tiny subplot; with no leftover axis, no legend is
    drawn.
    """
    rs = _first_last_blocks_table(trials, n_blocks=n_blocks, from_first_stop=from_first_stop)
    colors = colors if colors is not None else {True: "tab:orange", False: "tab:blue"}
    rng = np.random.default_rng(0)
    subject_ids = sorted(rs["subject_id"].unique())

    axes = list(axes)
    if len(subject_ids) > len(axes):
        raise ValueError(
            f"{len(subject_ids)} animals but only {len(axes)} axes provided"
        )

    bottom_of_column = {i % ncols: i for i in range(len(subject_ids))}
    legend_drawn = False

    for i, ax in enumerate(axes):
        if i >= len(subject_ids):
            if legend and not legend_drawn:
                ax.legend(
                    handles=_first_last_legend_handles(colors, n_blocks),
                    loc="center",
                    frameon=False,
                )
                legend_drawn = True
            ax.axis("off")
            continue

        subject_id = subject_ids[i]
        sub = rs[rs["subject_id"] == subject_id]
        for (is_rewarded, block_range), grp in sub.groupby(
            ["is_rewarded_odor", "block_range"]
        ):
            stats = bootstrap_group_stats(grp["has_choice"], grp["appearance"], rng)
            plot_mean_ci_band(
                ax,
                stats,
                color=colors[bool(is_rewarded)],
                marker="o",
                linestyle=BLOCK_RANGE_LINESTYLES[block_range],
            )

        ax.set_title(str(subject_id))
        ax.set_xticks(range(5))
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        if i % ncols != 0:
            ax.set_yticklabels([])
        if i != bottom_of_column[i % ncols]:
            ax.set_xticklabels([])


def plot_choice_by_block_position_raw(
    trials: pd.DataFrame,
    ax=None,
    legend: bool = True,
    from_first_stop: bool = False,
    colors: dict | None = None,
):
    """Plot ``P(Choice)`` against the raw within-block position, split by odor.

    Unlike :func:`plot_choice_by_block_position`, the x-axis here is the
    actual serial position within the block, grouped by
    :func:`analysis.features.assign_odor_blocks`'s contingency-detected
    ``odor_block`` -- not a per-odor occurrence count (0-4), and not
    :func:`analysis.features.assign_blocks`'s fixed-count ``block`` (wrong
    for curriculum stages whose true block isn't that length). The two series
    (rewarded / non-rewarded odor) are still split, but at a given raw
    position only the blocks where that odor actually landed there
    contribute a point to that series -- unlike the per-odor-indexed version,
    where every block contributes exactly one point to each series at each
    index, here the two series can draw on different (and position-varying)
    numbers of blocks.

    ``from_first_stop`` re-origins the index at the block's first stop, as in
    :func:`plot_choice_by_block_position`.

    Requires the ``odor_index`` and ``is_rewarded_odor`` columns to already
    be present on *trials* (``assign_odor_blocks`` is called internally).
    """
    rs = appearance_table(trials, from_first_stop=from_first_stop, by_odor=False)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    rng = np.random.default_rng(0)
    colors = colors if colors is not None else {True: "tab:orange", False: "tab:blue"}
    for is_rewarded, grp in rs.groupby("is_rewarded_odor"):
        stats = bootstrap_across_animals(grp, "has_choice", ["appearance"], rng)
        label = "Rewarded odor" if is_rewarded else "Non-rewarded odor"
        plot_mean_ci_band(ax, stats, color=colors[bool(is_rewarded)], label=label)

    ax.set_xlabel(
        "Position from first stop" if from_first_stop else "Position within block"
    )
    ax.set_ylabel("P(Choice)")
    ax.set_xticks(range(0, int(rs["appearance"].max()) + 1))
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    if legend:
        ax.legend()
    return ax


def plot_paired_appearance(
    trials: pd.DataFrame,
    pos_a: int,
    pos_b: int,
    ax=None,
    from_first_stop: bool = False,
    colors: dict | None = None,
    group_colors: dict | None = None,
):
    """Per-animal ``P(Choice)`` at two raw within-block positions, paired and
    connected, both odor reward statuses together on one shared axis.

    A "slopegraph" companion to :func:`plot_choice_by_block_position_raw`'s
    pooled band: each animal contributes one point at ``pos_a`` and one at
    ``pos_b`` (its own mean ``has_choice`` at that raw position), joined by a
    line -- a small inset showing whether the pooled effect holds
    animal-by-animal rather than only in the aggregate mean. ``pos_a``/
    ``pos_b`` are parameters rather than hard-coded to 0 vs 9 since this is
    meant to be reused for other position pairs.

    Both ``is_rewarded_odor`` groups (non-rewarded, then rewarded) are drawn
    side by side on this one axis instead of on two separate axes, so there's
    a single shared y-axis rather than two identical ones; which group is
    which is marked by a thin ``group_colors`` bar along the top of that
    group's x-range instead of a per-axis label/title.

    ``colors`` maps ``subject_id`` (as ``str``) to a hex color -- pass
    :func:`analysis.plotting_style.animal_palette`'s output; a subject absent
    from ``colors`` falls back to gray. ``group_colors`` maps
    ``{True: ..., False: ...}`` to the rewarded/non-rewarded indicator-bar
    color, defaulting to matplotlib's ``tab:orange``/``tab:blue``.

    Requires the ``odor_index``/``is_rewarded_odor`` columns
    (``assign_odor_blocks`` is called internally via :func:`appearance_table`).
    """
    rs = appearance_table(trials, from_first_stop=from_first_stop, by_odor=False)
    rs = rs[rs["appearance"].isin([pos_a, pos_b])]
    per_animal = (
        rs.groupby(["subject_id", "is_rewarded_odor", "appearance"])["has_choice"]
        .mean()
        .unstack("appearance")
    )
    if pos_a not in per_animal.columns or pos_b not in per_animal.columns:
        raise ValueError(f"no data at position {pos_a} and/or {pos_b}")

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    colors = colors or {}
    group_colors = group_colors or {True: "tab:orange", False: "tab:blue"}
    xticks: list[float] = []
    xticklabels: list[str] = []
    for group_index, is_rewarded in enumerate([False, True]):
        x0, x1 = group_index * 3, group_index * 3 + 1
        group = per_animal.xs(is_rewarded, level="is_rewarded_odor")
        for subject_id, row in group.iterrows():
            if pd.isna(row[pos_a]) or pd.isna(row[pos_b]):
                continue
            color = colors.get(str(subject_id), "gray")
            ax.plot([x0, x1], [row[pos_a], row[pos_b]], color=color, marker="o")
        ax.axvspan(
            x0 - 0.3, x1 + 0.3, ymin=0.97, ymax=1.0, color=group_colors[is_rewarded], linewidth=0
        )
        xticks += [x0, x1]
        xticklabels += [f"Pos {pos_a}", f"Pos {pos_b}"]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xticks[0] - 0.6, xticks[-1] + 0.6)
    ax.set_ylabel("P(Choice)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    return ax


def _plot_sessions_on_ax(
    sub: pd.DataFrame,
    ax,
    colors: dict,
    from_first_stop: bool = True,
) -> None:
    """Draw all sessions for one animal side-by-side on *ax*, separated by vertical lines."""
    N_APP = 5
    GAP = 2
    STRIDE = N_APP + GAP
    sessions = sorted(sub["session_id"].unique())
    seen = {True: False, False: False}
    rng = np.random.default_rng(0)

    for s_idx, session_id in enumerate(sessions):
        rs = appearance_table(
            sub[sub["session_id"] == session_id], from_first_stop=from_first_stop
        )
        x_offset = s_idx * STRIDE

        for is_rewarded, grp in rs.groupby("is_rewarded_odor"):
            stats = bootstrap_group_stats(grp["has_choice"], grp["appearance"], rng)
            if stats.empty:
                continue
            label = (
                ("Rewarded odor" if is_rewarded else "Non-rewarded odor")
                if not seen[bool(is_rewarded)]
                else "_nolegend_"
            )
            shifted = stats.set_axis(stats.index + x_offset)
            plot_mean_ci_band(
                ax, shifted, color=colors[bool(is_rewarded)], label=label, marker="o"
            )
            seen[bool(is_rewarded)] = True

        if s_idx < len(sessions) - 1:
            ax.axvline(
                x=x_offset + N_APP - 1 + GAP / 2,
                color="gray",
                linestyle="--",
                lw=0.8,
                alpha=0.6,
            )

        ax.text(
            x_offset + (N_APP - 1) / 2,
            1.01,
            session_id.split("_", 1)[1],
            ha="center",
            va="bottom",
            fontsize=7,
            transform=ax.get_xaxis_transform(),
        )

    all_ticks = [s * STRIDE + a for s in range(len(sessions)) for a in range(N_APP)]
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(
        [str(a) for _ in range(len(sessions)) for a in range(N_APP)], fontsize=7
    )
    ax.set_xlim(-0.5, len(sessions) * STRIDE - GAP - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(
        "Appearance from first stop" if from_first_stop else "Appearance within block"
    )


def plot_choice_by_block_position_raw_by_tercile(
    trials: pd.DataFrame,
    axes=None,
    legend: bool = True,
    from_first_stop: bool = False,
    colors: dict | None = None,
):
    """Three-panel version of :func:`plot_choice_by_block_position_raw`, one
    panel per chronological tercile (first/mid/last third) of each animal's
    blocks -- see :func:`analysis.features.block_tercile_tags` -- to show
    whether/how the within-block curve changes over training. Each panel is
    otherwise identical to :func:`plot_choice_by_block_position_raw`,
    restricted to that tercile's blocks; the y-axis is shared and only drawn
    once (leftmost panel) rather than repeated on all three.

    Tercile boundaries are computed on :func:`analysis.features.assign_odor_blocks`'s
    ``odor_block`` (the same grouping :func:`plot_choice_by_block_position_raw`
    itself uses), not :func:`analysis.features.assign_blocks`'s fixed-count
    ``block``.

    ``axes`` needs exactly 3 axes (e.g. from
    ``new_figure("wide", ncols=3, sharey=True)``); 3 side-by-side axes are
    created if not given.
    """
    trials_od = assign_odor_blocks(trials)
    tercile_tags = block_tercile_tags(trials_od, block_col="odor_block")

    rs = appearance_table(trials, from_first_stop=from_first_stop, by_odor=False)
    rs = rs.merge(
        tercile_tags[["subject_id", "session_id", "odor_block", "block_tercile"]],
        on=["subject_id", "session_id", "odor_block"],
        how="left",
    )

    if axes is None:
        _, axes = plt.subplots(ncols=3, figsize=(15, 4), sharey=True)

    rng = np.random.default_rng(0)
    colors = colors if colors is not None else {True: "tab:orange", False: "tab:blue"}
    xmax = int(rs["appearance"].max())
    for i, (ax, tercile) in enumerate(zip(axes, ["first", "mid", "last"])):
        sub = rs[rs["block_tercile"] == tercile]
        for is_rewarded, grp in sub.groupby("is_rewarded_odor"):
            stats = bootstrap_across_animals(grp, "has_choice", ["appearance"], rng)
            label = "Rewarded odor" if is_rewarded else "Non-rewarded odor"
            plot_mean_ci_band(ax, stats, color=colors[bool(is_rewarded)], label=label)
        ax.set_xlabel(f"{tercile.capitalize()} tercile")
        ax.set_xticks(range(0, xmax + 1))
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        if i == 0:
            ax.set_ylabel("P(Choice)")
        else:
            ax.tick_params(labelleft=False)
        if legend and i == 0:
            ax.legend()
    axes[0].figure.supxlabel(
        "Position from first stop" if from_first_stop else "Position within block"
    )
    return axes


def plot_choice_by_block_position_per_session(
    trials: pd.DataFrame,
    title_suffix: str = "",
    from_first_stop: bool = False,
    ax=None,
    single_axis: bool = False,
):
    """Per animal, plot :func:`plot_choice_by_block_position` for each session.

    Produces one figure per subject (parsed from ``session_id``), with that
    subject's sessions laid out chronologically as consecutive subplots sharing
    a common y-axis. ``title_suffix`` is appended to each figure's title.
    ``from_first_stop`` is forwarded to :func:`plot_choice_by_block_position`.
    Returns a ``{subject_id: figure}`` dict.

    If ``ax`` is given, the per-session/per-subject split is dropped: all data is
    pooled into that single axes via :func:`plot_choice_by_block_position` and
    the axes is returned instead of a figure dict.

    If ``single_axis`` is True, all sessions for each subject are drawn on a
    single axes with vertical separators between them and session date labels at
    the top, rather than one subplot per session.
    """
    if ax is not None:
        plot_choice_by_block_position(trials, ax=ax, from_first_stop=from_first_stop)
        if title_suffix:
            ax.set_title(title_suffix.lstrip(" —"))
        return ax

    trials = trials.copy()

    if single_axis:
        colors = {True: "tab:orange", False: "tab:blue"}
        figures = {}
        for subject, sub in trials.groupby("subject_id"):
            sessions = sorted(sub["session_id"].unique())
            fig, single_ax = plt.subplots(figsize=(max(10, 1.8 * len(sessions)), 4))
            _plot_sessions_on_ax(
                sub, single_ax, colors, from_first_stop=from_first_stop
            )
            single_ax.set_ylabel("P(Choice)")
            single_ax.legend()
            fig.suptitle(f"Subject {subject}{title_suffix}")
            fig.tight_layout()
            figures[subject] = fig

        return figures

    figures = {}
    for subject, sub in trials.groupby("subject_id"):
        sessions = sorted(
            sub["session_id"].unique()
        )  # session_id sorts chronologically
        # Cap total figure width to ~2200 px (regardless of the active dpi, which
        # a style context may raise) so all sessions stay visible in the notebook
        # rather than the figure overflowing the cell and clipping to session 1.
        dpi = plt.rcParams["figure.dpi"]
        per_session_w = min(4.0, (2200 / dpi) / len(sessions))
        fig, axes = plt.subplots(
            1,
            len(sessions),
            figsize=(per_session_w * len(sessions), 4),
            sharey=True,
            squeeze=False,
        )
        for ax, session_id in zip(axes[0], sessions):
            plot_choice_by_block_position(
                sub[sub["session_id"] == session_id],
                ax=ax,
                legend=False,
                from_first_stop=from_first_stop,
            )
            # title with the date/time part only (drop the redundant subject prefix)
            ax.set_title(session_id.split("_", 1)[1])
        axes[0][0].legend()
        fig.suptitle(f"Subject {subject}{title_suffix}")
        fig.tight_layout()
        figures[subject] = fig

    return figures


_FIRST_STOP_COLORS = {
    (True, True): "#c0392b",  # first-stop rewarded,     rewarded odor
    (True, False): "#e07b39",  # first-stop rewarded,     non-rewarded odor
    (False, True): "#1a5276",  # first-stop non-rewarded, rewarded odor
    (False, False): "#4f8fc0",  # first-stop non-rewarded, non-rewarded odor
}


def plot_choice_by_block_position_by_first_stop(trials: pd.DataFrame) -> dict:
    """Per-animal 2-row × N-sessions grid of P(choice), split by first-stop outcome.

    Top row: blocks whose first stop was rewarded.
    Bottom row: blocks whose first stop was not rewarded.
    One column per session; each cell calls plot_choice_by_block_position directly.

    Colors match the 4-condition palette used in the counterfactual analysis.
    Returns a {subject_id: figure} dict.
    """
    trials = label_first_stop(trials)

    CONDITIONS = [
        (True, "First stop: Rewarded"),
        (False, "First stop: Non-rewarded"),
    ]

    figures = {}
    for subject, sub in trials.groupby("subject_id"):
        sessions = sorted(sub["session_id"].unique())
        n_sess = len(sessions)

        fig, axes = plt.subplots(
            2,
            n_sess,
            figsize=(max(10, 3.5 * n_sess), 8),
            sharey=True,
            squeeze=False,
        )

        for row, (is_fsr, row_label) in enumerate(CONDITIONS):
            cond_colors = {
                True: _FIRST_STOP_COLORS[(is_fsr, True)],
                False: _FIRST_STOP_COLORS[(is_fsr, False)],
            }
            subset = sub[sub["first_stop_rewarded"] == is_fsr]
            for col, session_id in enumerate(sessions):
                ax = axes[row][col]
                plot_choice_by_block_position(
                    subset[subset["session_id"] == session_id],
                    ax=ax,
                    colors=cond_colors,
                    from_first_stop=True,
                    legend=False,
                )
                if row == 0:
                    ax.set_title(session_id.split("_", 1)[1], fontsize=8)
                if col > 0:
                    ax.set_ylabel("")
                if col == 0:
                    ax.annotate(
                        row_label,
                        xy=(0, 0.5),
                        xycoords="axes fraction",
                        xytext=(-0.35, 0.5),
                        textcoords="axes fraction",
                        rotation=90,
                        va="center",
                        ha="right",
                        fontsize=9,
                    )

        axes[0][-1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Subject {subject} — P(choice) by appearance from first stop")
        fig.tight_layout()
        figures[subject] = fig

    return figures


def plot_choice_by_block_position_by_first_stop_overlay(trials: pd.DataFrame):
    """:func:`plot_choice_by_block_position_by_first_stop`, days overlaid on one axes.

    Same analysis as the previous plot -- blocks are labelled by their
    first-stop outcome (:func:`analysis.features.label_first_stop`), aligned to
    the first stop, and ``P(Choice)`` is computed per odor as a function of
    appearance-from-first-stop. The only difference is presentation: instead of
    one subplot per session, *all* of a subject's sessions are overlaid. Each
    session (day) is drawn with a colour from a gradient that encodes its
    chronological order, and the rewarded vs non-rewarded odor are split into
    two side-by-side panels using *different* base colourmaps (oranges vs
    blues) so the shade still reads as the day. Returns a
    ``{(subject_id, condition): figure}`` dict.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    trials = label_first_stop(trials)
    trials = trials.copy()

    # one panel per odor, with its own gradient base colourmap
    # (matching the tab:orange / tab:blue of the non-overlay version)
    odor_panels = [
        (True, "Rewarded odor", "Oranges"),
        (False, "Non-rewarded odor", "Blues"),
    ]

    figures = {}
    for condition, is_first_stop_rewarded in [
        ("first-stop rewarded", True),
        ("first-stop non-rewarded", False),
    ]:
        cond = trials[trials["first_stop_rewarded"] == is_first_stop_rewarded]
        for subject, sub in cond.groupby("subject_id"):
            sessions = sorted(sub["session_id"].unique())  # sorts chronologically
            n = len(sessions)
            # map day index -> [0.35, 1.0] so even the earliest day stays visible
            shades = [0.35 + 0.65 * (i / max(n - 1, 1)) for i in range(n)]
            norm = Normalize(vmin=0, vmax=max(n - 1, 1))

            rng = np.random.default_rng(0)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
            for ax, (is_rewarded, title, cmap_name) in zip(axes, odor_panels):
                cmap = plt.get_cmap(cmap_name)
                for day, session_id in enumerate(sessions):
                    rs = appearance_table(
                        sub[sub["session_id"] == session_id], from_first_stop=True
                    )
                    grp = rs[rs["is_rewarded_odor"] == is_rewarded]
                    if grp.empty:
                        continue
                    stats = bootstrap_group_stats(
                        grp["has_choice"], grp["appearance"], rng
                    )
                    ax.errorbar(
                        stats.index,
                        stats["mean"],
                        yerr=ci_errorbar(stats),
                        marker="o",
                        capsize=2,
                        color=cmap(shades[day]),
                    )
                ax.set_xlabel("Appearance from first stop")
                ax.set_xticks(range(5))
                ax.set_ylim(0, 1.05)
                ax.set_title(title)
                # day gradient as a colourbar, no per-day text labels
                cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
                cb.set_label("Day")
                cb.set_ticks([])

            axes[0].set_ylabel("P(Choice)")
            fig.suptitle(f"Subject {subject} — {condition}")
            fig.tight_layout()
            figures[(subject, condition)] = fig

    return figures


def plot_naive_p_stop_first_last(trials: pd.DataFrame, n_blocks: int = 200):
    """:func:`plot_choice_by_block_position`'s curve -- P(Choice) against
    within-block appearance, split by odor reward status -- but averaged
    across animals (each animal contributes its own per-appearance mean, and
    the cohort mean +/- bootstrapped 95% CI across animals is drawn), and
    with one subplot for each animal's first vs last ``n_blocks`` blocks
    (see :func:`analysis.features.blocks_first_last_tags`) instead of per
    session.
    """
    tagged = appearance_table(blocks_first_last_tags(trials, n_blocks=n_blocks))

    rng = np.random.default_rng(0)
    colors = {True: "tab:orange", False: "tab:blue"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, block_range in zip(axes, ["first", "last"]):
        sub = tagged[tagged["block_range"] == block_range]
        for is_rewarded, grp in sub.groupby("is_rewarded_odor"):
            stats = bootstrap_across_animals(grp, "has_choice", ["appearance"], rng)
            label = "Rewarded odor" if is_rewarded else "Non-rewarded odor"
            plot_mean_ci_band(
                ax, stats, color=colors[bool(is_rewarded)], label=label, marker="o"
            )
        ax.set_xlabel("Appearance within block")
        ax.set_xticks(range(5))
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{block_range.capitalize()} {n_blocks} blocks")

    axes[0].set_ylabel("P(Choice)")
    axes[0].legend()
    fig.suptitle(
        f"P(Choice) by odor reward status, averaged across animals\n"
        f"(first vs last {n_blocks} blocks, pooled chronologically across sessions; "
        "shaded band = bootstrapped 95% CI across animals)"
    )
    fig.tight_layout()
    return fig
