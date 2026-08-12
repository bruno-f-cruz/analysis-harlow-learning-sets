"""Matplotlib plot-drawing / styling helpers shared across the notebook's cells.

Moved out of the old top-level ``viz_helpers.py`` and ``helpers.py`` (see
``docs/plans/2026-08-11-dockerized-analysis-environment.md``, Task 9.2). This
holds ``a_lot_of_style`` plus the generic "choice by block position" plotting
family, which is reused across several notebook cells and carries no
research-question-specific interpretation of its own (unlike the GLM-fitting /
bias / counterfactual analysis, which is intentionally left inline in
``workflows/analysis.py``, formerly ``demo_marimo.py``).
"""

from contextlib import contextmanager

import pandas as pd
import matplotlib.pyplot as plt

from analysis.features import _appearance_table, label_first_stop


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
    """Plot ``P(has_choice)`` against within-block appearance index, per odor.

    Restricting to RewardSite trials that belong to a kept block, the appearance
    index within each block is computed *separately* for the rewarded and the
    non-rewarded odor (0-4, since each odor appears 5 times per block of 10).
    The mean ``has_choice`` is then plotted against that index, giving two series
    -- one for ``is_rewarded_odor`` True and one for False -- with SEM error bars.

    When ``from_first_stop`` is True, the index origin is the block's first stop
    instead of the block start: trials before the first stop (first ``has_choice``)
    are dropped and the remaining trials are re-numbered from 0, still counted
    separately per odor. The odor the first stop landed on therefore has
    ``has_choice == 1`` at index 0 by construction.

    Requires the ``block`` (see :func:`analysis.features.assign_blocks`) and
    ``is_rewarded_odor`` columns to already be present on *trials*.
    """
    rs = _appearance_table(trials, from_first_stop=from_first_stop)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    colors = colors if colors is not None else {True: "tab:orange", False: "tab:blue"}
    for is_rewarded, grp in rs.groupby("is_rewarded_odor"):
        stats = grp.groupby("appearance")["has_choice"].agg(["mean", "sem"])
        label = "Rewarded odor" if is_rewarded else "Non-rewarded odor"
        ax.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["sem"],
            marker="o",
            capsize=3,
            label=label,
            color=colors[bool(is_rewarded)],
        )

    ax.set_xlabel(
        "Appearance from first stop" if from_first_stop else "Appearance within block"
    )
    ax.set_ylabel("P(has_choice)")
    ax.set_xticks(range(5))
    ax.set_ylim(0, 1.05)
    if legend:
        ax.legend()
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

    for s_idx, session_id in enumerate(sessions):
        rs = _appearance_table(
            sub[sub["session_id"] == session_id], from_first_stop=from_first_stop
        )
        x_offset = s_idx * STRIDE

        for is_rewarded, grp in rs.groupby("is_rewarded_odor"):
            stats = grp.groupby("appearance")["has_choice"].agg(["mean", "sem"])
            if stats.empty:
                continue
            label = (
                ("Rewarded odor" if is_rewarded else "Non-rewarded odor")
                if not seen[bool(is_rewarded)]
                else "_nolegend_"
            )
            ax.errorbar(
                stats.index + x_offset,
                stats["mean"],
                yerr=stats["sem"],
                marker="o",
                capsize=3,
                color=colors[bool(is_rewarded)],
                label=label,
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
            single_ax.set_ylabel("P(has_choice)")
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
    the first stop, and ``P(has_choice)`` is computed per odor as a function of
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

            fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
            for ax, (is_rewarded, title, cmap_name) in zip(axes, odor_panels):
                cmap = plt.get_cmap(cmap_name)
                for day, session_id in enumerate(sessions):
                    rs = _appearance_table(
                        sub[sub["session_id"] == session_id], from_first_stop=True
                    )
                    grp = rs[rs["is_rewarded_odor"] == is_rewarded]
                    if grp.empty:
                        continue
                    stats = grp.groupby("appearance")["has_choice"].agg(["mean", "sem"])
                    ax.errorbar(
                        stats.index,
                        stats["mean"],
                        yerr=stats["sem"],
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

            axes[0].set_ylabel("P(has_choice)")
            fig.suptitle(f"Subject {subject} — {condition}")
            fig.tight_layout()
            figures[(subject, condition)] = fig

    return figures
