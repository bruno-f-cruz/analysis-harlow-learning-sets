import numpy as np
import pandas as pd


def assign_blocks(trials: pd.DataFrame, block_size: int = 10) -> pd.DataFrame:
    """Add a ``block`` column grouping consecutive RewardSite trials into blocks.

    Within each session, RewardSite trials (in temporal order) are grouped into
    blocks of ``block_size``. Block numbering resets to 0 at the start of every
    session. The last block of each session is always dropped (set to NaN) since
    the session ends mid-block and those trials are never used -- this applies
    even when the final block happens to be complete. Non-RewardSite rows
    (InterSite / InterPatch) are always NaN.
    """
    block = pd.Series(pd.NA, index=trials.index, dtype="Int64")
    reward_sites = trials[trials["site_label"] == "RewardSite"]
    for _, grp in reward_sites.groupby("session_id", sort=False):
        ordered_idx = grp.sort_values("start_time").index
        local = (np.arange(len(ordered_idx)) // block_size).astype(float)
        local[local == local.max()] = np.nan  # drop the trailing block
        block.loc[ordered_idx] = pd.array(local, dtype="Int64")
    trials["block"] = block
    return trials


def trim_sessions(
    trials: pd.DataFrame, start_frac: float = 0.0, end_frac: float = 1.0
) -> pd.DataFrame:
    """Keep the ``[start_frac, end_frac]`` window of each session's rows.

    Within each session, rows are ordered chronologically (by ``start_time``)
    and only the fractional window from ``start_frac`` to ``end_frac`` is kept.
    Both are fractions of the session measured from its start, so e.g.
    ``start_frac=0.0, end_frac=0.7`` keeps the first 70% and
    ``start_frac=0.1, end_frac=0.9`` keeps the middle 80%. The default keeps the
    whole session. Returns a new trimmed DataFrame.
    """
    kept = []
    for _, grp in trials.groupby("session_id", sort=False):
        g = grp.sort_values("start_time")
        n = len(g)
        lo = int(round(n * start_frac))
        hi = int(round(n * end_frac))
        kept.append(g.iloc[lo:hi])
    return pd.concat(kept) if kept else trials.iloc[:0]


def _appearance_table(
    trials: pd.DataFrame, from_first_stop: bool = False
) -> pd.DataFrame:
    """Return RewardSite trials (kept blocks only) with an ``appearance`` column.

    ``appearance`` is the within-block, per-odor index (0-4). When
    ``from_first_stop`` is True the index origin is each block's first stop and
    trials before it are dropped (blocks with no stop are dropped entirely).
    Shared by the choice-by-block-position plots so they all count appearances
    identically.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    rs = rs.sort_values(["session_id", "block", "start_time"])

    if from_first_stop:
        # position of the first stop within each block, then keep only trials
        # at or after it (blocks with no stop have no first_stop_pos -> dropped)
        rs["_pos"] = rs.groupby(["session_id", "block"]).cumcount()
        first_stop_pos = (
            rs[rs["has_choice"]].groupby(["session_id", "block"])["_pos"].min()
        )
        rs = rs.join(
            first_stop_pos.rename("_first_stop_pos"), on=["session_id", "block"]
        )
        rs = rs[rs["_first_stop_pos"].notna() & (rs["_pos"] >= rs["_first_stop_pos"])]

    rs["appearance"] = rs.groupby(
        ["session_id", "block", "is_rewarded_odor"]
    ).cumcount()
    return rs


def plot_choice_by_block_position(
    trials: pd.DataFrame, ax=None, legend: bool = True, from_first_stop: bool = False
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

    Requires the ``block`` (see :func:`assign_blocks`) and ``is_rewarded_odor``
    columns to already be present on *trials*.
    """
    import matplotlib.pyplot as plt

    rs = _appearance_table(trials, from_first_stop=from_first_stop)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    colors = {True: "tab:orange", False: "tab:blue"}
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
    import matplotlib.pyplot as plt

    if ax is not None:
        plot_choice_by_block_position(trials, ax=ax, from_first_stop=from_first_stop)
        if title_suffix:
            ax.set_title(title_suffix.lstrip(" —"))
        return ax

    trials = trials.copy()
    trials["subject_id"] = trials["session_id"].str.split("_").str[0]

    if single_axis:
        n_app = 5   # appearances per odor per block (0-4)
        gap = 2     # x-units of blank space between sessions
        stride = n_app + gap
        colors = {True: "tab:orange", False: "tab:blue"}

        figures = {}
        for subject, sub in trials.groupby("subject_id"):
            sessions = sorted(sub["session_id"].unique())
            fig, single_ax = plt.subplots(figsize=(max(10, 1.8 * len(sessions)), 4))
            seen = {True: False, False: False}

            for s_idx, session_id in enumerate(sessions):
                rs = _appearance_table(
                    sub[sub["session_id"] == session_id],
                    from_first_stop=from_first_stop,
                )
                x_offset = s_idx * stride

                for is_rewarded, grp in rs.groupby("is_rewarded_odor"):
                    stats = grp.groupby("appearance")["has_choice"].agg(["mean", "sem"])
                    if stats.empty:
                        continue
                    color = colors[bool(is_rewarded)]
                    lbl = (
                        ("Rewarded odor" if is_rewarded else "Non-rewarded odor")
                        if not seen[bool(is_rewarded)]
                        else "_nolegend_"
                    )
                    single_ax.errorbar(
                        stats.index + x_offset,
                        stats["mean"],
                        yerr=stats["sem"],
                        marker="o",
                        capsize=3,
                        color=color,
                        label=lbl,
                    )
                    seen[bool(is_rewarded)] = True

                # vertical separator between sessions
                if s_idx < len(sessions) - 1:
                    single_ax.axvline(
                        x=x_offset + n_app - 1 + gap / 2,
                        color="gray",
                        linestyle="--",
                        lw=0.8,
                        alpha=0.6,
                    )

                # session label at top of each session block
                x_center = x_offset + (n_app - 1) / 2
                single_ax.text(
                    x_center,
                    1.01,
                    session_id.split("_", 1)[1],
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    transform=single_ax.get_xaxis_transform(),
                )

            all_ticks = [
                s * stride + a for s in range(len(sessions)) for a in range(n_app)
            ]
            single_ax.set_xticks(all_ticks)
            single_ax.set_xticklabels(
                [str(a) for _ in range(len(sessions)) for a in range(n_app)],
                fontsize=7,
            )
            single_ax.set_xlim(-0.5, len(sessions) * stride - gap - 0.5)
            single_ax.set_ylim(0, 1.05)
            single_ax.set_ylabel("P(has_choice)")
            single_ax.set_xlabel(
                "Appearance from first stop"
                if from_first_stop
                else "Appearance within block"
            )
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


def label_first_stop(trials: pd.DataFrame) -> pd.DataFrame:
    """Add ``first_stop_rewarded``: per block, whether the first stop was rewarded.

    For each ``(session_id, block)``, RewardSite trials are scanned in temporal
    order for the first one where the animal stopped (``has_choice``); that
    stop's ``has_reward`` becomes the block-level label (constant for every row
    of the block). Blocks where the animal never stopped get ``<NA>`` and are
    thus excluded from the conditioned plots. Requires ``block`` to be assigned.
    """
    trials = trials.copy()
    label = pd.Series(pd.NA, index=trials.index, dtype="boolean")
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    for _, grp in rs.groupby(["session_id", "block"]):
        ordered = grp.sort_values("start_time")
        stops = ordered[ordered["has_choice"]]
        if stops.empty:
            continue  # animal never stopped in this block -> leave <NA>
        label.loc[ordered.index] = bool(stops.iloc[0]["has_reward"])
    trials["first_stop_rewarded"] = label
    return trials


def plot_choice_by_block_position_by_first_stop(
    trials: pd.DataFrame, ax=None, single_axis: bool = False
):
    """:func:`plot_choice_by_block_position_per_session`, split by first-stop outcome.

    Labels each block by whether its first stop was rewarded
    (:func:`label_first_stop`), then produces the full per-animal/per-session
    figure set twice: once for first-stop-rewarded blocks and once for
    first-stop-non-rewarded blocks. Returns a ``{(subject_id, condition): figure}``
    dict.

    If ``ax`` is given, every split (subject, session, and condition) is dropped:
    all blocks that contain a stop are pooled and drawn (aligned to the first
    stop) into that single axes, which is returned.

    ``single_axis`` is forwarded to :func:`plot_choice_by_block_position_per_session`
    to draw all sessions on one axes per subject/condition with vertical separators.
    """
    trials = label_first_stop(trials)

    if ax is not None:
        pooled = trials[trials["first_stop_rewarded"].notna()]
        plot_choice_by_block_position(pooled, ax=ax, from_first_stop=True)
        return ax

    figures = {}
    for condition, is_rewarded in [
        ("first-stop rewarded", True),
        ("first-stop non-rewarded", False),
    ]:
        subset = trials[trials["first_stop_rewarded"] == is_rewarded]
        for subject, fig in plot_choice_by_block_position_per_session(
            subset,
            title_suffix=f" — {condition}",
            from_first_stop=True,
            single_axis=single_axis,
        ).items():
            figures[(subject, condition)] = fig

    return figures


def plot_choice_by_block_position_by_first_stop_overlay(trials: pd.DataFrame):
    """:func:`plot_choice_by_block_position_by_first_stop`, days overlaid on one axes.

    Same analysis as cell 6 -- blocks are labelled by their first-stop outcome
    (:func:`label_first_stop`), aligned to the first stop, and ``P(has_choice)``
    is computed per odor as a function of appearance-from-first-stop. The only
    difference is presentation: instead of one subplot per session, *all* of a
    subject's sessions are overlaid. Each session (day) is drawn with a colour
    from a gradient that encodes its chronological order, and the rewarded vs
    non-rewarded odor are split into two side-by-side panels using *different*
    base colourmaps (oranges vs blues) so the shade still reads as the day.
    Returns a ``{(subject_id, condition): figure}`` dict.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    trials = label_first_stop(trials)
    trials = trials.copy()
    trials["subject_id"] = trials["session_id"].str.split("_").str[0]

    # one panel per odor, with its own gradient base colourmap
    # (matching the tab:orange / tab:blue of cell 6)
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


def plot_odor_preference_ranking(trials: pd.DataFrame):
    """Rank odors by P(stop) using only paired within-block data.

    For each block, P(stop) is computed separately for the rewarded odor and
    the non-rewarded odor.  Those block-level estimates are then averaged per
    odor index and role.  Odors are ranked along the x-axis by their mean
    P(stop) when *non-rewarded* — the role where reward cannot explain the
    stopping — as this best reflects intrinsic odor attractiveness.

    Two bars per odor: orange = rewarded role, blue = non-rewarded role.
    Error bars are SEM across blocks.  One figure per subject.
    """
    import matplotlib.pyplot as plt

    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    rs["subject_id"] = rs["session_id"].str.split("_").str[0]
    rs["odor_index"] = rs["patch_label"].str.split("_").str[0].astype(int)
    rs["is_non_rewarded"] = rs["patch_label"].str.contains("NonRewarded")

    figures = {}
    for subject, sub in rs.groupby("subject_id"):
        # per-block, per-odor P(stop) — keeps data paired within blocks
        block_stats = (
            sub.groupby(["session_id", "block", "odor_index", "is_non_rewarded"])[
                "has_choice"
            ]
            .mean()
            .reset_index(name="p_stop")
        )

        summary = (
            block_stats.groupby(["odor_index", "is_non_rewarded"])["p_stop"]
            .agg(mean="mean", sem=lambda x: x.sem())
            .reset_index()
        )

        # rank by mean P(stop) in the non-rewarded role (intrinsic attractiveness)
        nr_means = summary[summary["is_non_rewarded"]].set_index("odor_index")["mean"]
        odor_order = nr_means.sort_values(ascending=False).index.tolist()

        x = np.arange(len(odor_order))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(6, len(odor_order) * 0.9), 4))
        for offset, (is_non_rew, label, color) in enumerate(
            [
                (False, "Rewarded role", "tab:orange"),
                (True, "Non-rewarded role", "tab:blue"),
            ]
        ):
            grp = summary[summary["is_non_rewarded"] == is_non_rew].set_index(
                "odor_index"
            )
            means = [
                grp.loc[o, "mean"] if o in grp.index else np.nan for o in odor_order
            ]
            sems = [grp.loc[o, "sem"] if o in grp.index else np.nan for o in odor_order]
            ax.bar(
                x + (offset - 0.5) * width,
                means,
                width,
                yerr=sems,
                capsize=3,
                label=label,
                color=color,
                alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Odor {o}" for o in odor_order])
        ax.set_ylabel("P(stop)")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Subject {subject} — odor preference ranking\n(ordered by non-rewarded P(stop))"
        )
        ax.legend()
        fig.tight_layout()
        figures[subject] = fig

    return figures


def plot_patch_type_delta_heatmap(trials: pd.DataFrame):
    """Heatmap of paired ΔP(stop) across valid Rewarded vs NonRewarded patch combinations.

    Each block contributes exactly one (rewarded_label, non_rewarded_label) pair.
    Within that block, P(stop) is computed separately for each type.  The block-
    level delta ``P(stop | rewarded) − P(stop | non-rewarded)`` is then averaged
    across all blocks that share the same label combination.

    This paired-block design is the only valid comparison: data from different
    blocks is never mixed, and same-odor pairs (which never co-occur in a block)
    remain masked.

    One figure per subject is returned as a ``{subject_id: figure}`` dict.
    """
    import matplotlib.pyplot as plt

    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    rs["subject_id"] = rs["session_id"].str.split("_").str[0]

    def _odor_idx(label: str) -> int:
        return int(label.split("_", 1)[0])

    figures = {}
    for subject, sub in rs.groupby("subject_id"):
        # compute per-block P(stop) for each patch type present in that block
        block_patch = (
            sub.groupby(["session_id", "block", "patch_label"])["has_choice"]
            .mean()
            .reset_index(name="p_stop")
        )

        # split into rewarded and non-rewarded sides, then join on the block key
        rew = block_patch[
            ~block_patch["patch_label"].str.contains("NonRewarded")
        ].rename(columns={"patch_label": "rew_label", "p_stop": "p_stop_rew"})
        non = block_patch[
            block_patch["patch_label"].str.contains("NonRewarded")
        ].rename(columns={"patch_label": "non_label", "p_stop": "p_stop_non"})
        paired = rew.merge(non, on=["session_id", "block"])
        paired["delta"] = paired["p_stop_rew"] - paired["p_stop_non"]

        # average paired deltas per (rewarded, non-rewarded) label combination
        combo = paired.groupby(["rew_label", "non_label"])["delta"].mean().reset_index()

        rew_labels = sorted(combo["rew_label"].unique().tolist(), key=_odor_idx)
        non_labels = sorted(combo["non_label"].unique().tolist(), key=_odor_idx)
        nr, nc = len(rew_labels), len(non_labels)

        matrix = np.full((nr, nc), np.nan)
        for _, row in combo.iterrows():
            i = rew_labels.index(row["rew_label"])
            j = non_labels.index(row["non_label"])
            matrix[i, j] = row["delta"]

        vmax = np.nanmax(np.abs(matrix))
        fig, ax = plt.subplots(figsize=(max(5, nc * 0.8), max(4, nr * 0.75)))
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        fig.colorbar(
            im,
            ax=ax,
            label="mean paired ΔP(stop)  [rewarded − non-rewarded]",
            fraction=0.046,
            pad=0.04,
        )

        ax.set_xticks(range(nc))
        ax.set_yticks(range(nr))
        ax.set_xticklabels([l.replace("_", "\n") for l in non_labels], fontsize=8)
        ax.set_yticklabels([l.replace("_", "\n") for l in rew_labels], fontsize=8)

        for i in range(nr):
            for j in range(nc):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if abs(matrix[i, j]) < 0.6 * vmax else "white",
                    )

        ax.set_xlabel("Non-rewarded patch type")
        ax.set_ylabel("Rewarded patch type")
        ax.set_title(
            f"Subject {subject} — mean paired ΔP(stop) per block\n"
            "(same-odor pairs never co-occur → masked)"
        )
        fig.tight_layout()
        figures[subject] = fig

    return figures


def plot_choice_difference_by_block_position_overlay(trials: pd.DataFrame):
    """Per-day difference (rewarded minus non-rewarded) in ``P(has_choice)``.

    For each subject and first-stop condition, each session (day) contributes
    one curve: ``P(has_choice | rewarded odor) - P(has_choice | non-rewarded
    odor)`` as a function of appearance-from-first-stop. Days are coloured by
    chronological order using a purple gradient. Returns a
    ``{(subject_id, condition): figure}`` dict.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    trials = label_first_stop(trials)
    trials = trials.copy()
    trials["subject_id"] = trials["session_id"].str.split("_").str[0]

    figures = {}
    for condition, is_first_stop_rewarded in [
        ("first-stop rewarded", True),
        ("first-stop non-rewarded", False),
    ]:
        cond = trials[trials["first_stop_rewarded"] == is_first_stop_rewarded]
        for subject, sub in cond.groupby("subject_id"):
            sessions = sorted(sub["session_id"].unique())
            n = len(sessions)
            shades = [0.35 + 0.65 * (i / max(n - 1, 1)) for i in range(n)]
            norm = Normalize(vmin=0, vmax=max(n - 1, 1))
            cmap = plt.get_cmap("Purples")

            fig, ax = plt.subplots(figsize=(7, 4))
            for day, session_id in enumerate(sessions):
                rs = _appearance_table(
                    sub[sub["session_id"] == session_id], from_first_stop=True
                )
                rewarded = (
                    rs[rs["is_rewarded_odor"]]
                    .groupby("appearance")["has_choice"]
                    .mean()
                )
                non_rewarded = (
                    rs[~rs["is_rewarded_odor"]]
                    .groupby("appearance")["has_choice"]
                    .mean()
                )
                diff = rewarded.sub(non_rewarded, fill_value=np.nan).dropna()
                if diff.empty:
                    continue
                ax.plot(
                    diff.index,
                    diff.values,
                    marker="o",
                    color=cmap(shades[day]),
                )

            ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_xlabel("Appearance from first stop")
            ax.set_ylabel("ΔP(has_choice)\n[rewarded − non-rewarded]")
            ax.set_xticks(range(5))
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cb.set_label("Day")
            cb.set_ticks([])
            fig.suptitle(f"Subject {subject} — {condition}")
            fig.tight_layout()
            figures[(subject, condition)] = fig

    return figures


# ─── Counterfactual learning ──────────────────────────────────────────────────
#
# Idea: after the *first* stop of a block the animal has one piece of evidence
# about the current odor mapping. A purely factual learner only updates the odor
# it just sampled; a counterfactual learner also updates the odor it did *not*
# sample (``rewarded here`` implies ``not rewarded there``, and vice versa).
#
# Each block is therefore split by whether its first stop was rewarded, and for
# each split we score the animal's decision at the *next* encounter of each odor
# type. Perfect behaviour is P(stop | rewarded odor) = 1 and
# P(stop | non-rewarded odor) = 0 in *both* splits.

#: Column order of the counterfactual matrix. Each entry is
#: ``(first_stop_rewarded, next_odor_is_rewarded, short_label, ideal_p_stop)``.
COUNTERFACTUAL_CELLS = [
    (True, True, "1st stop REW\n→ next REW", 1.0),
    (True, False, "1st stop REW\n→ next NOREW", 0.0),
    (False, True, "1st stop NOREW\n→ next REW", 1.0),
    (False, False, "1st stop NOREW\n→ next NOREW", 0.0),
]

COUNTERFACTUAL_CELL_KEYS = [(a, b) for a, b, _, _ in COUNTERFACTUAL_CELLS]

#: How each plottable value is rendered. ``ideal`` is the target value in each of
#: the four :data:`COUNTERFACTUAL_CELLS` columns; ``cmap`` is sequential for the
#: polarity-corrected ``accuracy`` and diverging (red = stop-ish, blue =
#: leave-ish) for the raw probabilities, which have opposite targets per column.
_COUNTERFACTUAL_VALUE_STYLES = {
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


def _counterfactual_style(value: str) -> dict:
    """Validate ``value`` and return its entry in :data:`_COUNTERFACTUAL_VALUE_STYLES`."""
    try:
        return _COUNTERFACTUAL_VALUE_STYLES[value]
    except KeyError:
        raise ValueError(
            f"value must be one of {sorted(_COUNTERFACTUAL_VALUE_STYLES)}, got {value!r}"
        ) from None


def _require_counterfactual_value(matrix: pd.DataFrame, value: str) -> None:
    """Fail readably when ``matrix`` was built before ``value`` was a column.

    A matrix cached in a long-running kernel (or produced by an older version of
    this module) is missing newer value columns, which would otherwise surface as
    a bare ``KeyError`` from deep inside the plotting code.
    """
    if value not in matrix.columns:
        available = sorted(
            c for c in matrix.columns if c in _COUNTERFACTUAL_VALUE_STYLES
        )
        raise KeyError(
            f"{value!r} is not a column of the supplied matrix (it has {available}). "
            "Rebuild it with counterfactual_session_matrix(trials) -- a matrix held "
            "over from an earlier kernel state will not have the newer columns."
        )


def _counterfactual_text_color(v: float, cmap: str) -> str:
    """Black on the light part of ``cmap``, white elsewhere, for cell annotations."""
    light = v > 0.75 if cmap == "viridis" else 0.35 < v < 0.65
    return "black" if light else "white"


def counterfactual_block_table(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per block with the animal's decision at the next odor of each type.

    For every ``(session_id, block)`` the RewardSite trials are walked in
    temporal order and the first stop (``has_choice``) is located. That stop's
    ``has_reward`` defines ``first_stop_rewarded``. Strictly *after* that trial
    we then find the first rewarded-odor site and the first non-rewarded-odor
    site and record whether the animal stopped there.

    Returns a frame with one row per block and columns

    ``subject_id, session_id, block, first_stop_rewarded, first_stop_pos,
    stop_next_good, stop_next_bad``

    where the two ``stop_next_*`` columns are nullable booleans -- ``<NA>`` when
    the block ended before that odor type reappeared. Blocks where the animal
    never stopped are omitted entirely (no split can be assigned).

    Requires ``block`` (:func:`assign_blocks`) and ``is_rewarded_odor``.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    rs = rs.sort_values(["session_id", "block", "start_time"])

    records = []
    for (session_id, block), grp in rs.groupby(["session_id", "block"], sort=False):
        choice = grp["has_choice"].to_numpy(dtype=bool)
        if not choice.any():
            continue  # never stopped -> block is unlabelled
        first = int(np.argmax(choice))

        rewarded_odor = grp["is_rewarded_odor"].to_numpy(dtype=bool)
        post_choice = choice[first + 1 :]
        post_type = rewarded_odor[first + 1 :]

        def _first_stop_of_type(is_rewarded: bool):
            """``has_choice`` at the first post-first-stop site of this odor type."""
            hits = np.flatnonzero(post_type == is_rewarded)
            if hits.size == 0:
                return pd.NA
            return bool(post_choice[hits[0]])

        records.append(
            {
                "subject_id": str(session_id).split("_")[0],
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


def counterfactual_session_matrix(
    trials: pd.DataFrame, min_blocks: int = 3
) -> pd.DataFrame:
    """Per-session P(stop) for the four counterfactual conditions.

    Aggregates :func:`counterfactual_block_table` over blocks within a session.
    Returns a long frame with columns ``subject_id, session_id, session_date,
    session_index, first_stop_rewarded, next_rewarded, p_stop, p_leave,
    n_blocks, accuracy`` where

    * ``p_stop`` is the fraction of blocks in that condition where the animal
      stopped at the next site of that odor type, and ``p_leave`` is its
      complement (ran through without stopping),
    * ``accuracy`` is ``p_stop`` for rewarded odors and ``1 - p_stop`` for
      non-rewarded ones, so all four conditions share a *higher-is-better*
      polarity and can be compared column-to-column,
    * ``session_index`` is the animal's 0-based chronological session number.

    Cells backed by fewer than ``min_blocks`` blocks are returned with
    ``p_stop = NaN`` (kept as rows so the heatmap keeps a stable 4-column grid).
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

    # Reindex onto the full (session x 4 conditions) grid so missing cells show
    # up as gaps in the heatmap instead of silently shifting columns.
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
    # cast off the nullable dtype inherited from the boolean mean so downstream
    # numpy/matplotlib code sees plain float NaN rather than pd.NA
    agg["p_stop"] = agg["p_stop"].astype(float)
    agg.loc[agg["n_blocks"] < min_blocks, "p_stop"] = np.nan

    # P(leave) is the exact complement: a site the animal did not stop at is one
    # it ran through. Ideal is 0 at a rewarded odor and 1 at a non-rewarded one.
    agg["p_leave"] = 1.0 - agg["p_stop"]
    agg["accuracy"] = np.where(agg["next_rewarded"], agg["p_stop"], 1.0 - agg["p_stop"])

    # session_id encodes the datetime -> lexicographic sort is chronological
    agg["session_date"] = agg["session_id"].str.split("_").str[1]
    agg = agg.sort_values(["subject_id", "session_id"])
    agg["session_index"] = agg.groupby("subject_id")["session_id"].transform(
        lambda s: s.rank(method="dense").astype(int) - 1
    )
    return agg


def _counterfactual_pivot(matrix: pd.DataFrame, subject: str, value: str):
    """``(n_sessions, 4)`` arrays of ``value`` and block counts for one subject."""
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
    trials: pd.DataFrame,
    value: str = "p_leave",
    min_blocks: int = 3,
    annotate: bool = True,
    align_rows: bool = True,
    matrix: pd.DataFrame | None = None,
):
    """Sessions x 4-condition heatmap of counterfactual behaviour, one panel per animal.

    Rows are the animal's sessions (chronological, top = first), columns are the
    four ``(first stop rewarded?) x (next odor rewarded?)`` conditions.

    ``value`` selects what is coloured:

    * ``"p_leave"`` (default) -- the probability of running the next site through
      without stopping, on a diverging colormap centred at 0.5. Ideal is 0 in the
      next-REW columns and 1 in the next-NOREW columns, i.e. blue/red alternating
      across the pairs.
    * ``"p_stop"`` -- the complement, ideal 1 / 0 / 1 / 0.
    * ``"accuracy"`` -- ``p_stop`` in the next-REW columns and ``p_leave`` in the
      next-NOREW ones, so higher is better everywhere on a sequential colormap. A general
      rule being acquired shows up as *all four* columns brightening together,
      condition-specific learning as columns brightening at different rates.

    ``annotate`` writes the value and the backing block count into each cell.
    ``align_rows`` pads every panel out to the animal with the most sessions so
    a row is the same height in all panels (comparable at the cost of trailing
    whitespace); set it False for compact per-animal panels. Pass a precomputed
    ``matrix`` from :func:`counterfactual_session_matrix` to avoid recomputing.
    Returns ``(fig, matrix)``.
    """
    from matplotlib import pyplot as plt

    style = _counterfactual_style(value)
    if matrix is None:
        matrix = counterfactual_session_matrix(trials, min_blocks=min_blocks)
    _require_counterfactual_value(matrix, value)

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
        grid, sessions, counts = _counterfactual_pivot(matrix, subject, value)
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
        ax.set_yticklabels([s.split("_")[1] for s in sessions], fontsize=6)
        if align_rows:
            # pad every panel out to the longest animal so one row = one session
            # at the same height everywhere, making the panels comparable by eye
            ax.set_ylim(max_rows - 0.5, -0.5)
        ax.axvline(1.5, color="white", lw=2.5)  # separate the two first-stop splits
        ax.set_title(f"Subject {subject}", fontsize=10)

        if annotate:
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    v = grid[r, c]
                    if np.isnan(v):
                        ax.text(
                            c, r, "·", ha="center", va="center", color="gray", fontsize=8
                        )
                        continue
                    ax.text(
                        c,
                        r,
                        f"{v:.2f}\nn{counts[r, c]}",
                        ha="center",
                        va="center",
                        fontsize=4.5,
                        color=_counterfactual_text_color(v, cmap),
                    )

    axes[0][0].set_ylabel("Session (chronological)")
    cb = fig.colorbar(images[0], ax=axes[0], fraction=0.02, pad=0.02)
    cb.set_label(style["label"])
    ideal = "  —  ideal: " + " / ".join(f"{i:g}" for i in style["ideal"])
    fig.suptitle(
        "Counterfactual learning: decision at the next odor of each type,\n"
        "split by whether the block's first stop was rewarded" + ideal,
        fontsize=11,
    )
    return fig, matrix


#: Per-animal series colours for the four counterfactual conditions.
COUNTERFACTUAL_COLORS = ["#c0392b", "#e07b39", "#1a5276", "#4f8fc0"]


def counterfactual_cohort_average(
    matrix: pd.DataFrame, value: str = "accuracy", min_animals: int = 1
) -> pd.DataFrame:
    """Average :func:`counterfactual_session_matrix` across mice per session number.

    Sessions are aligned by ``session_index`` -- each animal's own 0-based
    chronological session number -- and averaged across animals, so row ``k`` is
    "the cohort's k-th session". Each animal contributes at most one value per
    cell, so no animal is weighted by how many blocks it ran.

    Returns one row per ``(session_index, first_stop_rewarded, next_rewarded)``
    with ``mean``, ``sem`` and ``std`` across animals, ``n_animals``, and
    ``n_blocks`` (total blocks behind that cell). ``min_animals`` drops session
    indices backed by fewer than that many animals -- note the tail is thin,
    since only the longest-running animal reaches the highest indices.
    """
    _require_counterfactual_value(matrix, value)
    agg = (
        matrix.dropna(subset=[value])
        .groupby(["session_index", "first_stop_rewarded", "next_rewarded"])
        .agg(
            mean=(value, "mean"),
            sem=(value, "sem"),
            std=(value, "std"),
            n_animals=(value, "count"),
            n_blocks=("n_blocks", "sum"),
        )
        .reset_index()
    )
    agg["sem"] = agg["sem"].fillna(0.0)
    return agg[agg["n_animals"] >= min_animals].reset_index(drop=True)


def plot_counterfactual_cohort_average(
    matrix: pd.DataFrame,
    value: str = "p_leave",
    min_animals: int = 1,
    annotate: bool = True,
):
    """Cross-mouse counterfactual matrix, laid out with session number on the x axis.

    Two stacked panels sharing that x axis:

    * top -- ``4 conditions x session number`` heatmap, i.e.
      :func:`plot_counterfactual_heatmap` collapsed across animals and turned on
      its side so each condition reads left-to-right as one strip.
    * bottom -- the same four rows as timecourses (mean +- SEM across animals),
      which is where "do the conditions move together" is easiest to judge.

    ``value`` is ``"p_leave"`` (default, ideal 0 / 1 / 0 / 1 on a diverging
    colormap), ``"p_stop"`` (its complement) or ``"accuracy"`` (polarity
    corrected, higher = better everywhere, sequential colormap). Returns
    ``(fig, cohort)`` where ``cohort`` is the frame from
    :func:`counterfactual_cohort_average`.
    """
    from matplotlib import pyplot as plt

    style = _counterfactual_style(value)

    cohort = counterfactual_cohort_average(
        matrix, value=value, min_animals=min_animals
    )
    labels = [label for _, _, label, _ in COUNTERFACTUAL_CELLS]

    # dense contiguous session axis so the heatmap's extent lines up exactly with
    # the line plot below it (a skipped index would otherwise shift the strips)
    lo = int(cohort["session_index"].min())
    hi = int(cohort["session_index"].max())
    session_indices = list(range(lo, hi + 1))
    y = np.arange(len(COUNTERFACTUAL_CELL_KEYS))

    keys = ["session_index", "first_stop_rewarded", "next_rewarded"]
    means = cohort.set_index(keys)["mean"]
    sems = cohort.set_index(keys)["sem"]
    n_animals = cohort.set_index(keys)["n_animals"]

    def _strip(series):
        """``(4, n_sessions)`` array: one row per condition, one column per session."""
        return np.array(
            [
                [series.get((s, *key), np.nan) for s in session_indices]
                for key in COUNTERFACTUAL_CELL_KEYS
            ],
            dtype=float,
        )

    grid = _strip(means)
    grid_sem = _strip(sems)
    grid_n = _strip(n_animals)

    fig, (ax_hm, ax_ln) = plt.subplots(
        2,
        1,
        figsize=(0.46 * len(session_indices) + 4.5, 8.5),
        gridspec_kw={"height_ratios": [1, 1.6]},
        sharex=True,
        layout="constrained",
    )

    # ── top: conditions x session number heatmap ──────────────────────────────
    # extent puts cell centres on integer session numbers so the panel below,
    # which shares this x axis, lines up column-for-column with the strips
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
                    color=_counterfactual_text_color(v, style["cmap"]),
                )
    fig.colorbar(im, ax=(ax_hm, ax_ln), fraction=0.02, pad=0.01).set_label(
        style["label"], fontsize=9
    )

    # ── bottom: the same four rows as cohort timecourses ───────────────────────
    xs = np.asarray(session_indices, dtype=float)
    for r, ((_, _, label, _), color, ideal) in enumerate(
        zip(COUNTERFACTUAL_CELLS, COUNTERFACTUAL_COLORS, style["ideal"])
    ):
        ok = ~np.isnan(grid[r])
        ax_ln.errorbar(
            xs[ok],
            grid[r][ok],
            yerr=grid_sem[r][ok],
            marker="o",
            ms=5,
            lw=1.8,
            capsize=3,
            color=color,
            # the target differs per row for the raw probabilities, so name it
            label=f"{label.replace(chr(10), ' ')}  (ideal {ideal:g})",
        )

    # shade where fewer than half the animals still contribute, on both panels
    n_per_session = np.nanmax(grid_n, axis=0)
    thin = xs[n_per_session < np.nanmax(n_per_session) / 2]
    if thin.size:
        for ax, kwargs in ((ax_hm, {}), (ax_ln, {"label": "< half the cohort"})):
            ax.axvspan(
                thin.min() - 0.5, hi + 0.5, color="gray", alpha=0.12, zorder=0, **kwargs
            )
        ax_hm.axvline(thin.min() - 0.5, color="black", lw=1.2, alpha=0.6)
        ax_ln.axvline(thin.min() - 0.5, color="black", lw=1.2, alpha=0.6)

    ax_ln.axhline(0.5, color="gray", ls=":", lw=1)
    ax_ln.set_ylim(-0.03, 1.05)
    ax_ln.set_xlim(lo - 0.5, hi + 0.5)
    ax_ln.xaxis.get_major_locator().set_params(integer=True)
    ax_ln.set_xlabel("Session number (aligned across mice)")
    ax_ln.set_ylabel(style["label"])
    ax_ln.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)

    # animal count per session number along the top
    ax_top = ax_hm.secondary_xaxis("top")
    ax_top.set_xticks(session_indices)
    ax_top.set_xticklabels(
        [str(int(n)) if np.isfinite(n) else "" for n in n_per_session],
        fontsize=5.5,
    )
    ax_top.set_xlabel("animals contributing", fontsize=8)

    fig.suptitle(
        "Counterfactual learning, averaged across mice at the same session number\n"
        "(error bars = SEM across animals; n per session falls off as animals run out)",
        fontsize=11,
    )
    return fig, cohort


def counterfactual_session_trends(
    matrix: pd.DataFrame, value: str = "accuracy"
) -> pd.DataFrame:
    """Per-subject, per-condition OLS slope of ``value`` against session index.

    A candidate summary metric: if the animal is acquiring a *general* rule the
    four slopes should be similar; if it learns each condition separately the
    slopes diverge -- in particular the counterfactual cells, where the odor at
    ``t+1`` is *not* the one sampled at the first stop, would lag behind.

    Returns one row per ``(subject_id, first_stop_rewarded, next_rewarded)``
    with ``slope`` (units of ``value`` per session), ``intercept``, ``r``, and
    ``n_sessions``. Conditions with fewer than 3 usable sessions are skipped.
    """
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

