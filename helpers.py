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
    trials: pd.DataFrame, start_frac: float = 0.05, end_frac: float = 0.20
) -> pd.DataFrame:
    """Drop the first ``start_frac`` and last ``end_frac`` of each session's rows.

    Within each session, rows are ordered chronologically (by ``start_time``) and
    the leading ``start_frac`` and trailing ``end_frac`` fractions of rows are
    removed, keeping the middle portion. Returns a new trimmed DataFrame.
    """
    kept = []
    for _, grp in trials.groupby("session_id", sort=False):
        g = grp.sort_values("start_time")
        n = len(g)
        lo = int(round(n * start_frac))
        hi = n - int(round(n * end_frac))
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
    trials: pd.DataFrame, title_suffix: str = "", from_first_stop: bool = False, ax=None
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
    """
    import matplotlib.pyplot as plt

    if ax is not None:
        plot_choice_by_block_position(trials, ax=ax, from_first_stop=from_first_stop)
        if title_suffix:
            ax.set_title(title_suffix.lstrip(" —"))
        return ax

    trials = trials.copy()
    trials["subject_id"] = trials["session_id"].str.split("_").str[0]

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


def plot_choice_by_block_position_by_first_stop(trials: pd.DataFrame, ax=None):
    """:func:`plot_choice_by_block_position_per_session`, split by first-stop outcome.

    Labels each block by whether its first stop was rewarded
    (:func:`label_first_stop`), then produces the full per-animal/per-session
    figure set twice: once for first-stop-rewarded blocks and once for
    first-stop-non-rewarded blocks. Returns a ``{(subject_id, condition): figure}``
    dict.

    If ``ax`` is given, every split (subject, session, and condition) is dropped:
    all blocks that contain a stop are pooled and drawn (aligned to the first
    stop) into that single axes, which is returned.
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
            subset, title_suffix=f" — {condition}", from_first_stop=True
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
