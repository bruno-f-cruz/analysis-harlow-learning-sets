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

from analysis.features import block_window_index, expand_to_block_windows
from analysis.plotting import TWO_BY_TWO_COLORS, bootstrap_mean_ci

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
        raise ValueError(f"value must be one of {sorted(_VALUE_STYLES)}, got {value!r}") from None


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
    subject_map = rs.drop_duplicates("session_id").set_index("session_id")["subject_id"].to_dict()

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


def counterfactual_session_matrix(trials: pd.DataFrame, min_blocks: int = 3) -> pd.DataFrame:
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
        long.groupby(["subject_id", "session_id", "first_stop_rewarded", "next_rewarded"])[
            "stopped"
        ]
        .agg(p_stop="mean", n_blocks="count")
        .reset_index()
    )

    # Reindex onto the full (session x 4 conditions) grid so missing cells
    # show up as gaps in the heatmap instead of silently shifting columns.
    sessions = agg[["subject_id", "session_id"]].drop_duplicates()
    grid = sessions.merge(
        pd.DataFrame(COUNTERFACTUAL_CELL_KEYS, columns=["first_stop_rewarded", "next_rewarded"]),
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
        session_id=expanded["subject_id"].astype(str) + "_w" + expanded["window"].map("{:04d}".format),
        block=expanded["block_ordinal"],
    )
    matrix = counterfactual_session_matrix(rekeyed, min_blocks=min_blocks)
    matrix["window"] = matrix["session_id"].str.rsplit("_w", n=1).str[1].astype(int)
    bounds = expanded.drop_duplicates("window")[["window", "window_start", "window_end"]]
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
    max_rows = max(matrix[matrix["subject_id"] == s]["session_id"].nunique() for s in subjects)
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
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        )
        ax.grid(False)
        ax.set_xticks(range(len(COUNTERFACTUAL_CELLS)))
        ax.set_xticklabels(
            [label for _, _, label, _ in COUNTERFACTUAL_CELLS], rotation=45, ha="right", fontsize=7
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
                        ax.text(c, r, "·", ha="center", va="center", color="gray", fontsize=8)
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


def counterfactual_cohort_average(matrix, value="accuracy", min_animals=1, rng=None, n_boot=2000):
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
        columns=["session_index", "first_stop_rewarded", "next_rewarded", "mean", "ci_lo", "ci_hi", "n_animals", "n_blocks"],
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
            [[series.get((s, *key), np.nan) for s in session_indices] for key in COUNTERFACTUAL_CELL_KEYS],
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
                    s, r, f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color=_text_color(v, style["cmap"]),
                )
    fig.colorbar(im, ax=(ax_hm, ax_ln), fraction=0.02, pad=0.01).set_label(style["label"], fontsize=9)

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
            ax.axvspan(x0, x1, color="gray", alpha=0.12, zorder=0, label=legend_label if i == 0 else None)
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
    ax_top.set_xticklabels([str(int(n)) if np.isfinite(n) else "" for n in n_per_session], fontsize=5.5)
    ax_top.set_xlabel(
        "animals contributing (max over the 4 conditions; a cell below "
        "min_blocks drops out, so this can dip and recover)",
        fontsize=8,
    )

    fig.suptitle(title, fontsize=11)
    return fig, cohort, ax_ln


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
        1, len(COUNTERFACTUAL_CELLS), figsize=(6 * len(COUNTERFACTUAL_CELLS), 4),
        sharey=True, sharex=True, squeeze=False,
    )
    for ax, (fsr, nr, label, _), color, ideal in zip(
        axes[0], COUNTERFACTUAL_CELLS, COUNTERFACTUAL_COLORS, style["ideal"]
    ):
        cond = matrix[(matrix["first_stop_rewarded"] == fsr) & (matrix["next_rewarded"] == nr)]
        for subject in subjects:
            sub = cond[cond["subject_id"] == subject].dropna(subset=[value]).sort_values("session_index")
            ax.plot(
                sub["session_index"], sub[value], color=subject_colors[subject],
                linewidth=1, alpha=0.4, label=f"Subject {subject}",
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
    for (subject, fsr, nr), grp in matrix.groupby(["subject_id", "first_stop_rewarded", "next_rewarded"]):
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


def plot_counterfactual_trends_per_animal(cf_window: pd.DataFrame, window_blocks: int, skip_blocks: int):
    """Per-subject accuracy timecourse across block windows, with an OLS trend line.

    ``cf_window``'s ``session_index`` already equals its window number (see
    ``counterfactual_window_matrix``), but ``window`` is used directly here
    for clarity.
    """
    from matplotlib.ticker import MaxNLocator

    subjects = sorted(cf_window["subject_id"].unique())
    fig, axes = plt.subplots(1, len(subjects), figsize=(4.2 * len(subjects), 4), sharey=True, squeeze=False)
    for ax, subject in zip(axes[0], subjects):
        sub = cf_window[cf_window["subject_id"] == subject]
        for (fsr, nr, label, _), color in zip(COUNTERFACTUAL_CELLS, COUNTERFACTUAL_COLORS):
            g = (
                sub[(sub["first_stop_rewarded"] == fsr) & (sub["next_rewarded"] == nr)]
                .dropna(subset=["accuracy"])
                .sort_values("window")
            )
            ax.plot(g["window"], g["accuracy"], marker="o", ms=4, color=color, label=label.replace("\n", " "))
            if len(g) >= 3:
                x = g["window"].to_numpy(dtype=float)
                m, b = np.polyfit(x, g["accuracy"].to_numpy(dtype=float), 1)
                ax.plot(x, m * x + b, color=color, ls="--", lw=1.2, alpha=0.7)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(f"Block-window number ({window_blocks} blocks, stride {skip_blocks})")
        ax.set_title(f"Subject {subject}", fontsize=10)
    axes[0][0].set_ylabel("accuracy (higher = better)")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Counterfactual accuracy across block windows (dashed = OLS fit)", fontsize=11)
    fig.tight_layout()
    return fig


def first_site_chance_by_window(trials: pd.DataFrame, window_blocks: int, skip_blocks: int) -> pd.DataFrame:
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
    first_site = rs[rs["_block_pos"] == 0][["session_id", "block", "has_choice"]].rename(
        columns={"has_choice": "stopped_first_site"}
    )

    windows_map = block_window_index(trials, window_blocks, skip_blocks)
    blocks = windows_map.merge(first_site, on=["session_id", "block"], how="inner")
    return blocks.groupby(["subject_id", "window"])["stopped_first_site"].mean().reset_index()
