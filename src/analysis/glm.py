"""1-back within-block history GLM: feature construction, per-group fitting, and
its reward-cell timecourse plot."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from analysis.plotting import TWO_BY_TWO_COLORS, bootstrap_mean_ci
from analysis.plotting_style import animal_palette

#: Column order of the fitted history GLM. The two IsPrevChoice_* terms are
#: signed (+1 stopped / -1 ran through / 0 when the previous odor was of the
#: other kind); the four H_* terms are the one-hot (is_same_odor x
#: is_prev_rewarded) reward cells.
HISTORY_GLM_COEFS = [
    "IsPrevChoice_SameOdor",
    "IsPrevChoice_OtherOdor",
    "H_Same_Rew",
    "H_Same_NoRew",
    "H_Other_Rew",
    "H_Other_NoRew",
]

#: Display style for each HISTORY_GLM_COEFS term, split into the one-hot
#: reward cells and the choice-persistence bias terms so callers can group them.
#: Colors are TWO_BY_TWO_COLORS in (Same,Rew)/(Same,NoRew)/(Other,Rew)/
#: (Other,NoRew) order -- the same red/orange/blue/cyan convention the
#: counterfactual and bias plots use for their own two-boolean splits.
REWARD_CELL_STYLE = {
    "H_Same_Rew": {"label": "Same × Rew", "color": TWO_BY_TWO_COLORS[0]},
    "H_Same_NoRew": {"label": "Same × NoRew", "color": TWO_BY_TWO_COLORS[1]},
    "H_Other_Rew": {"label": "Other × Rew", "color": TWO_BY_TWO_COLORS[2]},
    "H_Other_NoRew": {"label": "Other × NoRew", "color": TWO_BY_TWO_COLORS[3]},
}
#: One-hot (is_same_odor x previous-trial outcome) cells, where the outcome is
#: stopped+rewarded / stopped+not rewarded / ran through ("Skip"). Exactly one
#: cell is 1 on every trial, so the six columns are orthogonal and (with no
#: intercept) fit one independent log-odds per cell: sigmoid(weight) is that
#: cell's empirical P(Choice). Row-major in a 2x3 grid = Same row, Other row.
ONEHOT_CELL_STYLE = {
    "Same_Rew": {"label": "Same × Rew", "color": TWO_BY_TWO_COLORS[0]},
    "Same_NoRew": {"label": "Same × NoRew", "color": TWO_BY_TWO_COLORS[1]},
    "Same_Skip": {"label": "Same × Skip", "color": "#2ca02c"},
    "Other_Rew": {"label": "Other × Rew", "color": TWO_BY_TWO_COLORS[2]},
    "Other_NoRew": {"label": "Other × NoRew", "color": TWO_BY_TWO_COLORS[3]},
    "Other_Skip": {"label": "Other × Skip", "color": "#9467bd"},
}
BIAS_TERM_STYLE = {
    "IsPrevChoice_SameOdor": {"label": "Prev choice (same odor)", "color": "#2ca02c"},
    "IsPrevChoice_OtherOdor": {"label": "Prev choice (other odor)", "color": "#9467bd"},
}


def history_glm_features(trials: pd.DataFrame) -> pd.DataFrame:
    """RewardSite trials with the 1-back within-block history regressors.

    The previous trial's odor, choice and reward are taken with a shift(1)
    *within* each real (session_id, block), so history never crosses a block
    or a session boundary and the first trial of every block drops out. Do
    this before any windowing -- a window groups blocks for fitting, it's
    never a boundary the 1-back history should see.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    rs = rs.sort_values(["session_id", "block", "start_time"])
    grp = rs.groupby(["session_id", "block"], sort=False)
    rs["prev_odor_index"] = grp["odor_index"].shift(1)
    rs["prev_has_choice"] = grp["has_choice"].shift(1)
    rs["prev_has_reward"] = grp["has_reward"].shift(1)
    rs = rs.dropna(subset=["prev_odor_index", "prev_has_choice", "prev_has_reward"])

    is_same = (rs["odor_index"] == rs["prev_odor_index"]).to_numpy()
    prev_choice = rs["prev_has_choice"].astype(bool).to_numpy()
    prev_rewarded = rs["prev_has_reward"].astype(bool).to_numpy()

    rs["IsPrevChoice_SameOdor"] = np.where(
        is_same, np.where(prev_choice, 1.0, -1.0), 0.0
    )
    rs["IsPrevChoice_OtherOdor"] = np.where(
        ~is_same, np.where(prev_choice, 1.0, -1.0), 0.0
    )
    rs["H_Same_Rew"] = (is_same & prev_rewarded).astype(float)
    rs["H_Same_NoRew"] = (is_same & ~prev_rewarded).astype(float)
    rs["H_Other_Rew"] = (~is_same & prev_rewarded).astype(float)
    rs["H_Other_NoRew"] = (~is_same & ~prev_rewarded).astype(float)
    # One-hot design: outcome of the previous trial split into rewarded /
    # stopped-not-rewarded / skipped (ran through), per same/other odor.
    prev_skip = ~prev_choice
    prev_norew = prev_choice & ~prev_rewarded
    for odor, mask in (("Same", is_same), ("Other", ~is_same)):
        rs[f"{odor}_Rew"] = (mask & prev_rewarded).astype(float)
        rs[f"{odor}_NoRew"] = (mask & prev_norew).astype(float)
        rs[f"{odor}_Skip"] = (mask & prev_skip).astype(float)
    rs["choice"] = rs["has_choice"].astype(int)
    return rs


def history_glm_after_first_stop(features: pd.DataFrame) -> pd.DataFrame:
    """One row per block: the trial right after the block's first stop.

    `features` is :func:`history_glm_features`'s output, ideally already
    expanded to block windows (:func:`analysis.features.expand_to_block_windows`)
    *before* this filter so windows are counted over all of an animal's
    blocks rather than only the ones that survive it (an overlapping-window
    block is kept once per window). That frame's
    ``prev_has_choice`` is False for every row before the first stop, so the
    first row per (session_id, block) with it True is exactly the trial that
    follows the first stop -- whose 1-back history is that stop. Blocks with
    no stop, or where the stop is the block's last trial, contribute nothing,
    so each block contributes at most one trial.

    Because the previous trial is a stop by construction, the two ``*_Skip``
    one-hot cells are always 0 here (empty columns) -- fit only the four
    ``Same_/Other_ x Rew/NoRew`` cells of :data:`ONEHOT_CELL_STYLE`.
    """
    hit = features[features["prev_has_choice"].astype(bool)]
    keys = ["session_id", "block"] + (["window"] if "window" in hit.columns else [])
    return hit.groupby(keys, sort=False).head(1)


def fit_history_glm(
    features: pd.DataFrame,
    unit_col: str | list[str] = "session_id",
    min_trials: int = 10,
    cv_folds: int = 5,
    coefs: list[str] | None = None,
) -> pd.DataFrame:
    """Fit the history GLM independently within each ``unit_col`` group.

    Unregularised logistic regression, no intercept. Groups with fewer than
    ``min_trials`` trials or no variance in ``choice`` are skipped, as is any
    group whose fit raises.

    Each coefficient gets a Wald 95% CI (``se``, ``ci_lo``, ``ci_hi``) from the
    inverse Fisher information at the MLE. ``cv_accuracy`` is the mean
    held-out accuracy of a stratified k-fold refit of the same group -- a
    sanity check that the unregularised fit isn't just memorising noise.

    ``coefs`` (default: all of :data:`HISTORY_GLM_COEFS`) selects which
    regressor columns to fit -- e.g. just the four one-hot reward cells when
    the choice-persistence terms are collinear with them.
    """
    coefs = list(HISTORY_GLM_COEFS) if coefs is None else list(coefs)
    unit_cols = [unit_col] if isinstance(unit_col, str) else list(unit_col)
    records = []
    for unit, sdf in features.groupby(unit_cols, sort=True):
        key = dict(zip(unit_cols, unit if isinstance(unit, tuple) else (unit,)))
        if len(sdf) < min_trials or sdf["choice"].nunique() < 2:
            continue
        X = sdf[coefs].to_numpy(dtype=float)
        y = sdf["choice"].to_numpy(dtype=int)
        try:
            clf = LogisticRegression(
                C=np.inf, solver="lbfgs", fit_intercept=False, max_iter=500
            )
            clf.fit(X, y)
        except Exception as exc:  # a single unusable group must not kill the sweep
            print(f"{key} failed: {exc}")
            continue

        # Wald CI: cov = (X^T W X)^-1 with W = diag(p(1-p)) at the MLE.
        p = clf.predict_proba(X)[:, 1]
        w = np.clip(p * (1 - p), 1e-6, None)
        info = (X * w[:, None]).T @ X
        try:
            se = np.sqrt(np.clip(np.diag(np.linalg.inv(info)), 0, None))
        except np.linalg.LinAlgError:
            se = np.full(len(coefs), np.nan)

        # Manual fold loop (not cross_val_score): this runs per group, and
        # cross_val_score's joblib dispatch overhead dominates at that scale
        # for a model this cheap to fit.
        cv_accuracy = np.nan
        n_splits = min(cv_folds, int(sdf["choice"].value_counts().min()))
        if n_splits >= 2:
            try:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
                fold_accuracies = []
                for train_idx, test_idx in skf.split(X, y):
                    fold_clf = LogisticRegression(
                        C=np.inf, solver="lbfgs", fit_intercept=False, max_iter=500
                    )
                    fold_clf.fit(X[train_idx], y[train_idx])
                    fold_accuracies.append(fold_clf.score(X[test_idx], y[test_idx]))
                cv_accuracy = float(np.mean(fold_accuracies))
            except Exception:
                pass  # degenerate fold -- leave as NaN rather than kill the sweep

        for i, name in enumerate(coefs):
            val = clf.coef_[0][i]
            records.append(
                {
                    **key,
                    "subject_id": sdf["subject_id"].iloc[0],
                    "coef": name,
                    "value": val,
                    "se": se[i],
                    "ci_lo": val - 1.96 * se[i],
                    "ci_hi": val + 1.96 * se[i],
                    "n_trials": len(sdf),
                    "cv_accuracy": cv_accuracy,
                }
            )
    return pd.DataFrame(records)


def plot_history_glm_reward_cells_by_window(
    coefs_window: pd.DataFrame,
    window_bounds: pd.DataFrame,
    window_blocks: int,
    min_animals: int = 2,
    n_boot: int = 2000,
    seed: int = 0,
):
    """One subplot per HISTORY_GLM_COEFS term: every animal's timecourse faint
    and unadorned, overlaid with the bootstrapped cohort mean +/- 95% CI
    *across animals* (not the per-animal Wald CI the fit itself produces).

    The shared y-axis is sized off the 90th percentile of the cohort CI
    bounds, not their max -- a single near-separated per-animal fit can send
    one window's CI far out on its own, and scaling off that would squash
    the rest. That kind of outlier window is instead drawn truncated (▲/▼).
    """
    all_terms = {**REWARD_CELL_STYLE, **BIAS_TERM_STYLE}
    terms = list(all_terms)
    subjects = sorted(coefs_window["subject_id"].unique())
    subject_colors = dict(zip(subjects, plt.cm.tab10.colors))
    rng = np.random.default_rng(seed)

    windows = sorted(coefs_window["window"].unique())
    x = np.array(windows, dtype=float)
    labels = [
        f"{int(window_bounds.loc[w, 'window_start'])}-{int(window_bounds.loc[w, 'window_end'])}"
        for w in windows
    ]

    cohort = {}
    for term in terms:
        cond = coefs_window[coefs_window["coef"] == term]
        by_window = (
            cond.groupby("window")["value"]
            .apply(lambda s: s.to_numpy(dtype=float))
            .reindex(windows)
        )
        mean = np.full(len(windows), np.nan)
        ci_lo = np.full(len(windows), np.nan)
        ci_hi = np.full(len(windows), np.nan)
        for i, vals_w in enumerate(by_window):
            vals_w = vals_w if isinstance(vals_w, np.ndarray) else np.array([])
            mean[i], ci_lo[i], ci_hi[i] = bootstrap_mean_ci(
                vals_w, rng, min_n=min_animals, n_boot=n_boot
            )
        cohort[term] = (mean, ci_lo, ci_hi)

    ci_bounds = np.concatenate(
        [np.concatenate([lo, hi]) for _, lo, hi in cohort.values()]
    )
    ci_bounds = ci_bounds[~np.isnan(ci_bounds)]
    val_max = np.nanpercentile(np.abs(ci_bounds), 90) if ci_bounds.size else 1.0
    ylim = max(val_max * 1.3, 1.0)

    fig, axes = plt.subplots(
        1,
        len(terms),
        figsize=(6 * len(terms), 4),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    for ai, (ax, term) in enumerate(zip(axes[0], terms)):
        style = all_terms[term]
        cond = coefs_window[coefs_window["coef"] == term]

        for subject in subjects:
            sub = cond[cond["subject_id"] == subject].sort_values("window")
            ax.plot(
                sub["window"],
                sub["value"],
                color=subject_colors[subject],
                linewidth=1,
                alpha=0.4,
                label=f"Subject {subject}",
            )

        mean, ci_lo, ci_hi = cohort[term]
        ok = ~np.isnan(mean)

        # Clip to ylim so the shaded band never fights the axis (matplotlib
        # would otherwise scale to it via autoscale/fill_between's default).
        mean_plot = np.clip(mean, -ylim, ylim)
        lo = np.clip(ci_lo, -ylim, ylim)
        hi = np.clip(ci_hi, -ylim, ylim)
        ax.plot(
            x[ok],
            mean_plot[ok],
            linewidth=2.2,
            color=style["color"],
            label="Cohort mean, bootstrapped 95% CI",
        )
        ax.fill_between(
            x[ok], lo[ok], hi[ok], color=style["color"], alpha=0.25, linewidth=0
        )
        clip_hi = ok & (ci_hi > ylim)
        clip_lo = ok & (ci_lo < -ylim)
        if np.any(clip_hi):
            ax.plot(
                x[clip_hi],
                np.full(clip_hi.sum(), ylim),
                marker="^",
                linestyle="none",
                color=style["color"],
                markersize=6,
                clip_on=False,
            )
        if np.any(clip_lo):
            ax.plot(
                x[clip_lo],
                np.full(clip_lo.sum(), -ylim),
                marker="v",
                linestyle="none",
                color=style["color"],
                markersize=6,
                clip_on=False,
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Block window (pooled block range)")
        ax.set_title(style["label"])
        if ai == len(REWARD_CELL_STYLE):
            # separates the one-hot reward cells (left) from the
            # choice-persistence bias terms (right)
            ax.spines["left"].set_linewidth(2.0)

    axes[0][0].set_ylabel("GLM coefficient")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(
        f"History GLM coefficients timecourse over {window_blocks}-block windows\n"
        "(faint = individual animals; bold line + shaded band = cohort mean and "
        f"bootstrapped 95% CI across animals, shown only where ≥{min_animals} animals "
        "contribute; ▲/▼ = CI truncated at axis limit)"
    )
    fig.tight_layout()
    return fig


def _history_glm_cohort_grid(
    work: pd.DataFrame,
    window_bounds: pd.DataFrame,
    terms_style: dict,
    ylabel: str,
    ref: float,
    fixed_ylim: tuple[float, float] | None,
    clip_percentile: bool,
    min_animals: int,
    n_boot: int,
    seed: int,
    animal_colors: dict | None,
    clip_pct: float = 90,
):
    """Shared one-subplot-per-term (2 rows) grid: faint per-animal
    timecourses + bold cohort mean/CI across animals, bootstrapped from
    `work`'s already-computed per-(subject_id, window, coef) ``metric``
    column. Used by both :func:`plot_history_glm_cohort_by_window` (GLM
    weights/model-implied probability) and
    :func:`plot_history_glm_empirical_by_window` (raw empirical P(choice))
    so the two share one grid/bootstrap/axis implementation and can't drift
    apart from each other.

    `clip_percentile` reproduces `plot_history_glm_reward_cells_by_window`'s
    fix for unbounded raw log-odds weights (scale off the 90th percentile of
    the cohort CI bounds, mark genuinely truncated ones with a clipped
    triangle) -- pass False for anything already bounded (a probability or a
    probability difference), where a single animal's fit can't blow up the
    same way and `fixed_ylim` (or plain autoscale) is enough.
    """
    from matplotlib.ticker import MaxNLocator

    terms = list(terms_style)
    subjects = sorted(work["subject_id"].unique())
    animal_colors = animal_colors if animal_colors is not None else animal_palette(subjects)
    rng = np.random.default_rng(seed)

    windows = sorted(work["window"].unique())
    # numeric "block number" x-position (each window's start, 1-based) --
    # a text label per window is illegible once there are dozens of them
    window_x = {w: int(window_bounds.loc[w, "window_start"]) + 1 for w in windows}
    work = work.copy()
    work["_x"] = work["window"].map(window_x)
    x = np.array([window_x[w] for w in windows], dtype=float)

    cohort = {}
    for term in terms:
        cond = work[work["coef"] == term]
        by_window = (
            cond.groupby("window")["metric"]
            .apply(lambda s: s.to_numpy(dtype=float))
            .reindex(windows)
        )
        mean = np.full(len(windows), np.nan)
        ci_lo = np.full(len(windows), np.nan)
        ci_hi = np.full(len(windows), np.nan)
        for i, vals_w in enumerate(by_window):
            vals_w = vals_w if isinstance(vals_w, np.ndarray) else np.array([])
            mean[i], ci_lo[i], ci_hi[i] = bootstrap_mean_ci(
                vals_w, rng, min_n=min_animals, n_boot=n_boot
            )
        cohort[term] = (mean, ci_lo, ci_hi)

    # Unbounded raw log-odds weights: a single animal's near-separated fit
    # in a sparse window can blow one point out to +/-10+ and drag the
    # shared y-axis (and the eye) away from everyone else. Scale off the
    # 90th percentile of the cohort CI bounds instead of their max; a
    # genuinely truncated cohort CI is marked with a clipped triangle
    # rather than silently cut off, and any per-animal excursion past it is
    # just cut off by the axis limit (as it would be regardless).
    clip_ylim = None
    if clip_percentile:
        ci_bounds = np.concatenate(
            [np.concatenate([lo, hi]) for _, lo, hi in cohort.values()]
        )
        ci_bounds = ci_bounds[~np.isnan(ci_bounds)]
        val_max = np.nanpercentile(np.abs(ci_bounds), clip_pct) if ci_bounds.size else 1.0
        clip_ylim = max(val_max * 1.3, 1.0)

    n_rows = 2
    n_cols = -(-len(terms) // n_rows)  # ceil div: 6 terms -> 3 cols, 4 terms -> 2 cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        sharey=True,
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    flat_axes = list(axes.flat)
    for ai, (ax, term) in enumerate(zip(flat_axes, terms)):
        style = terms_style[term]
        cond = work[work["coef"] == term]

        for subject_id in subjects:
            sub = cond[cond["subject_id"] == subject_id].sort_values("window")
            ax.plot(
                sub["_x"],
                sub["metric"],
                color=animal_colors[str(subject_id)],
                linewidth=1,
                alpha=0.4,
            )

        mean, ci_lo, ci_hi = cohort[term]
        ok = ~np.isnan(mean)
        if clip_ylim is not None:
            mean_plot = np.clip(mean, -clip_ylim, clip_ylim)
            lo_plot = np.clip(ci_lo, -clip_ylim, clip_ylim)
            hi_plot = np.clip(ci_hi, -clip_ylim, clip_ylim)
        else:
            mean_plot, lo_plot, hi_plot = mean, ci_lo, ci_hi
        ax.plot(x[ok], mean_plot[ok], linewidth=2.2, color=style["color"], marker="o")
        ax.fill_between(
            x[ok], lo_plot[ok], hi_plot[ok], color=style["color"], alpha=0.25, linewidth=0
        )
        if clip_ylim is not None:
            clip_hi = ok & (ci_hi > clip_ylim)
            clip_lo = ok & (ci_lo < -clip_ylim)
            if np.any(clip_hi):
                ax.plot(
                    x[clip_hi], np.full(clip_hi.sum(), clip_ylim), marker="^",
                    linestyle="none", color=style["color"], markersize=6, clip_on=False,
                )
            if np.any(clip_lo):
                ax.plot(
                    x[clip_lo], np.full(clip_lo.sum(), -clip_ylim), marker="v",
                    linestyle="none", color=style["color"], markersize=6, clip_on=False,
                )

        ax.axhline(ref, color="gray", linestyle=":", linewidth=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
        if ai // n_cols == n_rows - 1:
            ax.set_xlabel("Block number")
        ax.set_title(style["label"], fontsize=10)
        if ai == len(REWARD_CELL_STYLE) and term in BIAS_TERM_STYLE:
            # marks where the one-hot reward cells end and the
            # choice-persistence bias terms begin (only relevant when both
            # are on the same grid, i.e. the raw-weight view)
            ax.spines["left"].set_linewidth(2.0)

    for row in range(n_rows):
        axes[row][0].set_ylabel(ylabel)
    if fixed_ylim is not None:
        axes[0][0].set_ylim(*fixed_ylim)
    elif clip_ylim is not None:
        axes[0][0].set_ylim(-clip_ylim, clip_ylim)
    return fig, axes


def plot_history_glm_cohort_by_window(
    coefs_window: pd.DataFrame,
    window_bounds: pd.DataFrame,
    value: str = "weight",
    chance: pd.DataFrame | None = None,
    min_animals: int = 2,
    n_boot: int = 2000,
    seed: int = 0,
    animal_colors: dict | None = None,
):
    """Presentation-styled ("pretty") twin of
    :func:`plot_history_glm_reward_cells_by_window`: same one-subplot-per-term
    layout, faint per-animal timecourses + bold cohort mean/CI across
    animals, but with three interchangeable y-axis views selected by `value`:

    - ``"weight"`` (default): the raw GLM coefficient (log-odds), exactly
      what :func:`plot_history_glm_reward_cells_by_window` plots -- 0 means
      "no effect".
    - ``"p_choice"``: each (subject, window, term) coefficient passed
      through the logistic link, ``sigmoid(coef)`` -- the probability this
      term *alone* would predict if it were the only active regressor (this
      model has no intercept, so that baseline logit is exactly 0) -- a
      model-implied quantity, not the raw data; see
      :func:`plot_history_glm_empirical_by_window` for the empirical
      version. 0.5 (the dashed reference line) is "no effect", not "at
      chance" -- see ``"p_choice_minus_chance"`` for that.
    - ``"p_choice_minus_chance"``: the same ``sigmoid(coef)`` minus that
      (subject, window)'s empirical chance level -- P(Stop) at a block's
      very first site, before the animal has any evidence yet (`chance`,
      from :func:`analysis.counterfactual.first_site_chance_by_window` with
      matching window/skip/trials -- required for this mode, one row per
      (subject_id, window) with a ``stopped_first_site`` column). 0 is "no
      better than no-information guessing".

    Transform-then-aggregate throughout: the per-animal values are
    transformed (sigmoid, then chance-subtracted) *before*
    :func:`analysis.plotting.bootstrap_mean_ci` bootstraps the cohort mean
    across animals, not after -- so the plotted CI is a real CI on the
    transformed quantity, not a transformed CI on the coefficient.

    Individual animals are colored by :func:`analysis.plotting_style.animal_palette`
    (override via `animal_colors`) instead of `plot_history_glm_reward_cells_by_window`'s
    ``tab10``, matching every other plot in this notebook. Meant to be
    called inside :func:`analysis.plotting_style.presentation_style` -- no
    title, legend, or light-background styling of its own.
    """
    if value not in ("weight", "p_choice", "p_choice_minus_chance"):
        raise ValueError(
            f"value must be 'weight', 'p_choice', or 'p_choice_minus_chance', got {value!r}"
        )
    if value == "p_choice_minus_chance" and chance is None:
        raise ValueError("chance is required when value='p_choice_minus_chance'")

    work = coefs_window.copy()
    if value in ("p_choice", "p_choice_minus_chance"):
        work["metric"] = 1.0 / (1.0 + np.exp(-work["value"].to_numpy(dtype=float)))
        if value == "p_choice_minus_chance":
            work = work.merge(
                chance.rename(columns={"stopped_first_site": "_chance"}),
                on=["subject_id", "window"],
                how="left",
            )
            work["metric"] = work["metric"] - work["_chance"]
    else:
        work["metric"] = work["value"]

    # The choice-persistence bias terms are signed (+1/-1/0), not one-hot --
    # sigmoid(coef) for them doesn't carry the same "P(choice) if only this
    # cell were active" reading the one-hot reward cells get, so the two
    # probability views drop them and show just the 2x2 reward-cell grid;
    # the raw weight view keeps all 6 (log-odds is meaningful for both).
    # A fit on the one-hot design (:data:`ONEHOT_CELL_STYLE`) has every term
    # as a cell log-odds, so all six show in every view and sigmoid(weight)
    # is the cell's empirical P(Choice).
    onehot = bool(set(ONEHOT_CELL_STYLE) & set(work["coef"]))
    if onehot:
        terms_style = ONEHOT_CELL_STYLE
    else:
        terms_style = (
            {**REWARD_CELL_STYLE, **BIAS_TERM_STYLE} if value == "weight" else REWARD_CELL_STYLE
        )
    ylabel = {
        "weight": "GLM coefficient",
        "p_choice": "P(Choice)" if onehot else "P(Choice) implied by term",
        "p_choice_minus_chance": "P(Choice) − chance",
    }[value]
    ref = {"weight": 0.0, "p_choice": 0.5, "p_choice_minus_chance": 0.0}[value]
    fixed_ylim = (-0.1, 1.1) if value == "p_choice" else None

    present = set(work["coef"])
    terms_style = {k: v for k, v in terms_style.items() if k in present}
    return _history_glm_cohort_grid(
        work,
        window_bounds,
        terms_style,
        ylabel,
        ref,
        fixed_ylim=fixed_ylim,
        clip_percentile=(value == "weight"),
        clip_pct=98 if len(set(ONEHOT_CELL_STYLE) & present) == len(ONEHOT_CELL_STYLE) else 90,
        min_animals=min_animals,
        n_boot=n_boot,
        seed=seed,
        animal_colors=animal_colors,
    )


def history_glm_empirical_cells(glm_windows: pd.DataFrame) -> pd.DataFrame:
    """Empirical (not model-derived) P(Choice) for each of the four one-hot
    reward-history cells (:data:`REWARD_CELL_STYLE`), per animal per window.

    Unlike :func:`fit_history_glm`'s coefficients, this is nothing but the
    raw mean of ``choice`` among the trials where that cell is active --
    each trial belongs to exactly one of the four one-hot cells, so this is
    a plain conditional stop rate, no regression or logistic link involved.

    `glm_windows` is :func:`history_glm_features`'s output expanded to
    block windows (:func:`analysis.features.expand_to_block_windows`) --
    the same input :func:`fit_history_glm` takes. Returns long-format rows
    (subject_id, window, coef, metric), the same shape
    :func:`_history_glm_cohort_grid` expects.
    """
    rows = []
    for term in REWARD_CELL_STYLE:
        sub = glm_windows[glm_windows[term] == 1]
        per = sub.groupby(["subject_id", "window"])["choice"].mean().reset_index()
        per = per.rename(columns={"choice": "metric"})
        per["coef"] = term
        rows.append(per)
    return pd.concat(rows, ignore_index=True)


def plot_history_glm_empirical_by_window(
    glm_windows: pd.DataFrame,
    window_bounds: pd.DataFrame,
    value: str = "p_choice",
    chance: pd.DataFrame | None = None,
    min_animals: int = 2,
    n_boot: int = 2000,
    seed: int = 0,
    animal_colors: dict | None = None,
):
    """Empirical twin of :func:`plot_history_glm_cohort_by_window`'s
    ``"p_choice"``/``"p_choice_minus_chance"`` views: same 2x2 reward-cell
    grid, faint per-animal timecourses + bold cohort mean/CI, but the
    metric is the *raw observed* stop rate for each history cell
    (:func:`history_glm_empirical_cells`) -- no GLM fit, no logistic link,
    just what the animal actually did on those trials.

    ``value="p_choice"`` plots that empirical P(Choice) directly (0.5 dashed
    line is a neutral midpoint, not "at chance"). ``value="p_choice_minus_chance"``
    subtracts that (subject, window)'s empirical chance level -- P(Stop) at
    a block's very first site (`chance`, from
    :func:`analysis.counterfactual.first_site_chance_by_window` with
    matching window/skip/trials -- required for this mode). As with the
    GLM-based version, the per-animal values are chance-subtracted *before*
    :func:`analysis.plotting.bootstrap_mean_ci` bootstraps the cohort mean,
    not after.
    """
    if value not in ("p_choice", "p_choice_minus_chance"):
        raise ValueError(
            f"value must be 'p_choice' or 'p_choice_minus_chance', got {value!r}"
        )
    if value == "p_choice_minus_chance" and chance is None:
        raise ValueError("chance is required when value='p_choice_minus_chance'")

    work = history_glm_empirical_cells(glm_windows)
    if value == "p_choice_minus_chance":
        work = work.merge(
            chance.rename(columns={"stopped_first_site": "_chance"}),
            on=["subject_id", "window"],
            how="left",
        )
        work["metric"] = work["metric"] - work["_chance"]

    ylabel = {
        "p_choice": "P(Choice)",
        "p_choice_minus_chance": "P(Choice) − chance",
    }[value]
    ref = 0.5 if value == "p_choice" else 0.0
    fixed_ylim = (-0.1, 1.1) if value == "p_choice" else None

    return _history_glm_cohort_grid(
        work,
        window_bounds,
        REWARD_CELL_STYLE,
        ylabel,
        ref,
        fixed_ylim=fixed_ylim,
        clip_percentile=False,
        min_animals=min_animals,
        n_boot=n_boot,
        seed=seed,
        animal_colors=animal_colors,
    )
