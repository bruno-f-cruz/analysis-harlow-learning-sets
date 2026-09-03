"""1-back within-block history GLM: feature construction, per-group fitting, and
its reward-cell timecourse plot."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from analysis.plotting import TWO_BY_TWO_COLORS, bootstrap_mean_ci

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

    rs["IsPrevChoice_SameOdor"] = np.where(is_same, np.where(prev_choice, 1.0, -1.0), 0.0)
    rs["IsPrevChoice_OtherOdor"] = np.where(~is_same, np.where(prev_choice, 1.0, -1.0), 0.0)
    rs["H_Same_Rew"] = (is_same & prev_rewarded).astype(float)
    rs["H_Same_NoRew"] = (is_same & ~prev_rewarded).astype(float)
    rs["H_Other_Rew"] = (~is_same & prev_rewarded).astype(float)
    rs["H_Other_NoRew"] = (~is_same & ~prev_rewarded).astype(float)
    rs["choice"] = rs["has_choice"].astype(int)
    return rs


def fit_history_glm(
    features: pd.DataFrame,
    unit_col: str | list[str] = "session_id",
    min_trials: int = 10,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Fit the history GLM independently within each ``unit_col`` group.

    Unregularised logistic regression, no intercept. Groups with fewer than
    ``min_trials`` trials or no variance in ``choice`` are skipped, as is any
    group whose fit raises.

    Each coefficient gets a Wald 95% CI (``se``, ``ci_lo``, ``ci_hi``) from the
    inverse Fisher information at the MLE. ``cv_accuracy`` is the mean
    held-out accuracy of a stratified k-fold refit of the same group -- a
    sanity check that the unregularised fit isn't just memorising noise.
    """
    unit_cols = [unit_col] if isinstance(unit_col, str) else list(unit_col)
    records = []
    for unit, sdf in features.groupby(unit_cols, sort=True):
        key = dict(zip(unit_cols, unit if isinstance(unit, tuple) else (unit,)))
        if len(sdf) < min_trials or sdf["choice"].nunique() < 2:
            continue
        X = sdf[HISTORY_GLM_COEFS].to_numpy(dtype=float)
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
            se = np.full(len(HISTORY_GLM_COEFS), np.nan)

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

        for i, name in enumerate(HISTORY_GLM_COEFS):
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

    ci_bounds = np.concatenate([np.concatenate([lo, hi]) for _, lo, hi in cohort.values()])
    ci_bounds = ci_bounds[~np.isnan(ci_bounds)]
    val_max = np.nanpercentile(np.abs(ci_bounds), 90) if ci_bounds.size else 1.0
    ylim = max(val_max * 1.3, 1.0)

    fig, axes = plt.subplots(
        1, len(terms), figsize=(6 * len(terms), 4), sharey=True, sharex=True, squeeze=False
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
        ax.fill_between(x[ok], lo[ok], hi[ok], color=style["color"], alpha=0.25, linewidth=0)
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
