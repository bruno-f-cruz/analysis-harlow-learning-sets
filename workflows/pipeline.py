import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def figure_output(mo):
    import itertools
    import re
    from pathlib import Path as _Path

    from matplotlib import pyplot as plt

    SCRATCH_DIR = _Path("./scratch")
    IS_SCRIPT = mo.app_meta().mode == "script"

    def _figure_slug(fig, fallback):
        _title = fig.get_suptitle() if hasattr(fig, "get_suptitle") else ""
        if not _title and fig.axes:
            _title = fig.axes[0].get_title()
        _slug = re.sub(r"[^a-z0-9]+", "-", _title.split("\n")[0].strip().lower()).strip(
            "-"
        )
        return _slug[:60] or fallback

    _fig_counter = itertools.count()

    def _save_open_figures(*args, **kwargs):
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        for _num in plt.get_fignums():
            _figure = plt.figure(_num)
            _path = (
                SCRATCH_DIR
                / f"{next(_fig_counter):03d}-{_figure_slug(_figure, f'figure{_num}')}.png"
            )
            _figure.savefig(_path, dpi=150, bbox_inches="tight")
            print(f"saved {_path}")
        plt.close("all")

    if IS_SCRIPT:
        plt.switch_backend("Agg")
        plt.show = _save_open_figures
    return (plt,)


@app.cell
def imports_subprocess():
    import subprocess

    return (subprocess,)


@app.cell
def imports_data_loading():
    from pathlib import Path

    # `sync_open_data_sessions` (list-by-subject-and-date, then sync) is no
    # longer the entry point here -- `selection` below already resolves the
    # exact sessions to use from data_assets.json. We reuse its underlying
    # sync mechanism directly, aliased to a public-looking name since it's
    # the same "download these S3 prefixes to local disk" primitive
    # `sync_open_data_sessions` itself calls after listing.
    from analysis.io import _sync_uris_to_local as sync_uris_to_local

    return Path, sync_uris_to_local


@app.cell
def imports_provenance():
    from analysis.config import load_config
    from analysis.artifacts import artifact_store_for_uri
    from analysis.progress import ProgressWriter
    from analysis.run import (
        generate_run_id,
        build_manifest,
        git_commit,
        git_is_dirty,
        host_info,
    )
    from analysis.sessions import load_attached_datasets
    from analysis.io import build_inputs_manifest
    from datetime import datetime, timezone
    import os

    return (
        load_config,
        artifact_store_for_uri,
        ProgressWriter,
        generate_run_id,
        build_manifest,
        git_commit,
        git_is_dirty,
        host_info,
        load_attached_datasets,
        build_inputs_manifest,
        datetime,
        timezone,
        os,
    )


@app.cell
def run_setup(
    load_config, generate_run_id, artifact_store_for_uri, ProgressWriter, os, Path, datetime, timezone
):
    # Named `run_setup` rather than `setup` -- marimo reserves the literal cell
    # name `setup` for its own special zero-argument "setup cell" concept, and
    # rejects a `setup` cell that (like this one) depends on other cells.
    config = load_config(Path(__file__).parent.parent / "configs" / "default.yaml")
    run_id = os.environ.get("RUN_ID") or generate_run_id()
    started_at = datetime.now(timezone.utc).isoformat()
    store = artifact_store_for_uri(f"{config['artifact_uri']}/runs/{run_id}")
    # `store.uri(...)` returns a plain filesystem path for LocalArtifactStore but
    # an "s3://..." string for S3ArtifactStore -- ProgressWriter always opens a
    # real filesystem Path to append to, so this only works when the artifact
    # store is local. That's the only backend this project exercises end-to-end
    # so far (see the plan's "Still Open" section); routing progress.jsonl
    # writes through the `store` abstraction itself would be the more correct
    # long-term fix, but is out of scope here.
    progress_path = Path(store.uri("progress.jsonl"))
    progress = ProgressWriter(progress_path, run_id=run_id)
    progress.started(stage="run")
    return config, run_id, started_at, store, progress, progress_path


@app.cell
def selection(load_attached_datasets, build_inputs_manifest, store, progress, Path):
    # No live DocDB query here -- data_assets.json (repo root) is the pinned,
    # git-tracked source of truth for which sessions this run analyzes.
    # Refresh it separately with `uv run attach_datasets.py ...` when needed.
    attached = load_attached_datasets(Path(__file__).parent.parent / "data_assets.json")
    store.write_json("selection.json", {"attached_datasets": attached})

    inputs = build_inputs_manifest([entry["location"] for entry in attached])
    store.write_json("inputs.json", inputs)
    progress.log(f"resolved {len(attached)} attached sessions from data_assets.json")
    return attached, inputs


@app.cell
def sync_raw_data(Path, attached, subprocess, sync_uris_to_local):
    # Session selection now comes from `attached` (built in `selection` above,
    # itself read from data_assets.json) instead of a hardcoded subject/date
    # filter -- but the actual download-to-disk mechanism is unchanged.
    OUTPUT_ROOT = Path("./data")
    # `mount` mirrors the local session-dir naming "<subject>_<date>_<time>"
    # (analysis.sessions.build_attached_dataset_entries), so its first
    # underscore-delimited token is the subject id.
    SUBJECT_IDS = sorted({entry["mount"].split("_")[0] for entry in attached})
    uris = [entry["location"] for entry in attached]
    if True:
        sync_uris_to_local(uris, OUTPUT_ROOT, no_sign_request=True, confirm=False)
        #! uv run python process_sessions.py
        subprocess.call(["uv", "run", "python", "process_sessions.py"])
    return (SUBJECT_IDS,)


@app.cell
def load_and_prepare_trials():
    import pandas as pd
    import numpy as np
    from analysis.features import (
        add_subject_id,
        assign_blocks,
        report_subject_overrides,
        trim_sessions,
    )

    trials = pd.read_parquet("data/processed/trials.parquet")

    # A few sessions were acquired under the wrong subject, and upstream is not
    # renaming the assets (the name is only a unique key). Correct the subject
    # here, once, so every grouping below is right -- and print which overrides
    # actually matched rather than assuming they did.
    # https://github.com/AllenNeuralDynamics/aind-scientific-computing/issues/855
    print(report_subject_overrides(trials).to_string(index=False))
    trials = add_subject_id(trials)
    trials = assign_blocks(trials)

    # Drop sessions shorter than 15 minutes (first to last site timestamp)
    session_start = trials.groupby("session_id")["start_time"].min()
    session_end = trials.groupby("session_id")["start_time"].max()
    session_duration = session_end - session_start
    # Handle both timedelta and numeric (seconds) dtypes
    threshold = (
        pd.Timedelta(minutes=15)
        if pd.api.types.is_timedelta64_dtype(session_duration)
        else 15 * 60
    )
    long_sessions = session_duration[session_duration >= threshold].index
    trials = trials[trials["session_id"].isin(long_sessions)]

    trials = trim_sessions(trials, start_frac=0.0, end_frac=1.0)

    def is_rewarded(patch_label: str) -> bool:
        return "NonRewarded" not in patch_label

    def odor_index(odor_concentration: list[float] | np.ndarray) -> int:
        return np.argmax(np.array(odor_concentration))

    trials["is_rewarded_odor"] = trials["patch_label"].apply(is_rewarded)
    trials["odor_index"] = trials["odor_concentration"].apply(odor_index)

    # Per-block P(stay): fraction of a block's RewardSite trials the animal stopped.
    # Broadcast back onto every RewardSite row of the block; non-RewardSite / un-
    # blocked rows stay NaN (so they're never treated as degenerate below).
    rs_mask = (trials["site_label"] == "RewardSite") & trials["block"].notna()
    trials["p_stay_in_block"] = (
        trials[rs_mask].groupby(["session_id", "block"])["has_choice"].transform("mean")
    )

    # Snapshot before the degenerate-block filter below. The counterfactual analysis
    # at the end of the notebook needs this: dropping p_stay == 1 blocks removes
    # exactly the "stopped at everything" blocks, which are the clearest evidence of
    # *failed* discrimination, so filtering them would inflate the metric.
    #
    trials_all = trials.copy(deep=False)

    # Optional block filter: drop blocks with no within-block choice variability,
    # i.e. the animal always stopped (p_stay == 1) or always left (p_stay == 0).
    # Report how many blocks are excluded per session before dropping them.
    block_p_stay = trials[rs_mask].groupby(["session_id", "block"])["has_choice"].mean()
    excluded_per_session = (
        block_p_stay.isin([0.0, 1.0])
        .groupby(level="session_id")
        .agg(excluded="sum", total="count")
    )
    excluded_per_session["kept"] = (
        excluded_per_session["total"] - excluded_per_session["excluded"]
    )
    print("Blocks excluded per session (p_stay in {0, 1}):")
    print(excluded_per_session.to_string())
    print(
        f"\nTotal: excluding {excluded_per_session['excluded'].sum()} "
        f"of {excluded_per_session['total'].sum()} blocks"
    )

    trials = trials[~trials["p_stay_in_block"].isin([0.0, 1.0])]
    return np, pd, trials, trials_all


@app.cell
def sql_over_trials(mo, trials_all):
    # DuckDB query straight over the in-memory `trials_all` (pre-filter trials, with
    # the issue #855 subject corrections already applied). Swap the FROM for
    # read_parquet('data/processed/trials.parquet') to hit the file instead -- but
    # then `subject_id` is the raw session-name prefix, uncorrected.
    _sessions_per_animal = mo.sql(
        """
        SELECT subject_id, count(DISTINCT session_id) AS n_sessions
        FROM trials_all
        GROUP BY subject_id
        ORDER BY subject_id
        """,
        output=False,
    )

    # odor_concentration holds a 7-element vector per row, which the table widget
    # cannot serialise -- drop it and keep the derived scalar `odor_index`.
    mo.vstack(
        [
            mo.md("## Sessions per animal"),
            mo.ui.table(_sessions_per_animal, selection=None),
            mo.md(f"## Single trials ({len(trials_all):,} rows)"),
            mo.ui.table(
                trials_all.drop(columns=["odor_concentration"]),
                pagination=True,
                page_size=15,
                selection=None,
            ),
        ]
    )
    return


@app.cell
def choice_by_block_position_pooled(SUBJECT_IDS, plt, trials):
    from analysis.plotting import a_lot_of_style, plot_choice_by_block_position

    for _animal in SUBJECT_IDS:
        print(f"Animal {_animal}")
        _fig = plt.figure(figsize=(10, 6))
        _ax = _fig.add_subplot(111)
        with a_lot_of_style():
            plot_choice_by_block_position(
                trials[trials["subject_id"] == _animal], ax=_ax
            )
    plt.show()
    return (a_lot_of_style,)


@app.cell
def choice_by_block_position_per_session(a_lot_of_style, plt, trials):
    from analysis.plotting import plot_choice_by_block_position_per_session

    with a_lot_of_style():
        plot_choice_by_block_position_per_session(trials)

    plt.show()
    return


@app.cell
def choice_by_first_stop(a_lot_of_style, plt, trials):
    from analysis.plotting import plot_choice_by_block_position_by_first_stop

    with a_lot_of_style():
        plot_choice_by_block_position_by_first_stop(trials)

    plt.show()
    return


@app.cell
def choice_by_first_stop_overlay(a_lot_of_style, plt, trials):
    from analysis.plotting import plot_choice_by_block_position_by_first_stop_overlay

    with a_lot_of_style():
        plot_choice_by_block_position_by_first_stop_overlay(trials)

    plt.show()
    return


@app.cell
def choice_at_first_stops_across_sessions(
    SUBJECT_IDS,
    a_lot_of_style,
    np,
    plt,
    trials,
):
    # P(choice) at the very first trial of every block, averaged within session, plotted across sessions
    _rs = trials[
        (trials["site_label"] == "RewardSite") & trials["block"].notna()
    ].copy()
    _rs = _rs.sort_values(["session_id", "block", "start_time"])
    _rs["_block_pos"] = _rs.groupby(["session_id", "block"]).cumcount()
    _rs["session_date"] = _rs["session_id"].str.split("_").str[1]
    STOP_STYLES = {
        0: {"label": "1st stop", "color": "tab:blue"},
        1: {"label": "2nd stop", "color": "tab:orange"},
        2: {"label": "3rd stop", "color": "tab:green"},
    }
    # Add subject and session date label for plotting
    with a_lot_of_style():
        _fig, _axes = plt.subplots(
            1,
            len(SUBJECT_IDS),
            figsize=(5 * len(SUBJECT_IDS), 4),
            sharey=True,
            squeeze=False,
        )
        for _ax, _animal in zip(_axes[0], SUBJECT_IDS):
            for stop_pos, _style in STOP_STYLES.items():
                stop_trials = _rs[_rs["_block_pos"] == stop_pos].copy()
                session_means = (
                    stop_trials.groupby(["subject_id", "session_id", "session_date"])[
                        "has_choice"
                    ]
                    .mean()
                    .reset_index()
                )
                _sub = session_means[
                    session_means["subject_id"] == _animal
                ].sort_values("session_id")
                _x = np.arange(len(_sub))
                _ax.plot(
                    _x,
                    _sub["has_choice"],
                    marker="o",
                    color=_style["color"],
                    label=_style["label"],
                )
            first_stop_sub = _rs[
                (_rs["_block_pos"] == 0) & (_rs["subject_id"] == _animal)
            ]
            _dates = sorted(first_stop_sub["session_id"].unique())
            date_labels = [s.split("_")[1] for s in _dates]
            _ax.set_xticks(np.arange(len(_dates)))
            _ax.set_xticklabels(date_labels, rotation=45, ha="right")
            _ax.set_xlabel("Session")
            _ax.set_ylabel("P(choice)")
            _ax.set_ylim(0, 1.05)
            _ax.set_title(f"Subject {_animal}")
            _ax.legend(frameon=False, fontsize=8)
        _fig.suptitle(
            "P(choice) at 1st, 2nd, and 3rd trial of each block (session averages)"
        )
        _fig.tight_layout()
    plt.show()  # Use session dates from 1st stop for x-axis labels (most sessions should have it)
    return


@app.cell
def history_glm_per_session(a_lot_of_style, np, pd, plt, trials):
    from sklearn.linear_model import LogisticRegression
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    _rs = trials[
        (trials["site_label"] == "RewardSite") & trials["block"].notna()
    ].copy()
    _rs = _rs.sort_values(["session_id", "block", "start_time"])
    _grp = _rs.groupby(["session_id", "block"], sort=False)
    _rs["prev_odor_index"] = _grp["odor_index"].shift(1)
    _rs["prev_has_choice"] = _grp["has_choice"].shift(1)
    _rs["prev_has_reward"] = _grp["has_reward"].shift(1)
    _rs = _rs.dropna(subset=["prev_odor_index", "prev_has_choice", "prev_has_reward"])
    is_same = (_rs["odor_index"] == _rs["prev_odor_index"]).to_numpy()
    prev_choice = _rs["prev_has_choice"].astype(bool).to_numpy()
    prev_rewarded = _rs["prev_has_reward"].astype(bool).to_numpy()
    _rs["IsPrevChoice_SameOdor"] = np.where(
        is_same, np.where(prev_choice, 1.0, -1.0), 0.0
    )
    _rs["IsPrevChoice_OtherOdor"] = np.where(
        ~is_same, np.where(prev_choice, 1.0, -1.0), 0.0
    )
    _rs["H_Same_Rew"] = (is_same & prev_rewarded).astype(float)
    _rs["H_Same_NoRew"] = (is_same & ~prev_rewarded).astype(float)
    _rs["H_Other_Rew"] = (~is_same & prev_rewarded).astype(float)
    _rs["H_Other_NoRew"] = (~is_same & ~prev_rewarded).astype(float)
    _rs["choice"] = _rs["has_choice"].astype(int)
    PLOT_COEFS = [
        "IsPrevChoice_SameOdor",
        "IsPrevChoice_OtherOdor",
        "H_Same_Rew",
        "H_Same_NoRew",
        "H_Other_Rew",
        "H_Other_NoRew",
    ]
    FEATURE_COLS = PLOT_COEFS
    records = []
    for session_id, sdf in _rs.groupby("session_id"):
        if len(sdf) < 10 or sdf["choice"].nunique() < 2:
            continue
        X = sdf[FEATURE_COLS].to_numpy(dtype=float)
        y = sdf["choice"].to_numpy(dtype=int)
        try:
            clf = LogisticRegression(
                C=np.inf, solver="lbfgs", fit_intercept=False, max_iter=500
            )
            clf.fit(X, y)
            for name, val in zip(PLOT_COEFS, clf.coef_[0]):
                records.append(
                    dict(
                        session_id=session_id,
                        subject_id=sdf["subject_id"].iloc[0],
                        coef=name,
                        value=val,
                    )
                )
        except Exception as e:
            print(f"Session {session_id} failed: {e}")
    coefs = pd.DataFrame(records)
    SAME_COLOR = "#e07b39"
    OTHER_COLOR = "#4f8fc0"
    NEUTRAL_COLOR = "gray"

    def _coef_color(name):
        if name.startswith("H_Same"):
            return SAME_COLOR
        if name.startswith("H_Other"):
            return OTHER_COLOR
        if "Same" in name:
            return SAME_COLOR
        if "Other" in name:
            return OTHER_COLOR
        return NEUTRAL_COLOR

    TICK_LABELS = [
        "IsPrevChoice\n[same]",
        "IsPrevChoice\n[other]",
        "Same × Rew",
        "Same × NoRew",
        "Other × Rew",
        "Other × NoRew",
    ]
    subjects = sorted(coefs["subject_id"].unique())
    x_pos = np.arange(len(PLOT_COEFS))
    rng = np.random.default_rng(0)
    PAIR_GROUPS = [
        (["IsPrevChoice_SameOdor", "IsPrevChoice_OtherOdor"], "#f5f5f5"),
        (["H_Same_Rew", "H_Same_NoRew"], "#fff3eb"),
        (["H_Other_Rew", "H_Other_NoRew"], "#ebf3ff"),
    ]
    N_BOOTSTRAP = 2000
    with a_lot_of_style():
        _fig, _axes = plt.subplots(
            2,
            len(subjects),
            figsize=(8 * len(subjects), 10),
            sharey="row",
            squeeze=False,
        )
        for col, _subject in enumerate(subjects):
            _ax = _axes[0][col]
            for group_members, bg in PAIR_GROUPS:
                idxs = [PLOT_COEFS.index(_m) for _m in group_members]
                _ax.axvspan(min(idxs) - 0.4, max(idxs) + 0.4, color=bg, zorder=0)
            _sub = coefs[coefs["subject_id"] == _subject]
            _sessions = sorted(_sub["session_id"].unique())
            _n = len(_sessions)
            cmap = plt.get_cmap("viridis")
            norm = Normalize(vmin=0, vmax=max(_n - 1, 1))
            for day, session_id in enumerate(_sessions):
                sdata = _sub[_sub["session_id"] == session_id].set_index("coef")[
                    "value"
                ]
                _vals = [sdata.get(c, np.nan) for c in PLOT_COEFS]
                jx = x_pos + rng.uniform(-0.15, 0.15, len(x_pos))
                _ax.scatter(
                    jx, _vals, color=cmap(norm(day)), s=40, zorder=3, alpha=0.85
                )
            means = _sub.groupby("coef")["value"].mean().reindex(PLOT_COEFS)
            sems = _sub.groupby("coef")["value"].sem().reindex(PLOT_COEFS)
            for _xi, coef in enumerate(PLOT_COEFS):
                _ax.errorbar(
                    _xi,
                    means[coef],
                    yerr=sems[coef],
                    fmt="o",
                    color=_coef_color(coef),
                    ms=8,
                    lw=2.5,
                    capsize=5,
                    zorder=5,
                )
            _ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            _ax.set_xticks(x_pos)
            _ax.set_xticklabels(TICK_LABELS, rotation=30, ha="right", fontsize=9)
            _ax.set_title(f"Subject {_subject}")
            _ax.set_xlabel("Regressor")
            cb = _fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=_ax)
            cb.set_label("Session (day)")
            cb.set_ticks([0, max(_n - 1, 1)])
            cb.set_ticklabels(["first", "last"])
            ax2 = _axes[1][col]
            for group_members, bg in PAIR_GROUPS:
                idxs = [PLOT_COEFS.index(_m) for _m in group_members]
                ax2.axvspan(min(idxs) - 0.4, max(idxs) + 0.4, color=bg, zorder=0)
            session_list = sorted(_sub["session_id"].unique())
            coef_matrix = np.array(
                [
                    [
                        _sub[_sub["session_id"] == sid]
                        .set_index("coef")["value"]
                        .reindex(PLOT_COEFS)
                        .values
                        for sid in session_list
                    ]
                ]
            ).squeeze(0)
            observed_mean = np.nanmean(coef_matrix, axis=0)
            boot_rng = np.random.default_rng(42)
            boot_means = np.array(
                [
                    np.nanmean(
                        coef_matrix[
                            boot_rng.integers(
                                0, len(session_list), size=len(session_list)
                            )
                        ],
                        axis=0,
                    )
                    for _ in range(N_BOOTSTRAP)
                ]
            )
            ci_lo = np.nanpercentile(boot_means, 2.5, axis=0)
            ci_hi = np.nanpercentile(boot_means, 97.5, axis=0)
            for _xi, coef in enumerate(PLOT_COEFS):
                _color = _coef_color(coef)
                ax2.bar(
                    _xi,
                    observed_mean[_xi],
                    color=_color,
                    alpha=0.75,
                    width=0.6,
                    zorder=2,
                )
                ax2.errorbar(
                    _xi,
                    observed_mean[_xi],
                    yerr=[
                        [observed_mean[_xi] - ci_lo[_xi]],
                        [ci_hi[_xi] - observed_mean[_xi]],
                    ],
                    fmt="none",
                    color="black",
                    capsize=5,
                    lw=1.5,
                    zorder=3,
                )
            ax2.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(TICK_LABELS, rotation=30, ha="right", fontsize=9)
            ax2.set_title(f"Subject {_subject} — bootstrap mean")
            ax2.set_xlabel("Regressor")
        _axes[0][0].set_ylabel("GLM coefficient (per session)")
        _axes[1][0].set_ylabel("GLM coefficient (bootstrap mean ± 95% CI)")
        _fig.suptitle(
            "Logistic GLM: P(choice) — within-block, 1-trial history\nReward encoding: one-hot (is_same × is_prev_rewarded), no intercept"
        )
        _fig.tight_layout()
    plt.show()
    return coefs, subjects


@app.cell
def history_glm_reward_cells(a_lot_of_style, coefs, np, plt, subjects):
    # ── One-hot reward cells timecourse across sessions ───────────────────────────
    # 4 distinct colors so solid/dashed ambiguity is avoided entirely
    REWARD_CELLS = {
        "H_Same_Rew": {"label": "Same × Rew", "color": "#e07b39", "marker": "o"},
        "H_Same_NoRew": {"label": "Same × NoRew", "color": "#f5c18a", "marker": "o"},
        "H_Other_Rew": {"label": "Other × Rew", "color": "#2a6496", "marker": "s"},
        "H_Other_NoRew": {"label": "Other × NoRew", "color": "#9ecae1", "marker": "s"},
    }
    with a_lot_of_style():
        _fig, _axes = plt.subplots(
            1, len(subjects), figsize=(6 * len(subjects), 4), sharey=True, squeeze=False
        )
        for _ax, _subject in zip(_axes[0], subjects):
            _sub = coefs[coefs["subject_id"] == _subject]
            _sessions = sorted(_sub["session_id"].unique())
            _x = np.arange(len(_sessions))
            _dates = [s.split("_")[1] for s in _sessions]
            for term, _style in REWARD_CELLS.items():
                rows = _sub[_sub["coef"] == term].set_index("session_id")
                _vals = [
                    rows.loc[sid, "value"] if sid in rows.index else np.nan
                    for sid in _sessions
                ]
                _ax.plot(
                    _x,
                    _vals,
                    marker=_style["marker"],
                    color=_style["color"],
                    linewidth=2,
                    markersize=7,
                    label=_style["label"],
                )
            _ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_dates, rotation=45, ha="right", fontsize=8)
            _ax.set_xlabel("Session")
            _ax.set_title(f"Subject {_subject}")
            _ax.legend(frameon=False, fontsize=8)
        _axes[0][0].set_ylabel("GLM coefficient")
        _fig.suptitle("One-hot reward cells timecourse")
        _fig.tight_layout()
    plt.show()
    return


@app.cell
def window_controls(mo):
    window_blocks = mo.ui.slider(
        2, 100, 1, value=100, label="Window size (blocks)", show_value=True
    )
    skip_blocks = mo.ui.slider(
        1, 100, 1, value=20, label="Skip / stride (blocks)", show_value=True
    )
    min_blocks_window = mo.ui.slider(
        1, 20, 1, value=3, label="Min blocks per counterfactual cell", show_value=True
    )
    mo.vstack([window_blocks, skip_blocks, min_blocks_window])
    return min_blocks_window, skip_blocks, window_blocks


@app.cell
def history_glm_per_window(np, pd, skip_blocks, trials, window_blocks):
    from analysis.features import expand_to_block_windows as _expand_to_block_windows

    # Regressors of the 1-back within-block logistic GLM on P(choice). The two
    # IsPrevChoice_* terms are signed (+1 stopped / -1 ran through / 0 when the
    # previous odor was of the other kind); the four H_* terms are the one-hot
    # (is_same_odor x is_prev_rewarded) reward cells. No intercept is fit. This
    # mirrors history_glm_per_session above, but fit per block-window instead of
    # per session -- kept inline (rather than a shared library function) since
    # the GLM-fitting composition is the analysis itself, not a generic utility.
    HISTORY_GLM_COEFS = [
        "IsPrevChoice_SameOdor",
        "IsPrevChoice_OtherOdor",
        "H_Same_Rew",
        "H_Same_NoRew",
        "H_Other_Rew",
        "H_Other_NoRew",
    ]

    def history_glm_features(trials):
        """RewardSite trials with the 1-back within-block history regressors.

        The previous trial's odor, choice and reward are taken with a shift(1)
        *within* each real (session_id, block), so history never crosses a block
        or a session boundary and the first trial of every block drops out. This
        is deliberately done before any windowing: a window is a way of grouping
        blocks for fitting, never a boundary the 1-back history should see.
        """
        rs = trials[
            (trials["site_label"] == "RewardSite") & trials["block"].notna()
        ].copy()
        rs = rs.sort_values(["session_id", "block", "start_time"])
        grp = rs.groupby(["session_id", "block"], sort=False)
        rs["prev_odor_index"] = grp["odor_index"].shift(1)
        rs["prev_has_choice"] = grp["has_choice"].shift(1)
        rs["prev_has_reward"] = grp["has_reward"].shift(1)
        rs = rs.dropna(
            subset=["prev_odor_index", "prev_has_choice", "prev_has_reward"]
        )

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
        rs["choice"] = rs["has_choice"].astype(int)
        return rs

    def fit_history_glm(features, unit_col="session_id", min_trials=10):
        """Fit the history GLM independently within each ``unit_col`` group.

        ``unit_col`` is a column name or a list of them -- ``["subject_id",
        "window"]`` for this per-block-window fit (``window`` alone would pool
        animals). Unregularised logistic regression, no intercept. Groups with
        fewer than ``min_trials`` trials or no variance in ``choice`` are
        skipped, as is any group whose fit raises.
        """
        from sklearn.linear_model import LogisticRegression

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
            for name, val in zip(HISTORY_GLM_COEFS, clf.coef_[0]):
                records.append(
                    {
                        **key,
                        "subject_id": sdf["subject_id"].iloc[0],
                        "coef": name,
                        "value": val,
                        "n_trials": len(sdf),
                    }
                )
        return pd.DataFrame(records)

    # Same regressors and same unregularised fit as the per-session GLM above, with
    # a window of blocks as the unit of analysis instead of a session. The 1-back
    # history is built *before* windowing, so a window boundary is never mistaken
    # for a block boundary. Windows are cut over the blocks `trials` still holds,
    # i.e. the ones that survive the degenerate-block filter the session fit uses.
    glm_features = history_glm_features(trials)
    glm_windows = _expand_to_block_windows(
        glm_features, window_blocks.value, skip_blocks.value
    )
    coefs_window = fit_history_glm(glm_windows, unit_col=["subject_id", "window"])
    window_bounds = (
        glm_windows.drop_duplicates("window")
        .set_index("window")[["window_start", "window_end"]]
        .sort_index()
    )
    print(
        f"{window_blocks.value}-block windows, stride {skip_blocks.value} -> {len(window_bounds)} window positions"
    )
    print(
        coefs_window.groupby("subject_id")
        .agg(n_windows=("window", "nunique"), trials_per_window=("n_trials", "mean"))
        .round(1)
        .to_string()
    )
    return coefs_window, window_bounds


@app.cell
def history_glm_reward_cells_by_window(
    a_lot_of_style,
    coefs_window,
    np,
    plt,
    window_blocks,
    window_bounds,
):
    # Per-window version of the reward-cell timecourse above: x is the window
    # number (labelled with its pooled block range) rather than the session date.
    REWARD_CELLS_WINDOW = {
        "H_Same_Rew": {"label": "Same × Rew", "color": "#e07b39", "marker": "o"},
        "H_Same_NoRew": {"label": "Same × NoRew", "color": "#f5c18a", "marker": "o"},
        "H_Other_Rew": {"label": "Other × Rew", "color": "#2a6496", "marker": "s"},
        "H_Other_NoRew": {"label": "Other × NoRew", "color": "#9ecae1", "marker": "s"},
    }
    subjects_window = sorted(coefs_window["subject_id"].unique())
    with a_lot_of_style():
        _fig, _axes = plt.subplots(
            1,
            len(subjects_window),
            figsize=(6 * len(subjects_window), 4),
            sharey=True,
            squeeze=False,
        )
        for _ax, _subject in zip(_axes[0], subjects_window):
            _sub = coefs_window[coefs_window["subject_id"] == _subject]
            _windows = sorted(_sub["window"].unique())
            _x = np.arange(len(_windows))
            _labels = [
                f"{int(window_bounds.loc[w, 'window_start'])}-{int(window_bounds.loc[w, 'window_end'])}"
                for w in _windows
            ]
            for _term, _cell_style in REWARD_CELLS_WINDOW.items():
                _rows = _sub[_sub["coef"] == _term].set_index("window")
                _vals = [
                    _rows.loc[w, "value"] if w in _rows.index else np.nan
                    for w in _windows
                ]
                _ax.plot(
                    _x,
                    _vals,
                    marker=_cell_style["marker"],
                    color=_cell_style["color"],
                    linewidth=2,
                    markersize=7,
                    label=_cell_style["label"],
                )
            _ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_labels, rotation=45, ha="right", fontsize=7)
            _ax.set_xlabel("Block window (pooled block range)")
            _ax.set_title(f"Subject {_subject}")
            _ax.legend(frameon=False, fontsize=8)
        _axes[0][0].set_ylabel("GLM coefficient")
        _fig.suptitle(
            f"One-hot reward cells timecourse over {window_blocks.value}-block windows (sessions pooled per animal)"
        )
        _fig.tight_layout()
    plt.show()
    return


@app.cell
def bias_by_odor_identity(a_lot_of_style, pd, plt, trials):
    _rs = trials[
        (trials["site_label"] == "RewardSite") & trials["block"].notna()
    ].copy()
    odor_block = (
        _rs.groupby(["subject_id", "session_id", "block", "odor_index"])
        .agg(
            p_stop=("has_choice", "mean"),
            n_trials=("has_choice", "count"),
            is_rewarded_odor=("is_rewarded_odor", "first"),
        )
        .reset_index()
    )
    odor_block = odor_block.sort_values(
        ["subject_id", "odor_index", "session_id", "block"]
    )
    pair_records = []
    for (_subject, odor), _grp in odor_block.groupby(["subject_id", "odor_index"]):
        _grp = _grp.reset_index(drop=True)
        for i in range(len(_grp) - 1):
            prev = _grp.iloc[i]
            curr = _grp.iloc[i + 1]
            pair_records.append(
                {
                    "subject_id": _subject,
                    "odor_index": int(odor),
                    "prev_rewarded": bool(prev["is_rewarded_odor"]),
                    "curr_rewarded": bool(curr["is_rewarded_odor"]),
                    "p_stop_curr": curr["p_stop"],
                    "n_trials_curr": int(curr["n_trials"]),
                }
            )
    pairs_df = pd.DataFrame(pair_records)
    CONDITIONS = [
        (True, True, "Prev Rew\n-> Curr Rew", "#c0392b"),
        (True, False, "Prev Rew\n-> Curr NoRew", "#e07b39"),
        (False, True, "Prev NoRew\n-> Curr Rew", "#1a5276"),
        (False, False, "Prev NoRew\n-> Curr NoRew", "#4f8fc0"),
    ]
    subjects_1 = sorted(pairs_df["subject_id"].unique())
    with a_lot_of_style():
        _fig, _axes = plt.subplots(
            1,
            len(subjects_1),
            figsize=(5 * len(subjects_1), 5),
            sharey=True,
            squeeze=False,
        )
        _fig.suptitle(
            "P(stop) at next odor encounter — 4 conditions\n(prev block rewarded / not) x (curr block rewarded / not)",
            fontsize=10,
        )
        for ai, _subject in enumerate(subjects_1):
            _ax = _axes[0][ai]
            _sub = pairs_df[pairs_df["subject_id"] == _subject]
            _ax.axvspan(-0.5, 1.5, color="#fff0eb", zorder=0)
            _ax.axvspan(1.5, 3.5, color="#eaf3fb", zorder=0)
            for _xi, (prev_rew, curr_rew, _label, _color) in enumerate(CONDITIONS):
                _grp = _sub[
                    (_sub["prev_rewarded"] == prev_rew)
                    & (_sub["curr_rewarded"] == curr_rew)
                ]["p_stop_curr"]
                _n = len(_grp)
                _m = _grp.mean() if _n > 0 else float("nan")
                se = _grp.sem() if _n > 1 else 0.0
                _ax.bar(_xi, _m, color=_color, alpha=0.85, width=0.65, zorder=2)
                if _m == _m:
                    _ax.errorbar(
                        _xi,
                        _m,
                        yerr=se,
                        fmt="none",
                        color="black",
                        capsize=5,
                        lw=1.5,
                        zorder=3,
                    )
                    _ax.text(
                        _xi,
                        min(_m + se + 0.04, 1.18),
                        f"n={_n}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        zorder=4,
                    )
            _ax.axvline(1.5, color="black", lw=1.0, alpha=0.3, zorder=1)
            _ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
            _ax.set_xticks(range(len(CONDITIONS)))
            _ax.set_xticklabels([c[2] for c in CONDITIONS], fontsize=8)
            _ax.set_ylim(0, 1.35)
            _ax.set_title(f"Subject {_subject}", fontsize=10)
            if ai == 0:
                _ax.set_ylabel("P(stop) in next block encounter")
            _ax.text(
                0.5,
                1.28,
                "Prev: Rewarded",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#8b0000",
                transform=_ax.get_xaxis_transform(),
            )
            _ax.text(
                2.5,
                1.28,
                "Prev: Not Rewarded",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#1a5276",
                transform=_ax.get_xaxis_transform(),
            )
        _fig.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def md_counterfactual(mo):
    mo.md(r"""
    ## Counterfactual learning

    After the **first stop of a block** the animal holds exactly one piece of evidence about the
    current odor mapping. A purely *factual* learner updates only the odor it just sampled. A
    *counterfactual* learner also updates the odor it did **not** sample — "rewarded here" implies
    "not rewarded there", and vice versa.

    So: split blocks by whether the first stop was rewarded, then score the decision at the **next
    encounter of each odor type** (strictly after that first stop). Four conditions per block:

    | first stop | next odor | ideal P(leave) | inference required |
    |---|---|---|---|
    | REW | next REW | 0 (blue) | factual — same odor, stop again |
    | REW | next NOREW | 1 (red) | **counterfactual** — the other odor must be unrewarded |
    | NOREW | next REW | 0 (blue) | **counterfactual** — the other odor must be rewarded |
    | NOREW | next NOREW | 1 (red) | factual — same odor, keep avoiding |

    The heatmaps show `p_leave` on a diverging map, so **perfect behaviour reads blue / red / blue /
    red** down the four conditions. If all four reach their target together the animal is acquiring a
    general rule; if the two counterfactual rows lag, learning is condition-specific.

    > Uses `trials_all` (pre `p_stay ∈ {0,1}` filter) — dropping `p_stay == 1` blocks would remove
    > exactly the "stopped at everything" blocks, i.e. the clearest failures of discrimination.
    """)
    return


@app.cell
def counterfactual_matrix(np, pd, trials_all):
    # ── Counterfactual learning ────────────────────────────────────────────────
    #
    # Idea: after the *first* stop of a block the animal has one piece of evidence
    # about the current odor mapping. A purely factual learner only updates the
    # odor it just sampled; a counterfactual learner also updates the odor it did
    # *not* sample (`rewarded here` implies `not rewarded there`, and vice versa).
    #
    # Each block is therefore split by whether its first stop was rewarded, and for
    # each split we score the animal's decision at the *next* encounter of each odor
    # type. Perfect behaviour is P(stop | rewarded odor) = 1 and
    # P(stop | non-rewarded odor) = 0 in *both* splits.
    #
    # This whole apparatus is kept inline (not a shared library function) since it
    # *is* the counterfactual analysis, not a generic utility -- downstream cells
    # receive the functions/constants they need as ordinary marimo cell outputs.

    #: Column order of the counterfactual matrix. Each entry is
    #: (first_stop_rewarded, next_odor_is_rewarded, short_label, ideal_p_stop).
    COUNTERFACTUAL_CELLS = [
        (True, True, "1st stop REW\n→ next REW", 1.0),
        (True, False, "1st stop REW\n→ next NOREW", 0.0),
        (False, True, "1st stop NOREW\n→ next REW", 1.0),
        (False, False, "1st stop NOREW\n→ next NOREW", 0.0),
    ]

    COUNTERFACTUAL_CELL_KEYS = [(a, b) for a, b, _, _ in COUNTERFACTUAL_CELLS]

    #: How each plottable value is rendered. `ideal` is the target value in each of
    #: the four COUNTERFACTUAL_CELLS columns; `cmap` is sequential for the
    #: polarity-corrected `accuracy` and diverging (red = stop-ish, blue =
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

    def _counterfactual_style(value):
        """Validate `value` and return its entry in _COUNTERFACTUAL_VALUE_STYLES."""
        try:
            return _COUNTERFACTUAL_VALUE_STYLES[value]
        except KeyError:
            raise ValueError(
                f"value must be one of {sorted(_COUNTERFACTUAL_VALUE_STYLES)}, got {value!r}"
            ) from None

    def _require_counterfactual_value(matrix, value):
        """Fail readably when `matrix` was built before `value` was a column."""
        if value not in matrix.columns:
            available = sorted(
                c for c in matrix.columns if c in _COUNTERFACTUAL_VALUE_STYLES
            )
            raise KeyError(
                f"{value!r} is not a column of the supplied matrix (it has {available}). "
                "Rebuild it with counterfactual_session_matrix(trials) -- a matrix held "
                "over from an earlier kernel state will not have the newer columns."
            )

    def _counterfactual_text_color(v, cmap):
        """Black on the light part of `cmap`, white elsewhere, for cell annotations."""
        light = v > 0.75 if cmap == "viridis" else 0.35 < v < 0.65
        return "black" if light else "white"

    def counterfactual_block_table(trials):
        """One row per block with the animal's decision at the next odor of each type.

        For every (session_id, block) the RewardSite trials are walked in temporal
        order and the first stop (has_choice) is located. That stop's has_reward
        defines first_stop_rewarded. Strictly *after* that trial we then find the
        first rewarded-odor site and the first non-rewarded-odor site and record
        whether the animal stopped there. Blocks where the animal never stopped are
        omitted entirely (no split can be assigned).
        """
        rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
        rs = rs.sort_values(["session_id", "block", "start_time"])
        # resolve the subject once per session rather than per block
        subject_map = (
            rs.drop_duplicates("session_id")
            .set_index("session_id")["subject_id"]
            .to_dict()
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
                """`has_choice` at the first post-first-stop site of this odor type."""
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

    def counterfactual_session_matrix(trials, min_blocks=3):
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

        # Reindex onto the full (session x 4 conditions) grid so missing cells show
        # up as gaps in the heatmap instead of silently shifting columns.
        sessions = agg[["subject_id", "session_id"]].drop_duplicates()
        grid = sessions.merge(
            pd.DataFrame(
                COUNTERFACTUAL_CELL_KEYS,
                columns=["first_stop_rewarded", "next_rewarded"],
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
        agg["accuracy"] = np.where(
            agg["next_rewarded"], agg["p_stop"], 1.0 - agg["p_stop"]
        )

        # session_id encodes the datetime -> lexicographic sort is chronological
        agg["session_date"] = agg["session_id"].str.split("_").str[1]
        agg = agg.sort_values(["subject_id", "session_id"])
        agg["session_index"] = agg.groupby("subject_id")["session_id"].transform(
            lambda s: s.rank(method="dense").astype(int) - 1
        )
        return agg

    def _counterfactual_pivot(matrix, subject, value):
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
    ):
        """Sessions x 4-condition heatmap of counterfactual behaviour, one panel per animal."""
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

    def counterfactual_cohort_average(matrix, value="accuracy", min_animals=1):
        """Average counterfactual_session_matrix across mice per session number.

        Sessions are aligned by session_index -- each animal's own 0-based
        chronological session number -- and averaged across animals.
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
        matrix,
        value="p_leave",
        min_animals=1,
        annotate=True,
        x_label="Session number (aligned across mice)",
        title=(
            "Counterfactual learning, averaged across mice at the same session number\n"
            "(error bars = SEM across animals; n per session falls off as animals run out)"
        ),
    ):
        """Cross-mouse counterfactual matrix, laid out with session number on the x axis."""
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
            """(4, n_sessions) array: one row per condition, one column per session."""
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
                label=f"{label.replace(chr(10), ' ')}  (ideal {ideal:g})",
            )

        # shade where fewer than half the animals still contribute, on both panels.
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

        # animal count per session number along the top
        ax_top = ax_hm.secondary_xaxis("top")
        ax_top.set_xticks(session_indices)
        ax_top.set_xticklabels(
            [str(int(n)) if np.isfinite(n) else "" for n in n_per_session],
            fontsize=5.5,
        )
        ax_top.set_xlabel(
            "animals contributing (max over the 4 conditions; a cell below "
            "min_blocks drops out, so this can dip and recover)",
            fontsize=8,
        )

        fig.suptitle(title, fontsize=11)
        return fig, cohort

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

    # One row per block: the first-stop split + whether the animal stopped at the
    # next rewarded-odor site and the next non-rewarded-odor site after it.
    cf_blocks = counterfactual_block_table(trials_all)

    # Aggregate over blocks within a session -> the four probabilities per session.
    cf = counterfactual_session_matrix(trials_all, min_blocks=3)

    print(f"{len(cf_blocks)} usable blocks over {cf['session_id'].nunique()} sessions")
    print("\nPooled across all sessions:")
    print(
        cf.groupby(["first_stop_rewarded", "next_rewarded"])[
            ["p_stop", "p_leave", "accuracy", "n_blocks"]
        ]
        .mean()
        .round(3)
        .to_string()
    )
    cf_blocks.head()
    return (
        COUNTERFACTUAL_CELLS,
        cf,
        counterfactual_session_matrix,
        counterfactual_session_trends,
        plot_counterfactual_cohort_average,
        plot_counterfactual_heatmap,
    )


@app.cell
def counterfactual_heatmap_per_animal(
    a_lot_of_style,
    cf,
    plot_counterfactual_heatmap,
    plt,
    trials_all,
):
    # Sessions (rows) x 4 conditions (columns), one panel per animal.
    # P(leave) on a diverging map: ideal is 0 (blue) in the "next REW" columns and
    # 1 (red) in the "next NOREW" columns, so perfect behaviour reads
    # blue / red / blue / red.
    with a_lot_of_style():
        _fig, _ = plot_counterfactual_heatmap(trials_all, value="p_leave", matrix=cf)
    plt.show()
    return


@app.cell
def counterfactual_trends_per_animal(
    COUNTERFACTUAL_CELLS, a_lot_of_style, cf, counterfactual_session_trends, np, plt
):
    from matplotlib.ticker import MaxNLocator

    CF_COLORS = ["#c0392b", "#e07b39", "#1a5276", "#4f8fc0"]
    subjects_2 = sorted(cf["subject_id"].unique())
    trends = counterfactual_session_trends(cf, value="accuracy")
    print(trends.round(4).to_string(index=False))
    with a_lot_of_style():
        _fig, _axes = plt.subplots(
            1,
            len(subjects_2),
            figsize=(4.2 * len(subjects_2), 4),
            sharey=True,
            squeeze=False,
        )
        for _ax, _subject in zip(_axes[0], subjects_2):
            _sub = cf[cf["subject_id"] == _subject]
            for (fsr, nr, _label, _), _color in zip(COUNTERFACTUAL_CELLS, CF_COLORS):
                g = (
                    _sub[
                        (_sub["first_stop_rewarded"] == fsr)
                        & (_sub["next_rewarded"] == nr)
                    ]
                    .dropna(subset=["accuracy"])
                    .sort_values("session_index")
                )
                _ax.plot(
                    g["session_index"],
                    g["accuracy"],
                    marker="o",
                    ms=4,
                    color=_color,
                    label=_label.replace("\n", " "),
                )
                if len(g) >= 3:
                    _x = g["session_index"].to_numpy(dtype=float)
                    _m, b = np.polyfit(_x, g["accuracy"].to_numpy(dtype=float), 1)
                    _ax.plot(_x, _m * _x + b, color=_color, ls="--", lw=1.2, alpha=0.7)
            _ax.axhline(0.5, color="gray", ls=":", lw=1)
            _ax.set_ylim(0, 1.05)
            _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            _ax.set_xlabel("Session index")
            _ax.set_title(f"Subject {_subject}", fontsize=10)
        _axes[0][0].set_ylabel("accuracy (higher = better)")
        _axes[0][-1].legend(frameon=False, fontsize=7, loc="lower right")
        _fig.suptitle(
            "Counterfactual accuracy across sessions (dashed = OLS fit)", fontsize=11
        )
        _fig.tight_layout()
    plt.show()
    return


@app.cell
def counterfactual_cohort_by_session(
    a_lot_of_style, cf, plot_counterfactual_cohort_average, plt
):
    # Average across mice at the same session number (each animal's own 0-based
    # session_index), so column k is "the cohort's k-th session". Each animal
    # contributes at most one value per cell -> no weighting by blocks run.
    # Only the longest-running animal reaches the highest indices, so the right-hand
    # columns are thin; the animal count runs along the top and the shaded region
    # marks where fewer than half the cohort remains.
    with a_lot_of_style():
        _fig, cf_cohort = plot_counterfactual_cohort_average(
            cf, value="p_leave", min_animals=1
        )
    plt.show()
    print(
        cf_cohort.pivot_table(
            index="session_index",
            columns=["first_stop_rewarded", "next_rewarded"],
            values=["mean", "n_animals"],
        )
        .round(3)
        .to_string()
    )
    return


@app.cell
def counterfactual_cohort_by_window(
    a_lot_of_style,
    counterfactual_session_matrix,
    min_blocks_window,
    plot_counterfactual_cohort_average,
    plt,
    skip_blocks,
    trials,
    window_blocks,
):
    from analysis.features import expand_to_block_windows as _expand_to_block_windows

    def counterfactual_window_matrix(trials, window_blocks, skip_blocks, min_blocks=3):
        """Per-block-window analogue of counterfactual_session_matrix (see the
        counterfactual_matrix cell above). Re-keys the trials rather than
        reimplementing the aggregation: session_id becomes a synthetic
        "{subject}_w{window:04d}" window key and block becomes the animal's
        global block ordinal, so the result has exactly the columns of
        counterfactual_session_matrix with session_id holding the window key.
        """
        expanded = _expand_to_block_windows(trials, window_blocks, skip_blocks)
        rekeyed = expanded.assign(
            session_id=expanded["subject_id"].astype(str)
            + "_w"
            + expanded["window"].map("{:04d}".format),
            block=expanded["block_ordinal"],
        )
        matrix = counterfactual_session_matrix(rekeyed, min_blocks=min_blocks)
        matrix["window"] = (
            matrix["session_id"].str.rsplit("_w", n=1).str[1].astype(int)
        )
        bounds = expanded.drop_duplicates("window")[
            ["window", "window_start", "window_end"]
        ]
        return matrix.merge(bounds, on="window", how="left")

    plot_cf_cohort = plot_counterfactual_cohort_average

    cf_window = counterfactual_window_matrix(
        trials,
        window_blocks.value,
        skip_blocks.value,
        min_blocks=min_blocks_window.value,
    )
    with a_lot_of_style():
        _fig, cf_cohort_window = plot_cf_cohort(
            cf_window,
            value="p_leave",
            min_animals=1,
            x_label=f"Block-window number ({window_blocks.value} blocks, stride {skip_blocks.value}; aligned across mice)",
            title=f"Counterfactual learning, averaged across mice at the same {window_blocks.value}-block window\n(sessions pooled per animal; error bars = SEM across animals; n falls off as animals run out of blocks)",
        )
    plt.show()
    _have = cf_window.groupby("session_index")["subject_id"].nunique()
    _contributing = (
        cf_cohort_window.groupby("session_index")["n_animals"]
        .max()
        .reindex(_have.index, fill_value=0)
    )
    _short = (_have - _contributing).pipe(lambda s: s[s > 0])
    print(
        f"{len(_short)} of {len(_have)} windows lose an animal to min_blocks={min_blocks_window.value} (window: animals lost)"
    )
    print(_short.to_string() if len(_short) else "  none")
    print(
        cf_cohort_window.pivot_table(
            index="session_index",
            columns=["first_stop_rewarded", "next_rewarded"],
            values=["mean", "n_animals"],
        )
        .round(3)
        .to_string()
    )
    return


@app.cell
def finalize(
    store,
    progress,
    run_id,
    started_at,
    build_manifest,
    git_commit,
    git_is_dirty,
    host_info,
    os,
    datetime,
    timezone,
):
    manifest = build_manifest(
        run_id=run_id,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        status="completed",
        git_commit=git_commit(),
        container_image=os.environ.get("CONTAINER_IMAGE"),
        python_version=host_info()["python_version"],
        # NOT `**host_info()` here -- host_info()'s "python_version" key
        # collides with the reserved key already passed above via the
        # explicit `python_version=` argument, and build_manifest's
        # `_reject_reserved` guard on `extra` raises ValueError on any such
        # collision. Pull out only the non-reserved field(s).
        extra={"git_dirty": git_is_dirty(), "hostname": host_info()["hostname"]},
    )
    store.write_json("manifest.json", manifest)
    progress.completed(stage="run")
    return (manifest,)


if __name__ == "__main__":
    app.run()
