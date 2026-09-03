import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def figure_output(figures_dir, mo):
    import itertools
    import re

    from matplotlib import pyplot as plt

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
        figures_dir.mkdir(parents=True, exist_ok=True)
        for _num in plt.get_fignums():
            _figure = plt.figure(_num)
            _path = (
                figures_dir
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
def imports_pathlib():
    from pathlib import Path

    return (Path,)


@app.cell
def imports_provenance():
    from analysis.artifacts import artifact_store_for_uri
    from analysis.run import (
        generate_run_id,
        build_manifest,
        git_commit,
        git_is_dirty,
        host_info,
    )
    from analysis.sessions import load_attached_datasets, build_inputs_manifest
    from datetime import datetime, timezone
    import os

    return (
        artifact_store_for_uri,
        build_inputs_manifest,
        build_manifest,
        datetime,
        generate_run_id,
        git_commit,
        git_is_dirty,
        host_info,
        load_attached_datasets,
        os,
        timezone,
    )


@app.cell
def run_setup(artifact_store_for_uri, datetime, generate_run_id, os, timezone):
    import logging
    import sys
    from pathlib import Path as _Path
    from analysis.logger import log as _log
    from obstore.store import S3Store

    # Named `run_setup` rather than `setup` -- marimo reserves the literal cell
    # name `setup` for its own special zero-argument "setup cell" concept, and
    # rejects a `setup` cell that (like this one) depends on other cells.

    # Route pipeline logs to stdout so they appear in marimo's cell-output area
    # (which only captures stdout) and are picked up by the `tee` in
    # scripts/start_prod.sh.  Idempotent: re-running this cell in dev won't
    # stack duplicate handlers.
    if not _log.handlers:
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        _log.addHandler(_handler)
    _log.setLevel(logging.INFO)

    run_id = os.environ.get("RUN_ID") or generate_run_id()
    started_at = datetime.now(timezone.utc).isoformat()
    artifact_uri = os.environ.get("ARTIFACT_URI", "./artifacts")
    artifact_store = artifact_store_for_uri(f"{artifact_uri}/runs/{run_id}")

    _local_root = getattr(artifact_store, "root", None)
    figures_dir = (
        (_local_root / "figures")
        if _local_root is not None
        else _Path("./scratch/figures")
    )

    _log.info("run_id=%s  artifacts → %s/runs/%s", run_id, artifact_uri, run_id)

    import json as _json

    _dataset_location = _json.loads(
        (_Path(__file__).parent.parent / "data_assets.json").read_text()
    )["attached_datasets"][0]["location"]
    store = S3Store.from_url(_dataset_location, region="us-west-2", skip_signature=True)
    return artifact_store, figures_dir, run_id, started_at, store


@app.cell
def session_viewer(mo, store):
    import io
    import pandas as _pd
    import obstore as _obs

    df = _pd.read_parquet(
        io.BytesIO(bytes(_obs.get(store, "session.parquet").bytes()))
    )
    mo.ui.table(df, pagination=True, page_size=20, selection=None, hidden_columns=["rig", "task_logic", "session", "trainer_state"])
    return (df,)


@app.cell
def _(df):
    print(df["curriculum_stage_name"].unique())
    return


@app.cell
def selection(
    Path,
    artifact_store,
    build_inputs_manifest,
    load_attached_datasets,
):
    from analysis.logger import log as _log

    # data_assets.json points at the already-processed dataset (see
    # scripts/attach_datasets.py and scripts/sync_and_process.py to regenerate it).
    attached = load_attached_datasets(Path(__file__).parent.parent / "data_assets.json")
    artifact_store.write_json("selection.json", {"attached_datasets": attached})

    inputs = build_inputs_manifest([entry["location"] for entry in attached])
    artifact_store.write_json("inputs.json", inputs)
    _log.info("resolved %d attached dataset(s) from data_assets.json", len(attached))
    return (attached,)


@app.cell
def load_and_prepare_trials(attached, df):
    import pandas as pd
    import numpy as np
    from analysis.features import prepare_trials
    from analysis.logger import log as _log
    from analysis.sessions import load_processed_table

    _log.info("loading and preparing trials…")
    # "sites" == trials (one row per site). The analysis doesn't need to know
    # whether the dataset lives on local disk or S3, signed or unsigned --
    # that's load_processed_table's job, not the analysis's.
    trials = load_processed_table(attached[0]["location"], "sites")
    trials, trials_all = prepare_trials(trials, df, degenerate_margin=0.1, end_frac=0.8)

    SUBJECT_IDS = sorted(trials["subject_id"].unique())
    return SUBJECT_IDS, np, pd, trials, trials_all


@app.cell
def curriculum_stage_datasets(df, mo, trials, trials_all):
    FULL_STAGES = [
        "LearningSets",
        "manual_LearningSets_v1.2.0_2Contrasts_5Rew_5NonRew",
        "manual_LearningSets_v1.2.0_2Contrasts_8Rew_10NonRew",
    ]
    ABREVERSAL_STAGES = ["manual_LearningSets_v1.2.0_ABReversal_odor0v1_5Rew_5NonRew"]

    _full_sessions = df.loc[df["curriculum_stage_name"].isin(FULL_STAGES), "session_id"]
    _abreversal_sessions = df.loc[
        df["curriculum_stage_name"].isin(ABREVERSAL_STAGES), "session_id"
    ]

    trials_full = trials[trials["session_id"].isin(_full_sessions)]
    trials_abreversal = trials[trials["session_id"].isin(_abreversal_sessions)]
    trials_all_full = trials_all[trials_all["session_id"].isin(_full_sessions)]
    trials_all_abreversal = trials_all[
        trials_all["session_id"].isin(_abreversal_sessions)
    ]

    print(
        f"Full: {trials_full['session_id'].nunique()} sessions, "
        f"{len(trials_full):,} trials"
    )
    print(
        f"ABReversal: {trials_abreversal['session_id'].nunique()} sessions, "
        f"{len(trials_abreversal):,} trials"
    )

    # Everything below this point analyses whichever curriculum stage is picked here.
    # Defaults to `--dataset` on the command line (e.g. `python workflows/pipeline.py
    # --dataset ABReversal`, or `marimo run workflows/pipeline.py -- --dataset ABReversal`);
    # in the interactive editor mo.cli_args() is empty, so it falls back to "Full" and
    # stays switchable via the radio.
    _DATASET_OPTIONS = ["Full", "ABReversal"]
    _cli_dataset = mo.cli_args().get("dataset", "Full")
    if _cli_dataset not in _DATASET_OPTIONS:
        raise ValueError(f"--dataset must be one of {_DATASET_OPTIONS}, got {_cli_dataset!r}")

    dataset_toggle = mo.ui.radio(
        options=_DATASET_OPTIONS,
        value=_cli_dataset,
        label="Curriculum stage dataset (used by all analysis cells below)",
    )
    dataset_toggle
    return (
        dataset_toggle,
        trials_abreversal,
        trials_all_abreversal,
        trials_all_full,
        trials_full,
    )


@app.cell
def dataset_selection(
    dataset_toggle,
    trials_abreversal,
    trials_all_abreversal,
    trials_all_full,
    trials_full,
):
    trials_selected = (
        trials_full if dataset_toggle.value == "Full" else trials_abreversal
    )
    trials_all_selected = (
        trials_all_full if dataset_toggle.value == "Full" else trials_all_abreversal
    )
    SUBJECT_IDS_SELECTED = sorted(trials_selected["subject_id"].unique())
    print(
        f"Analysing '{dataset_toggle.value}': "
        f"{trials_selected['session_id'].nunique()} sessions, "
        f"{len(trials_selected):,} trials, "
        f"{len(SUBJECT_IDS_SELECTED)} subjects"
    )
    return SUBJECT_IDS_SELECTED, trials_all_selected, trials_selected


@app.cell
def sql_over_trials(mo, trials_all):
    # Query over the in-memory `trials_all`.
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
def choice_by_block_position_pooled(SUBJECT_IDS_SELECTED, plt, trials_selected):
    from analysis.plotting import a_lot_of_style, plot_choice_by_block_position

    for _animal in SUBJECT_IDS_SELECTED:
        print(f"Animal {_animal}")
        _fig = plt.figure(figsize=(10, 6))
        _ax = _fig.add_subplot(111)
        with a_lot_of_style():
            plot_choice_by_block_position(
                trials_selected[trials_selected["subject_id"] == _animal], ax=_ax
            )
    plt.show()
    return (a_lot_of_style,)


@app.cell
def choice_by_block_position_per_session(a_lot_of_style, plt, trials_selected):
    from analysis.plotting import plot_choice_by_block_position_per_session

    with a_lot_of_style():
        plot_choice_by_block_position_per_session(trials_selected)

    plt.show()
    return


@app.cell
def choice_by_first_stop(a_lot_of_style, plt, trials_selected):
    from analysis.plotting import plot_choice_by_block_position_by_first_stop

    with a_lot_of_style():
        plot_choice_by_block_position_by_first_stop(trials_selected)

    plt.show()
    return


@app.cell
def choice_by_first_stop_overlay(a_lot_of_style, plt, trials_selected):
    from analysis.plotting import plot_choice_by_block_position_by_first_stop_overlay

    with a_lot_of_style():
        plot_choice_by_block_position_by_first_stop_overlay(trials_selected)

    plt.show()
    return


@app.cell
def choice_at_first_stops_across_sessions(
    SUBJECT_IDS_SELECTED,
    a_lot_of_style,
    np,
    plt,
    trials_selected,
):
    # P(choice) at the very first trial of every block, averaged within session, plotted across sessions
    _rs = trials_selected[
        (trials_selected["site_label"] == "RewardSite") & trials_selected["block"].notna()
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
            len(SUBJECT_IDS_SELECTED),
            figsize=(5 * len(SUBJECT_IDS_SELECTED), 4),
            sharey=True,
            squeeze=False,
        )
        for _ax, _animal in zip(_axes[0], SUBJECT_IDS_SELECTED):
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
def history_glm_per_window(skip_blocks, trials_selected, window_blocks):
    from analysis.features import expand_to_block_windows
    from analysis.glm import fit_history_glm, history_glm_features

    # 1-back within-block history GLM, fit per block-window instead of per
    # session. History is built *before* windowing, so a window boundary is
    # never mistaken for a block boundary (see history_glm_features).
    glm_features = history_glm_features(trials_selected)
    glm_windows = expand_to_block_windows(
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
        .agg(
            n_windows=("window", "nunique"),
            trials_per_window=("n_trials", "mean"),
            mean_cv_accuracy=("cv_accuracy", "mean"),
        )
        .round(3)
        .to_string()
    )
    return coefs_window, window_bounds


@app.cell
def history_glm_reward_cells_by_window(
    a_lot_of_style, coefs_window, plt, window_blocks, window_bounds
):
    from analysis.glm import plot_history_glm_reward_cells_by_window

    with a_lot_of_style():
        plot_history_glm_reward_cells_by_window(
            coefs_window, window_bounds, window_blocks.value
        )
    plt.show()
    return


@app.cell
def bias_by_odor_identity(a_lot_of_style, plt, trials_selected):
    from analysis.bias import odor_identity_bias_pairs, plot_bias_by_odor_identity

    pairs_df = odor_identity_bias_pairs(trials_selected)
    with a_lot_of_style():
        plot_bias_by_odor_identity(pairs_df)
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
def counterfactual_matrix(trials_all_selected):
    from analysis.counterfactual import counterfactual_block_table, counterfactual_session_matrix
    from analysis.logger import log as _log

    _log.info("computing counterfactual matrix…")
    cf_blocks = counterfactual_block_table(trials_all_selected)
    cf = counterfactual_session_matrix(trials_all_selected, min_blocks=3)

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
    return


@app.cell
def counterfactual_cohort_by_window(
    a_lot_of_style,
    min_blocks_window,
    plt,
    skip_blocks,
    trials_selected,
    window_blocks,
):
    from analysis.counterfactual import (
        counterfactual_cohort_average,
        counterfactual_window_matrix,
        plot_counterfactual_cohort_average,
        plot_counterfactual_cohort_by_condition,
    )

    cf_window = counterfactual_window_matrix(
        trials_selected,
        window_blocks.value,
        skip_blocks.value,
        min_blocks=min_blocks_window.value,
    )
    # min_animals=4: cut the 5-animal cohort average once fewer than 4 remain,
    # rather than trailing off on a shrinking sample (see the excl-864845
    # cell below for the same plot over the other 4 animals' full range).
    with a_lot_of_style():
        plot_counterfactual_cohort_average(
            cf_window,
            value="p_leave",
            min_animals=4,
            x_label=f"Block-window number ({window_blocks.value} blocks, stride {skip_blocks.value}; aligned across mice)",
            title=f"Counterfactual learning, averaged across mice at the same {window_blocks.value}-block window\n(error bars = bootstrapped 95% CI across animals; truncated once fewer than 4 animals remain)",
        )
    plt.show()
    with a_lot_of_style():
        plot_counterfactual_cohort_by_condition(
            cf_window,
            value="p_leave",
            min_animals=4,
            x_label=f"Block-window number ({window_blocks.value} blocks, stride {skip_blocks.value}; aligned across mice)",
            title=f"Counterfactual learning, averaged across mice at the same {window_blocks.value}-block window\n(faint = individual animals; bold line + shaded band = cohort mean & 95% CI; truncated once fewer than 4 animals remain)",
        )
    plt.show()
    cf_cohort_window = counterfactual_cohort_average(cf_window, value="p_leave", min_animals=4)
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
    return (cf_window,)


@app.cell
def counterfactual_cohort_by_window_with_chance(
    a_lot_of_style,
    cf_window,
    np,
    plt,
    skip_blocks,
    trials_selected,
    window_blocks,
):
    from analysis.counterfactual import first_site_chance_by_window
    from analysis.counterfactual import plot_counterfactual_cohort_average as _plot_cf_cohort_heatmap
    from analysis.counterfactual import plot_counterfactual_cohort_by_condition as _plot_cf_cohort
    from analysis.plotting import bootstrap_group_stats

    _MIN_ANIMALS = 4  # match the main cohort plot's cutoff for a like-for-like x-range
    _chance = first_site_chance_by_window(
        trials_selected, window_blocks.value, skip_blocks.value
    )
    _rng = np.random.default_rng(0)
    _chance_cohort = bootstrap_group_stats(
        _chance["stopped_first_site"], _chance["window"], _rng
    ).reset_index()
    _chance_cohort["n_animals"] = (
        _chance.groupby("window")["stopped_first_site"]
        .count()
        .reindex(_chance_cohort["window"])
        .to_numpy()
    )
    _chance_cohort = _chance_cohort[_chance_cohort["n_animals"] >= _MIN_ANIMALS]
    # Plotted as P(leave) -- the same quantity every subplot uses -- so it
    # shares the axis honestly: 1 - P(stop) at the block's first site. The
    # chance level isn't condition-specific, so the same line is overlaid on
    # all 4 subplots (and on the heatmap plot's line axis below).
    _chance_x = _chance_cohort["window"]
    _chance_mean = 1.0 - _chance_cohort["mean"]
    _chance_ci_lo = 1.0 - _chance_cohort["ci_hi"]
    _chance_ci_hi = 1.0 - _chance_cohort["ci_lo"]

    with a_lot_of_style():
        _fig0, _cohort0, _ax_ln = _plot_cf_cohort_heatmap(
            cf_window,
            value="p_leave",
            min_animals=_MIN_ANIMALS,
            x_label=f"Block-window number ({window_blocks.value} blocks, stride {skip_blocks.value}; aligned across mice)",
            title=f"Counterfactual learning, averaged across mice at the same {window_blocks.value}-block window\n(error bars = bootstrapped 95% CI across animals; dotted = chance level)",
        )
        _ax_ln.plot(
            _chance_x, _chance_mean, color="black", ls=":", lw=1.6, zorder=5,
            label="Chance (1 − P(stop) @ first site of block)",
        )
        _ax_ln.fill_between(
            _chance_x, _chance_ci_lo, _chance_ci_hi,
            color="black", alpha=0.12, linewidth=0, zorder=4,
        )
        _ax_ln.legend(frameon=False, fontsize=7, loc="best")
    plt.show()

    with a_lot_of_style():
        _fig, _axes = _plot_cf_cohort(
            cf_window,
            value="p_leave",
            min_animals=_MIN_ANIMALS,
            x_label=f"Block-window number ({window_blocks.value} blocks, stride {skip_blocks.value}; aligned across mice)",
            title=f"Counterfactual learning, averaged across mice at the same {window_blocks.value}-block window\n(faint = individual animals; bold + shaded = cohort mean & 95% CI; dotted = chance level)",
        )
        for _ax in _axes:
            _ax.plot(
                _chance_x, _chance_mean, color="black", ls=":", lw=1.6, zorder=5,
                label="Chance (1 − P(stop) @ first site of block)",
            )
            _ax.fill_between(
                _chance_x, _chance_ci_lo, _chance_ci_hi,
                color="black", alpha=0.12, linewidth=0, zorder=4,
            )
        _axes[-1].legend(frameon=False, fontsize=7, loc="best")
    plt.show()
    return


@app.cell
def counterfactual_cohort_by_window_excl_864845(
    a_lot_of_style,
    cf_window,
    plt,
    skip_blocks,
    window_blocks,
):
    from analysis.counterfactual import plot_counterfactual_cohort_average as _plot_cf_cohort_heatmap
    from analysis.counterfactual import plot_counterfactual_cohort_by_condition as _plot_cf_cohort

    # Same cohort average as above, but with subject 864845 dropped -- the
    # noisiest of the five (see history_glm_reward_cells_by_window) and part
    # of why min_animals=4 truncates the main plot early.
    _cf_window_excl = cf_window[cf_window["subject_id"].astype(str) != "864845"]
    with a_lot_of_style():
        _plot_cf_cohort_heatmap(
            _cf_window_excl,
            value="p_leave",
            min_animals=1,
            x_label=f"Block-window number ({window_blocks.value} blocks, stride {skip_blocks.value}; aligned across mice)",
            title=f"Exclude animal 864845 — counterfactual learning averaged across mice at the same {window_blocks.value}-block window\n(error bars = bootstrapped 95% CI across animals)",
        )
    plt.show()
    with a_lot_of_style():
        _plot_cf_cohort(
            _cf_window_excl,
            value="p_leave",
            min_animals=1,
            x_label=f"Block-window number ({window_blocks.value} blocks, stride {skip_blocks.value}; aligned across mice)",
            title=f"Exclude animal 864845 — counterfactual learning averaged across mice at the same {window_blocks.value}-block window\n(faint = individual animals; bold line + shaded band = cohort mean & 95% CI)",
        )
    plt.show()
    return


@app.cell
def counterfactual_trends_per_animal(
    a_lot_of_style, cf_window, plt, skip_blocks, window_blocks
):
    from analysis.counterfactual import (
        counterfactual_session_trends,
        plot_counterfactual_trends_per_animal,
    )

    # counterfactual_session_trends is generic over whatever "session_index"
    # column its matrix has, and cf_window's session_index already equals its
    # window number, so it's reused as-is; only the "n_sessions" column in
    # its output is really "n_windows" here.
    trends = counterfactual_session_trends(cf_window, value="accuracy").rename(
        columns={"n_sessions": "n_windows"}
    )
    print(trends.round(4).to_string(index=False))
    with a_lot_of_style():
        plot_counterfactual_trends_per_animal(
            cf_window, window_blocks.value, skip_blocks.value
        )
    plt.show()
    return


@app.cell
def counterfactual_heatmap_per_animal(
    a_lot_of_style, cf_window, plt, skip_blocks, trials_selected, window_blocks
):
    from analysis.counterfactual import plot_counterfactual_heatmap

    # Rows are block-windows -- the same window/stride the GLM-by-window fit
    # uses -- rather than raw (variable-length) sessions, so animals share an
    # "amount of experience" axis the way the GLM weight plots do.
    with a_lot_of_style():
        _fig, _ = plot_counterfactual_heatmap(
            trials_selected,
            value="p_leave",
            matrix=cf_window,
            ylabel=f"Block-window ({window_blocks.value} blocks, stride {skip_blocks.value})",
            row_label_fn=lambda s: s.rsplit("_w", 1)[1],
        )
    plt.show()
    return


@app.cell
def finalize(
    artifact_store,
    build_manifest,
    datetime,
    git_commit,
    git_is_dirty,
    host_info,
    os,
    run_id,
    started_at,
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
    from analysis.logger import log as _log

    artifact_store.write_json("manifest.json", manifest)
    _log.info("complete — run_id=%s", run_id)
    return


if __name__ == "__main__":
    app.run()
