import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def load_dataset():
    from pathlib import Path

    import marimo as mo
    from analysis.sessions import Dataset

    root = Path(__file__).parent.parent
    dataset = Dataset.from_manifests(
        root / "data_assets.json",
        root / "raw_sessions.json",
    )
    sessions, sites = dataset.session, dataset.sites
    return dataset, mo, sessions, sites


@app.cell
def preprocess(sessions, sites):
    from analysis.features import prepare_trials

    trials, trials_all = prepare_trials(
        sites, sessions, degenerate_margin=0.1, end_frac=0.8
    )
    return trials, trials_all


@app.cell
def curriculum_selector(mo, sessions):
    from analysis.dataset_selection import (
        DATASET_OPTIONS,
        curriculum_stage_session_ids,
    )

    session_ids_by_dataset = curriculum_stage_session_ids(sessions)

    dataset_toggle = mo.ui.radio(
        options=DATASET_OPTIONS,
        value="Full",
        label="Curriculum stage dataset",
    )
    dataset_toggle
    return dataset_toggle, session_ids_by_dataset


@app.cell
def select_curriculum(
    dataset_toggle,
    session_ids_by_dataset,
    trials,
    trials_all,
):
    from analysis.dataset_selection import select_trials_by_session

    selected_sessions = session_ids_by_dataset[dataset_toggle.value]
    selected_trials = select_trials_by_session(trials, selected_sessions)
    selected_trials_all = select_trials_by_session(trials_all, selected_sessions)
    print(
        f"{dataset_toggle.value}: {selected_sessions.nunique()} sessions, "
        f"{len(selected_trials):,} preprocessed trials"
    )
    return selected_trials, selected_trials_all


@app.cell
def dataset_summary(selected_trials):
    from analysis.dataset_selection import summarize_dataset

    summary = summarize_dataset(selected_trials)
    return (summary,)


@app.cell
def show_dataset_summary(mo, summary):
    mo.vstack(
        [
            mo.md("## Days and block transitions by animal"),
            mo.ui.table(summary, selection=None),
            mo.md(
                "A block transition is a change from one block to the next within a session. "
                "Counts use the selected trials, with degenerate blocks removed."
            ),
        ]
    )
    return


@app.cell
def choice_by_block_position_raw(mo, selected_trials_all):
    from analysis.plotting import (
        plot_choice_by_block_position_raw as _plot_choice_by_block_position_raw,
    )
    from analysis.plotting import plot_paired_appearance as _plot_paired_appearance
    from analysis.plotting_style import animal_palette as _animal_palette
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    # This plot only: use the pre-degenerate-filter trials (still trimmed to
    # the first 80% of each session) rather than `selected_trials` -- the
    # degenerate p_stay<=0.1/>=0.9 block filter isn't driving the position-0
    # odor-identity-bias effect, so it's left in here for the raw picture.
    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_choice_by_block_position_raw(
            selected_trials_all, ax=_ax, colors={True: _orange, False: _blue}
        )
        _fig.tight_layout()

        # Small paired (slopegraph) inset: P(Choice) at position 0 vs 9, one
        # point per animal, connected -- both odor reward statuses on one
        # shared axis (a top color bar marks which group is which) rather
        # than two axes, to check the pooled band above holds
        # animal-by-animal.
        _fig_paired, _ax_paired = _new_figure("standard")
        _plot_paired_appearance(
            selected_trials_all,
            0,
            9,
            ax=_ax_paired,
            colors=_animal_palette(),
            group_colors={True: _orange, False: _blue},
        )
        _fig_paired.tight_layout()

    mo.vstack([_fig, _fig_paired])
    return


@app.cell
def choice_by_block_position_raw_by_tercile(selected_trials_all):
    from analysis.plotting import (
        plot_choice_by_block_position_raw_by_tercile as _plot_choice_by_block_position_raw_by_tercile,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _new_figure("wide", ncols=3, sharey=True)
        _blue, _orange = _categorical(2)
        _plot_choice_by_block_position_raw_by_tercile(
            selected_trials_all, axes=_axes, colors={True: _orange, False: _blue}
        )
        for _ax in _axes:
            _xlabel = _ax.get_xlabel()
            if _xlabel:
                _ax.set_title(_xlabel)
                _ax.set_xlabel("")
        _fig.tight_layout()
    _fig
    return


@app.cell
def choice_by_odor_appearance(selected_trials):
    from analysis.plotting import (
        plot_choice_by_odor_appearance as _plot_choice_by_odor_appearance,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_choice_by_odor_appearance(
            selected_trials, ax=_ax, colors={True: _orange, False: _blue}, n_blocks=30
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def choice_by_odor_appearance_by_animal(selected_trials):
    from analysis.plotting import (
        plot_choice_by_odor_appearance_by_animal as _plot_choice_by_odor_appearance_by_animal,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _new_figure("square", nrows=2, ncols=3)
        _blue, _orange = _categorical(2)
        _plot_choice_by_odor_appearance_by_animal(
            selected_trials, axes=_axes.flat, colors={True: _orange, False: _blue}, n_blocks=30
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def choice_by_odor_appearance_from_first_stop(selected_trials):
    from analysis.plotting import (
        plot_choice_by_odor_appearance as _plot_choice_by_odor_appearance,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_choice_by_odor_appearance(
            selected_trials,
            ax=_ax,
            colors={True: _orange, False: _blue},
            from_first_stop=True,
            n_blocks=30,
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def choice_by_odor_appearance_first_stop_rewarded(selected_trials):
    from analysis.features import label_first_stop as _label_first_stop
    from analysis.plotting import (
        plot_choice_by_odor_appearance as _plot_choice_by_odor_appearance,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    _labeled_trials = _label_first_stop(selected_trials)
    _cond_trials = _labeled_trials[_labeled_trials["first_stop_rewarded"] == True]

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_choice_by_odor_appearance(
            _cond_trials,
            ax=_ax,
            colors={True: _orange, False: _blue},
            from_first_stop=True,
            n_blocks=None,
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def choice_by_odor_appearance_first_stop_nonrewarded(selected_trials):
    from analysis.features import label_first_stop as _label_first_stop
    from analysis.plotting import (
        plot_choice_by_odor_appearance as _plot_choice_by_odor_appearance,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    _labeled_trials = _label_first_stop(selected_trials)
    _cond_trials = _labeled_trials[_labeled_trials["first_stop_rewarded"] == False]

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_choice_by_odor_appearance(
            _cond_trials,
            ax=_ax,
            colors={True: _orange, False: _blue},
            from_first_stop=True,
            n_blocks=None,
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def choice_by_odor_appearance_first_stop_rewarded_by_blocks(selected_trials):
    from analysis.features import label_first_stop as _label_first_stop
    from analysis.plotting import (
        plot_choice_by_odor_appearance as _plot_choice_by_odor_appearance,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    _labeled_trials = _label_first_stop(selected_trials)
    _cond_trials = _labeled_trials[_labeled_trials["first_stop_rewarded"] == True]

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_choice_by_odor_appearance(
            _cond_trials,
            ax=_ax,
            colors={True: _orange, False: _blue},
            from_first_stop=True,
            n_blocks=30,
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def choice_by_odor_appearance_first_stop_nonrewarded_by_blocks(
    selected_trials,
):
    from analysis.features import label_first_stop as _label_first_stop
    from analysis.plotting import (
        plot_choice_by_odor_appearance as _plot_choice_by_odor_appearance,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    _labeled_trials = _label_first_stop(selected_trials)
    _cond_trials = _labeled_trials[_labeled_trials["first_stop_rewarded"] == False]

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_choice_by_odor_appearance(
            _cond_trials,
            ax=_ax,
            colors={True: _orange, False: _blue},
            from_first_stop=True,
            n_blocks=30,
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def counterfactual_p_stop_after_bad_first_stop_by_window(selected_trials):
    from analysis.counterfactual import (
        plot_counterfactual_p_stop_by_window as _plot_counterfactual_p_stop_by_window,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_counterfactual_p_stop_by_window(
            selected_trials,
            first_stop_rewarded=False,
            window_blocks=30,
            skip_blocks=15,
            ax=_ax,
            reward_colors={True: _orange, False: _blue},
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def counterfactual_p_stop_after_good_first_stop_by_window(selected_trials):
    from analysis.counterfactual import (
        plot_counterfactual_p_stop_by_window as _plot_counterfactual_p_stop_by_window,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_counterfactual_p_stop_by_window(
            selected_trials,
            first_stop_rewarded=True,
            window_blocks=30,
            skip_blocks=15,
            ax=_ax,
            reward_colors={True: _orange, False: _blue},
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def counterfactual_p_stop_first_last_scatter(selected_trials):
    from analysis.counterfactual import (
        plot_counterfactual_p_stop_first_last_scatter as _plot_counterfactual_p_stop_first_last_scatter,
    )
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _ax = _new_figure("square")
        _blue, _orange = _categorical(2)
        _plot_counterfactual_p_stop_first_last_scatter(
            selected_trials,
            n_blocks=30,
            last_n_blocks=100,
            ax=_ax,
            reward_colors={True: _orange, False: _blue},
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def bias_first_site(selected_trials):
    from analysis.bias import bias_first_site_pairs as _bias_first_site_pairs
    from analysis.bias import plot_bias_first_site as _plot_bias_first_site
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    _pairs_df = _bias_first_site_pairs(selected_trials)

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _blue, _orange = _categorical(2)
        _plot_bias_first_site(
            _pairs_df, ax=_ax, reward_colors={True: _orange, False: _blue}
        )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def p_stop_first_site_by_session(selected_trials):
    from analysis.counterfactual import (
        plot_p_stop_hazard_by_session as _plot_p_stop_hazard_by_session,
    )
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _plot_p_stop_hazard_by_session(selected_trials, position=0, ax=_ax)
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def p_stop_hazard_second_site_by_session(selected_trials):
    from analysis.counterfactual import (
        plot_p_stop_hazard_by_session as _plot_p_stop_hazard_by_session,
    )
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _plot_p_stop_hazard_by_session(selected_trials, position=1, ax=_ax)
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def p_stop_hazard_third_site_by_session(selected_trials):
    from analysis.counterfactual import (
        plot_p_stop_hazard_by_session as _plot_p_stop_hazard_by_session,
    )
    from analysis.plotting_style import new_figure as _new_figure
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _ax = _new_figure("standard")
        _plot_p_stop_hazard_by_session(selected_trials, position=2, ax=_ax)
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def history_glm_fit(selected_trials):
    from analysis.counterfactual import first_site_chance_by_window
    from analysis.features import expand_to_block_windows
    from analysis.glm import ONEHOT_CELL_STYLE, fit_history_glm, history_glm_features

    glm_window_blocks = 100
    glm_skip_blocks = 20

    _glm_features = history_glm_features(selected_trials)
    glm_windows = expand_to_block_windows(_glm_features, glm_window_blocks, glm_skip_blocks)
    glm_coefs_window = fit_history_glm(
        glm_windows, unit_col=["subject_id", "window"], coefs=list(ONEHOT_CELL_STYLE)
    )
    glm_window_bounds = (
        glm_windows.drop_duplicates("window")
        .set_index("window")[["window_start", "window_end"]]
        .sort_index()
    )
    glm_chance = first_site_chance_by_window(selected_trials, glm_window_blocks, glm_skip_blocks)
    print(
        f"{glm_window_blocks}-block windows, stride {glm_skip_blocks} -> "
        f"{len(glm_window_bounds)} window positions"
    )
    return (
        glm_chance,
        glm_coefs_window,
        glm_skip_blocks,
        glm_window_blocks,
        glm_window_bounds,
        glm_windows,
    )


@app.cell(hide_code=True)
def history_glm_weights(glm_coefs_window, glm_window_bounds):
    from analysis.glm import plot_history_glm_cohort_by_window as _plot_history_glm_cohort_by_window
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot_history_glm_cohort_by_window(
            glm_coefs_window, glm_window_bounds, value="weight"
        )
    _fig
    return


@app.cell(hide_code=True)
def history_glm_p_choice(glm_window_bounds, glm_windows):
    from analysis.glm import plot_history_glm_empirical_by_window as _plot_history_glm_empirical_by_window
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot_history_glm_empirical_by_window(
            glm_windows, glm_window_bounds, value="p_choice"
        )
    _fig
    return


@app.cell(hide_code=True)
def history_glm_p_choice_minus_chance(
    glm_chance,
    glm_window_bounds,
    glm_windows,
):
    from analysis.glm import plot_history_glm_empirical_by_window as _plot_history_glm_empirical_by_window
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot_history_glm_empirical_by_window(
            glm_windows, glm_window_bounds, value="p_choice_minus_chance", chance=glm_chance
        )
    _fig
    return


@app.cell(hide_code=True)
def counterfactual_cohort_minus_chance(selected_trials):
    import matplotlib.pyplot as _plt
    from analysis.counterfactual import (
        plot_counterfactual_cohort_minus_chance as _plot_counterfactual_cohort_minus_chance,
    )
    from analysis.plotting_style import presentation_style as _presentation_style

    # columns: all animals / excluding 864845; rows: minus chance / raw P(Stop)
    with _presentation_style():
        _fig, _axes = _plt.subplots(2, 2, figsize=(16, 9), sharex="col")
        for _col, _exclude in enumerate([(), ["864845"]]):
            for _row, _subtract in enumerate([True, False]):
                _plot_counterfactual_cohort_minus_chance(
                    selected_trials, window_blocks=100, skip_blocks=20,
                    exclude_subjects=_exclude, subtract_chance=_subtract,
                    ax=_axes[_row][_col],
                )
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def history_glm_first_stop_fit(
    glm_skip_blocks,
    glm_window_blocks,
    selected_trials,
):
    from analysis.features import expand_to_block_windows as _expand_to_block_windows
    from analysis.glm import ONEHOT_CELL_STYLE as _ONEHOT_CELL_STYLE
    from analysis.glm import fit_history_glm as _fit_history_glm
    from analysis.glm import history_glm_after_first_stop as _after_first_stop
    from analysis.glm import history_glm_features as _history_glm_features

    # one trial per block: the trial right after the block's first stop. Windows
    # are built over all blocks first, then filtered, so they still span 100
    # blocks each. Only the four Rew/NoRew one-hot cells are fit -- the Skip
    # cells are empty when the previous trial is a stop by construction.
    _feats = _history_glm_features(selected_trials)
    glm1_windows = _after_first_stop(
        _expand_to_block_windows(_feats, glm_window_blocks, glm_skip_blocks)
    )
    glm1_coefs_window = _fit_history_glm(
        glm1_windows, unit_col=["subject_id", "window"], coefs=[_c for _c in _ONEHOT_CELL_STYLE if not _c.endswith("_Skip")]
    )
    glm1_window_bounds = (
        glm1_windows.drop_duplicates("window")
        .set_index("window")[["window_start", "window_end"]]
        .sort_index()
    )
    return glm1_coefs_window, glm1_window_bounds, glm1_windows


@app.cell(hide_code=True)
def history_glm_first_stop_weights(glm1_coefs_window, glm1_window_bounds):
    from analysis.glm import plot_history_glm_cohort_by_window as _plot
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot(glm1_coefs_window, glm1_window_bounds, value="weight")
    _fig
    return


@app.cell(hide_code=True)
def history_glm_first_stop_p_choice(glm1_window_bounds, glm1_windows):
    from analysis.glm import plot_history_glm_empirical_by_window as _plot
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot(glm1_windows, glm1_window_bounds, value="p_choice")
    _fig
    return


@app.cell(hide_code=True)
def history_glm_first_stop_p_choice_minus_chance(
    glm1_window_bounds,
    glm1_windows,
    glm_chance,
):
    from analysis.glm import plot_history_glm_empirical_by_window as _plot
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot(glm1_windows, glm1_window_bounds, value="p_choice_minus_chance", chance=glm_chance)
    _fig
    return


@app.cell
def block_identity_counts(selected_trials):
    import matplotlib.pyplot as _plt
    from analysis.bias import block_identity_table as _block_identity_table
    from analysis.bias import plot_block_identity_counts as _plot_block_identity_counts
    from analysis.plotting_style import presentation_style as _presentation_style

    _blocks = _block_identity_table(selected_trials)

    with _presentation_style():
        _fig, _axes = _plt.subplots(2, 3, figsize=(15, 9))
        _plot_block_identity_counts(_blocks, axes=_axes)
        _fig.tight_layout()
    _fig
    return


@app.cell
def include_degenerate_toggle(mo):
    include_degenerate = mo.ui.switch(label="Include blocks with p_stay <= 0.1 / >= 0.9 (no degenerate-block exclusion)")
    include_degenerate
    return (include_degenerate,)


@app.cell
def occurrence_vs_second_trial(
    include_degenerate,
    selected_trials,
    selected_trials_all,
):
    import matplotlib.pyplot as _plt
    from analysis.bias import block_identity_second_trial_table as _second_trial_table
    from analysis.bias import plot_occurrence_vs_second_trial as _plot_occ_vs_second
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import presentation_style as _presentation_style

    _trials = selected_trials_all if include_degenerate.value else selected_trials
    _table = _second_trial_table(_trials)

    with _presentation_style():
        _fig, _axes = _plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
        _blue, _orange = _categorical(2)
        _plot_occ_vs_second(_table, colors=(_blue, _orange), axes=_axes)
        _fig.tight_layout()
    _fig
    return


@app.cell
def session_subset_toggle(mo):
    session_subset = mo.ui.radio(
        options=["All sessions", "Last tercile of sessions"],
        value="All sessions",
        label="Sessions per animal",
    )
    session_subset
    return (session_subset,)


@app.cell
def session_subset_trials(selected_trials, session_subset):
    # Each animal's sessions in chronological order (session ids sort by date);
    # the last tercile is its final third of them.
    if session_subset.value == "All sessions":
        analysis_trials = selected_trials
    else:
        _keep = []
        for _subject, _sids in selected_trials.groupby("subject_id")["session_id"]:
            _ordered = sorted(_sids.unique())
            _keep += _ordered[len(_ordered) - -(-len(_ordered) // 3) :]
        analysis_trials = selected_trials[selected_trials["session_id"].isin(_keep)]
    print(
        f"{session_subset.value}: {analysis_trials['session_id'].nunique()} sessions "
        f"({analysis_trials.groupby('subject_id')['session_id'].nunique().to_dict()})"
    )
    return (analysis_trials,)


@app.cell
def velocity_aligned_to_site(analysis_trials, dataset):
    import numpy as np
    import pandas as pd

    VEL_PRE, VEL_POST, VEL_DT = 1.0, 3.0, 0.02  # seconds around site onset
    vel_t = np.arange(-VEL_PRE, VEL_POST + VEL_DT / 2, VEL_DT)

    # Reward sites are the only ones with a choice; odor_index is 0..6.
    _sub = analysis_trials[analysis_trials["site_label"] == "RewardSite"]

    # One row per (animal, session, odor, has_choice): trial-mean velocity trace.
    _rows = []
    for (_subject, _sid), _tr in _sub.groupby(["subject_id", "session_id"]):
        _v = dataset.load_table("position_velocity", [_sid])
        _ts, _vel = _v["timestamp"].to_numpy(), _v["velocity"].to_numpy()
        _idx = np.searchsorted(_ts, _tr["start_time"].to_numpy()[:, None] + vel_t)
        _idx = np.clip(_idx, 0, len(_ts) - 1)
        _mat = _vel[_idx]  # (n_trials, n_time)
        for (_odor, _choice), _g in _tr.reset_index(drop=True).groupby(
            ["odor_index", "has_choice"]
        ):
            _rows.append(
                (_subject, _sid, int(_odor), bool(_choice), len(_g), _mat[_g.index].mean(axis=0))
            )

    vel_by_session = pd.DataFrame(
        _rows,
        columns=["subject_id", "session_id", "odor_index", "has_choice", "n_trials", "trace"],
    )
    print(vel_by_session.groupby("subject_id")["session_id"].nunique())
    return np, pd, vel_by_session, vel_t


@app.cell
def velocity_aligned_plot(np, pd, vel_by_session, vel_t):
    import matplotlib.pyplot as _plt
    from analysis.plotting import bootstrap_group_stats as _bootstrap_group_stats
    from analysis.plotting import plot_mean_ci_band as _plot_mean_ci_band
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import presentation_style as _presentation_style

    _rng = np.random.default_rng(0)
    _blue, _orange = _categorical(2)
    _colors = {True: _orange, False: _blue}
    _labels = {True: "Choice", False: "No choice"}
    _subjects = sorted(vel_by_session["subject_id"].unique())

    with _presentation_style():
        _fig, _axes = _plt.subplots(
            len(_subjects), 7, figsize=(24, 3 * len(_subjects)), sharex=True, sharey="row",
            squeeze=False,
        )
        for _row, _subject in enumerate(_subjects):
            for _ax, _odor in zip(_axes[_row], range(7)):
                for _choice in (False, True):
                    _d = vel_by_session[
                        (vel_by_session["subject_id"] == _subject)
                        & (vel_by_session["odor_index"] == _odor)
                        & (vel_by_session["has_choice"] == _choice)
                    ]
                    if _d.empty:
                        continue
                    # One animal per row, so there is no across-animal N: the CI
                    # bootstraps this animal's per-session means instead.
                    _traces = np.stack(_d["trace"].to_numpy())  # (n_sessions, n_time)
                    _long = pd.DataFrame({"v": _traces.ravel(), "t": np.tile(vel_t, len(_d))})
                    _stats = _bootstrap_group_stats(_long["v"], _long["t"], _rng, n_boot=500)
                    _plot_mean_ci_band(_ax, _stats, _colors[_choice], label=_labels[_choice])
                _ax.axvline(0, color="#898781", lw=1)
                # Odor label as in-axes text (presentation figures carry no titles).
                if _row == 0:
                    _ax.text(0.5, 0.97, f"Odor {_odor}", transform=_ax.transAxes,
                             ha="center", va="top")
                if _row == len(_subjects) - 1:
                    _ax.set_xlabel("Time from site onset (s)")
            _axes[_row, 0].set_ylabel(f"{_subject} velocity")
        _axes[0, -1].legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
        _fig.tight_layout()
    _fig
    return


@app.cell
def velocity_by_condition_plot(np, pd, vel_by_session, vel_t):
    import matplotlib.pyplot as _plt
    from analysis.plotting import bootstrap_group_stats as _bootstrap_group_stats
    from analysis.plotting import plot_mean_ci_band as _plot_mean_ci_band
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import presentation_style as _presentation_style

    _rng = np.random.default_rng(0)
    _odor_colors = dict(zip(range(7), _categorical(7)))  # same color per odor everywhere
    _conditions = {False: "No choice", True: "Choice"}
    _subjects = sorted(vel_by_session["subject_id"].unique())

    with _presentation_style():
        _fig, _axes = _plt.subplots(
            2, len(_subjects), figsize=(4 * len(_subjects), 7), sharex=True, sharey="col",
            squeeze=False,
        )
        for _col, _subject in enumerate(_subjects):
            for _ax, (_choice, _cond_label) in zip(_axes[:, _col], _conditions.items()):
                for _odor, _color in _odor_colors.items():
                    _d = vel_by_session[
                        (vel_by_session["subject_id"] == _subject)
                        & (vel_by_session["odor_index"] == _odor)
                        & (vel_by_session["has_choice"] == _choice)
                    ]
                    if _d.empty:
                        continue
                    # One animal per column, so there is no across-animal N: the CI
                    # bootstraps this animal's per-session means instead.
                    _traces = np.stack(_d["trace"].to_numpy())  # (n_sessions, n_time)
                    _long = pd.DataFrame({"v": _traces.ravel(), "t": np.tile(vel_t, len(_d))})
                    _stats = _bootstrap_group_stats(_long["v"], _long["t"], _rng, n_boot=500)
                    _plot_mean_ci_band(_ax, _stats, _color, label=f"Odor {_odor}")
                _ax.axvline(0, color="#898781", lw=1)
                if _cond_label == "No choice":
                    _ax.text(0.5, 0.97, _subject, transform=_ax.transAxes,
                             ha="center", va="top")
                if _col == 0:
                    _ax.set_ylabel(f"{_cond_label} velocity")
            _axes[-1, _col].set_xlabel("Time from site onset (s)")
        _axes[0, -1].legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), ncol=2)
        _fig.tight_layout()
    _fig
    return


@app.cell
def lick_rate_aligned_to_choice_cue(analysis_trials, dataset, np, pd):
    LICK_PRE, LICK_POST, LICK_BIN = 4.0, 5.0, 0.1  # seconds around choice cue
    lick_edges = np.arange(-LICK_PRE, LICK_POST + LICK_BIN / 2, LICK_BIN)
    lick_t = lick_edges[:-1] + LICK_BIN / 2  # bin centers

    _chosen = analysis_trials[analysis_trials["has_choice"]]

    # One row per (animal, session, rewarded): trial-mean lick rate (Hz) per bin.
    _rows = []
    for (_subject, _sid), _tr in _chosen.groupby(["subject_id", "session_id"]):
        _l = dataset.load_table("licks", [_sid])
        _onsets = np.sort(_l.loc[_l["is_lick_onset"], "timestamp"].to_numpy())
        _tr = _tr.reset_index(drop=True)
        _edges = _tr["choice_cue_time"].to_numpy()[:, None] + lick_edges  # (n_trials, n_edges)
        _counts = np.diff(np.searchsorted(_onsets, _edges), axis=1)  # (n_trials, n_bins)
        for _rewarded, _g in _tr.groupby("has_reward"):
            _rows.append(
                (_subject, _sid, bool(_rewarded), len(_g), _counts[_g.index].mean(axis=0) / LICK_BIN)
            )

    lick_by_session = pd.DataFrame(
        _rows, columns=["subject_id", "session_id", "rewarded", "n_trials", "trace"]
    )
    print(lick_by_session.groupby(["subject_id", "rewarded"])["session_id"].nunique().unstack())
    return LICK_BIN, lick_by_session, lick_edges, lick_t


@app.cell
def lick_rate_plot(lick_by_session, lick_t, np, pd):
    import matplotlib.pyplot as _plt
    from analysis.plotting import bootstrap_group_stats as _bootstrap_group_stats
    from analysis.plotting import plot_mean_ci_band as _plot_mean_ci_band
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import presentation_style as _presentation_style

    _rng = np.random.default_rng(0)
    _blue, _orange = _categorical(2)
    _conditions = {True: ("Rewarded", _orange), False: ("Not rewarded", _blue)}
    _subjects = sorted(lick_by_session["subject_id"].unique())

    with _presentation_style():
        _fig, _axes = _plt.subplots(
            1, len(_subjects), figsize=(4 * len(_subjects), 4), sharex=True, sharey=True,
            squeeze=False,
        )
        for _ax, _subject in zip(_axes[0], _subjects):
            for _rewarded, (_label, _color) in _conditions.items():
                _d = lick_by_session[
                    (lick_by_session["subject_id"] == _subject)
                    & (lick_by_session["rewarded"] == _rewarded)
                ]
                if _d.empty:
                    continue
                # One animal per panel, so there is no across-animal N: the CI
                # bootstraps this animal's per-session means instead.
                _traces = np.stack(_d["trace"].to_numpy())  # (n_sessions, n_bins)
                _long = pd.DataFrame({"v": _traces.ravel(), "t": np.tile(lick_t, len(_d))})
                _stats = _bootstrap_group_stats(_long["v"], _long["t"], _rng, n_boot=500)
                _plot_mean_ci_band(_ax, _stats, _color, label=_label)
            _ax.axvline(0, color="#898781", lw=1)
            _ax.text(0.5, 0.97, _subject, transform=_ax.transAxes, ha="center", va="top")
            _ax.set_xlabel("Time from choice cue (s)")
        _axes[0, 0].set_ylabel("Lick rate (Hz)")
        _axes[0, -1].legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
        _fig.tight_layout()
    _fig
    return


@app.cell
def second_site_trials(analysis_trials, np):
    # Second reward site of every block, labelled by what happened at the first.
    # Needs a stop at the first site (otherwise there is no "first stop" outcome).
    _rs = analysis_trials[
        (analysis_trials["site_label"] == "RewardSite") & analysis_trials["block"].notna()
    ].sort_values(["session_id", "block", "start_time"])
    _rs = _rs.assign(pos=_rs.groupby(["session_id", "block"]).cumcount())
    _first = _rs[_rs["pos"] == 0].set_index(["session_id", "block"])
    _second = _rs[_rs["pos"] == 1].set_index(["session_id", "block"])

    second_site = _second.join(
        _first[["odor_index", "has_choice", "has_reward"]].add_prefix("first_"), how="inner"
    ).reset_index()
    second_site = second_site[second_site["first_has_choice"]].assign(
        first_stop_rewarded=lambda d: d["first_has_reward"].astype(bool),
        same_odor=lambda d: d["odor_index"] == d["first_odor_index"],
        stopped=lambda d: d["has_choice"].astype(bool),
    )

    # No-stop trials have no choice cue to align licks to, so give them a stand-in:
    # site onset + that animal's median onset-to-cue latency on stop trials.
    _latency = (
        second_site[second_site["stopped"]]
        .assign(lat=lambda d: d["choice_cue_time"] - d["start_time"])
        .groupby("subject_id")["lat"]
        .median()
    )
    second_site["lick_align_time"] = np.where(
        second_site["stopped"],
        second_site["choice_cue_time"],
        second_site["start_time"] + second_site["subject_id"].map(_latency),
    )
    print(
        second_site.groupby(["subject_id", "first_stop_rewarded", "same_odor", "stopped"])
        .size()
        .unstack("subject_id")
    )
    return (second_site,)


@app.cell
def second_site_traces(
    LICK_BIN,
    dataset,
    lick_edges,
    np,
    pd,
    second_site,
    vel_t,
):
    _keys = ["first_stop_rewarded", "same_odor", "stopped"]
    _vel_rows, _lick_rows = [], []
    for (_subject, _sid), _tr in second_site.groupby(["subject_id", "session_id"]):
        _tr = _tr.reset_index(drop=True)

        _v = dataset.load_table("position_velocity", [_sid])
        _ts, _vel = _v["timestamp"].to_numpy(), _v["velocity"].to_numpy()
        _idx = np.searchsorted(_ts, _tr["start_time"].to_numpy()[:, None] + vel_t)
        _vel_mat = _vel[np.clip(_idx, 0, len(_ts) - 1)]  # (n_trials, n_time)

        _l = dataset.load_table("licks", [_sid])
        _onsets = np.sort(_l.loc[_l["is_lick_onset"], "timestamp"].to_numpy())
        _edges = _tr["lick_align_time"].to_numpy()[:, None] + lick_edges
        _lick_mat = np.diff(np.searchsorted(_onsets, _edges), axis=1) / LICK_BIN  # Hz

        for _cond, _g in _tr.groupby(_keys):
            _vel_rows.append((_subject, _sid, *_cond, len(_g), _vel_mat[_g.index].mean(axis=0)))
            _lick_rows.append((_subject, _sid, *_cond, len(_g), _lick_mat[_g.index].mean(axis=0)))

    _cols = ["subject_id", "session_id", *_keys, "n_trials", "trace"]
    second_site_vel = pd.DataFrame(_vel_rows, columns=_cols)
    second_site_lick = pd.DataFrame(_lick_rows, columns=_cols)
    print(second_site_vel.groupby(["subject_id", *_keys])["session_id"].nunique().unstack("subject_id"))
    return second_site_lick, second_site_vel


@app.cell
def second_site_plot_helper(np, pd):
    import matplotlib.pyplot as _plt
    from analysis.plotting import bootstrap_group_stats as _bootstrap_group_stats
    from analysis.plotting import plot_mean_ci_band as _plot_mean_ci_band
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import presentation_style as _presentation_style


    def plot_second_site_grid(by_session, t, xlabel, ylabel):
        """Animals as columns; one row for stops and one for skips. Color is the
        first stop's outcome (orange rewarded / blue unrewarded, as elsewhere in
        the notebook); solid = same odor as the first site, dashed = other odor.
        The CI bootstraps each animal's per-session means (one animal per panel,
        so no across-animal N)."""
        rng = np.random.default_rng(0)
        blue, orange = _categorical(2)
        first_colors = {True: orange, False: blue}
        first_labels = {True: "First stop rewarded", False: "First stop unrewarded"}
        odor_styles = {True: ("same odor", "-"), False: ("other odor", "--")}
        rows = [(True, "Stop"), (False, "Skip")]
        subjects = sorted(by_session["subject_id"].unique())
        with _presentation_style():
            fig, axes = _plt.subplots(
                len(rows), len(subjects), figsize=(4 * len(subjects), 3.5 * len(rows)),
                sharex=True, sharey="col", squeeze=False,
            )
            for c, subject in enumerate(subjects):
                for ax, (stopped, row_label) in zip(axes[:, c], rows):
                    for rewarded, color in first_colors.items():
                        for same, (odor_label, linestyle) in odor_styles.items():
                            d = by_session[
                                (by_session["subject_id"] == subject)
                                & (by_session["first_stop_rewarded"] == rewarded)
                                & (by_session["same_odor"] == same)
                                & (by_session["stopped"] == stopped)
                            ]
                            if len(d) < 2:  # too few sessions for a CI
                                continue
                            traces = np.stack(d["trace"].to_numpy())
                            long = pd.DataFrame({"v": traces.ravel(), "t": np.tile(t, len(d))})
                            stats = _bootstrap_group_stats(long["v"], long["t"], rng, n_boot=500)
                            _plot_mean_ci_band(
                                ax, stats, color, linestyle=linestyle,
                                label=f"{first_labels[rewarded]}, {odor_label}",
                            )
                    ax.axvline(0, color="#898781", lw=1)
                    if c == 0:
                        ax.set_ylabel(f"{row_label}  {ylabel}")
                axes[0, c].text(0.5, 1.0, subject, transform=axes[0, c].transAxes,
                                ha="center", va="bottom")
                axes[-1, c].set_xlabel(xlabel)
            axes[0, -1].legend(loc="upper right", bbox_to_anchor=(1.0, 0.85), fontsize=9)
            fig.tight_layout()
        return fig

    return (plot_second_site_grid,)


@app.cell
def second_site_velocity_plot(plot_second_site_grid, second_site_vel, vel_t):
    plot_second_site_grid(second_site_vel, vel_t, "Time from site onset (s)", "velocity")
    return


@app.cell
def second_site_lick_plot(lick_t, plot_second_site_grid, second_site_lick):
    plot_second_site_grid(second_site_lick, lick_t, "Time from choice cue (s)", "lick rate (Hz)")
    return


@app.cell
def first_vs_later_stop_traces(
    LICK_BIN,
    analysis_trials,
    dataset,
    lick_edges,
    np,
    pd,
    vel_t,
):
    # Every stop at a reward site, flagged as the block's first stop or a later one.
    _rs = analysis_trials[
        (analysis_trials["site_label"] == "RewardSite") & analysis_trials["block"].notna()
    ].sort_values(["session_id", "block", "start_time"])
    _stops = _rs[_rs["has_choice"]].copy()
    _stops["first_stop"] = ~_stops.duplicated(["session_id", "block"])

    _vel_rows, _lick_rows = [], []
    for (_subject, _sid), _tr in _stops.groupby(["subject_id", "session_id"]):
        _tr = _tr.reset_index(drop=True)

        _v = dataset.load_table("position_velocity", [_sid])
        _ts, _vel = _v["timestamp"].to_numpy(), _v["velocity"].to_numpy()
        _idx = np.searchsorted(_ts, _tr["start_time"].to_numpy()[:, None] + vel_t)
        _vel_mat = _vel[np.clip(_idx, 0, len(_ts) - 1)]  # aligned to site onset

        _l = dataset.load_table("licks", [_sid])
        _onsets = np.sort(_l.loc[_l["is_lick_onset"], "timestamp"].to_numpy())
        _edges = _tr["choice_cue_time"].to_numpy()[:, None] + lick_edges
        _lick_mat = np.diff(np.searchsorted(_onsets, _edges), axis=1) / LICK_BIN  # Hz

        for _first, _g in _tr.groupby("first_stop"):
            _vel_rows.append((_subject, _sid, bool(_first), len(_g), _vel_mat[_g.index].mean(axis=0)))
            _lick_rows.append((_subject, _sid, bool(_first), len(_g), _lick_mat[_g.index].mean(axis=0)))

    _cols = ["subject_id", "session_id", "first_stop", "n_trials", "trace"]
    first_vs_later_vel = pd.DataFrame(_vel_rows, columns=_cols)
    first_vs_later_lick = pd.DataFrame(_lick_rows, columns=_cols)
    print(_stops.groupby(["subject_id", "first_stop"]).size().unstack("first_stop"))
    return first_vs_later_lick, first_vs_later_vel


@app.cell
def first_vs_later_stop_plot(
    first_vs_later_lick,
    first_vs_later_vel,
    lick_t,
    np,
    pd,
    vel_t,
):
    import matplotlib.pyplot as _plt
    from analysis.plotting import bootstrap_group_stats as _bootstrap_group_stats
    from analysis.plotting import plot_mean_ci_band as _plot_mean_ci_band
    from analysis.plotting_style import categorical as _categorical
    from analysis.plotting_style import presentation_style as _presentation_style

    _rng = np.random.default_rng(0)
    _slots = _categorical(4)
    _lines = {True: ("First stop in block", _slots[2]), False: ("Later stops", _slots[3])}
    _modalities = [
        (first_vs_later_vel, vel_t, "Time from site onset (s)", "Velocity"),
        (first_vs_later_lick, lick_t, "Time from choice cue (s)", "Lick rate (Hz)"),
    ]
    _subjects = sorted(first_vs_later_vel["subject_id"].unique())

    with _presentation_style():
        _fig, _axes = _plt.subplots(
            2, len(_subjects), figsize=(4 * len(_subjects), 7), sharey="row", squeeze=False,
        )
        for _row, (_by_session, _t, _xlabel, _ylabel) in enumerate(_modalities):
            for _ax, _subject in zip(_axes[_row], _subjects):
                for _first, (_label, _color) in _lines.items():
                    _d = _by_session[
                        (_by_session["subject_id"] == _subject)
                        & (_by_session["first_stop"] == _first)
                    ]
                    if len(_d) < 2:
                        continue
                    # One animal per column, so the CI bootstraps this animal's
                    # per-session means (no across-animal N).
                    _traces = np.stack(_d["trace"].to_numpy())
                    _long = pd.DataFrame({"v": _traces.ravel(), "t": np.tile(_t, len(_d))})
                    _stats = _bootstrap_group_stats(_long["v"], _long["t"], _rng, n_boot=500)
                    _plot_mean_ci_band(_ax, _stats, _color, label=_label)
                _ax.axvline(0, color="#898781", lw=1)
                _ax.set_xlabel(_xlabel)
                if _row == 0:
                    _ax.text(0.5, 1.0, _subject, transform=_ax.transAxes, ha="center", va="bottom")
            _axes[_row, 0].set_ylabel(_ylabel)
        _axes[0, -1].legend(loc="upper right", bbox_to_anchor=(1.0, 0.85))
        _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
