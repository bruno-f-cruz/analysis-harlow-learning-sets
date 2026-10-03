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
    return mo, sessions, sites


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
        value="ABReversal",
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
            selected_trials,
            axes=_axes.flat,
            colors={True: _orange, False: _blue},
            n_blocks=30,
        )
        _fig.tight_layout()
    _fig


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


@app.cell(hide_code=True)
def history_glm_fit(selected_trials):
    from analysis.counterfactual import first_site_chance_by_window
    from analysis.features import expand_to_block_windows
    from analysis.glm import ONEHOT_CELL_STYLE, fit_history_glm, history_glm_features

    glm_window_blocks = 100
    glm_skip_blocks = 20

    _glm_features = history_glm_features(selected_trials)
    glm_windows = expand_to_block_windows(
        _glm_features, glm_window_blocks, glm_skip_blocks
    )
    glm_coefs_window = fit_history_glm(
        glm_windows, unit_col=["subject_id", "window"], coefs=list(ONEHOT_CELL_STYLE)
    )
    glm_window_bounds = (
        glm_windows.drop_duplicates("window")
        .set_index("window")[["window_start", "window_end"]]
        .sort_index()
    )
    glm_chance = first_site_chance_by_window(
        selected_trials, glm_window_blocks, glm_skip_blocks
    )
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
    from analysis.glm import (
        plot_history_glm_cohort_by_window as _plot_history_glm_cohort_by_window,
    )
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot_history_glm_cohort_by_window(
            glm_coefs_window, glm_window_bounds, value="weight"
        )
    _fig


@app.cell(hide_code=True)
def history_glm_p_choice(glm_window_bounds, glm_windows):
    from analysis.glm import (
        plot_history_glm_empirical_by_window as _plot_history_glm_empirical_by_window,
    )
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot_history_glm_empirical_by_window(
            glm_windows, glm_window_bounds, value="p_choice"
        )
    _fig


@app.cell(hide_code=True)
def history_glm_p_choice_minus_chance(
    glm_chance,
    glm_window_bounds,
    glm_windows,
):
    from analysis.glm import (
        plot_history_glm_empirical_by_window as _plot_history_glm_empirical_by_window,
    )
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot_history_glm_empirical_by_window(
            glm_windows,
            glm_window_bounds,
            value="p_choice_minus_chance",
            chance=glm_chance,
        )
    _fig


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
                    selected_trials,
                    window_blocks=100,
                    skip_blocks=20,
                    exclude_subjects=_exclude,
                    subtract_chance=_subtract,
                    ax=_axes[_row][_col],
                )
        _fig.tight_layout()
    _fig


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
        glm1_windows,
        unit_col=["subject_id", "window"],
        coefs=[_c for _c in _ONEHOT_CELL_STYLE if not _c.endswith("_Skip")],
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


@app.cell(hide_code=True)
def history_glm_first_stop_p_choice(glm1_window_bounds, glm1_windows):
    from analysis.glm import plot_history_glm_empirical_by_window as _plot
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot(glm1_windows, glm1_window_bounds, value="p_choice")
    _fig


@app.cell(hide_code=True)
def history_glm_first_stop_p_choice_minus_chance(
    glm1_window_bounds,
    glm1_windows,
    glm_chance,
):
    from analysis.glm import plot_history_glm_empirical_by_window as _plot
    from analysis.plotting_style import presentation_style as _presentation_style

    with _presentation_style():
        _fig, _axes = _plot(
            glm1_windows,
            glm1_window_bounds,
            value="p_choice_minus_chance",
            chance=glm_chance,
        )
    _fig


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


@app.cell
def include_degenerate_toggle(mo):
    include_degenerate = mo.ui.switch(
        label="Include blocks with p_stay <= 0.1 / >= 0.9 (no degenerate-block exclusion)"
    )
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


if __name__ == "__main__":
    app.run()
