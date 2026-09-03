import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def imports_pathlib():
    from pathlib import Path

    return (Path,)


@app.cell
def load_data(Path):
    from analysis.features import prepare_trials
    from analysis.sessions import load_attached_datasets, load_processed_table

    attached = load_attached_datasets(Path(__file__).parent.parent / "data_assets.json")
    location = attached[0]["location"]
    df = load_processed_table(location, "session")
    trials = load_processed_table(location, "sites")
    _trials, trials_all = prepare_trials(trials, df)
    print(f"Loaded {trials_all['session_id'].nunique()} sessions, {len(trials_all):,} trials")
    return df, trials_all


@app.cell
def curriculum_stage_selection(df, mo, trials_all):
    # Same curriculum-stage split as workflows/pipeline.py -- Full and
    # ABReversal are different tasks, and mixing them would confound the
    # time-since-session-start axis with a task-difficulty difference.
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
    trials_all_full = trials_all[trials_all["session_id"].isin(_full_sessions)]
    trials_all_abreversal = trials_all[trials_all["session_id"].isin(_abreversal_sessions)]

    # Defaults to `--dataset` on the command line (e.g. `python
    # workflows/within_session_performance.py --dataset ABReversal`); in the
    # interactive editor mo.cli_args() is empty, so it falls back to "Full"
    # and stays switchable via the radio.
    _DATASET_OPTIONS = ["Full", "ABReversal"]
    _cli_dataset = mo.cli_args().get("dataset", "Full")
    if _cli_dataset not in _DATASET_OPTIONS:
        raise ValueError(f"--dataset must be one of {_DATASET_OPTIONS}, got {_cli_dataset!r}")

    dataset_toggle = mo.ui.radio(
        options=_DATASET_OPTIONS, value=_cli_dataset, label="Curriculum stage dataset"
    )
    dataset_toggle
    return dataset_toggle, trials_all_abreversal, trials_all_full


@app.cell
def dataset_selection(dataset_toggle, trials_all_abreversal, trials_all_full):
    trials_all_selected = (
        trials_all_full if dataset_toggle.value == "Full" else trials_all_abreversal
    )
    print(
        f"Analysing '{dataset_toggle.value}': "
        f"{trials_all_selected['session_id'].nunique()} sessions, "
        f"{len(trials_all_selected):,} trials"
    )
    return (trials_all_selected,)


@app.cell
def bin_controls(mo):
    bin_minutes = mo.ui.slider(
        5, 60, 5, value=15, label="Time bin size (minutes)", show_value=True
    )
    bin_minutes
    return (bin_minutes,)


@app.cell
def within_session_performance(bin_minutes, trials_all_selected):
    import matplotlib.pyplot as plt
    from analysis.plotting import a_lot_of_style
    from analysis.within_session import (
        plot_within_session_performance,
        within_session_performance_gap,
    )

    # Uses trials_all_selected -- *before* the degenerate-block filter --
    # since a block where the animal stops at everything (or nothing) late
    # in a session is exactly the performance-decay signal this is after,
    # not noise to discard.
    per_session_bin = within_session_performance_gap(
        trials_all_selected, bin_minutes=bin_minutes.value
    )
    with a_lot_of_style():
        plot_within_session_performance(per_session_bin, bin_minutes=bin_minutes.value)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
