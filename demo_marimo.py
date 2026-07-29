import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def imports_subprocess():
    import subprocess

    return (subprocess,)


@app.cell
def imports_data_loading():
    # Edits to helpers.py / viz_helpers.py are picked up by marimo's own module
    # autoreloading -- no IPython %autoreload needed. Unlike %autoreload, marimo
    # re-executes the affected cells rather than hot-patching function bodies, so
    # adding a new module-level name works too (that was the case that produced
    # `NameError: name '_foo' is not defined` in the Jupyter version).
    #
    # Enable it once via Settings -> "Module autoreloading" -> autorun, or via the
    # [tool.marimo.runtime] auto_reload setting already added to pyproject.toml.
    from pathlib import Path

    from data_loading import sync_open_data_sessions

    return Path, sync_open_data_sessions


@app.cell
def sync_raw_data(Path, subprocess, sync_open_data_sessions):
    SUBJECT_IDS = ["841312", "841299", "866063", "864846", "864845"]
    START_DATE = "2026-06-01"
    OUTPUT_ROOT = Path("./data")
    if False:
        sync_open_data_sessions(
            subject_ids=SUBJECT_IDS,
            start_date=START_DATE,
            output_root=OUTPUT_ROOT,
            confirm=False,
        )
        #! uv run python process_sessions.py
        subprocess.call(['uv', 'run', 'python', 'process_sessions.py'])
    return (SUBJECT_IDS,)


@app.cell
def load_and_prepare_trials():
    import pandas as pd
    import numpy as np
    from helpers import (
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
    print("Subject overrides (see helpers.SESSION_SUBJECT_OVERRIDES):")
    print(report_subject_overrides(trials).to_string(index=False))
    trials = add_subject_id(trials)

    trials = assign_blocks(trials)

    # Drop sessions shorter than 15 minutes (first to last site timestamp)
    session_start = trials.groupby("session_id")["start_time"].min()
    session_end = trials.groupby("session_id")["start_time"].max()
    session_duration = session_end - session_start
    # Handle both timedelta and numeric (seconds) dtypes
    threshold = pd.Timedelta(minutes=15) if pd.api.types.is_timedelta64_dtype(session_duration) else 15 * 60
    long_sessions = session_duration[session_duration >= threshold].index
    trials = trials[trials["session_id"].isin(long_sessions)]

    # Keep the [start_frac, end_frac] window of each session (fractions from its
    # start). end_frac=1.0 keeps the whole session; end_frac=0.7 would keep the
    # first 70%.
    trials = trim_sessions(trials, start_frac=0.0, end_frac=1.0)
    trials = trials.copy()


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
def choice_by_block_position_pooled(SUBJECT_IDS, trials):
    from helpers import plot_choice_by_block_position
    from viz_helpers import a_lot_of_style
    from matplotlib import pyplot as plt
    for _animal in SUBJECT_IDS:
        print(f'Animal {_animal}')
        _fig = plt.figure(figsize=(10, 6))
        _ax = _fig.add_subplot(111)
        with a_lot_of_style():
            plot_choice_by_block_position(trials[trials['subject_id'] == _animal], ax=_ax)
    plt.show()
    return a_lot_of_style, plt


@app.cell
def choice_by_block_position_per_session(a_lot_of_style, plt, trials):
    from helpers import plot_choice_by_block_position_per_session

    with a_lot_of_style():
        plot_choice_by_block_position_per_session(trials)

    plt.show()
    return


@app.cell
def choice_by_first_stop(a_lot_of_style, plt, trials):
    from helpers import plot_choice_by_block_position_by_first_stop

    with a_lot_of_style():
        plot_choice_by_block_position_by_first_stop(trials, single_axis=True)

    plt.show()
    return


@app.cell
def choice_by_first_stop_overlay(a_lot_of_style, plt, trials):
    from helpers import plot_choice_by_block_position_by_first_stop_overlay

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
    _rs = trials[(trials['site_label'] == 'RewardSite') & trials['block'].notna()].copy()
    _rs = _rs.sort_values(['session_id', 'block', 'start_time'])
    _rs['_block_pos'] = _rs.groupby(['session_id', 'block']).cumcount()
    _rs['session_date'] = _rs['session_id'].str.split('_').str[1]
    STOP_STYLES = {0: {'label': '1st stop', 'color': 'tab:blue'}, 1: {'label': '2nd stop', 'color': 'tab:orange'}, 2: {'label': '3rd stop', 'color': 'tab:green'}}
    # Add subject and session date label for plotting
    with a_lot_of_style():
        _fig, _axes = plt.subplots(1, len(SUBJECT_IDS), figsize=(5 * len(SUBJECT_IDS), 4), sharey=True, squeeze=False)
        for _ax, _animal in zip(_axes[0], SUBJECT_IDS):
            for stop_pos, _style in STOP_STYLES.items():
                stop_trials = _rs[_rs['_block_pos'] == stop_pos].copy()
                session_means = stop_trials.groupby(['subject_id', 'session_id', 'session_date'])['has_choice'].mean().reset_index()
                _sub = session_means[session_means['subject_id'] == _animal].sort_values('session_id')
                _x = np.arange(len(_sub))
                _ax.plot(_x, _sub['has_choice'], marker='o', color=_style['color'], label=_style['label'])
            first_stop_sub = _rs[(_rs['_block_pos'] == 0) & (_rs['subject_id'] == _animal)]
            _dates = sorted(first_stop_sub['session_id'].unique())
            date_labels = [s.split('_')[1] for s in _dates]
            _ax.set_xticks(np.arange(len(_dates)))
            _ax.set_xticklabels(date_labels, rotation=45, ha='right')
            _ax.set_xlabel('Session')
            _ax.set_ylabel('P(choice)')
            _ax.set_ylim(0, 1.05)
            _ax.set_title(f'Subject {_animal}')
            _ax.legend(frameon=False, fontsize=8)
        _fig.suptitle('P(choice) at 1st, 2nd, and 3rd trial of each block (session averages)')
        _fig.tight_layout()
    plt.show()  # Use session dates from 1st stop for x-axis labels (most sessions should have it)
    return


@app.cell
def history_glm_per_session(a_lot_of_style, np, pd, plt, trials):
    from sklearn.linear_model import LogisticRegression
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    _rs = trials[(trials['site_label'] == 'RewardSite') & trials['block'].notna()].copy()
    _rs = _rs.sort_values(['session_id', 'block', 'start_time'])
    _grp = _rs.groupby(['session_id', 'block'], sort=False)
    _rs['prev_odor_index'] = _grp['odor_index'].shift(1)
    _rs['prev_has_choice'] = _grp['has_choice'].shift(1)
    _rs['prev_has_reward'] = _grp['has_reward'].shift(1)
    _rs = _rs.dropna(subset=['prev_odor_index', 'prev_has_choice', 'prev_has_reward'])
    is_same = (_rs['odor_index'] == _rs['prev_odor_index']).to_numpy()
    prev_choice = _rs['prev_has_choice'].astype(bool).to_numpy()
    prev_rewarded = _rs['prev_has_reward'].astype(bool).to_numpy()
    _rs['IsPrevChoice_SameOdor'] = np.where(is_same, np.where(prev_choice, 1.0, -1.0), 0.0)
    _rs['IsPrevChoice_OtherOdor'] = np.where(~is_same, np.where(prev_choice, 1.0, -1.0), 0.0)
    _rs['H_Same_Rew'] = (is_same & prev_rewarded).astype(float)
    _rs['H_Same_NoRew'] = (is_same & ~prev_rewarded).astype(float)
    _rs['H_Other_Rew'] = (~is_same & prev_rewarded).astype(float)
    _rs['H_Other_NoRew'] = (~is_same & ~prev_rewarded).astype(float)
    _rs['choice'] = _rs['has_choice'].astype(int)
    PLOT_COEFS = ['IsPrevChoice_SameOdor', 'IsPrevChoice_OtherOdor', 'H_Same_Rew', 'H_Same_NoRew', 'H_Other_Rew', 'H_Other_NoRew']
    FEATURE_COLS = PLOT_COEFS
    records = []
    for session_id, sdf in _rs.groupby('session_id'):
        if len(sdf) < 10 or sdf['choice'].nunique() < 2:
            continue
        X = sdf[FEATURE_COLS].to_numpy(dtype=float)
        y = sdf['choice'].to_numpy(dtype=int)
        try:
            clf = LogisticRegression(C=np.inf, solver='lbfgs', fit_intercept=False, max_iter=500)
            clf.fit(X, y)
            for name, val in zip(PLOT_COEFS, clf.coef_[0]):
                records.append(dict(session_id=session_id, subject_id=sdf['subject_id'].iloc[0], coef=name, value=val))
        except Exception as e:
            print(f'Session {session_id} failed: {e}')
    coefs = pd.DataFrame(records)
    SAME_COLOR = '#e07b39'
    OTHER_COLOR = '#4f8fc0'
    NEUTRAL_COLOR = 'gray'

    def _coef_color(name):
        if name.startswith('H_Same'):
            return SAME_COLOR
        if name.startswith('H_Other'):
            return OTHER_COLOR
        if 'Same' in name:
            return SAME_COLOR
        if 'Other' in name:
            return OTHER_COLOR
        return NEUTRAL_COLOR
    TICK_LABELS = ['IsPrevChoice\n[same]', 'IsPrevChoice\n[other]', 'Same × Rew', 'Same × NoRew', 'Other × Rew', 'Other × NoRew']
    subjects = sorted(coefs['subject_id'].unique())
    x_pos = np.arange(len(PLOT_COEFS))
    rng = np.random.default_rng(0)
    PAIR_GROUPS = [(['IsPrevChoice_SameOdor', 'IsPrevChoice_OtherOdor'], '#f5f5f5'), (['H_Same_Rew', 'H_Same_NoRew'], '#fff3eb'), (['H_Other_Rew', 'H_Other_NoRew'], '#ebf3ff')]
    N_BOOTSTRAP = 2000
    with a_lot_of_style():
        _fig, _axes = plt.subplots(2, len(subjects), figsize=(8 * len(subjects), 10), sharey='row', squeeze=False)
        for col, _subject in enumerate(subjects):
            _ax = _axes[0][col]
            for group_members, bg in PAIR_GROUPS:
                idxs = [PLOT_COEFS.index(_m) for _m in group_members]
                _ax.axvspan(min(idxs) - 0.4, max(idxs) + 0.4, color=bg, zorder=0)
            _sub = coefs[coefs['subject_id'] == _subject]
            _sessions = sorted(_sub['session_id'].unique())
            _n = len(_sessions)
            cmap = plt.get_cmap('viridis')
            norm = Normalize(vmin=0, vmax=max(_n - 1, 1))
            for day, session_id in enumerate(_sessions):
                sdata = _sub[_sub['session_id'] == session_id].set_index('coef')['value']
                _vals = [sdata.get(c, np.nan) for c in PLOT_COEFS]
                jx = x_pos + rng.uniform(-0.15, 0.15, len(x_pos))
                _ax.scatter(jx, _vals, color=cmap(norm(day)), s=40, zorder=3, alpha=0.85)
            means = _sub.groupby('coef')['value'].mean().reindex(PLOT_COEFS)
            sems = _sub.groupby('coef')['value'].sem().reindex(PLOT_COEFS)
            for _xi, coef in enumerate(PLOT_COEFS):
                _ax.errorbar(_xi, means[coef], yerr=sems[coef], fmt='o', color=_coef_color(coef), ms=8, lw=2.5, capsize=5, zorder=5)
            _ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            _ax.set_xticks(x_pos)
            _ax.set_xticklabels(TICK_LABELS, rotation=30, ha='right', fontsize=9)
            _ax.set_title(f'Subject {_subject}')
            _ax.set_xlabel('Regressor')
            cb = _fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=_ax)
            cb.set_label('Session (day)')
            cb.set_ticks([0, max(_n - 1, 1)])
            cb.set_ticklabels(['first', 'last'])
            ax2 = _axes[1][col]
            for group_members, bg in PAIR_GROUPS:
                idxs = [PLOT_COEFS.index(_m) for _m in group_members]
                ax2.axvspan(min(idxs) - 0.4, max(idxs) + 0.4, color=bg, zorder=0)
            session_list = sorted(_sub['session_id'].unique())
            coef_matrix = np.array([[_sub[_sub['session_id'] == sid].set_index('coef')['value'].reindex(PLOT_COEFS).values for sid in session_list]]).squeeze(0)
            observed_mean = np.nanmean(coef_matrix, axis=0)
            boot_rng = np.random.default_rng(42)
            boot_means = np.array([np.nanmean(coef_matrix[boot_rng.integers(0, len(session_list), size=len(session_list))], axis=0) for _ in range(N_BOOTSTRAP)])
            ci_lo = np.nanpercentile(boot_means, 2.5, axis=0)
            ci_hi = np.nanpercentile(boot_means, 97.5, axis=0)
            for _xi, coef in enumerate(PLOT_COEFS):
                _color = _coef_color(coef)
                ax2.bar(_xi, observed_mean[_xi], color=_color, alpha=0.75, width=0.6, zorder=2)
                ax2.errorbar(_xi, observed_mean[_xi], yerr=[[observed_mean[_xi] - ci_lo[_xi]], [ci_hi[_xi] - observed_mean[_xi]]], fmt='none', color='black', capsize=5, lw=1.5, zorder=3)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(TICK_LABELS, rotation=30, ha='right', fontsize=9)
            ax2.set_title(f'Subject {_subject} — bootstrap mean')
            ax2.set_xlabel('Regressor')
        _axes[0][0].set_ylabel('GLM coefficient (per session)')
        _axes[1][0].set_ylabel('GLM coefficient (bootstrap mean ± 95% CI)')
        _fig.suptitle('Logistic GLM: P(choice) — within-block, 1-trial history\nReward encoding: one-hot (is_same × is_prev_rewarded), no intercept')
        _fig.tight_layout()
    plt.show()
    return coefs, subjects


@app.cell
def history_glm_reward_cells(a_lot_of_style, coefs, np, plt, subjects):
    # ── One-hot reward cells timecourse across sessions ───────────────────────────
    # 4 distinct colors so solid/dashed ambiguity is avoided entirely
    REWARD_CELLS = {'H_Same_Rew': {'label': 'Same × Rew', 'color': '#e07b39', 'marker': 'o'}, 'H_Same_NoRew': {'label': 'Same × NoRew', 'color': '#f5c18a', 'marker': 'o'}, 'H_Other_Rew': {'label': 'Other × Rew', 'color': '#2a6496', 'marker': 's'}, 'H_Other_NoRew': {'label': 'Other × NoRew', 'color': '#9ecae1', 'marker': 's'}}
    with a_lot_of_style():
        _fig, _axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 4), sharey=True, squeeze=False)
        for _ax, _subject in zip(_axes[0], subjects):
            _sub = coefs[coefs['subject_id'] == _subject]
            _sessions = sorted(_sub['session_id'].unique())
            _x = np.arange(len(_sessions))
            _dates = [s.split('_')[1] for s in _sessions]
            for term, _style in REWARD_CELLS.items():
                rows = _sub[_sub['coef'] == term].set_index('session_id')
                _vals = [rows.loc[sid, 'value'] if sid in rows.index else np.nan for sid in _sessions]
                _ax.plot(_x, _vals, marker=_style['marker'], color=_style['color'], linewidth=2, markersize=7, label=_style['label'])
            _ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_dates, rotation=45, ha='right', fontsize=8)
            _ax.set_xlabel('Session')
            _ax.set_title(f'Subject {_subject}')
            _ax.legend(frameon=False, fontsize=8)
        _axes[0][0].set_ylabel('GLM coefficient')
        _fig.suptitle('One-hot reward cells timecourse')
        _fig.tight_layout()
    plt.show()
    return


@app.cell
def bias_by_odor_identity(a_lot_of_style, pd, plt, trials):
    _rs = trials[(trials['site_label'] == 'RewardSite') & trials['block'].notna()].copy()
    odor_block = _rs.groupby(['subject_id', 'session_id', 'block', 'odor_index']).agg(p_stop=('has_choice', 'mean'), n_trials=('has_choice', 'count'), is_rewarded_odor=('is_rewarded_odor', 'first')).reset_index()
    odor_block = odor_block.sort_values(['subject_id', 'odor_index', 'session_id', 'block'])
    pair_records = []
    for (_subject, odor), _grp in odor_block.groupby(['subject_id', 'odor_index']):
        _grp = _grp.reset_index(drop=True)
        for i in range(len(_grp) - 1):
            prev = _grp.iloc[i]
            curr = _grp.iloc[i + 1]
            pair_records.append({'subject_id': _subject, 'odor_index': int(odor), 'prev_rewarded': bool(prev['is_rewarded_odor']), 'curr_rewarded': bool(curr['is_rewarded_odor']), 'p_stop_curr': curr['p_stop'], 'n_trials_curr': int(curr['n_trials'])})
    pairs_df = pd.DataFrame(pair_records)
    CONDITIONS = [(True, True, 'Prev Rew\n-> Curr Rew', '#c0392b'), (True, False, 'Prev Rew\n-> Curr NoRew', '#e07b39'), (False, True, 'Prev NoRew\n-> Curr Rew', '#1a5276'), (False, False, 'Prev NoRew\n-> Curr NoRew', '#4f8fc0')]
    subjects_1 = sorted(pairs_df['subject_id'].unique())
    with a_lot_of_style():
        _fig, _axes = plt.subplots(1, len(subjects_1), figsize=(5 * len(subjects_1), 5), sharey=True, squeeze=False)
        _fig.suptitle('P(stop) at next odor encounter — 4 conditions\n(prev block rewarded / not) x (curr block rewarded / not)', fontsize=10)
        for ai, _subject in enumerate(subjects_1):
            _ax = _axes[0][ai]
            _sub = pairs_df[pairs_df['subject_id'] == _subject]
            _ax.axvspan(-0.5, 1.5, color='#fff0eb', zorder=0)
            _ax.axvspan(1.5, 3.5, color='#eaf3fb', zorder=0)
            for _xi, (prev_rew, curr_rew, _label, _color) in enumerate(CONDITIONS):
                _grp = _sub[(_sub['prev_rewarded'] == prev_rew) & (_sub['curr_rewarded'] == curr_rew)]['p_stop_curr']
                _n = len(_grp)
                _m = _grp.mean() if _n > 0 else float('nan')
                se = _grp.sem() if _n > 1 else 0.0
                _ax.bar(_xi, _m, color=_color, alpha=0.85, width=0.65, zorder=2)
                if _m == _m:
                    _ax.errorbar(_xi, _m, yerr=se, fmt='none', color='black', capsize=5, lw=1.5, zorder=3)
                    _ax.text(_xi, min(_m + se + 0.04, 1.18), f'n={_n}', ha='center', va='bottom', fontsize=7, zorder=4)
            _ax.axvline(1.5, color='black', lw=1.0, alpha=0.3, zorder=1)
            _ax.axhline(0.5, color='gray', linestyle='--', lw=0.8, alpha=0.5)
            _ax.set_xticks(range(len(CONDITIONS)))
            _ax.set_xticklabels([c[2] for c in CONDITIONS], fontsize=8)
            _ax.set_ylim(0, 1.35)
            _ax.set_title(f'Subject {_subject}', fontsize=10)
            if ai == 0:
                _ax.set_ylabel('P(stop) in next block encounter')
            _ax.text(0.5, 1.28, 'Prev: Rewarded', ha='center', va='top', fontsize=7.5, color='#8b0000', transform=_ax.get_xaxis_transform())
            _ax.text(2.5, 1.28, 'Prev: Not Rewarded', ha='center', va='top', fontsize=7.5, color='#1a5276', transform=_ax.get_xaxis_transform())
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
def counterfactual_matrix(trials_all):
    import helpers

    # Guard against a stale `helpers` module. marimo's module autoreloading makes
    # this far less likely than it was under IPython %autoreload, but it still
    # fires if autoreloading is off and helpers.py changed under a live session.
    _REQUIRED = [
        "counterfactual_block_table",
        "counterfactual_session_matrix",
        "plot_counterfactual_heatmap",
        "plot_counterfactual_cohort_average",
        "_counterfactual_style",
        "_COUNTERFACTUAL_VALUE_STYLES",
    ]
    _missing = [name for name in _REQUIRED if not hasattr(helpers, name)]
    if _missing:
        raise RuntimeError(
            f"stale `helpers` module (missing {_missing}). Enable Settings -> "
            "'Module autoreloading' -> autorun, or restart the marimo kernel."
        )
    print(f"helpers loaded from {helpers.__file__}")

    from helpers import (
        counterfactual_block_table,
        counterfactual_session_matrix,
        plot_counterfactual_heatmap,
    )

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
    return cf, plot_counterfactual_heatmap


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
        _fig, _ = plot_counterfactual_heatmap(trials_all, value='p_leave', matrix=cf)
    plt.show()
    return


@app.cell
def counterfactual_trends_per_animal(a_lot_of_style, cf, np, plt):
    from matplotlib.ticker import MaxNLocator
    from helpers import COUNTERFACTUAL_CELLS, counterfactual_session_trends
    CF_COLORS = ['#c0392b', '#e07b39', '#1a5276', '#4f8fc0']
    subjects_2 = sorted(cf['subject_id'].unique())
    trends = counterfactual_session_trends(cf, value='accuracy')
    print(trends.round(4).to_string(index=False))
    with a_lot_of_style():
        _fig, _axes = plt.subplots(1, len(subjects_2), figsize=(4.2 * len(subjects_2), 4), sharey=True, squeeze=False)
        for _ax, _subject in zip(_axes[0], subjects_2):
            _sub = cf[cf['subject_id'] == _subject]
            for (fsr, nr, _label, _), _color in zip(COUNTERFACTUAL_CELLS, CF_COLORS):
                g = _sub[(_sub['first_stop_rewarded'] == fsr) & (_sub['next_rewarded'] == nr)].dropna(subset=['accuracy']).sort_values('session_index')
                _ax.plot(g['session_index'], g['accuracy'], marker='o', ms=4, color=_color, label=_label.replace('\n', ' '))
                if len(g) >= 3:
                    _x = g['session_index'].to_numpy(dtype=float)
                    _m, b = np.polyfit(_x, g['accuracy'].to_numpy(dtype=float), 1)
                    _ax.plot(_x, _m * _x + b, color=_color, ls='--', lw=1.2, alpha=0.7)
            _ax.axhline(0.5, color='gray', ls=':', lw=1)
            _ax.set_ylim(0, 1.05)
            _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            _ax.set_xlabel('Session index')
            _ax.set_title(f'Subject {_subject}', fontsize=10)
        _axes[0][0].set_ylabel('accuracy (higher = better)')
        _axes[0][-1].legend(frameon=False, fontsize=7, loc='lower right')
        _fig.suptitle('Counterfactual accuracy across sessions (dashed = OLS fit)', fontsize=11)
        _fig.tight_layout()
    plt.show()
    return


@app.cell
def counterfactual_cohort_by_session(a_lot_of_style, cf, plt):
    # Average across mice at the same session number (each animal's own 0-based
    # session_index), so column k is "the cohort's k-th session". Each animal
    # contributes at most one value per cell -> no weighting by blocks run.
    # Only the longest-running animal reaches the highest indices, so the right-hand
    # columns are thin; the animal count runs along the top and the shaded region
    # marks where fewer than half the cohort remains.
    from helpers import plot_counterfactual_cohort_average
    with a_lot_of_style():
        _fig, cf_cohort = plot_counterfactual_cohort_average(cf, value='p_leave', min_animals=1)
    plt.show()
    print(cf_cohort.pivot_table(index='session_index', columns=['first_stop_rewarded', 'next_rewarded'], values=['mean', 'n_animals']).round(3).to_string())
    return


if __name__ == "__main__":
    app.run()
