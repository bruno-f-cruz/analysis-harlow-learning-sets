import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def intro():
    import marimo as mo

    mo.md(
        """
        # Harlow learning sets: tabular Q-learning agent

        Simulated animals on the stay/leave task from `packages/harlow-rl`. State is
        `(current odor, previous odor, previous outcome)`; the previous-trial memory
        resets at each block. All CIs are 95% bootstrapped **across animals**.
        """
    )
    return (mo,)


@app.cell
def controls(mo):
    params = mo.ui.dictionary(
        {
            "n_animals": mo.ui.slider(
                4, 64, value=20, step=4, label="animals", debounce=True
            ),
            "n_steps": mo.ui.slider(
                10_000,
                200_000,
                value=80_000,
                step=10_000,
                label="trials",
                debounce=True,
            ),
            "bin_blocks": mo.ui.slider(
                50, 500, value=200, step=50, label="blocks / bin", debounce=True
            ),
            "first_last_blocks": mo.ui.slider(
                50, 1000, value=100, step=50, label="first / last blocks", debounce=True
            ),
            "state_mode": mo.ui.dropdown(
                {
                    "full: odor, previous odor, R/U/S": "full",
                    "binary: odor, previous odor, R/not R": "binary",
                    "relational: same odor?, R/not R": "relational",
                },
                value="binary: odor, previous odor, R/not R",
                label="state",
            ),
            "alpha": mo.ui.slider(
                0.01, 0.5, value=0.15, step=0.01, label="alpha", debounce=True
            ),
            "gamma": mo.ui.slider(
                0.5, 1.0, value=0.97, step=0.01, label="gamma (per s)", debounce=True
            ),
            "epsilon": mo.ui.slider(
                0.0, 0.5, value=0.02, step=0.01, label="epsilon (training)", debounce=True
            ),
            "eval_epsilon": mo.ui.slider(
                0.0, 0.5, value=0.0, step=0.01, label="epsilon (evaluation)", debounce=True
            ),
            "q_init": mo.ui.slider(
                0.0, 2.0, value=1.0, step=0.1, label="Q init", debounce=True
            ),
            "dwell_s": mo.ui.number(0.0, 10.0, value=3.0, step=0.5, label="dwell (s)"),
            "travel_s": mo.ui.number(
                0.0, 20.0, value=4.0, step=0.5, label="travel (s)"
            ),
            "seed": mo.ui.number(0, 10_000, value=0, step=1, label="seed"),
        }
    )
    params
    return (params,)


@app.cell
def simulate_agents(params):
    import numpy as np
    import pandas as pd
    from harlow_rl import HarlowEnv, QAgent, TaskConfig, simulate

    p = params.value
    cfg = TaskConfig(
        dwell_s=p["dwell_s"], travel_s=p["travel_s"], state_mode=p["state_mode"]
    )
    env = HarlowEnv(cfg, n_envs=p["n_animals"], seed=p["seed"])
    agent = QAgent(
        p["n_animals"],
        env.n_states,
        alpha=p["alpha"],
        gamma=p["gamma"],
        epsilon=p["epsilon"],
        q_init=p["q_init"],
        seed=p["seed"],
    )
    out = simulate(env, agent, p["n_steps"])

    # Long table, one row per (animal, trial). The previous trial's odor/outcome
    # are rebuilt from the shifted rollout; absent on the first trial of a block.

    def rollout_to_df(out, late_start=0):
        """Long table, one row per (animal, trial); see comments below."""
        _n_steps, _n_animals = out["action"].shape
        _first = out["trial_in_block"] == 0

        def _shift(a):
            _prev = np.vstack([np.full((1, _n_animals), -1), a[:-1]])
            _prev[_first] = -1
            return _prev

        _prev_odor, _prev_outcome = _shift(out["odor"]), _shift(out["outcome"])
        _cols = {
            "subject_id": np.broadcast_to(np.arange(_n_animals), (_n_steps, _n_animals)),
            "trial": np.broadcast_to(np.arange(_n_steps)[:, None], (_n_steps, _n_animals)),
            "block": out["block"],
            "trial_in_block": out["trial_in_block"],
            "stay": out["action"],
            "reward": out["reward"],
            "duration": out["duration"],
            "odor": out["odor"],
            "prev_odor": _prev_odor,
            "prev_outcome": _prev_outcome,
            "odor_rewarded": out["odor"] == out["rewarded_odor"],
        }
        df = pd.DataFrame({k: np.asarray(v).ravel() for k, v in _cols.items()})
        df["stay"] = df["stay"].astype(float)
        df["same_as_prev"] = (df["odor"] == df["prev_odor"]) & (df["prev_odor"] >= 0)
        df["train_block"] = df["block"] // p["bin_blocks"] * p["bin_blocks"]
        df["late"] = df["trial"] >= late_start
        return df

    df = rollout_to_df(out, late_start=int(p["n_steps"] * 0.8))

    # Evaluation: freeze the trained Q-tables (learn=False) and roll out with
    # eval_epsilon. Learning-independent plots use this table. The env continues
    # from where training stopped, so drop the (partial) first block.
    _train_epsilon = agent.epsilon
    agent.epsilon = p["eval_epsilon"]
    _out_eval = simulate(env, agent, p["n_steps"] // 4, learn=False)
    agent.epsilon = _train_epsilon
    df_eval = rollout_to_df(_out_eval)
    df_eval = df_eval[df_eval["block"] > df_eval["block"].min()].copy()
    df_eval["late"] = True

    return HarlowEnv, agent, cfg, df, df_eval, env, np, p, pd, simulate


@app.cell
def optimal_policy(HarlowEnv, cfg, np, p, simulate):
    class OptimalAgent:
        """Knows the rule. Stays on the first trial of a block (it has to sample), then:
        first odor rewarded -> stay only on it; first odor not rewarded -> stay only on
        the other odor."""

        def __init__(self, env):
            self.env = env
            n = env.n_envs
            self.first_odor = np.zeros(n, dtype=np.int64)
            self.first_rewarded = np.zeros(n, dtype=bool)

        def act(self, states, odors):
            first = self.env.trial == 0
            self.first_odor = np.where(first, odors, self.first_odor)
            on_first = odors == self.first_odor
            return (first | (on_first == self.first_rewarded)).astype(np.int64)

        def update(self, states, odors, actions, rewards, next_states, next_odors, durations):
            # env.trial has already advanced: 1 means the action was the block's first.
            self.first_rewarded = np.where(
                self.env.trial == 1, rewards > 0, self.first_rewarded
            )

    _opt_env = HarlowEnv(cfg, n_envs=p["n_animals"], seed=p["seed"] + 1)
    _opt_out = simulate(_opt_env, OptimalAgent(_opt_env), p["n_steps"] // 4)
    _rate = _opt_out["reward"].sum(axis=0) / _opt_out["duration"].sum(axis=0)
    optimal_rate = float(_rate.mean())

    return (optimal_rate,)


@app.cell
def plot_helpers(np):
    from analysis.plotting import (
        BLOCK_RANGE_LINESTYLES,
        bootstrap_across_animals,
        bootstrap_mean_ci,
        ci_errorbar,
        plot_mean_ci_band,
    )
    from analysis.plotting import (
        _first_last_legend_handles as first_last_legend_handles,
    )
    from analysis.plotting_style import (
        INK_MUTED,
        INK_SECONDARY,
        categorical,
        diverging_cmap,
        new_figure,
        presentation_style,
    )

    rng = np.random.default_rng(0)

    def prob_axis(ax, label="P(Choice)"):
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylabel(label)

    def categorical_points(ax, stats, series, colors, x_labels):
        """Mean + CI dots: a categorical x-axis has no line to hang a band on."""
        offsets = np.linspace(-0.12, 0.12, len(series))
        for off, (key, label), color in zip(offsets, series, colors):
            s = stats.xs(key, level=1).reindex(x_labels)
            ax.errorbar(
                np.arange(len(x_labels)) + off,
                s["mean"],
                yerr=ci_errorbar(s),
                fmt="o",
                color=color,
                label=label,
                capsize=0,
            )
        ax.set_xticks(np.arange(len(x_labels)), x_labels)
        ax.set_xlim(-0.5, len(x_labels) - 0.5)

    return (
        BLOCK_RANGE_LINESTYLES,
        INK_MUTED,
        INK_SECONDARY,
        bootstrap_across_animals,
        bootstrap_mean_ci,
        categorical,
        categorical_points,
        diverging_cmap,
        first_last_legend_handles,
        new_figure,
        plot_mean_ci_band,
        presentation_style,
        prob_axis,
        rng,
    )


@app.cell
def summary(bootstrap_across_animals, cfg, df, mo, optimal_rate, rng):
    _late = df[df["late"] & (df["trial_in_block"] > 0)]
    _prec = bootstrap_across_animals(_late[_late["stay"] == 1], "reward", ["late"], rng)
    _stay = bootstrap_across_animals(_late, "stay", ["odor_rewarded"], rng)

    _rate = (
        df[df["late"]]
        .groupby("subject_id")
        .apply(lambda d: d["reward"].sum() / d["duration"].sum(), include_groups=False)
    )
    _dur_oracle = cfg.travel_s + cfg.dwell_s * 0.5
    _oracle = cfg.reward * 0.5 / _dur_oracle
    _always = cfg.reward * 0.5 / (cfg.travel_s + cfg.dwell_s)

    mo.md(
        f"""
        ## Summary (last 20% of trials, trials 2-10 of each block)

        | | mean [95% CI] |
        |---|---|
        | P(stay \\| rewarded odor) | {_stay.loc[True, "mean"]:.2f} [{_stay.loc[True, "ci_lo"]:.2f}, {_stay.loc[True, "ci_hi"]:.2f}] |
        | P(stay \\| unrewarded odor) | {_stay.loc[False, "mean"]:.2f} [{_stay.loc[False, "ci_lo"]:.2f}, {_stay.loc[False, "ci_hi"]:.2f}] |
        | P(reward \\| stay) | {_prec.iloc[0]["mean"]:.2f} [{_prec.iloc[0]["ci_lo"]:.2f}, {_prec.iloc[0]["ci_hi"]:.2f}] |
        | reward rate (per s), all trials | {_rate.mean():.3f} |
        | reference: optimal policy (knows the rule) | {optimal_rate:.3f} |
        | reference: perfect knowledge (unattainable) / always stay | {_oracle:.3f} / {_always:.3f} |

        The rule is only revealed by the previous trial when it was stayed at
        (rewarded or not), so P(reward | stay) cannot reach 1 after a skip.
        """
    )
    return


@app.cell
def choice_by_training(
    bootstrap_across_animals,
    categorical,
    df,
    new_figure,
    plot_mean_ci_band,
    presentation_style,
    prob_axis,
    rng,
):
    _blue, _orange, _aqua = categorical(3)
    _later = bootstrap_across_animals(
        df[df["trial_in_block"] > 0], "stay", ["train_block", "odor_rewarded"], rng
    )
    _first = bootstrap_across_animals(
        df[df["trial_in_block"] == 0], "stay", ["train_block"], rng
    )

    with presentation_style():
        _fig, _ax = new_figure("standard")
        plot_mean_ci_band(_ax, _later.xs(True, level=1), _orange, label="Rewarded odor")
        plot_mean_ci_band(
            _ax, _later.xs(False, level=1), _blue, label="Non-rewarded odor"
        )
        plot_mean_ci_band(_ax, _first, _aqua, label="First trial of block")
        _ax.set_xlabel("Blocks of training")
        prob_axis(_ax)
        _ax.legend(loc="center right")
        _fig.tight_layout()
    _fig
    return


@app.cell
def choice_by_block_position(
    bootstrap_across_animals,
    categorical,
    df_eval,
    new_figure,
    plot_mean_ci_band,
    presentation_style,
    prob_axis,
    rng,
):
    _blue, _orange = categorical(2)
    _stats = bootstrap_across_animals(
        df_eval, "stay", ["trial_in_block", "odor_rewarded"], rng
    )

    with presentation_style():
        _fig, _ax = new_figure("standard")
        plot_mean_ci_band(_ax, _stats.xs(True, level=1), _orange, label="Rewarded odor")
        plot_mean_ci_band(
            _ax, _stats.xs(False, level=1), _blue, label="Non-rewarded odor"
        )
        _ax.set_xlabel("Position within block")
        _ax.set_xticks(range(10))
        prob_axis(_ax)
        _ax.legend()
        _fig.tight_layout()
    _fig
    return


@app.cell
def choice_by_odor_appearance(
    BLOCK_RANGE_LINESTYLES,
    bootstrap_across_animals,
    categorical,
    df,
    first_last_legend_handles,
    new_figure,
    np,
    params,
    plot_mean_ci_band,
    presentation_style,
    prob_axis,
    rng,
):
    _n = params.value["first_last_blocks"]
    _block = df["block"].to_numpy()
    _d = df.assign(
        appearance=df.groupby(["subject_id", "block", "odor"]).cumcount(),
        block_range=np.select(
            [_block < _n, _block >= _block.max() + 1 - _n], ["first", "last"], ""
        ),
    )
    _d = _d[(_d["block_range"] != "") & (_d["appearance"] < 5)]

    _blue, _orange = categorical(2)
    _colors = {True: _orange, False: _blue}
    with presentation_style():
        _fig, _ax = new_figure("standard")
        for (_rewarded, _range), _grp in _d.groupby(["odor_rewarded", "block_range"]):
            _stats = bootstrap_across_animals(_grp, "stay", ["appearance"], rng)
            plot_mean_ci_band(
                _ax,
                _stats,
                _colors[bool(_rewarded)],
                marker="o",
                linestyle=BLOCK_RANGE_LINESTYLES[_range],
            )
        _ax.set_xlabel("Appearance within block")
        _ax.set_xticks(range(5))
        prob_axis(_ax)
        _ax.legend(
            handles=first_last_legend_handles(_colors, _n), ncol=2, loc="lower right"
        )
        _fig.tight_layout()
    _fig
    return


@app.cell
def first_stop_trials(df_eval):
    from analysis.plotting import plot_choice_by_odor_appearance
    from analysis.simulation import trials_from_rollout

    # Whole blocks from the frozen-agent evaluation rollout (eval epsilon).
    sim_trials = trials_from_rollout(df_eval)
    return plot_choice_by_odor_appearance, sim_trials


@app.cell
def choice_by_odor_appearance_first_stop_rewarded(
    categorical,
    new_figure,
    plot_choice_by_odor_appearance,
    presentation_style,
    sim_trials,
):
    _cond = sim_trials[sim_trials["first_stop_rewarded"] == True]  # noqa: E712
    _blue, _orange = categorical(2)
    with presentation_style():
        _fig, _ax = new_figure("standard")
        plot_choice_by_odor_appearance(
            _cond,
            ax=_ax,
            colors={True: _orange, False: _blue},
            from_first_stop=True,
            n_blocks=None,
        )
        _fig.tight_layout()
    _fig
    return


@app.cell
def choice_by_odor_appearance_first_stop_nonrewarded(
    categorical,
    new_figure,
    plot_choice_by_odor_appearance,
    presentation_style,
    sim_trials,
):
    _cond = sim_trials[sim_trials["first_stop_rewarded"] == False]  # noqa: E712
    _blue, _orange = categorical(2)
    with presentation_style():
        _fig, _ax = new_figure("standard")
        plot_choice_by_odor_appearance(
            _cond,
            ax=_ax,
            colors={True: _orange, False: _blue},
            from_first_stop=True,
            n_blocks=None,
        )
        _fig.tight_layout()
    _fig
    return


@app.cell
def choice_by_previous_trial(
    bootstrap_across_animals,
    categorical,
    categorical_points,
    df_eval,
    new_figure,
    presentation_style,
    prob_axis,
    rng,
):
    _names = {0: "Rewarded", 1: "Not rewarded", 2: "Skipped"}
    _d = df_eval[df_eval["prev_outcome"] >= 0].assign(
        prev=lambda d: d["prev_outcome"].map(_names)
    )
    _stats = bootstrap_across_animals(_d, "stay", ["prev", "same_as_prev"], rng)
    _aqua, _yellow = categorical(4)[2:]

    with presentation_style():
        _fig, _ax = new_figure("standard")
        categorical_points(
            _ax,
            _stats,
            [(True, "Same odor as previous trial"), (False, "Different odor")],
            [_aqua, _yellow],
            list(_names.values()),
        )
        _ax.set_xlabel("Previous trial outcome")
        prob_axis(_ax)
        _ax.legend()
        _fig.tight_layout()
    _fig
    return


@app.cell
def reward_rate(
    INK_MUTED,
    INK_SECONDARY,
    bootstrap_across_animals,
    categorical,
    cfg,
    df,
    new_figure,
    optimal_rate,
    plot_mean_ci_band,
    presentation_style,
    rng,
):
    _per_bin = (
        df.groupby(["train_block", "subject_id"])
        .agg(reward=("reward", "sum"), duration=("duration", "sum"))
        .assign(rate=lambda d: d["reward"] / d["duration"])
        .reset_index()
    )
    _stats = bootstrap_across_animals(_per_bin, "rate", ["train_block"], rng)
    _half = cfg.reward * 0.5
    _aqua = categorical(3)[2]

    with presentation_style():
        _fig, _ax = new_figure("standard")
        plot_mean_ci_band(_ax, _stats, _aqua, label="Agent")
        _ax.axhline(
            _half / (cfg.travel_s + cfg.dwell_s * 0.5),
            color=INK_SECONDARY,
            linestyle="--",
            label="Perfect knowledge (unattainable)",
        )
        _ax.axhline(
            optimal_rate,
            color=INK_SECONDARY,
            linestyle=":",
            label="Optimal policy (knows the rule)",
        )
        _ax.axhline(
            _half / (cfg.travel_s + cfg.dwell_s),
            color=INK_MUTED,
            linestyle="--",
            label="Always stay",
        )
        _ax.set_xlabel("Blocks of training")
        _ax.set_ylabel("Reward rate (per s)")
        _ax.legend(loc="lower right")
        _fig.tight_layout()
    _fig
    return


@app.cell
def q_values(
    agent,
    bootstrap_mean_ci,
    categorical,
    categorical_points,
    diverging_cmap,
    env,
    new_figure,
    np,
    pd,
    presentation_style,
    rng,
):
    from analysis.plotting_style import sequential_cmap
    from harlow_rl.state import state_features

    _, _same, _prev_outcome = state_features(env.config.state_mode, env.config.n_odors)
    _names = ["Rewarded", "Not rewarded", "Skipped"][: env.n_outcomes]
    _adv = agent.q[:, :, 1] - agent.q[:, :, 0]  # (animals, states): Q(stay) - Q(leave)

    _rows = []
    for _k, _name in enumerate(_names):
        for _is_same in (True, False):
            _mask = (_prev_outcome == _k) & (_same == _is_same)
            _mean, _lo, _hi = bootstrap_mean_ci(_adv[:, _mask].mean(axis=1), rng)
            _rows.append((_name, _is_same, _mean, _lo, _hi))
    _stats = pd.DataFrame(
        _rows, columns=["prev", "same", "mean", "ci_lo", "ci_hi"]
    ).set_index(["prev", "same"])
    _aqua, _yellow = categorical(4)[2:]

    # Heatmaps: rows are current odors (one row when the state has no odor identity).
    def _as_grid(v):
        if env.config.state_mode == "relational":
            return v[None, :]
        return v.reshape(env.config.n_odors, 1 + env.config.n_odors * env.n_outcomes)

    if env.config.state_mode == "relational":
        _xticks = ["First trial", "Same, R", "Same, not R", "Diff, R", "Diff, not R"]
        _xlabel, _yticks = "Previous trial", [""]
    else:
        _letters = "RUS" if env.n_outcomes == 3 else "RN"
        _xticks = ["none"] + [
            f"{o}{_letters[k]}"
            for o in range(env.config.n_odors)
            for k in range(env.n_outcomes)
        ]
        _xlabel, _yticks = "Previous trial (odor, outcome)", range(env.config.n_odors)
    _grid = _as_grid(_adv.mean(axis=0))
    _lim = np.abs(_grid).max()
    _q_stay = _as_grid(agent.q[:, :, 1].mean(axis=0))
    _q_leave = _as_grid(agent.q[:, :, 0].mean(axis=0))
    _q_max = max(_q_stay.max(), _q_leave.max())
    _ylabel = "Current odor" if env.config.state_mode != "relational" else ""

    def _heatmap(ax, grid, cmap, vmin, vmax):
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(_xticks)), _xticks, rotation=90, fontsize=7)
        ax.set_yticks(range(len(_yticks)), [str(y) for y in _yticks])
        ax.set_xlabel(_xlabel)
        ax.set_ylabel(_ylabel)
        return im

    with presentation_style():
        # Top row: difference (bar summary + heatmap). Bottom row: the raw Q-values
        # behind it, on one shared colour scale.
        _fig = new_figure("wide")[0]
        _fig.clf()
        _fig.set_size_inches(10, 9)
        _fig.set_layout_engine("constrained")
        _gs = _fig.add_gridspec(2, 6)
        _ax_bar = _fig.add_subplot(_gs[0, :2])
        _ax_map = _fig.add_subplot(_gs[0, 2:])
        _ax_stay = _fig.add_subplot(_gs[1, :3])
        _ax_leave = _fig.add_subplot(_gs[1, 3:])

        categorical_points(
            _ax_bar,
            _stats,
            [(True, "Same odor as previous trial"), (False, "Different odor")],
            [_aqua, _yellow],
            _names,
        )
        _ax_bar.axhline(0, color="white", linewidth=0.8)
        _ax_bar.set_xlabel("Previous trial outcome")
        _ax_bar.tick_params(axis="x", labelrotation=30)
        _ax_bar.set_ylabel("Q(stay) - Q(leave)")
        _ax_bar.set_ylim(-0.4, 2.0)
        _ax_bar.legend(loc="upper center")

        _im = _heatmap(_ax_map, _grid, diverging_cmap(), -_lim, _lim)
        _fig.colorbar(_im, ax=_ax_map, label="Q(stay) - Q(leave)")

        _im_raw = _heatmap(_ax_stay, _q_stay, sequential_cmap("blue"), 0, _q_max)
        _heatmap(_ax_leave, _q_leave, sequential_cmap("blue"), 0, _q_max)
        _ax_stay.set_ylabel("Q(stay)")
        _ax_leave.set_ylabel("Q(leave)")
        _fig.colorbar(_im_raw, ax=[_ax_stay, _ax_leave], label="Q (mean over animals)")
    _fig
    return


if __name__ == "__main__":
    app.run()
