import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def intro():
    import marimo as mo

    mo.md(
        """
        # Working-memory + sensory agent mixture

        Two tabular Q-learners blended at decision time, `Q = w * Q_WM + (1 - w) * Q_sensory`,
        with a softmax on the blend. Each is trained on its own TD error.

        - **WM agent:** state = (odor, previous odor, rewarded / not rewarded), forgotten at
          every block boundary, learns fast.
        - **Sensory agent:** state = the current odor only. Its Q-table is never reset, so what
          it learned about an odor in an earlier block biases later blocks, where the odor's
          rewarded status has been re-randomised.

        All CIs are 95% bootstrapped **across animals**.
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
                value=60_000,
                step=10_000,
                label="trials",
                debounce=True,
            ),
            "bin_blocks": mo.ui.slider(
                50, 500, value=200, step=50, label="blocks / bin", debounce=True
            ),
            "w": mo.ui.slider(
                0.0, 1.0, value=0.5, step=0.05, label="w (WM weight)", debounce=True
            ),
            "beta": mo.ui.slider(
                1.0, 20.0, value=10.0, step=1.0, label="beta", debounce=True
            ),
            "alpha_wm": mo.ui.slider(
                0.01, 0.5, value=0.1, step=0.01, label="alpha WM", debounce=True
            ),
            "alpha_sens": mo.ui.slider(
                0.01, 0.5, value=0.1, step=0.01, label="alpha sensory", debounce=True
            ),
            "gamma_wm": mo.ui.slider(
                0.5, 1.0, value=0.95, step=0.01, label="gamma WM (per s)", debounce=True
            ),
            "gamma_sens": mo.ui.slider(
                0.5,
                1.0,
                value=0.95,
                step=0.01,
                label="gamma sensory (per s)",
                debounce=True,
            ),
            "q_init": mo.ui.slider(
                0.0, 2.0, value=1.0, step=0.1, label="Q init", debounce=True
            ),
            "dwell_s": mo.ui.number(0.0, 10.0, value=2.0, step=0.5, label="dwell (s)"),
            "travel_s": mo.ui.number(
                0.0, 20.0, value=5.0, step=0.5, label="travel (s)"
            ),
            "seed": mo.ui.number(0, 10_000, value=0, step=1, label="seed"),
        }
    )
    params
    return (params,)


@app.cell
def helpers():
    import numpy as np
    import pandas as pd
    from harlow_rl import HarlowEnv, MixtureAgent, TaskConfig, simulate

    from analysis.plotting import (
        bootstrap_across_animals,
        bootstrap_mean_ci,
        ci_errorbar,
        plot_mean_ci_band,
    )
    from analysis.plotting_style import (
        INK_MUTED,
        INK_SECONDARY,
        categorical,
        new_figure,
        presentation_style,
    )

    rng = np.random.default_rng(0)

    def run_mixture(p, w=None, n_steps=None):
        """Simulate ``p["n_animals"]`` animals; ``w`` / ``n_steps`` override the controls."""
        cfg = TaskConfig(
            dwell_s=p["dwell_s"], travel_s=p["travel_s"], state_mode="binary"
        )
        env = HarlowEnv(cfg, n_envs=p["n_animals"], seed=p["seed"])
        agent = MixtureAgent(
            p["n_animals"],
            env.n_states,
            cfg.n_odors,
            w=p["w"] if w is None else w,
            beta=p["beta"],
            alpha_wm=p["alpha_wm"],
            alpha_sens=p["alpha_sens"],
            gamma_wm=p["gamma_wm"],
            gamma_sens=p["gamma_sens"],
            q_init=p["q_init"],
            seed=p["seed"],
        )
        out = simulate(env, agent, p["n_steps"] if n_steps is None else n_steps)
        return cfg, env, agent, out

    def rollout_frame(out, bin_blocks):
        """Long table, one row per (animal, trial).

        ``prior_label`` is the outcome of the odor's last *stay* in an earlier block
        (``Rewarded`` / ``Not rewarded``), or ``No previous stay``. It only looks at
        earlier blocks, so the current block's own history cannot leak into it.
        """
        n_steps, n_animals = out["action"].shape
        cols = {
            "subject_id": np.broadcast_to(np.arange(n_animals), (n_steps, n_animals)),
            "trial": np.broadcast_to(np.arange(n_steps)[:, None], (n_steps, n_animals)),
            "block": out["block"],
            "trial_in_block": out["trial_in_block"],
            "stay": out["action"],
            "reward": out["reward"],
            "duration": out["duration"],
            "odor": out["odor"],
            "outcome": out["outcome"],
            "odor_rewarded": out["odor"] == out["rewarded_odor"],
        }
        df = pd.DataFrame({k: np.asarray(v).ravel() for k, v in cols.items()})
        df["stay"] = df["stay"].astype(float)
        df["train_block"] = df["block"] // bin_blocks * bin_blocks

        keys = ["subject_id", "odor", "block"]
        last_stay = (
            df[df["stay"] == 1].groupby(keys)["outcome"].last().rename("last_stay")
        )
        table = df[keys].drop_duplicates().set_index(keys).join(last_stay).sort_index()
        table["prior"] = table.groupby(level=["subject_id", "odor"])[
            "last_stay"
        ].transform(lambda s: s.ffill().shift(1))
        df = df.join(table["prior"], on=keys)
        df["prior_label"] = (
            df["prior"]
            .map({0.0: "Rewarded", 1.0: "Not rewarded"})
            .fillna("No previous stay")
        )

        # Previous trial, when it was a stay: same / other odor x rewarded / not rewarded.
        first = df["trial_in_block"] == 0
        by_animal = df.groupby("subject_id")
        prev_odor = by_animal["odor"].shift(1).mask(first)
        prev_outcome = by_animal["outcome"].shift(1).mask(first)
        df["past"] = df["odor"].eq(prev_odor).map(
            {True: "Same", False: "Other"}
        ) + prev_outcome.map({0.0: ", rewarded", 1.0: ", not rewarded"})
        df["late"] = df["trial"] >= n_steps // 2
        return df

    def prob_axis(ax, label="P(Choice)"):
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylabel(label)

    return (
        INK_MUTED,
        INK_SECONDARY,
        bootstrap_across_animals,
        bootstrap_mean_ci,
        categorical,
        ci_errorbar,
        new_figure,
        np,
        pd,
        plot_mean_ci_band,
        presentation_style,
        prob_axis,
        rng,
        rollout_frame,
        run_mixture,
    )


@app.cell
def simulate_agents(params, rollout_frame, run_mixture):
    p = params.value
    cfg, env, agent, out = run_mixture(p)
    df = rollout_frame(out, p["bin_blocks"])
    return agent, cfg, df, env, p


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
    _blue, _orange = categorical(2)
    _stats = bootstrap_across_animals(
        df[df["trial_in_block"] > 0], "stay", ["train_block", "odor_rewarded"], rng
    )

    with presentation_style():
        _fig, _ax = new_figure("standard")
        plot_mean_ci_band(_ax, _stats.xs(True, level=1), _orange, label="Rewarded odor")
        plot_mean_ci_band(
            _ax, _stats.xs(False, level=1), _blue, label="Non-rewarded odor"
        )
        _ax.set_xlabel("Blocks of training")
        prob_axis(_ax)
        _ax.legend(loc="center right")
        _fig.tight_layout()
    _fig


@app.cell
def choice_by_past_trial(
    bootstrap_across_animals,
    categorical,
    ci_errorbar,
    df,
    new_figure,
    np,
    presentation_style,
    prob_axis,
    rng,
):
    _conds = [
        "Same, rewarded",
        "Same, not rewarded",
        "Other, rewarded",
        "Other, not rewarded",
    ]
    _ideal = [1, 0, 0, 1]  # stay iff the previous stay implies this odor is rewarded
    _stats = bootstrap_across_animals(
        df[df["late"] & df["past"].notna()], "stay", ["past"], rng
    ).reindex(_conds)
    _aqua, _yellow = categorical(4)[2:]

    with presentation_style():
        _fig, _ax = new_figure("standard")
        for _sl, _color, _label in (
            (slice(0, 2), _aqua, "Previous trial: same odor"),
            (slice(2, 4), _yellow, "Previous trial: other odor"),
        ):
            _s = _stats.iloc[_sl]
            _ax.errorbar(
                np.arange(4)[_sl],
                _s["mean"],
                yerr=ci_errorbar(_s),
                fmt="o",
                color=_color,
                label=_label,
                capsize=0,
            )
        _ax.scatter(range(4), _ideal, marker="_", s=900, color="white", label="Ideal")
        _ax.set_xticks(range(4), [c.replace(", ", "\n") for c in _conds])
        _ax.set_xlim(-0.5, 3.5)
        _ax.set_xlabel("Previous trial (stays only)")
        prob_axis(_ax)
        _ax.legend(loc="center right", fontsize=11)
        _fig.tight_layout()
    _fig


@app.cell
def first_stop_trials(df):
    from analysis.plotting import plot_choice_by_odor_appearance
    from analysis.simulation import trials_from_rollout

    # Whole blocks from the late (trained) part of the run only.
    sim_trials = trials_from_rollout(
        df[df["block"] > df.loc[df["late"], "block"].min()]
    )
    return plot_choice_by_odor_appearance, sim_trials


@app.cell
def choice_by_odor_appearance_first_stop_rewarded(
    categorical,
    new_figure,
    plot_choice_by_odor_appearance,
    presentation_style,
    sim_trials,
):
    _cond = sim_trials[sim_trials["first_stop_rewarded"] == True]
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


@app.cell
def choice_by_odor_appearance_first_stop_nonrewarded(
    categorical,
    new_figure,
    plot_choice_by_odor_appearance,
    presentation_style,
    sim_trials,
):
    _cond = sim_trials[sim_trials["first_stop_rewarded"] == False]
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


@app.cell
def bias_by_position_nonrewarded(
    bootstrap_across_animals,
    categorical,
    df,
    new_figure,
    plot_mean_ci_band,
    presentation_style,
    prob_axis,
    rng,
):
    _colors = dict(
        zip(["Rewarded", "Not rewarded", "No previous stay"], categorical(5)[2:])
    )
    _d = df[df["late"] & ~df["odor_rewarded"]]
    _stats = bootstrap_across_animals(
        _d, "stay", ["trial_in_block", "prior_label"], rng
    )

    with presentation_style():
        _fig, _ax = new_figure("standard")
        for _label, _color in _colors.items():
            if _label in _stats.index.get_level_values(1):
                plot_mean_ci_band(
                    _ax,
                    _stats.xs(_label, level=1),
                    _color,
                    label=f"Last stay at this odor: {_label.lower()}",
                )
        _ax.set_xlabel("Position within block (non-rewarded odor)")
        _ax.set_xticks(range(10))
        prob_axis(_ax)
        _ax.legend(loc="upper right", fontsize=11)
        _fig.tight_layout()
    _fig


@app.cell
def bias_by_position_rewarded(
    bootstrap_across_animals,
    categorical,
    df,
    new_figure,
    plot_mean_ci_band,
    presentation_style,
    prob_axis,
    rng,
):
    _colors = dict(
        zip(["Rewarded", "Not rewarded", "No previous stay"], categorical(5)[2:])
    )
    _d = df[df["late"] & df["odor_rewarded"]]
    _stats = bootstrap_across_animals(
        _d, "stay", ["trial_in_block", "prior_label"], rng
    )

    with presentation_style():
        _fig, _ax = new_figure("standard")
        for _label, _color in _colors.items():
            if _label in _stats.index.get_level_values(1):
                plot_mean_ci_band(
                    _ax,
                    _stats.xs(_label, level=1),
                    _color,
                    label=f"Last stay at this odor: {_label.lower()}",
                )
        _ax.set_xlabel("Position within block (rewarded odor)")
        _ax.set_xticks(range(10))
        prob_axis(_ax)
        _ax.legend(loc="lower right", fontsize=11)
        _fig.tight_layout()
    _fig


@app.cell
def sweep_w(
    bootstrap_across_animals, bootstrap_mean_ci, p, pd, rng, rollout_frame, run_mixture
):
    _rows_bias, _rows_rate, _rows_past = [], [], []
    for _w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        _, _, _, _out = run_mixture(p, w=_w)
        _d = rollout_frame(_out, p["bin_blocks"])
        _late = _d[_d["late"]]

        # Bias: P(Choice) at non-rewarded odors when the odor paid last time minus when
        # it did not, one difference per animal.
        _err = _late[~_late["odor_rewarded"] & (_late["trial_in_block"] > 0)]
        _per_animal = (
            _err[_err["prior_label"].isin(["Rewarded", "Not rewarded"])]
            .groupby(["subject_id", "prior_label"])["stay"]
            .mean()
            .unstack()
        )
        _rows_bias.append(
            (
                _w,
                *bootstrap_mean_ci(
                    _per_animal["Rewarded"] - _per_animal["Not rewarded"], rng
                ),
            )
        )

        _rate = _late.groupby("subject_id").apply(
            lambda g: g["reward"].sum() / g["duration"].sum(), include_groups=False
        )
        _rows_rate.append((_w, *bootstrap_mean_ci(_rate, rng)))

        _past = bootstrap_across_animals(
            _late[_late["past"].notna()], "stay", ["past"], rng
        )
        _rows_past += [(_w, _cond, *_row) for _cond, _row in _past.iterrows()]

    bias_stats = pd.DataFrame(
        _rows_bias, columns=["w", "mean", "ci_lo", "ci_hi"]
    ).set_index("w")
    rate_stats = pd.DataFrame(
        _rows_rate, columns=["w", "mean", "ci_lo", "ci_hi"]
    ).set_index("w")
    past_stats = pd.DataFrame(
        _rows_past, columns=["w", "past", "mean", "ci_lo", "ci_hi"]
    ).set_index("w")
    return bias_stats, past_stats, rate_stats


@app.cell
def plot_sweep_bias(
    bias_stats, categorical, new_figure, plot_mean_ci_band, presentation_style
):
    _aqua = categorical(3)[2]
    with presentation_style():
        _fig, _ax = new_figure("standard")
        plot_mean_ci_band(_ax, bias_stats, _aqua, marker="o")
        _ax.axhline(0, color="white", linewidth=0.8)
        _ax.set_xlabel("WM weight w")
        _ax.set_ylabel("Bias: paid last time - did not")
        _fig.tight_layout()
    _fig


@app.cell
def plot_sweep_past(
    categorical, new_figure, past_stats, plot_mean_ci_band, presentation_style
):
    _aqua, _yellow = categorical(4)[2:]
    # same odor = aqua, other odor = yellow; solid = rewarded, dashed = not rewarded
    _styles = {
        "Same, rewarded": (_aqua, "-"),
        "Same, not rewarded": (_aqua, "--"),
        "Other, rewarded": (_yellow, "-"),
        "Other, not rewarded": (_yellow, "--"),
    }
    with presentation_style():
        _fig, _ax = new_figure("standard")
        for _cond, (_color, _ls) in _styles.items():
            plot_mean_ci_band(
                _ax,
                past_stats[past_stats["past"] == _cond],
                _color,
                label=f"Previous: {_cond.lower()}",
                marker="o",
                linestyle=_ls,
            )
        _ax.set_xlabel("WM weight w")
        _ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        _ax.set_ylim(-0.1, 1.1)
        _ax.set_ylabel("P(Choice)")
        _ax.legend(loc="center left", fontsize=10)
        _fig.tight_layout()
    _fig


@app.cell
def plot_sweep_rate(
    categorical, new_figure, plot_mean_ci_band, presentation_style, rate_stats
):
    _aqua = categorical(3)[2]
    with presentation_style():
        _fig, _ax = new_figure("standard")
        plot_mean_ci_band(_ax, rate_stats, _aqua, marker="o")
        _ax.set_xlabel("WM weight w")
        _ax.set_ylabel("Reward rate (per s)")
        _fig.tight_layout()
    _fig


@app.cell
def sensory_values(
    agent,
    bootstrap_mean_ci,
    categorical,
    ci_errorbar,
    new_figure,
    np,
    pd,
    presentation_style,
    rng,
):
    _adv = agent.q_sens[:, :, 1] - agent.q_sens[:, :, 0]  # (animals, odors)
    _stats = pd.DataFrame(
        [bootstrap_mean_ci(_adv[:, o], rng) for o in range(_adv.shape[1])],
        columns=["mean", "ci_lo", "ci_hi"],
    )
    _aqua = categorical(3)[2]

    with presentation_style():
        _fig, _ax = new_figure("standard")
        _ax.errorbar(
            np.arange(len(_stats)),
            _stats["mean"],
            yerr=ci_errorbar(_stats),
            fmt="o",
            color=_aqua,
        )
        _ax.axhline(0, color="white", linewidth=0.8)
        _ax.set_xticks(range(len(_stats)))
        _ax.set_xlabel("Odor")
        _ax.set_ylabel("Sensory Q(stay) - Q(leave)")
        _fig.tight_layout()
    _fig


if __name__ == "__main__":
    app.run()
