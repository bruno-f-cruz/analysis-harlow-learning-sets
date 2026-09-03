"""Generic, reusable data-prep / feature-construction helpers for the trials frame.

Pure data transformation, no plotting. The GLM-fitting, odor-identity-bias
and counterfactual analyses each have their own module
(:mod:`analysis.glm`, :mod:`analysis.bias`, :mod:`analysis.counterfactual`).
"""

import numpy as np
import pandas as pd


def assign_blocks(trials: pd.DataFrame, block_size: int = 10) -> pd.DataFrame:
    """Add a ``block`` column grouping consecutive RewardSite trials into blocks.

    Within each session, RewardSite trials (in temporal order) are grouped into
    blocks of ``block_size``. Block numbering resets to 0 at the start of every
    session. The last block of each session is always dropped (set to NaN) since
    the session ends mid-block and those trials are never used -- this applies
    even when the final block happens to be complete. Non-RewardSite rows
    (InterSite / InterPatch) are always NaN.
    """
    block = pd.Series(pd.NA, index=trials.index, dtype="Int64")
    reward_sites = trials[trials["site_label"] == "RewardSite"]
    for _, grp in reward_sites.groupby("session_id", sort=False):
        ordered_idx = grp.sort_values("start_time").index
        local = (np.arange(len(ordered_idx)) // block_size).astype(float)
        local[local == local.max()] = np.nan  # drop the trailing block
        block.loc[ordered_idx] = pd.array(local, dtype="Int64")
    trials["block"] = block
    return trials


def trim_sessions(
    trials: pd.DataFrame, start_frac: float = 0.0, end_frac: float = 1.0
) -> pd.DataFrame:
    """Keep the ``[start_frac, end_frac]`` window of each session's rows.

    Within each session, rows are ordered chronologically (by ``start_time``)
    and only the fractional window from ``start_frac`` to ``end_frac`` is kept.
    Both are fractions of the session measured from its start, so e.g.
    ``start_frac=0.0, end_frac=0.7`` keeps the first 70% and
    ``start_frac=0.1, end_frac=0.9`` keeps the middle 80%. The default keeps the
    whole session. Returns a new trimmed DataFrame.
    """
    kept = []
    for _, grp in trials.groupby("session_id", sort=False):
        g = grp.sort_values("start_time")
        n = len(g)
        lo = int(round(n * start_frac))
        hi = int(round(n * end_frac))
        kept.append(g.iloc[lo:hi])
    return pd.concat(kept) if kept else trials.iloc[:0]


def prepare_trials(
    trials: pd.DataFrame,
    session_table: pd.DataFrame,
    min_session_minutes: float = 15.0,
    degenerate_margin: float = 0.1,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge in ``subject_id``, assign blocks, and derive the columns most
    analyses need, from a raw ``sites`` table plus its session-level table.

    ``start_frac``/``end_frac`` trim each session to that fractional window
    (see :func:`trim_sessions`) before anything else is derived -- e.g.
    ``start_frac=0.1, end_frac=0.9`` drops the first and last 10% of every
    session.

    Returns ``(trials, trials_all)``: ``trials`` has degenerate blocks
    (``p_stay_in_block`` within ``degenerate_margin`` of 0 or 1) dropped;
    ``trials_all`` is the same frame *before* that filter, for analyses that
    need the degenerate blocks too (e.g. within-session performance decay,
    where "stops at everything" late in a session is exactly the effect
    being looked for, not noise to discard).
    """
    trials = trials.merge(
        session_table[["session_id", "subject_id"]],
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    missing_subject = trials["subject_id"].isna()
    if missing_subject.any():
        raise ValueError(
            "No subject_id in the session table for session_id(s): "
            f"{sorted(trials.loc[missing_subject, 'session_id'].unique())}"
        )
    trials = assign_blocks(trials)

    # Drop sessions shorter than min_session_minutes (first to last site timestamp)
    session_start = trials.groupby("session_id")["start_time"].min()
    session_end = trials.groupby("session_id")["start_time"].max()
    session_duration = session_end - session_start
    threshold = (
        pd.Timedelta(minutes=min_session_minutes)
        if pd.api.types.is_timedelta64_dtype(session_duration)
        else min_session_minutes * 60
    )
    long_sessions = session_duration[session_duration >= threshold].index
    trials = trials[trials["session_id"].isin(long_sessions)]
    trials = trim_sessions(trials, start_frac=start_frac, end_frac=end_frac)

    def _is_rewarded(patch_label: str) -> bool:
        return "NonRewarded" not in patch_label

    def _odor_index(odor_concentration) -> int:
        return np.argmax(np.array(odor_concentration))

    trials["is_rewarded_odor"] = trials["patch_label"].apply(_is_rewarded)
    trials["odor_index"] = trials["odor_concentration"].apply(_odor_index)

    # Per-block P(stay): fraction of a block's RewardSite trials the animal stopped.
    rs_mask = (trials["site_label"] == "RewardSite") & trials["block"].notna()
    trials["p_stay_in_block"] = (
        trials[rs_mask].groupby(["session_id", "block"])["has_choice"].transform("mean")
    )

    trials_all = trials.copy(deep=False)

    block_p_stay = trials[rs_mask].groupby(["session_id", "block"])["has_choice"].mean()
    is_degenerate_block = (block_p_stay <= degenerate_margin) | (
        block_p_stay >= 1 - degenerate_margin
    )
    excluded_per_session = is_degenerate_block.groupby(level="session_id").agg(
        excluded="sum", total="count"
    )
    excluded_per_session["kept"] = (
        excluded_per_session["total"] - excluded_per_session["excluded"]
    )
    print(
        f"Blocks excluded per session (p_stay <= {degenerate_margin} or "
        f">= {1 - degenerate_margin}):"
    )
    print(excluded_per_session.to_string())
    print(
        f"\nTotal: excluding {excluded_per_session['excluded'].sum()} "
        f"of {excluded_per_session['total'].sum()} blocks"
    )

    # NaN (non-RewardSite / unblocked rows) must survive this filter -- both
    # comparisons below are False for NaN, so `~` keeps them.
    row_is_degenerate = (trials["p_stay_in_block"] <= degenerate_margin) | (
        trials["p_stay_in_block"] >= 1 - degenerate_margin
    )
    trials = trials[~row_is_degenerate]

    return trials, trials_all


#: Trials beyond the 5th occurrence of an odor type within a block are
#: dropped -- most blocks cap each type at 5, and stages with a higher cap
#: (e.g. 8Rew/10NonRew) would otherwise skew appearance-indexed plots with
#: occurrences the majority of blocks never reach.
MAX_APPEARANCE = 5


def appearance_table(
    trials: pd.DataFrame, from_first_stop: bool = False
) -> pd.DataFrame:
    """Return RewardSite trials (kept blocks only) with an ``appearance`` column.

    ``appearance`` is the within-block, per-odor index (0-4; occurrences past
    ``MAX_APPEARANCE`` are dropped). When ``from_first_stop`` is True the
    index origin is each block's first stop and trials before it are dropped
    (blocks with no stop are dropped entirely). Shared by the
    choice-by-block-position plots so they all count appearances identically.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()].copy()
    rs = rs.sort_values(["session_id", "block", "start_time"])

    if from_first_stop:
        # position of the first stop within each block, then keep only trials
        # at or after it (blocks with no stop have no first_stop_pos -> dropped)
        rs["_pos"] = rs.groupby(["session_id", "block"]).cumcount()
        first_stop_pos = (
            rs[rs["has_choice"]].groupby(["session_id", "block"])["_pos"].min()
        )
        rs = rs.join(
            first_stop_pos.rename("_first_stop_pos"), on=["session_id", "block"]
        )
        rs = rs[rs["_first_stop_pos"].notna() & (rs["_pos"] >= rs["_first_stop_pos"])]

    rs["appearance"] = rs.groupby(
        ["session_id", "block", "is_rewarded_odor"]
    ).cumcount()
    return rs[rs["appearance"] < MAX_APPEARANCE]


def label_first_stop(trials: pd.DataFrame) -> pd.DataFrame:
    """Add ``first_stop_rewarded``: per block, whether the first stop was rewarded.

    For each ``(session_id, block)``, RewardSite trials are scanned in temporal
    order for the first one where the animal stopped (``has_choice``); that
    stop's ``has_reward`` becomes the block-level label (constant for every row
    of the block). Blocks where the animal never stopped get ``<NA>`` and are
    thus excluded from the conditioned plots. Requires ``block`` to be assigned.
    """
    trials = trials.copy()
    label = pd.Series(pd.NA, index=trials.index, dtype="boolean")
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    for _, grp in rs.groupby(["session_id", "block"]):
        ordered = grp.sort_values("start_time")
        stops = ordered[ordered["has_choice"]]
        if stops.empty:
            continue  # animal never stopped in this block -> leave <NA>
        label.loc[ordered.index] = bool(stops.iloc[0]["has_reward"])
    trials["first_stop_rewarded"] = label
    return trials


# ─── Block windows ────────────────────────────────────────────────────────────
#
# Session boundaries are an artefact of how the data was collected, not of how the
# animal learns. These helpers treat each animal as if all of its data came from a
# single long session: every block is pooled in chronological order and that stream
# is then cut into sliding windows of a fixed number of blocks. ``skip == window``
# gives non-overlapping windows, ``skip < window`` overlapping ones, and
# ``skip > window`` leaves gaps. Only *full* windows are emitted.
#
# Used by both the per-window GLM fit and the per-window counterfactual matrix
# in the notebook, which is why this lives here rather than inline in either.


def pooled_block_ordinal(trials: pd.DataFrame) -> pd.DataFrame:
    """Chronological 0-based block number per animal, pooled across sessions.

    Only blocks actually present in ``trials`` are ranked, so a frame that has
    already had blocks filtered out (e.g. the degenerate ``p_stay in {0, 1}``
    blocks) yields a dense ordinal over the survivors rather than a gappy one.

    Returns one row per ``(subject_id, session_id, block)`` plus
    ``block_ordinal``. ``session_id`` encodes the acquisition datetime, so
    sorting it lexicographically is chronological.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    keys = (
        rs[["subject_id", "session_id", "block"]]
        .drop_duplicates()
        .sort_values(["subject_id", "session_id", "block"])
        .reset_index(drop=True)
    )
    keys["block_ordinal"] = keys.groupby("subject_id").cumcount()
    return keys


def blocks_first_last_tags(trials: pd.DataFrame, n_blocks: int = 200) -> pd.DataFrame:
    """RewardSite trials from each animal's first/last ``n_blocks`` blocks
    (pooled chronologically across sessions, see :func:`pooled_block_ordinal`).

    Adds a ``block_range`` column ("first" or "last"); trials in neither
    window are dropped. An animal with fewer than ``2 * n_blocks`` blocks has
    its middle blocks claimed by "first" before "last" is considered, so the
    "last" window comes up short there rather than the two overlapping.
    """
    ordinals = pooled_block_ordinal(trials)
    max_ordinal = ordinals.groupby("subject_id")["block_ordinal"].transform("max")
    ordinals = ordinals.assign(
        block_range=np.select(
            [
                ordinals["block_ordinal"] < n_blocks,
                ordinals["block_ordinal"] > max_ordinal - n_blocks,
            ],
            ["first", "last"],
            default=None,
        )
    )
    ordinals = ordinals[ordinals["block_range"].notna()]

    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    return rs.merge(
        ordinals[["subject_id", "session_id", "block", "block_range"]],
        on=["subject_id", "session_id", "block"],
        how="inner",
    )


def block_window_index(
    trials: pd.DataFrame, window_blocks: int, skip_blocks: int
) -> pd.DataFrame:
    """Map every block to the sliding windows that contain it.

    Windows start at the block ordinals ``0, skip_blocks, 2 * skip_blocks, ...``
    and span ``window_blocks`` blocks each. Only windows with the full
    ``window_blocks`` are emitted, so an animal's trailing blocks are dropped
    when its count is not an exact multiple of the stride and every window rests
    on the same amount of data.

    Returns ``(subject_id, session_id, block, block_ordinal, window,
    window_start, window_end)`` -- one row per (block, window) pair, so an
    overlapping window layout lists a block once per window containing it.
    """
    if window_blocks < 1 or skip_blocks < 1:
        raise ValueError(
            f"window_blocks and skip_blocks must both be >= 1, "
            f"got {window_blocks} and {skip_blocks}"
        )

    keys = pooled_block_ordinal(trials)
    parts = []
    for _, grp in keys.groupby("subject_id", sort=False):
        starts = range(0, len(grp) - window_blocks + 1, skip_blocks)
        for window, start in enumerate(starts):
            end = start + window_blocks - 1
            part = grp[grp["block_ordinal"].between(start, end)].copy()
            part["window"] = window
            part["window_start"] = start
            part["window_end"] = end
            parts.append(part)

    if not parts:
        longest = keys.groupby("subject_id").size().max() if len(keys) else 0
        raise ValueError(
            f"no animal has {window_blocks} blocks to fill a window "
            f"(the longest-running one has {longest})"
        )
    return pd.concat(parts, ignore_index=True)


def expand_to_block_windows(
    trials: pd.DataFrame, window_blocks: int, skip_blocks: int
) -> pd.DataFrame:
    """RewardSite trials tagged with the ``window`` they belong to.

    Rows are duplicated once per containing window when the windows overlap, so
    grouping the result by ``["subject_id", "window"]`` gives each window's
    trials. Windowing is done over the blocks present in ``trials``
    (see :func:`pooled_block_ordinal`).
    """
    windows = block_window_index(trials, window_blocks, skip_blocks)
    rs = trials[(trials["site_label"] == "RewardSite") & trials["block"].notna()]
    return rs.merge(windows, on=["subject_id", "session_id", "block"], how="inner")
