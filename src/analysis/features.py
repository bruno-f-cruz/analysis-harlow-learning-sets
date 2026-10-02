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


def assign_odor_blocks(trials: pd.DataFrame, max_block_size: int = 18) -> pd.DataFrame:
    """Add ``odor_block`` and ``site_in_odor_block``: block/position detected
    from the actual odor-reward contingency, not a fixed trial count.

    ``assign_blocks`` chunks every ``block_size`` (default 10) RewardSite
    trials regardless of curriculum stage. That matches the true block length
    for a "5Rew_5NonRew" stage (10 trials), but not "8Rew_10NonRew" (18
    trials) -- there, a fixed 10-count block is half of one real block, so
    "position 0" of the second half is really the 11th trial of an
    already-half-learned block, not a fresh one. There's no ground-truth
    column for this in the raw data (``block_index`` is constant 0 throughout
    -- unused by this task's software -- and ``site_index_in_block`` just
    mirrors the running site count).

    This instead starts a new ``odor_block`` whenever, for the current
    RewardSite trial's ``odor_index``: it isn't one of the (at most 2) odors
    already active in the current block, or it *is* one of them but its
    ``is_rewarded_odor`` now differs from what it was earlier in the same
    block (an immediate reward-mapping flip on an otherwise-repeated pair).
    ``site_in_odor_block`` is the 0-based position within that block.

    Undetectable case: a pair can immediately repeat with the *same* mapping
    too, and nothing observable changes at that boundary, so it would
    otherwise merge into one over-long block. ``max_block_size`` (default 18,
    the largest true block length seen in this dataset -- 8Rew_10NonRew) is a
    hard cap that forces a new block once ``pos`` reaches it regardless, so
    an undetected repeat is split too long by at most one block's worth
    rather than merged without limit.

    Kept as its own opt-in columns rather than replacing ``block`` -- nothing
    that already depends on ``block`` / ``p_stay_in_block`` /
    ``appearance_table``'s default ``by_odor=True`` path is affected.
    """
    trials = trials.copy()
    odor_block = pd.Series(pd.NA, index=trials.index, dtype="Int64")
    site_in_odor_block = pd.Series(pd.NA, index=trials.index, dtype="Int64")

    reward_sites = trials[trials["site_label"] == "RewardSite"]
    for _, grp in reward_sites.groupby("session_id", sort=False):
        ordered = grp.sort_values("start_time")
        block_id = 0
        pos = 0
        active: dict = {}  # odor_index -> is_rewarded_odor, within the current block
        block_ids = []
        positions = []
        for odor_index, is_rewarded in zip(
            ordered["odor_index"], ordered["is_rewarded_odor"]
        ):
            is_new_odor = odor_index not in active
            is_flip = (not is_new_odor) and (active[odor_index] != is_rewarded)
            if (is_new_odor and len(active) >= 2) or is_flip or pos >= max_block_size:
                block_id += 1
                pos = 0
                active = {}
            block_ids.append(block_id)
            positions.append(pos)
            active[odor_index] = is_rewarded
            pos += 1
        # drop the trailing block (session ends mid-block; see assign_blocks)
        block_ids = [
            bid if bid != block_id else np.nan for bid in block_ids
        ]
        odor_block.loc[ordered.index] = pd.array(block_ids, dtype="Int64")
        site_in_odor_block.loc[ordered.index] = pd.array(positions, dtype="Int64")

    trials["odor_block"] = odor_block
    trials["site_in_odor_block"] = site_in_odor_block
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


MAX_BLOCK_POSITION = 10


def appearance_table(
    trials: pd.DataFrame, from_first_stop: bool = False, by_odor: bool = True
) -> pd.DataFrame:
    """Return RewardSite trials (kept blocks only) with an ``appearance`` column.

    ``by_odor=True`` (default): ``appearance`` is the within-block, per-odor
    index (0-4; occurrences past ``MAX_APPEARANCE`` are dropped) -- how many
    times *that specific odor* has appeared in the block so far. Grouped by
    :func:`assign_blocks`'s fixed-count ``block``.
    ``by_odor=False``: ``appearance`` is instead the raw within-block position,
    pooling both odor types at each position -- the serial-position effect
    independent of which odor is up. Grouped by :func:`assign_odor_blocks`'s
    contingency-detected ``odor_block`` instead of ``block``, since ``block``'s
    fixed trial count is wrong for curriculum stages whose true block isn't
    that length (see :func:`assign_odor_blocks`); positions past
    ``MAX_BLOCK_POSITION`` (10) are dropped, same reasoning as ``MAX_APPEARANCE``
    above -- only the longest curriculum stage's blocks reach that far, so
    beyond it the pooled stat is thinning out rather than describing the whole
    dataset.

    When ``from_first_stop`` is True the index origin is each block's first
    stop and trials before it are dropped (blocks with no stop are dropped
    entirely). Shared by the choice-by-block-position plots so they all count
    appearances identically.
    """
    if by_odor:
        rs = trials[
            (trials["site_label"] == "RewardSite") & trials["block"].notna()
        ].copy()
        block_col = "block"
    else:
        rs = assign_odor_blocks(trials)
        rs = rs[
            (rs["site_label"] == "RewardSite") & rs["odor_block"].notna()
        ].copy()
        block_col = "odor_block"

    rs = rs.sort_values(["session_id", block_col, "start_time"])

    if from_first_stop:
        # position of the first stop within each block, then keep only trials
        # at or after it (blocks with no stop have no first_stop_pos -> dropped)
        rs["_pos"] = rs.groupby(["session_id", block_col]).cumcount()
        first_stop_pos = (
            rs[rs["has_choice"]].groupby(["session_id", block_col])["_pos"].min()
        )
        rs = rs.join(
            first_stop_pos.rename("_first_stop_pos"), on=["session_id", block_col]
        )
        rs = rs[rs["_first_stop_pos"].notna() & (rs["_pos"] >= rs["_first_stop_pos"])]

    group_keys = (
        ["session_id", block_col, "is_rewarded_odor"]
        if by_odor
        else ["session_id", block_col]
    )
    rs["appearance"] = rs.groupby(group_keys).cumcount()
    if by_odor:
        rs = rs[rs["appearance"] < MAX_APPEARANCE]
    else:
        rs = rs[rs["appearance"] < MAX_BLOCK_POSITION]
    return rs


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


def pooled_block_ordinal(trials: pd.DataFrame, block_col: str = "block") -> pd.DataFrame:
    """Chronological 0-based block number per animal, pooled across sessions.

    Only blocks actually present in ``trials`` are ranked, so a frame that has
    already had blocks filtered out (e.g. the degenerate ``p_stay in {0, 1}``
    blocks) yields a dense ordinal over the survivors rather than a gappy one.
    ``block_col`` defaults to :func:`assign_blocks`'s fixed-count ``block``;
    pass ``"odor_block"`` to rank :func:`assign_odor_blocks`'s
    contingency-detected blocks instead (``trials`` must already have that
    column in that case).

    Returns one row per ``(subject_id, session_id, block_col)`` plus
    ``block_ordinal``. ``session_id`` encodes the acquisition datetime, so
    sorting it lexicographically is chronological.
    """
    rs = trials[(trials["site_label"] == "RewardSite") & trials[block_col].notna()]
    keys = (
        rs[["subject_id", "session_id", block_col]]
        .drop_duplicates()
        .sort_values(["subject_id", "session_id", block_col])
        .reset_index(drop=True)
    )
    keys["block_ordinal"] = keys.groupby("subject_id").cumcount()
    return keys


def block_tercile_tags(trials: pd.DataFrame, block_col: str = "block") -> pd.DataFrame:
    """Tag every block with which chronological third of that animal's blocks
    it falls in: ``"first"``, ``"mid"``, or ``"last"``.

    Boundaries are drawn per-animal (thirds of that animal's own total block
    count via :func:`pooled_block_ordinal`, pooled chronologically across
    sessions) rather than a global block count, so an animal with fewer
    blocks still gets a fair first/mid/last split instead of being dominated
    by animals with more. Meant for a "does this curve change over training"
    facet -- see :func:`analysis.plotting.plot_choice_by_block_position_raw_by_tercile`.

    Returns one row per ``(subject_id, session_id, block_col)`` plus
    ``block_ordinal`` and ``block_tercile``.
    """
    ordinals = pooled_block_ordinal(trials, block_col=block_col)
    n_blocks = ordinals.groupby("subject_id")["block_ordinal"].transform("max") + 1
    frac = (ordinals["block_ordinal"] + 0.5) / n_blocks
    ordinals["block_tercile"] = pd.cut(
        frac, bins=[0, 1 / 3, 2 / 3, 1], labels=["first", "mid", "last"], include_lowest=True
    )
    return ordinals


def blocks_first_last_tags(
    trials: pd.DataFrame, n_blocks: int = 200, last_n_blocks: int | None = None
) -> pd.DataFrame:
    """RewardSite trials from each animal's first/last blocks (pooled
    chronologically across sessions, see :func:`pooled_block_ordinal`).

    ``n_blocks`` sizes the "first" window; ``last_n_blocks`` (default: same
    as ``n_blocks``) sizes the "last" window independently, for an
    asymmetric comparison (e.g. first 30 vs last 100 blocks).

    Adds a ``block_range`` column ("first" or "last"); trials in neither
    window are dropped. An animal with fewer than ``n_blocks + last_n_blocks``
    blocks has its middle blocks claimed by "first" before "last" is
    considered, so the "last" window comes up short there rather than the
    two overlapping.
    """
    last_n_blocks = n_blocks if last_n_blocks is None else last_n_blocks
    ordinals = pooled_block_ordinal(trials)
    max_ordinal = ordinals.groupby("subject_id")["block_ordinal"].transform("max")
    ordinals = ordinals.assign(
        block_range=np.select(
            [
                ordinals["block_ordinal"] < n_blocks,
                ordinals["block_ordinal"] > max_ordinal - last_n_blocks,
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
