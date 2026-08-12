"""Generic, reusable data-prep / feature-construction helpers for the trials frame.

Moved out of the old top-level ``helpers.py`` (see
``docs/plans/2026-08-11-dockerized-analysis-environment.md``, Task 9.2). Only
the pieces that are pure data transformation (no plotting) and used by more
than one downstream consumer live here. The GLM-fitting / bias / counterfactual
*analysis* itself is intentionally **not** here -- that composition stays
inline in the ``workflows/analysis.py`` (formerly ``demo_marimo.py``) notebook
cells that use it.
"""

import numpy as np
import pandas as pd

# ─── Subject corrections ──────────────────────────────────────────────────────
#
# A session_id here is the data asset name, which happens to start with a subject
# id. For a few sessions that prefix is wrong -- the session was acquired under
# the wrong subject in the rig metadata. Upstream deliberately does NOT rename the
# assets (the name is only a unique key and is not meant to be parsed), so the
# correction has to be applied at load time on our side.
#
# https://github.com/AllenNeuralDynamics/aind-scientific-computing/issues/855

#: ``{session_id: correct_subject_id}``. Keep the session_id keys exactly as they
#: appear in the parquet; entries that match nothing are harmless, so a fix can be
#: recorded here before the corresponding session has been synced locally.
SESSION_SUBJECT_OVERRIDES = {
    # issue #855, first report: metadata subject was 841312
    "841312_2026-07-07_20-38-00": "866063",
    # issue #855, follow-up comment: metadata subject was 841299
    "841299_2026-07-29_17-02-58": "864846",
}


def subject_id_for(session_ids):
    """Correct subject id(s) for ``session_ids``, applying the #855 overrides.

    Accepts a single session_id or any Series/sequence of them, and returns the
    same shape (a ``str`` for scalar input, a ``Series`` of ``str`` otherwise).
    Never parse the subject out of a session_id directly -- use this, or the
    ``subject_id`` column that :func:`add_subject_id` writes.
    """
    if isinstance(session_ids, str):
        return SESSION_SUBJECT_OVERRIDES.get(session_ids, session_ids.split("_")[0])
    s = pd.Series(session_ids, dtype="object").astype(str)
    return (
        s.str.split("_")
        .str[0]
        .mask(s.isin(SESSION_SUBJECT_OVERRIDES), s.map(SESSION_SUBJECT_OVERRIDES))
    )


def add_subject_id(df: pd.DataFrame, session_col: str = "session_id") -> pd.DataFrame:
    """Return ``df`` with a corrected ``subject_id`` column (overwrites if present).

    This is the *only* place the correction is applied: call it once, right after
    loading the trials, and every frame derived from that one carries the column.
    The helpers below just group by ``subject_id`` and never re-derive it, so a
    frame that skipped this step raises ``KeyError`` there rather than quietly
    grouping by the wrong name prefix.
    """
    out = df.copy()
    out["subject_id"] = subject_id_for(out[session_col]).to_numpy()
    return out


def report_subject_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Which :data:`SESSION_SUBJECT_OVERRIDES` actually applied to ``df``.

    Returns one row per override with ``session_id``, ``name_prefix`` (the wrong
    subject still in the asset name), ``corrected_to`` and ``n_rows`` (0 when that
    session is not present locally). Print it after loading so a correction that
    silently stops matching -- e.g. because an asset was renamed after all -- is
    visible rather than assumed.
    """
    present = set(df["session_id"].unique())
    return pd.DataFrame(
        [
            {
                "session_id": sid,
                "name_prefix": sid.split("_")[0],
                "corrected_to": subject,
                "n_rows": int((df["session_id"] == sid).sum()),
                "present": sid in present,
            }
            for sid, subject in SESSION_SUBJECT_OVERRIDES.items()
        ]
    )


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


def _appearance_table(
    trials: pd.DataFrame, from_first_stop: bool = False
) -> pd.DataFrame:
    """Return RewardSite trials (kept blocks only) with an ``appearance`` column.

    ``appearance`` is the within-block, per-odor index (0-4). When
    ``from_first_stop`` is True the index origin is each block's first stop and
    trials before it are dropped (blocks with no stop are dropped entirely).
    Shared by the choice-by-block-position plots so they all count appearances
    identically.
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
