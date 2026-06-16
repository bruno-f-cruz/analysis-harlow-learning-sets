"""Process all sessions in the ``data/`` directory and write per-modality
concatenated Parquet files to ``data/processed/``.

Each output file gets a ``session_id`` column so rows can always be traced back
to their origin session.

Usage
-----
::

    uv run python process_sessions.py             # process all sessions
    uv run python process_sessions.py --force     # re-process even if output exists

Tables written
--------------
data/processed/trials.parquet
data/processed/position.parquet
data/processed/licks.parquet
data/processed/sniffing.parquet
data/processed/sessions.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
_log = logging.getLogger(__name__)

_VELOCITY_SMOOTH_WINDOW = 11

# ---------------------------------------------------------------------------
# Per-table extractors (mirror the PackagingProcessingAdapter logic)
# ---------------------------------------------------------------------------


def _build_trials(ds) -> tuple[pd.DataFrame, str]:
    try:
        from aind_behavior_vr_foraging_packaging.processing._trial_table import (
            TrialTableProcessor,
        )

        proc = TrialTableProcessor(ds, raise_on_error=False)
        dataset_version = str(proc.dataset_version)
        if proc.dataset_version != proc.parser_version:
            _log.warning(
                "Dataset version %s != parser version %s; values may be coerced",
                proc.dataset_version,
                proc.parser_version,
            )
        sites = proc.process_to_sites()
        if not sites:
            _log.warning("TrialTableProcessor returned no sites")
            return pd.DataFrame(), dataset_version
        return pd.DataFrame([s.model_dump() for s in sites]), dataset_version
    except Exception:
        _log.warning("Failed to build trials table", exc_info=True)
        return pd.DataFrame(), "unknown"


def _build_position(ds) -> pd.DataFrame:
    try:
        raw = ds.at("Behavior").at("OperationControl").at("CurrentPosition").load().data
        series = raw["Position"].sort_index()
        series = series[~series.index.duplicated(keep="first")]
        if len(series) < 2:
            return pd.DataFrame(columns=["time", "position", "velocity"])
        t = series.index.to_numpy(dtype=float)
        p = series.to_numpy(dtype=float)
        raw_vel = np.gradient(p, t)
        smoothed = (
            pd.Series(raw_vel, index=series.index)
            .rolling(_VELOCITY_SMOOTH_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
        return pd.DataFrame({"time": t, "position": p, "velocity": smoothed})
    except Exception:
        _log.warning("Failed to build position table", exc_info=True)
        return pd.DataFrame()


def _build_licks(ds) -> pd.DataFrame:
    try:
        raw = ds.at("Behavior").at("HarpLickometer").at("LickState").load().data
        channel = raw["Channel0"].astype(bool).sort_index()
        onsets = channel[channel & ~channel.shift(1, fill_value=False)]
        return pd.DataFrame({"time": onsets.index.to_numpy(dtype=float), "channel": 0})
    except Exception:
        _log.warning("Failed to build licks table", exc_info=True)
        return pd.DataFrame()


def _build_sniffing(ds) -> pd.DataFrame:
    try:
        from aind_behavior_vr_foraging_packaging.processing._sniffing import (
            SniffingProcessor,
        )

        proc = SniffingProcessor(ds)
        sniff_signal, sampling_rate = proc.compute_sniff_signal(ds)
        return pd.DataFrame(
            {
                "time": sniff_signal.index.to_numpy(dtype=float),
                "sniff_signal": sniff_signal.to_numpy(dtype=float),
                "sampling_rate_hz": float(sampling_rate),
            }
        )
    except Exception:
        _log.warning("Failed to build sniffing table", exc_info=True)
        return pd.DataFrame()


def _build_sessions(local_dir: Path, session_id: str, n_trials: int) -> pd.DataFrame:
    row: dict = {
        "session_id": session_id,
        "experimenter": None,
        "rig_name": None,
        "task_version": None,
        "session_name": None,
        "notes": None,
        "n_trials": n_trials,
    }

    logs_dir = _find_logs_dir(local_dir)
    if logs_dir is None:
        _log.warning("Behavior/Logs directory not found under %s", local_dir)
        return pd.DataFrame([row])

    session_fp = logs_dir / "session_input.json"
    if session_fp.exists():
        try:
            from aind_behavior_services.session import Session

            session = Session.model_validate_json(session_fp.read_text())
            exp = session.experimenter
            row["experimenter"] = (
                exp
                if isinstance(exp, str)
                else (", ".join(str(e) for e in exp) if exp else None)
            )
            row["session_name"] = str(getattr(session, "session_name", "") or "")
            row["notes"] = str(getattr(session, "notes", "") or "")
        except Exception:
            _log.debug("Could not parse session_input.json", exc_info=True)

    task_fp = logs_dir / "tasklogic_input.json"
    if task_fp.exists():
        try:
            task_data = json.loads(task_fp.read_text())
            row["task_version"] = str(task_data.get("version", "") or "")
        except Exception:
            _log.debug("Could not parse tasklogic_input.json", exc_info=True)

    rig_fp = logs_dir / "rig_input.json"
    if rig_fp.exists():
        try:
            rig_data = json.loads(rig_fp.read_text())
            row["rig_name"] = str(rig_data.get("rig_name", "") or "")
        except Exception:
            _log.debug("Could not parse rig_input.json", exc_info=True)

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_logs_dir(local_dir: Path) -> Path | None:
    for behavior in ("behavior", "Behavior"):
        candidate = local_dir / behavior / "Logs"
        if candidate.is_dir():
            return candidate
    return None


def _session_dirs(data_root: Path) -> list[Path]:
    """Return subdirectories of *data_root* that look like session folders
    (``<subject_id>_<YYYY-MM-DD>_<HH-MM-SS>``).
    """
    pattern = re.compile(r"^\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
    return sorted(
        p for p in data_root.iterdir() if p.is_dir() and pattern.match(p.name)
    )


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

_TABLE_NAMES = ("trials", "position", "licks", "sniffing", "sessions")


def process_all(data_root: Path, force: bool = False) -> None:
    from aind_behavior_vr_foraging.data_contract import dataset as load_dataset

    processed_dir = data_root / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Check which tables already exist so we can skip if not forced
    existing = {
        name for name in _TABLE_NAMES if (processed_dir / f"{name}.parquet").exists()
    }
    if existing and not force:
        _log.info(
            "Found existing output for: %s. Pass --force to recompute.",
            ", ".join(sorted(existing)),
        )

    session_dirs = _session_dirs(data_root)
    if not session_dirs:
        _log.error("No session directories found under %s", data_root)
        return

    _log.info("Found %d session(s) under %s", len(session_dirs), data_root)

    accumulators: dict[str, list[pd.DataFrame]] = {name: [] for name in _TABLE_NAMES}

    for session_dir in tqdm(session_dirs, desc="Sessions", unit="session"):
        session_id = session_dir.name
        _log.info("Processing %s", session_id)

        try:
            ds = load_dataset(str(session_dir))
        except Exception:
            _log.warning("Failed to load dataset for %s", session_id, exc_info=True)
            continue

        trials_df, _ = _build_trials(ds)
        pos_df = _build_position(ds)
        licks_df = _build_licks(ds)
        sniff_df = _build_sniffing(ds)
        sessions_df = _build_sessions(session_dir, session_id, n_trials=len(trials_df))

        for df, name in (
            (trials_df, "trials"),
            (pos_df, "position"),
            (licks_df, "licks"),
            (sniff_df, "sniffing"),
            (sessions_df, "sessions"),
        ):
            if df.empty:
                continue
            # Prepend session_id to every table except sessions (already has it)
            if "session_id" not in df.columns:
                df = df.copy()
                df.insert(0, "session_id", session_id)
            accumulators[name].append(df)

    # Concatenate and write
    for name in _TABLE_NAMES:
        parts = accumulators[name]
        if not parts:
            _log.warning("No data collected for table '%s' — skipping", name)
            continue
        combined = pd.concat(parts, ignore_index=True)
        out_path = processed_dir / f"{name}.parquet"
        combined.to_parquet(out_path, index=False)
        _log.info("Wrote %s rows → %s", len(combined), out_path)

    _log.info("Done. Output in %s", processed_dir)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process all sessions → data/processed/*.parquet"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Root directory containing session folders (default: data/ next to this script)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even when output parquet files already exist",
    )
    args = parser.parse_args()
    process_all(args.data_dir, force=args.force)
