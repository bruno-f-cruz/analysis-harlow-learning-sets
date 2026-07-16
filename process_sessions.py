"""Process all sessions in the ``data/`` directory and write per-modality
concatenated Parquet files to ``data/processed/``.

Each output file gets a ``session_id`` column so rows can always be traced back
to their origin session.

Trials and position/velocity are built via the version-dispatching factory
functions in ``aind_behavior_vr_foraging_packaging.pipeline``
(``get_trial_table_processor`` / ``get_position_velocity_processor``), which
select the legacy or current processor variant based on each session's
dataset version. Licks has no legacy variant, so its processor is used
directly. Each processor's ``.compute()`` returns a DataFrame indexed by harp
time (except ``trials``, which is one row per site) and stamps provenance
(package / data-contract / dataset versions) into ``df.attrs``.

Usage
-----
::

    uv run python process_sessions.py             # process all sessions
    uv run python process_sessions.py --force     # re-process even if output exists

Tables written
--------------
data/processed/trials.parquet            one row per site
data/processed/position_velocity.parquet indexed by harp time; position (cm), velocity (cm/s)
data/processed/licks.parquet              indexed by harp time; is_lick_onset (True=onset, False=offset)
data/processed/sessions.parquet          one row per session with metadata + versions

Following the upstream package's own convention (see
``scripts/example_parquet_pipeline.py``), the harp-time index on
position_velocity/licks is written as-is via ``to_parquet()`` (``index=True``)
rather than flattened into a plain column; trials/sessions have no meaningful
index, so they're written with ``index=False``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
_log = logging.getLogger(__name__)


def _packaging_version() -> str | None:
    try:
        return _pkg_version("aind-behavior-vr-foraging-packaging")
    except PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Per-table extractors
# ---------------------------------------------------------------------------


def _build_sessions(
    local_dir: Path,
    session_id: str,
    n_trials: int,
    provenance: dict,
) -> pd.DataFrame:
    row: dict = {
        "session_id": session_id,
        "experimenter": None,
        "rig_name": None,
        "task_version": None,
        "session_name": None,
        "notes": None,
        "n_trials": n_trials,
        "dataset_version": provenance.get("dataset_version"),
        "data_contract_version": provenance.get("data_contract_version"),
        "packaging_version": provenance.get("packaging_version") or _packaging_version(),
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

_TABLE_NAMES = ("trials", "position_velocity", "licks", "sessions")
#: Tables indexed by harp time; written with their pandas index intact
#: (``to_parquet()`` default), matching the upstream package's own convention.
_INDEXED_TABLES = frozenset({"position_velocity", "licks"})


def process_all(data_root: Path, force: bool = False) -> None:
    from aind_behavior_vr_foraging.data_contract import dataset as load_dataset
    from aind_behavior_vr_foraging_packaging.pipeline import (
        get_position_velocity_processor,
        get_trial_table_processor,
    )
    from aind_behavior_vr_foraging_packaging.processing import LicksProcessor

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
            ds = load_dataset(session_dir)
        except Exception:
            _log.warning("Failed to load dataset for %s", session_id, exc_info=True)
            continue

        try:
            trials_df = get_trial_table_processor(ds, raise_on_error=False).compute()
            pos_df = get_position_velocity_processor(ds, raise_on_error=False).compute()
            licks_df = LicksProcessor(ds, raise_on_error=False).compute()
        except Exception:
            _log.warning("Failed to process %s — skipping session", session_id, exc_info=True)
            continue

        sessions_df = _build_sessions(
            session_dir, session_id, n_trials=len(trials_df), provenance=trials_df.attrs
        )

        for name, df in (
            ("trials", trials_df),
            ("position_velocity", pos_df),
            ("licks", licks_df),
            ("sessions", sessions_df),
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
        out_path = processed_dir / f"{name}.parquet"
        if name in _INDEXED_TABLES:
            # Preserve each session's harp-time index rather than flattening it
            # into a column (mirrors the upstream package's own to_parquet() usage).
            combined = pd.concat(parts)
            combined.to_parquet(out_path)
        else:
            combined = pd.concat(parts, ignore_index=True)
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
