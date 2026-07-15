"""Process all sessions in the ``data/`` directory and write per-modality
concatenated Parquet files to ``data/processed/``.

Each output file gets a ``session_id`` column so rows can always be traced back
to their origin session.

The per-modality tables are produced by the processor classes shipped in
``aind_behavior_vr_foraging_packaging.processing`` (``TrialTableProcessor``,
``PositionAndVelocityProcessor``, ``LicksProcessor``, ``SniffingProcessor``).
Each processor exposes ``.compute()``, which returns a DataFrame indexed by
harp time and stamps provenance (package / data-contract / dataset versions)
into ``df.attrs``.

Usage
-----
::

    uv run python process_sessions.py             # process all sessions
    uv run python process_sessions.py --force     # re-process even if output exists

Tables written
--------------
data/processed/trials.parquet     one row per site (from TrialTableProcessor)
data/processed/position.parquet   time, position (cm), velocity (cm/s)
data/processed/licks.parquet      time, is_lick_onset (True=onset, False=offset)
data/processed/sniffing.parquet   time, voltage (V), sampling_rate_hz
data/processed/sessions.parquet   one row per session with metadata + versions
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
# Per-table extractors (thin wrappers over the packaging processor classes)
# ---------------------------------------------------------------------------


def _run_processor(processor_cls, ds, **kwargs) -> pd.DataFrame:
    """Instantiate *processor_cls*, run ``.compute()`` and return a DataFrame
    with the harp-time index promoted to a ``time`` column.

    Any scalar provenance stamped into ``df.attrs`` by the processor (e.g.
    ``sampling_rate_hz`` from :class:`SniffingProcessor`) is carried into a
    column. Returns an empty frame on failure.
    """
    try:
        df = processor_cls(ds, **kwargs).compute()
    except Exception:
        _log.warning("Failed to build %s table", processor_cls.__name__, exc_info=True)
        return pd.DataFrame()

    attrs = dict(df.attrs)
    df = df.rename_axis("time").reset_index()
    if "sampling_rate_hz" in attrs:
        df["sampling_rate_hz"] = float(attrs["sampling_rate_hz"])
    return df


def _build_trials(processor_cls, ds) -> tuple[pd.DataFrame, dict]:
    """Return the per-site trial table and a provenance dict for the session."""
    provenance: dict = {
        "dataset_version": None,
        "data_contract_version": None,
        "packaging_version": _packaging_version(),
    }
    try:
        proc = processor_cls(ds, raise_on_error=False)
        provenance["dataset_version"] = str(proc.dataset_version)
        provenance["data_contract_version"] = str(proc.parser_version)
        if proc.dataset_version != proc.parser_version:
            _log.warning(
                "Dataset version %s != parser version %s; values may be coerced",
                proc.dataset_version,
                proc.parser_version,
            )
        sites = proc.process_to_sites()
        if not sites:
            _log.warning("TrialTableProcessor returned no sites")
            return pd.DataFrame(), provenance
        return pd.DataFrame([s.model_dump() for s in sites]), provenance
    except Exception:
        _log.warning("Failed to build trials table", exc_info=True)
        return pd.DataFrame(), provenance


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
        "packaging_version": provenance.get("packaging_version"),
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
    from aind_behavior_vr_foraging_packaging.processing import (
        LicksProcessor,
        PositionAndVelocityProcessor,
        SniffingProcessor,
        TrialTableProcessor,
    )

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

        trials_df, provenance = _build_trials(TrialTableProcessor, ds)
        pos_df = _run_processor(PositionAndVelocityProcessor, ds)
        licks_df = _run_processor(LicksProcessor, ds)
        sniff_df = _run_processor(SniffingProcessor, ds)
        sessions_df = _build_sessions(
            session_dir, session_id, n_trials=len(trials_df), provenance=provenance
        )

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
