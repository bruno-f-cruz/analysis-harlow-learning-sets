# Harlow learning sets

## Task

Based on Harlow's learning sets. In each block of trials (usually 10), two odors are available: one is always rewarded, the other never is. Once the subject learns which odor is rewarded, the optimal policy is to stop at the rewarded odor and skip the other. The question is whether subjects learn the general rule ("if one odor is rewarded, the other isn't") and apply it to new pairs in later blocks.

There are 7 odors. Each block draws a pair; the next block draws from the remaining 5, after which the previous pair returns to the pool.

## Running

Interactive (marimo at `http://localhost:2718`):

```bash
# uv
uv run marimo edit workflows/pipeline.py --host 0.0.0.0 --port 2718

# docker
docker compose up -d dev
docker compose exec dev uv run marimo edit workflows/pipeline.py --host 0.0.0.0 --port 2718
```

Batch run (writes to `artifacts/runs/<run_id>/`):

```bash
# uv
uv run python workflows/pipeline.py
uv run python workflows/pipeline.py --dataset ABReversal   # curriculum stage: Full (default) or ABReversal

# docker (also tees the log to out.log in the run folder)
docker compose up prod
```

`workflows/within_session_performance.py` runs the same way.

## Data

Data is read directly from the shared, processed VR-foraging dataset on S3. No local data and no AWS credentials needed. Two git-tracked files pin what's read:

- `data_assets.json`: where the dataset lives (`s3://aind-scratch-data/vr-foraging/vr-foraging-dataset`).
- `raw_sessions.json`: which sessions to use (each entry's `mount` is a `session_id`).

Loading fails if any listed session is missing from the dataset.

### API

```python
from analysis.sessions import Dataset

dataset = Dataset.from_manifests("data_assets.json", "raw_sessions.json")
dataset.session  # one row per session
dataset.sites  # one row per site (trial)

# Per-animal tables, loaded on demand:
dataset.load_licks("841312")
dataset.load_position("841312")
dataset.load_sniffing("841312")

# Any per-session table (licks, position_velocity, sniffing, software_events) for any sessions:
dataset.load_table("software_events", ["841312_2026-06-04_20-19-36"])
```

Per-session tables come back stacked, with a `session_id` column. Position and sniffing are ~1M rows per session, so narrow the session list when you can. Pass `include_config=True` to also load the session table's large JSON config columns.

### Updating sessions

Edit `SUBJECT_IDS` / `START_DATE` in `scripts/attach_datasets.py`, then:

```bash
uv run scripts/attach_datasets.py          # add new matching sessions
uv run scripts/attach_datasets.py --prune  # replace the list with current matches
```

Commit the updated `raw_sessions.json`.

## Run outputs

Each batch run writes to `artifacts/runs/<run_id>/`:

- `manifest.json`: git commit, python version, status, timestamps
- `inputs.json`: dataset location, session ids, and size/etag of every dataset file read
- `dataset/`: copies of the dataset's `data_description.json` and `processing.json`
- `figures/`, `results/`, and `out.log` (docker only)

Output location is set by `ARTIFACT_URI` (default `./artifacts`). Writing to a private S3 bucket uses the standard AWS credential chain.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
```
