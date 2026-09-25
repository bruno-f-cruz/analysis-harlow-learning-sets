## Experiment

<add new details to this file as you see fit>
The experiment is based on Harlow's learning sets.O
On each block of trials (usually 10 but sometimes varies), two odors are available. One of these odors is ALWAYS rewarded, the other one is not. The optimal policy should be to choose the rewarded odor and "skip" non-rewarded odors once the subject has learned which odor is rewarded. The experiment is designed to test the subject's ability to learn and adapt to changing reward contingencies over time, while keeping the ability to learn the simple rule: if one odor is rewarded, the other is not. The subject should be able to learn this rule and apply it to new pairs of odors presented in subsequent blocks of trials.

We do not have infinite many odors as a result we use 7 distinct odors. In each block we draw a pair. The next block we will draw from the remaining 5 odors, and add the previously used pair to the pool of available odors. This way, the subject will be presented with new pairs of odors in each block, while still being able to apply the learned rule from previous blocks.

## Quickstart (local)

```bash
git clone <repo> && cd <repo>
docker compose up -d dev
docker compose exec dev bash   # or: attach VS Code via "Reopen in Container"
uv run marimo edit workflows/pipeline.py --host 0.0.0.0 --port 2718
```

Open `http://localhost:2718` to explore the pipeline/analysis interactively.

## Running the analysis non-interactively

Either of these are equivalent:

```bash
docker compose up prod
# or, without Docker:
uv run python workflows/pipeline.py
# or, via the documented entrypoint shim:
uv run python scripts/run.py
```

`docker compose up prod` uses `scripts/start_prod.sh`, which generates the `run_id`, pre-creates the run directory, and tees all stdout/stderr to `out.log` there while still printing to the terminal.

## Codespaces

Open this repository in GitHub Codespaces. It uses the same devcontainer (`.devcontainer/devcontainer.json`) as local VS Code — port `2718` (marimo) auto-forwards. All the commands above work identically.

## What gets analyzed

The data comes from the shared, already-processed VR-foraging dataset (`session.parquet` + `sites.parquet`, covering every VR-foraging session, not just this experiment's). This repo does no processing of its own and keeps no local data. Two git-tracked manifests (Code-Ocean-style `attached_datasets` lists of `{id, mount, location}`) pin what gets read:

- **`data_assets.json`**: *where* the dataset lives. It holds a single entry, currently `s3://aind-scratch-data/vr-foraging/vr-foraging-dataset`. Hand-edit it to point at a different build.
- **`raw_sessions.json`**: *which* sessions of it this analysis uses. Each entry's `mount` is a `session_id`.

`analysis.sessions.Dataset` ties them together:

```python
from analysis.sessions import Dataset

dataset = Dataset.from_manifests("data_assets.json", "raw_sessions.json")
dataset.session   # one row per selected session
dataset.sites     # one row per site (trial)
```

Only the selected sessions are read (the filter is pushed down into the Parquet scan), and loading **raises if any listed session is missing** from the dataset, so an analysis never quietly runs on a subset.

### Per-session tables (licks, position, sniffing)

Some tables aren't in the shared dataset. They only exist in each session's own processed asset (e.g. `s3://aind-open-data/879295_2026-09-24_16-51-46_processed_2026-09-25_11-27-26/licks.parquet`), whose location is the session table's `source_s3_location` column. `Dataset` loads them on demand, per animal:

```python
licks = dataset.load_licks("841312")        # is_lick_onset, timestamp, session_id
position = dataset.load_position("841312")  # position, velocity, timestamp, session_id
sniffing = dataset.load_sniffing("841312")  # voltage, timestamp, session_id

# Any sessions, any per-session table (incl. software_events):
events = dataset.load_table("software_events", [some_session_id])
```

Each result stacks the requested sessions and adds a categorical `session_id` column for joining back to `session`/`sites`. Timestamps share the clock of `sites.start_time`. Position and sniffing are ~1M rows *per session*, so a whole animal can be several GB; use `load_table` with a narrower list of sessions when you don't need everything.

To refresh `raw_sessions.json`:

```bash
uv run scripts/attach_datasets.py
# --prune replaces the whole list instead of merging into it:
uv run scripts/attach_datasets.py --prune
```

Which animals/dates to query are hard-coded constants at the top of `scripts/attach_datasets.py` (`SUBJECT_IDS`/`START_DATE`) rather than CLI flags. Edit those directly when you need to change them. It queries the DocDB via `version="v2"` (the default `"v1"` returns nothing for these sessions: they're indexed under the newer aind-data-schema layout, where the timestamp field also moved from `session.session_start_time` to `acquisition.acquisition_start_time`), filtered to `data_description.data_level: "raw"`. By default, newly matched sessions are *added* to the existing list; existing entries are kept even if they no longer match the query. If a newly attached session hasn't been added to the shared dataset yet, the workflows will fail until it is.

## Configuration

Plain env vars, read directly where they're used — no config file: `ARTIFACT_URI` (run output location, default `./artifacts`), `AWS_REGION`, `RUN_ID`.

## AWS credentials — you probably don't need any

Every *read* in this repo is public/unsigned: the shared dataset in `aind-scratch-data` allows anonymous access, so `analysis.sessions.Dataset` (Polars, via `storage_options={"skip_signature": "true"}`) and `analysis.sessions.build_inputs_manifest` (boto3, via `Config(signature_version=UNSIGNED)`) never need credentials.

Credentials only come into play for *writes*: `ARTIFACT_URI` pointed at a private S3 bucket for writing run outputs in production. That uses the standard AWS SDK credential chain (local `~/.aws/config`, or an IAM instance role on EC2) — never keys in the repo. This repo currently exercises only the local-filesystem artifact-store path end-to-end for run *outputs* (`ARTIFACT_URI=./artifacts`); `S3ArtifactStore` exists and is unit-tested but isn't wired to a real output bucket yet.

## Run artifacts & provenance

Every run gets an immutable `run_id` (`<UTC timestamp>-<suffix>`) and writes to `artifacts/runs/<run_id>/`:

- `manifest.json` — run identity/provenance: git commit, container image, python version, status, timestamps
- `selection.json` — the exact `data_assets.json` and `raw_sessions.json` content this run used
- `inputs.json` — every object under the shared dataset's location, with size/etag, resolved before processing starts
- `dataset/` — verbatim copies of the dataset's `data_description.json` and `processing.json`, recording what generated the tables this run read
- `out.log` — full stdout/stderr of the pipeline run (written by `scripts/start_prod.sh` via `tee`; not present for local `uv run` invocations)
- `figures/` — saved plots
- `results/` — analysis outputs

A completed run is never modified. To reproduce a past run, inspect its `manifest.json`/`selection.json`/`inputs.json` — they pin exactly what code, config, sessions, and S3 objects were used.

## EC2 deployment

```bash
git clone <repo> && cd <repo>
docker compose up -d prod
```

Input reads need no AWS role at all (see AWS credentials above). If writing artifacts to S3, attach an IAM instance role scoped to that bucket. Avoid exposing port `2718` publicly — prefer an SSH tunnel:

```bash
ssh -L 2718:localhost:2718 user@ec2-host
```

## Testing

```bash
uv run pytest
```

This is a unit-test suite only — there's deliberately no integration/e2e fixture dataset (see the implementation plan's Phase 12 rationale). `tests/conftest.py` has shared fixtures ready for future tests that want one.
