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
docker compose run --rm analysis
# or, without Docker:
uv run python workflows/pipeline.py
# or, via the documented entrypoint shim:
uv run python scripts/run.py
```

## Progress dashboard

```bash
PROGRESS_PATH=artifacts/runs/<run_id>/progress.jsonl uv run uvicorn server.app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` for a live-polling view of the current run's status and recent events (`GET /api/status`, `GET /api/events` back it). State is derived entirely from `progress.jsonl` — restarting the server (or the container) doesn't lose it.

## Codespaces

Open this repository in GitHub Codespaces. It uses the same devcontainer (`.devcontainer/devcontainer.json`) as local VS Code — ports `2718` (marimo) and `8080` (progress dashboard) auto-forward. All the commands above work identically.

## What gets analyzed — `data_assets.json`

The repo root has a git-tracked `data_assets.json` (a Code-Ocean-style `attached_datasets` list of `{id, mount, location}` entries) that pins exactly what this analysis reads. Unlike the raw-session model this used to follow, it now holds a *single* entry: the already-processed dataset (`session.parquet` + `sites.parquet`, built by `scripts/sync_and_process.py`) sitting in the scratch bucket. `workflows/pipeline.py` mounts that location directly — no local sync, no local re-processing, no DocDB query at run time.

```json
{"version": 1, "attached_datasets": [{"id": "harlow-experiment-processed", "mount": "processed", "location": "s3://aind-scratch-data/vr-foraging/harlow-experiments/harlow-experiment"}]}
```

To point the analysis at a different processed dataset, hand-edit this file (there's no query behind it — it's just a literal location) and commit the change.

## Regenerating the processed dataset

Most of the time you don't need this — you're just reading the already-processed dataset via `data_assets.json` above. Regenerate it only when new raw sessions need to be folded in:

1. **Pin which raw sessions to use** — `raw_sessions.json` (git-tracked) is the durable record of which raw `aind-open-data` sessions feed the processed dataset, refreshed via:

   ```bash
   uv run scripts/attach_datasets.py
   # --prune replaces the whole list instead of merging into it:
   uv run scripts/attach_datasets.py --prune
   ```

   Which animals/dates to query are hard-coded constants at the top of `scripts/attach_datasets.py` (`SUBJECT_IDS`/`START_DATE`) rather than CLI flags — edit those directly when you need to change them. It queries the DocDB via `version="v2"` (the default `"v1"` returns nothing for these sessions — they're indexed under the newer aind-data-schema layout, where the timestamp field also moved from `session.session_start_time` to `acquisition.acquisition_start_time`), filtered to `data_description.data_level: "raw"` so derived/processed assets are excluded. By default, newly matched sessions are *added* to the existing list; existing entries are kept even if they no longer match the query.

2. **Sync those raw sessions to local disk and rebuild the processed dataset**, then upload the result:

   ```bash
   uv run python scripts/sync_and_process.py --upload
   ```

   This is a thin script, not a custom pipeline: for each session in `raw_sessions.json`, if `data/raw/<session>/` already exists it's assumed complete and skipped outright (no `aws s3 sync` call at all — pass `--force-sync` to re-sync anyway); otherwise it's downloaded via `aws s3 sync`. It then calls `aind_behavior_vr_foraging_packaging.export_pipeline`'s own `process_sessions`/`aggregate` functions directly on `data/raw/` to (re)build `data/processed/` — excluding the `sniffing` processor, since this analysis doesn't use it. `--upload` syncs `data/processed/` to the scratch bucket via `aws s3 sync`. `data_assets.json`'s location should already point at that same scratch-bucket path; update it if you changed the destination. Unlike every read in this repo, `--upload` needs real AWS credentials — see below.

## Configuration

Plain env vars, read directly where they're used — no config file: `ARTIFACT_URI` (run output location, default `./artifacts`), `AWS_REGION`, `RUN_ID`.

## AWS credentials — you probably don't need any

Every *read* in this repo is public/unsigned: the raw `aind-open-data` sessions and the processed dataset in `aind-scratch-data` both allow anonymous access — `analysis.sessions.load_processed_table` (Polars, via `storage_options={"skip_signature": "true"}`) and `analysis.sessions.build_inputs_manifest` (boto3, via `Config(signature_version=UNSIGNED)`) never need credentials to read either bucket.

Credentials only come into play for *writes*: `scripts/sync_and_process.py --upload` (writing the processed dataset to the scratch bucket — not anonymous even though reading it is), or `ARTIFACT_URI` pointed at a private S3 bucket for writing run outputs in production. Both cases use the standard AWS SDK credential chain (local `~/.aws/config`, or an IAM instance role on EC2) — never keys in the repo. This repo currently exercises only the local-filesystem artifact-store path end-to-end for run *outputs* (`ARTIFACT_URI=./artifacts`); `S3ArtifactStore` exists and is unit-tested but isn't wired to a real output bucket yet.

## Run artifacts & provenance

Every run gets an immutable `run_id` (`<UTC timestamp>-<suffix>`) and writes to `artifacts/runs/<run_id>/`:

- `manifest.json` — run identity/provenance: git commit, container image, python version, status, timestamps
- `selection.json` — the exact `data_assets.json` content this run used
- `inputs.json` — every object under the processed dataset's location, with size/etag, resolved before processing starts
- `progress.jsonl` — the append-only event log the dashboard reads
- `results/` — analysis outputs

A completed run is never modified. To reproduce a past run, inspect its `manifest.json`/`selection.json`/`inputs.json` — they pin exactly what code, config, sessions, and S3 objects were used.

## EC2 deployment

```bash
git clone <repo> && cd <repo>
docker compose up -d
```

Input reads need no AWS role at all (see AWS credentials above). If writing artifacts to S3, or running `scripts/sync_and_process.py --upload`, attach an IAM instance role scoped to that bucket. Avoid exposing `2718`/`8080` publicly — prefer an SSH tunnel:

```bash
ssh -L 2718:localhost:2718 -L 8080:localhost:8080 user@ec2-host
```

## Testing

```bash
uv run pytest
```

This is a unit-test suite only — there's deliberately no integration/e2e fixture dataset (see the implementation plan's Phase 12 rationale). `tests/conftest.py` has shared fixtures ready for future tests that want one.
