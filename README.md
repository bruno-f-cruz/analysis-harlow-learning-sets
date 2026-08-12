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

## Which sessions get analyzed — `data_assets.json`

The repo root has a git-tracked `data_assets.json` (a Code-Ocean-style `attached_datasets` list of `{id, mount, location}` entries) that pins exactly which sessions this analysis targets. `workflows/pipeline.py` only ever reads this file — it never queries the AIND metadata DocDB itself at run time.

To change which sessions are attached, run the self-contained refresh script (its dependencies are declared inline via PEP 723 and installed into a throwaway environment by `uv run`, independent of this repo's own `uv.lock`) and commit the resulting diff:

```bash
uv run attach_datasets.py --subject-ids 841299 841312 --start-date 2026-06-01
# --prune replaces the whole list instead of merging into it:
uv run attach_datasets.py --subject-ids 841299 --start-date 2026-06-01 --prune
```

By default, newly matched sessions are *added* to the existing list; existing entries are kept even if they no longer match the query. Caveat: if you hand-remove an entry for a session that still matches your query criteria, re-running with the same query will silently re-add it — there's currently no way to permanently exclude a still-matching session short of changing the query or using `--prune`.

## Configuration

`configs/default.yaml` holds `data_root`, `artifact_uri`, `aws_region`, and processing flags. Env var overrides: `DATASET_URI` (local raw-data root — *not* related to session selection, which lives in `data_assets.json` above), `ARTIFACT_URI`, `AWS_REGION`, `RUN_ID`.

## AWS credentials — you probably don't need any

Input session data lives in `aind-open-data`, a public S3 bucket accessed anonymously (unsigned requests) — reading inputs works identically with zero AWS setup on a laptop, in Codespaces, or on EC2.

Credentials only come into play if `ARTIFACT_URI` is pointed at a private S3 bucket for writing run outputs in production; in that case, use the normal AWS SDK credential chain (local `~/.aws/config`, or an IAM instance role on EC2) — never keys in the repo. This repo currently exercises only the local-filesystem artifact-store path end-to-end (`ARTIFACT_URI=./artifacts`); `S3ArtifactStore` exists and is unit-tested but isn't wired to a real bucket yet.

## Run artifacts & provenance

Every run gets an immutable `run_id` (`<UTC timestamp>-<suffix>`) and writes to `artifacts/runs/<run_id>/`:

- `manifest.json` — run identity/provenance: git commit, container image, python version, status, timestamps
- `selection.json` — the exact `data_assets.json` content this run used
- `inputs.json` — every S3 object under each attached session's prefix, with size/etag, resolved before processing starts
- `progress.jsonl` — the append-only event log the dashboard reads
- `results/` — analysis outputs

A completed run is never modified. To reproduce a past run, inspect its `manifest.json`/`selection.json`/`inputs.json` — they pin exactly what code, config, sessions, and S3 objects were used.

## EC2 deployment

```bash
git clone <repo> && cd <repo>
docker compose up -d
```

Input reads need no AWS role at all (see above). If writing artifacts to S3, attach an IAM instance role scoped to that bucket. Avoid exposing `2718`/`8080` publicly — prefer an SSH tunnel:

```bash
ssh -L 2718:localhost:2718 -L 8080:localhost:8080 user@ec2-host
```

## Testing

```bash
uv run pytest
```

This is a unit-test suite only — there's deliberately no integration/e2e fixture dataset (see the implementation plan's Phase 12 rationale). `tests/conftest.py` has shared fixtures ready for future tests that want one.
