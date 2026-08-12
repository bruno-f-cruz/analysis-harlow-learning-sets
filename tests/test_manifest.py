import re

from analysis.run import generate_run_id, build_manifest


def test_generate_run_id_matches_expected_format():
    run_id = generate_run_id(now="2026-08-11T18:55:00", suffix="a81f42c")
    assert run_id == "20260811T185500-a81f42c"


def test_build_manifest_contains_required_fields():
    manifest = build_manifest(
        run_id="20260811T185500-a81f42c",
        started_at="2026-08-11T18:55:00Z",
        completed_at="2026-08-11T19:02:00Z",
        status="completed",
        git_commit="d5b3367",
        container_image="analysis:latest",
        python_version="3.13.0",
    )
    required = {
        "run_id", "started_at", "completed_at", "status", "git_commit",
        "container_image", "python_version", "config", "inputs", "selection",
    }
    assert required.issubset(manifest.keys())
    assert manifest["config"] == "config.yaml"
    assert manifest["inputs"] == "inputs.json"
    assert manifest["selection"] == "selection.json"
