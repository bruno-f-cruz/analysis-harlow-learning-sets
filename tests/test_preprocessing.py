import sys
from pathlib import Path

import analysis.preprocessing as preprocessing


def test_default_data_dir_resolves_to_repo_root_data(monkeypatch):
    """--data-dir's default is computed by walking up parent directories from
    preprocessing.py's own location. Exercise the real CLI default (via
    main(), with process_all() stubbed out) and compare it against the repo
    root's data/ dir computed independently from this test file's own
    location, so a wrong parent count — e.g. after moving the module again —
    fails loudly instead of only "working" by accident.
    """
    captured = {}
    monkeypatch.setattr(
        preprocessing, "process_all", lambda data_root, **kw: captured.update(data_root=data_root)
    )
    monkeypatch.setattr(sys, "argv", ["scripts/process_sessions.py"])

    preprocessing.main()

    repo_root = Path(__file__).resolve().parent.parent  # tests/ -> repo root
    assert (repo_root / "pyproject.toml").is_file()
    assert captured["data_root"] == repo_root / "data"
