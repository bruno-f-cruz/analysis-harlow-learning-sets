"""Unit tests for the ``scripts/attach_datasets.py`` ops script.

It lives outside the ``analysis`` package and starts with a PEP 723
``# /// script`` header — that header is just a comment block to Python, so
it doesn't interfere with a normal import. We load the module dynamically
via ``importlib`` rather than adding ``scripts/`` to ``sys.path``, to avoid
any chance of shadowing/clobbering other top-level modules during the test
session.

Only the pure functions (``build_entries``, ``merge``) are covered here.
``query_sessions`` requires live network/DocDB access and ``main`` is a
CLI/file-IO integration point — both are out of scope for unit tests.
"""

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).parent.parent / "scripts" / "attach_datasets.py"
_spec = importlib.util.spec_from_file_location("attach_datasets", _MODULE_PATH)
attach_datasets = importlib.util.module_from_spec(_spec)
sys.modules["attach_datasets"] = attach_datasets
_spec.loader.exec_module(attach_datasets)

build_entries = attach_datasets.build_entries
merge = attach_datasets.merge


RECORDS = [
    {
        "_id": "9e2b1c3a",
        "name": "841312_2026-06-04_20-19-36",
        "location": "s3://aind-open-data/841312_2026-06-04_20-19-36",
    },
    {
        "_id": "c16d7200",
        "name": "841299_2026-06-05_19-13-19",
        "location": "s3://aind-open-data/841299_2026-06-05_19-13-19",
    },
]


def test_build_entries_maps_id_mount_location():
    entries = build_entries(RECORDS)
    assert entries[0] == {
        "id": "c16d7200",
        "mount": "841299_2026-06-05_19-13-19",
        "location": "s3://aind-open-data/841299_2026-06-05_19-13-19",
    }


def test_build_entries_sorted_by_mount():
    entries = build_entries(RECORDS)
    assert [e["mount"] for e in entries] == [
        "841299_2026-06-05_19-13-19",
        "841312_2026-06-04_20-19-36",
    ]


def test_build_entries_empty_input():
    assert build_entries([]) == []


def test_merge_same_id_in_fresh_overwrites_existing_entry():
    existing = [{"id": "a1", "mount": "old-mount", "location": "s3://old"}]
    fresh = [{"id": "a1", "mount": "new-mount", "location": "s3://new"}]

    result = merge(existing, fresh, prune=False)

    assert result == [{"id": "a1", "mount": "new-mount", "location": "s3://new"}]


def test_merge_disjoint_ids_accumulate():
    existing = [{"id": "a1", "mount": "mount-a", "location": "s3://a"}]
    fresh = [{"id": "b2", "mount": "mount-b", "location": "s3://b"}]

    result = merge(existing, fresh, prune=False)

    assert result == [
        {"id": "a1", "mount": "mount-a", "location": "s3://a"},
        {"id": "b2", "mount": "mount-b", "location": "s3://b"},
    ]


def test_merge_prune_replaces_list_ignoring_existing():
    existing = [{"id": "a1", "mount": "mount-a", "location": "s3://a"}]
    fresh = [{"id": "b2", "mount": "mount-b", "location": "s3://b"}]

    result = merge(existing, fresh, prune=True)

    assert result == [{"id": "b2", "mount": "mount-b", "location": "s3://b"}]


def test_merge_prune_sorts_fresh_by_mount():
    existing = []
    fresh = [
        {"id": "b2", "mount": "z-mount", "location": "s3://z"},
        {"id": "a1", "mount": "a-mount", "location": "s3://a"},
    ]

    result = merge(existing, fresh, prune=True)

    assert [e["mount"] for e in result] == ["a-mount", "z-mount"]
