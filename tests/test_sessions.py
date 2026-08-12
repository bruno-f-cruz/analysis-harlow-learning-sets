import json

from analysis.sessions import build_attached_dataset_entries, load_attached_datasets


RECORDS = [
    {"_id": "9e2b1c3a", "name": "841312_2026-06-04_20-19-36", "location": "s3://aind-open-data/841312_2026-06-04_20-19-36"},
    {"_id": "c16d7200", "name": "841299_2026-06-05_19-13-19", "location": "s3://aind-open-data/841299_2026-06-05_19-13-19"},
]


def test_build_attached_dataset_entries_maps_id_mount_location():
    entries = build_attached_dataset_entries(RECORDS)
    assert entries[0] == {
        "id": "c16d7200",
        "mount": "841299_2026-06-05_19-13-19",
        "location": "s3://aind-open-data/841299_2026-06-05_19-13-19",
    }


def test_build_attached_dataset_entries_sorted_by_mount_for_stable_diffs():
    entries = build_attached_dataset_entries(RECORDS)
    assert [e["mount"] for e in entries] == [
        "841299_2026-06-05_19-13-19",
        "841312_2026-06-04_20-19-36",
    ]


def test_build_attached_dataset_entries_empty_input():
    assert build_attached_dataset_entries([]) == []


def test_load_attached_datasets_reads_manifest(tmp_path):
    path = tmp_path / "data_assets.json"
    path.write_text(json.dumps({
        "version": 1,
        "attached_datasets": [
            {"id": "c16d7200", "mount": "841299_2026-06-05_19-13-19", "location": "s3://aind-open-data/841299_2026-06-05_19-13-19"},
        ],
    }))

    entries = load_attached_datasets(path)

    assert entries == [
        {"id": "c16d7200", "mount": "841299_2026-06-05_19-13-19", "location": "s3://aind-open-data/841299_2026-06-05_19-13-19"},
    ]


def test_load_attached_datasets_missing_file_returns_empty_list(tmp_path):
    assert load_attached_datasets(tmp_path / "does-not-exist.json") == []
