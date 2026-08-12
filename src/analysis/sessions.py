import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def build_attached_dataset_entries(
    records: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Map raw DocDB session records into the ``data_assets.json`` entry shape
    (``id``/``mount``/``location``), sorted by mount name for stable git diffs.

    ``mount`` mirrors the local session directory naming (``<subject>_<date>_<time>``,
    i.e. the record's ``name``) so the attachment file reads like a Code Ocean
    ``attached_datasets`` list — a human can tell what's attached at a glance.
    """
    entries = [
        {"id": str(record["_id"]), "mount": record["name"], "location": record["location"]}
        for record in records
    ]
    return sorted(entries, key=lambda entry: entry["mount"])


def load_attached_datasets(path: Path | str = "data_assets.json") -> List[Dict[str, Any]]:
    """Read the repo's ``data_assets.json`` — the durable declaration of what
    this analysis reads, not resolved via any live query at run time.
    """
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("attached_datasets", [])
