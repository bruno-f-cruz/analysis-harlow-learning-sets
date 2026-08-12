"""Config loading (design spec section 14): YAML defaults, env-var overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ENV_OVERRIDES = {
    "DATASET_URI": "data_root",
    "ARTIFACT_URI": "artifact_uri",
    "AWS_REGION": "aws_region",
}


def load_config(path: Path | str) -> dict[str, Any]:
    """Load YAML config from ``path``, then apply env-var overrides.

    Only env vars present in :data:`_ENV_OVERRIDES` are consulted, and each
    only takes effect when actually set in the environment — an unset
    override leaves the corresponding YAML value untouched.
    """
    config = yaml.safe_load(Path(path).read_text()) or {}
    for env_var, config_key in _ENV_OVERRIDES.items():
        if env_var in os.environ:
            config[config_key] = os.environ[env_var]
    return config
