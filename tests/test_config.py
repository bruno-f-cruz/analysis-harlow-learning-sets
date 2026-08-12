from analysis.config import load_config


def test_load_config_reads_yaml_defaults(tmp_path):
    config_path = tmp_path / "default.yaml"
    config_path.write_text("data_root: ./data\nartifact_uri: ./artifacts\n")

    config = load_config(config_path)

    assert config["data_root"] == "./data"
    assert config["artifact_uri"] == "./artifacts"


def test_load_config_env_vars_override_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "default.yaml"
    config_path.write_text("data_root: ./data\nartifact_uri: ./artifacts\n")
    monkeypatch.setenv("DATASET_URI", "/mnt/override-data")

    config = load_config(config_path)

    assert config["data_root"] == "/mnt/override-data"
    assert config["artifact_uri"] == "./artifacts"  # unset env var doesn't clobber yaml
