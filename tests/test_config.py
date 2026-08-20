"""Tests for src/config.py."""

from pathlib import Path

import yaml

from src.config import get_config, load_yaml_config, resolve_path


class TestLoadYamlConfig:
    def test_loads_valid_config(self, tmp_path):
        config = {"camera": {"device": 1}}
        path = tmp_path / "test.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        result = load_yaml_config(path)
        assert result["camera"]["device"] == 1

    def test_returns_default_for_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.yaml"
        result = load_yaml_config(path)
        assert "camera" in result
        assert "detection" in result
        assert "security" in result

    def test_returns_default_for_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        result = load_yaml_config(path)
        assert result["camera"]["device"] == 0


class TestGetConfig:
    def test_returns_dict(self, isolated_config):
        config = get_config()
        assert isinstance(config, dict)
        assert "camera" in config

    def test_singleton_returns_same_instance(self, isolated_config):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestResolvePath:
    def test_creates_directory(self, tmp_path, monkeypatch):
        import src.config
        monkeypatch.setattr(src.config, "ROOT_DIR", tmp_path)

        path = resolve_path("./test_dir")
        assert path.exists()
        assert path.is_dir()

    def test_handles_absolute_path(self, tmp_path):
        abs_path = str(tmp_path / "absolute_test")
        path = resolve_path(abs_path)
        assert path.exists()
        assert str(path) == abs_path
