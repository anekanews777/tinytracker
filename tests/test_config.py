"""Tests for config module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tinytracker.config import _parse_simple_toml, load_config, _find_config_file


class TestParseSimpleToml:
    def test_string_value(self):
        result = _parse_simple_toml('name = "my_project"')
        assert result == {"name": "my_project"}

    def test_int_value(self):
        result = _parse_simple_toml("count = 42")
        assert result == {"count": 42}

    def test_float_value(self):
        result = _parse_simple_toml("rate = 0.001")
        assert result == {"rate": 0.001}

    def test_bool_values(self):
        result = _parse_simple_toml("enabled = true\ndisabled = false")
        assert result == {"enabled": True, "disabled": False}

    def test_ignores_comments(self):
        result = _parse_simple_toml("# comment\nkey = value")
        assert result == {"key": "value"}

    def test_ignores_sections(self):
        result = _parse_simple_toml("[section]\nkey = value")
        assert result == {"key": "value"}


class TestLoadConfig:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TINYTRACKER_PROJECT", "env_project")
        # Mock no config file found
        with patch("tinytracker.config._find_config_file", return_value=None):
            config = load_config()
        assert config.get("default_project") == "env_project"

    def test_parses_config_content(self):
        # Test the parsing directly without file system
        content = 'default_project = "test_proj"'
        result = _parse_simple_toml(content)
        assert result == {"default_project": "test_proj"}
