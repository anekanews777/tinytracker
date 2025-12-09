"""Configuration file support."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

# Try tomllib (3.11+), fall back to tomli, fall back to basic parsing
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

CONFIG_FILENAME = ".tinytracker.toml"
ENV_PREFIX = "TINYTRACKER_"


def _find_config_file() -> Optional[Path]:
    """Search for config file in cwd and parents."""
    path = Path.cwd()
    while path != path.parent:
        config_path = path / CONFIG_FILENAME
        if config_path.exists():
            return config_path
        path = path.parent
    return None


def _parse_simple_toml(content: str) -> Dict[str, Any]:
    """Basic TOML parser for simple key=value pairs (fallback option)."""
    result: Dict[str, Any] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Type conversion
            if value.lower() == "true":
                result[key] = True
            elif value.lower() == "false":
                result[key] = False
            else:
                try:
                    result[key] = int(value)
                except ValueError:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    """Load config from file and environment variables."""
    config: Dict[str, Any] = {}

    # Load from file
    config_path = _find_config_file()
    if config_path:
        content = config_path.read_text()
        if tomllib:
            config = tomllib.loads(content)
        else:
            config = _parse_simple_toml(content)

    # Environment variables override file config
    env_map = {
        "TINYTRACKER_PROJECT": "default_project",
        "TINYTRACKER_DB_PATH": "db_path",
    }
    for env_var, config_key in env_map.items():
        if env_var in os.environ:
            config[config_key] = os.environ[env_var]

    return config


def get_default_project() -> Optional[str]:
    """Get default project from config."""
    return load_config().get("default_project")


def get_db_path_override() -> Optional[Path]:
    """Get custom DB path from config."""
    path = load_config().get("db_path")
    return Path(path) if path else None
