"""
Centralized configuration loader.

Loads config/config.yaml once and exposes it as a singleton `SETTINGS` object
with attribute-style access, plus a `get_config()` accessor for explicit use
in scripts/tests that want a fresh read.
"""
from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class _DotDict(dict):
    """dict that also supports attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict):
            return _DotDict(value)
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> _DotDict:
    """Load YAML config from disk into a dot-accessible dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    return _DotDict(raw)


def get_config(path: str | Path = DEFAULT_CONFIG_PATH) -> _DotDict:
    """Explicit, always-fresh config read (useful in tests)."""
    return load_config(path)


# Module-level singleton for convenient `from config.settings import SETTINGS`
SETTINGS = load_config()

# Resolve absolute paths and ensure directories exist
for _key in ("raw_data_dir", "processed_data_dir", "checkpoint_dir", "log_dir"):
    _rel = SETTINGS["paths"][_key]
    _abs = str(PROJECT_ROOT / _rel)
    SETTINGS["paths"][_key] = _abs
    os.makedirs(_abs, exist_ok=True)
