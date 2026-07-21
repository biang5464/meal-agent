"""Runtime-writable and seed-path resolution.

All helpers read from environment variables at call time; absolute defaults
are derived from the project root so they work regardless of cwd.
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(key: str, default: Path) -> Path:
    val = os.getenv(key)
    return Path(val) if val else default


def chroma_dir() -> Path:
    return _env_path("CHROMA_PERSIST_DIR", _PROJECT_ROOT / "data" / "chroma")


def sqlite_path() -> Path:
    return _env_path("SQLITE_DB_PATH", _PROJECT_ROOT / "data" / "users.db")


def dead_letter_path() -> Path:
    return _env_path("DEAD_LETTER_DB_PATH", _PROJECT_ROOT / "data" / "dead_letter.db")


def nutrition_dir() -> Path:
    return _env_path("NUTRITION_DIR", _PROJECT_ROOT / "data" / "nutrition")


def food_safety_dir() -> Path:
    return _env_path("FOOD_SAFETY_DIR", _PROJECT_ROOT / "data" / "food_safety")


def ensure_runtime_dirs() -> None:
    """Create writable runtime parent directories if they don't exist."""
    chroma_dir().mkdir(parents=True, exist_ok=True)
    sqlite_path().parent.mkdir(parents=True, exist_ok=True)
    dead_letter_path().parent.mkdir(parents=True, exist_ok=True)


def env_flag(name: str, *, default: bool = False) -> bool:
    """Parse an environment variable as a boolean flag.

    Accepts (case-insensitive): true/1/yes/on → True; false/0/no/off → False.
    Raises ValueError for any other non-empty value.
    """
    val = os.getenv(name)
    if val is None:
        return default
    normalised = val.strip().lower()
    if normalised in ("true", "1", "yes", "on"):
        return True
    if normalised in ("false", "0", "no", "off"):
        return False
    raise ValueError(
        f"Environment variable {name!r} has unexpected boolean value {val!r}. "
        "Expected: true/false/1/0/yes/no/on/off."
    )
