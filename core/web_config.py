"""Web configuration: CORS origin parsing and validation."""
from __future__ import annotations

import os

_DEV_DEFAULTS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def parse_allowed_origins(
    raw: str | None = None,
    *,
    app_env: str | None = None,
) -> list[str]:
    """
    Parse ALLOWED_ORIGINS into a validated list of CORS origins.

    Development (APP_ENV != "production"):
      - No ALLOWED_ORIGINS → ["http://localhost:3000", "http://127.0.0.1:3000"]
      - ALLOWED_ORIGINS set → parse and return as-is (dev is permissive)

    Production (APP_ENV == "production"):
      - ALLOWED_ORIGINS must be non-empty
      - Wildcards (*) are rejected
      - Each origin must start with http:// or https://
      Raises ValueError on misconfiguration; messages never contain secret values.
    """
    if app_env is None:
        app_env = os.getenv("APP_ENV", "development")
    if raw is None:
        raw = os.getenv("ALLOWED_ORIGINS")

    is_production = app_env.strip().lower() == "production"

    if not raw or not raw.strip():
        if is_production:
            raise ValueError(
                "ALLOWED_ORIGINS must be set in production. "
                "Example: ALLOWED_ORIGINS=https://your-app.vercel.app"
            )
        return list(_DEV_DEFAULTS)

    origins: list[str] = []
    for part in raw.split(","):
        origin = part.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            if is_production:
                raise ValueError(
                    "ALLOWED_ORIGINS='*' is not permitted in production. "
                    "Specify explicit origins such as https://your-app.vercel.app"
                )
            origins.append(origin)
            continue
        if not (origin.startswith("http://") or origin.startswith("https://")):
            scheme = origin.split("://")[0] if "://" in origin else origin
            raise ValueError(
                f"Invalid CORS origin: must start with http:// or https://, "
                f"got scheme {scheme!r}"
            )
        origins.append(origin)

    # stable dedup — preserve first occurrence
    seen: set[str] = set()
    unique: list[str] = []
    for o in origins:
        if o not in seen:
            seen.add(o)
            unique.append(o)

    return unique


def get_allowed_origins() -> list[str]:
    """Read ALLOWED_ORIGINS and APP_ENV from environment and return parsed origins."""
    return parse_allowed_origins()
