"""API authentication: configuration, key verification, and ASGI middleware.

Design decisions:
- Pure ASGI middleware (not BaseHTTPMiddleware) to avoid buffering SSE streams.
- secrets.compare_digest for constant-time key comparison.
- OPTIONS preflight and public paths bypass auth unconditionally.
- Production fail-close: raises ValueError at startup if misconfigured.
- Development default: auth disabled so existing tests are unaffected.
"""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Paths that never require authentication
_PUBLIC_PATHS: frozenset[str] = frozenset([
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
])

# Prefixes whose sub-paths are all public (e.g. /docs/oauth2-redirect)
_PUBLIC_PREFIXES: tuple[str, ...] = ("/docs/", "/redoc/")

_PLACEHOLDER_KEYS: frozenset[str] = frozenset([
    "", "placeholder", "changeme", "...", "change-me", "your-key-here",
    "sk-...", "<your-key>",
])


@dataclass(frozen=True)
class ApiSecurityConfig:
    enabled: bool
    api_key: str
    app_env: str


def get_api_security_config() -> ApiSecurityConfig:
    """Read auth config from environment. Raises ValueError on production misconfiguration."""
    app_env = os.getenv("APP_ENV", "development").strip().lower()

    # Default: enabled in production, disabled in development
    default_enabled = "true" if app_env == "production" else "false"
    enabled_raw = os.getenv("API_AUTH_ENABLED", default_enabled).strip().lower()
    enabled = enabled_raw in ("true", "1", "yes", "on")

    api_key = os.getenv("MEAL_AGENT_API_KEY", "")

    if app_env == "production" and enabled:
        if not api_key:
            raise ValueError(
                "MEAL_AGENT_API_KEY must be set and non-empty when "
                "APP_ENV=production and API_AUTH_ENABLED=true."
            )
        if api_key.strip().lower() in _PLACEHOLDER_KEYS:
            raise ValueError(
                "MEAL_AGENT_API_KEY appears to be a placeholder value. "
                "Set a strong random key before deploying to production."
            )

    if enabled and api_key:
        logger.info("[api_security] Authentication enabled (app_env=%s)", app_env)
    else:
        logger.info(
            "[api_security] Authentication disabled (app_env=%s, enabled=%s)",
            app_env, enabled,
        )

    return ApiSecurityConfig(enabled=enabled, api_key=api_key, app_env=app_env)


def is_public_path(path: str) -> bool:
    """Return True if the path is exempt from authentication."""
    if path in _PUBLIC_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES)


def verify_api_key(provided: str, expected: str) -> bool:
    """Constant-time comparison of two API keys. Returns False if either is empty."""
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided.encode(), expected.encode())


# ---------- Pure ASGI middleware ----------

_UNAUTHORIZED_BODY = b'{"detail":"Unauthorized"}'
_UNAUTHORIZED_HEADERS = [
    (b"content-type", b"application/json"),
    (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
]


class AuthMiddleware:
    """Pure ASGI middleware that enforces API key authentication.

    Does NOT use BaseHTTPMiddleware to avoid buffering SSE/streaming responses.
    Exempt: OPTIONS preflights, public paths, and when auth is disabled.
    """

    def __init__(self, app, *, config: ApiSecurityConfig) -> None:
        self._app = app
        self._config = config

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method: str = scope.get("method", "")
        path: str = scope.get("path", "")

        # Pass through: OPTIONS preflight
        if method == "OPTIONS":
            await self._app(scope, receive, send)
            return

        # Pass through: public paths
        if is_public_path(path):
            await self._app(scope, receive, send)
            return

        # Pass through: auth disabled
        if not self._config.enabled:
            await self._app(scope, receive, send)
            return

        # Extract X-API-Key header (case-insensitive via lower-cased bytes)
        headers: dict[bytes, bytes] = {k.lower(): v for k, v in scope.get("headers", [])}
        provided = headers.get(b"x-api-key", b"").decode("utf-8", errors="replace")

        if verify_api_key(provided, self._config.api_key):
            await self._app(scope, receive, send)
            return

        # Reject with 401 — never echo the provided key or reveal whether it was
        # missing vs wrong (both produce identical responses)
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": _UNAUTHORIZED_HEADERS,
        })
        await send({
            "type": "http.response.body",
            "body": _UNAUTHORIZED_BODY,
            "more_body": False,
        })
