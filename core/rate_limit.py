"""Unified rate limiting: Redis atomic counters, per-client hashing, ASGI middleware.

Design decisions:
- Pure ASGI middleware to avoid buffering SSE responses.
- Redis Lua script for atomic INCR + EXPIRE (no GET→SET race condition).
- Client identity: X-Client-IP header (forwarded by Vercel Route Handler),
  hashed with SHA-256 before use in Redis keys — raw IPs never stored.
- rate_limit_write policy: retries=0 (non-idempotent counter), timeout=0.5s.
- Production Redis failure on cost-bearing endpoints → 503 fail-close.
- Development/test: RATE_LIMIT_ENABLED=false → pass through unconditionally.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis as _redis_mod

logger = logging.getLogger(__name__)

# Lua script: atomic INCR + conditional EXPIRE.
# Returns [count, ttl_seconds].
# KEYS[1] = redis key
# ARGV[1] = window seconds (integer)
_LUA_RATE_LIMIT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
return {count, ttl}
"""

# Paths that trigger LLM / Graph / Chroma — fail-close on Redis error in prod
_COST_BEARING_PREFIXES: tuple[str, ...] = (
    "/recommend",
    "/api/daily-recommendation",
)

# Paths that are never rate-limited
_RATE_LIMIT_EXEMPT_PATHS: frozenset[str] = frozenset([
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
])
_RATE_LIMIT_EXEMPT_PREFIXES: tuple[str, ...] = ("/docs/", "/redoc/")

# Fallback bucket when no client identifier is available
_ANON_BUCKET = "anon"


@dataclass(frozen=True)
class RateLimitConfig:
    enabled: bool
    recommend_per_minute: int
    read_per_minute: int
    app_env: str


def get_rate_limit_config() -> RateLimitConfig:
    """Read rate limit config from environment variables."""
    app_env = os.getenv("APP_ENV", "development").strip().lower()
    default_enabled = "true" if app_env == "production" else "false"
    enabled_raw = os.getenv("RATE_LIMIT_ENABLED", default_enabled).strip().lower()
    enabled = enabled_raw in ("true", "1", "yes", "on")

    recommend_per_minute = int(os.getenv("RATE_LIMIT_RECOMMEND_PER_MINUTE", "10"))
    read_per_minute = int(os.getenv("RATE_LIMIT_READ_PER_MINUTE", "60"))

    return RateLimitConfig(
        enabled=enabled,
        recommend_per_minute=recommend_per_minute,
        read_per_minute=read_per_minute,
        app_env=app_env,
    )


def hash_client_id(raw_ip: str) -> str:
    """Return SHA-256 hex digest of a raw IP. Raw IP is never stored."""
    return hashlib.sha256(raw_ip.encode()).hexdigest()[:32]


def _get_client_id(scope) -> str:
    """Extract and hash the client identifier from ASGI scope headers.

    Reads X-Client-IP (forwarded by Vercel Route Handler) or falls back to the
    direct connection IP. If neither is available, returns the anonymous bucket.
    Raw IP addresses are hashed immediately and never logged.
    """
    headers: dict[bytes, bytes] = {k.lower(): v for k, v in scope.get("headers", [])}

    # Prefer the proxy-forwarded client IP set by Vercel Route Handler
    client_ip_header = headers.get(b"x-client-ip", b"").decode("utf-8", errors="replace").strip()
    if client_ip_header:
        return hash_client_id(client_ip_header)

    # Fall back to direct connection IP (dev/direct access)
    client = scope.get("client")
    if client and client[0]:
        return hash_client_id(client[0])

    return _ANON_BUCKET


def _is_exempt_path(path: str) -> bool:
    if path in _RATE_LIMIT_EXEMPT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _RATE_LIMIT_EXEMPT_PREFIXES)


def _is_cost_bearing(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _COST_BEARING_PREFIXES)


def _resolve_limit(path: str, config: RateLimitConfig) -> int:
    """Return the per-minute request limit for a given path."""
    if _is_cost_bearing(path):
        return config.recommend_per_minute
    return config.read_per_minute


def _run_lua(redis_client, key: str, window_seconds: int):
    """Execute the rate-limit Lua script synchronously. Returns (count, ttl)."""
    result = redis_client.eval(_LUA_RATE_LIMIT, 1, key, window_seconds)
    count = int(result[0])
    ttl = int(result[1])
    return count, ttl


# ---------- Pre-built response bytes ----------

def _build_429_body(limit: int, retry_after: int) -> bytes:
    return (
        '{"detail":"Too many requests. Please try again later."}'.encode()
    )


def _build_429_headers(limit: int, retry_after: int) -> list[tuple[bytes, bytes]]:
    body = _build_429_body(limit, retry_after)
    return [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"retry-after", str(retry_after).encode()),
        (b"x-ratelimit-limit", str(limit).encode()),
        (b"x-ratelimit-remaining", b"0"),
    ]


_503_BODY = b'{"detail":"Service temporarily unavailable. Please retry shortly."}'
_503_HEADERS = [
    (b"content-type", b"application/json"),
    (b"content-length", str(len(_503_BODY)).encode()),
]


# ---------- Pure ASGI middleware ----------

class RateLimitMiddleware:
    """Pure ASGI rate-limit middleware using Redis atomic counters.

    Does NOT use BaseHTTPMiddleware to avoid buffering SSE/streaming responses.
    Exempt: OPTIONS, /health, and other public paths.
    Fail-close in production for cost-bearing endpoints on Redis error.
    Non-idempotent counter: no retry on network error (rate_limit_write policy).
    """

    def __init__(self, app, *, config: RateLimitConfig) -> None:
        self._app = app
        self._config = config

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method: str = scope.get("method", "")
        path: str = scope.get("path", "")

        # OPTIONS preflight — never rate-limited
        if method == "OPTIONS":
            await self._app(scope, receive, send)
            return

        # Exempt paths
        if _is_exempt_path(path):
            await self._app(scope, receive, send)
            return

        # Rate limiting disabled (default in dev/test)
        if not self._config.enabled:
            await self._app(scope, receive, send)
            return

        limit = _resolve_limit(path, self._config)
        window_seconds = 60
        client_id = _get_client_id(scope)
        redis_key = f"ratelimit:{path.strip('/')}:{client_id}"

        try:
            from core.cache import _client as redis_client  # existing singleton
            count, ttl = await asyncio.wait_for(
                asyncio.to_thread(_run_lua, redis_client, redis_key, window_seconds),
                timeout=0.5,  # rate_limit_write policy timeout
            )
        except Exception as exc:
            logger.warning("[rate_limit] Redis error path=%s err=%s", path, exc)
            if self._config.app_env == "production" and _is_cost_bearing(path):
                # fail-close: do NOT call LLM/Graph
                await send({
                    "type": "http.response.start",
                    "status": 503,
                    "headers": _503_HEADERS,
                })
                await send({
                    "type": "http.response.body",
                    "body": _503_BODY,
                    "more_body": False,
                })
                return
            # dev or non-cost-bearing: fail-open
            await self._app(scope, receive, send)
            return

        if count > limit:
            retry_after = max(ttl, 1)
            headers = _build_429_headers(limit, retry_after)
            body = _build_429_body(limit, retry_after)
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": headers,
            })
            await send({
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            })
            return

        await self._app(scope, receive, send)
