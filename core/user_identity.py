"""Backend anonymous user identity validation.

The proxy (Cloudflare/Next.js Route Handler) resolves the anonymous session from
the signed HttpOnly cookie and injects X-Meal-Agent-User-ID into every request
before forwarding to Railway. This module validates that header and provides a
FastAPI dependency that endpoints use as their sole source of user identity.

Trust model:
  TRUSTED  — X-Meal-Agent-User-ID (injected server-side by the proxy)
  IGNORED  — request body user_id, query param user_id, cookies, X-Forwarded-For
"""
from __future__ import annotations

import re
from typing import Annotated

from fastapi import Header, HTTPException

# Accepted format: anon_ + canonical UUID (lowercase hex with dashes)
_ANON_RE = re.compile(
    r"^anon_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_MAX_USER_ID_LEN = 64


def validate_anonymous_user_id(value: str) -> str:
    """Validate and return a canonical anonymous user ID.

    Raises ValueError on any format violation so callers can decide whether to
    raise HTTPException or handle the error differently.

    Never logs the invalid value to prevent accidental exposure.
    """
    if not isinstance(value, str):
        raise ValueError("user_id must be a string")
    if len(value) > _MAX_USER_ID_LEN:
        raise ValueError("user_id exceeds maximum length")
    if not _ANON_RE.match(value):
        raise ValueError("user_id format invalid")
    return value


def require_current_user_id(
    x_meal_agent_user_id: Annotated[str | None, Header()] = None,
) -> str:
    """FastAPI dependency: return the current user ID from the trusted proxy header.

    Returns the validated user ID string on success.
    Raises HTTP 401 if the header is absent or does not match the expected format.

    This dependency is the ONLY authoritative source of user identity in route
    handlers. Body/query user_id fields are ignored; they must not be used.
    """
    if not x_meal_agent_user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        return validate_anonymous_user_id(x_meal_agent_user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Unauthorized")
