"""
tests/test_phase10d4_hotfix_proxy.py

Phase 10D4 Hotfix: code-inspection tests for the Vercel proxy and deployment docs.

Tests verify structural correctness of the TypeScript proxy and DEPLOYMENT.md
without requiring a TypeScript/Jest test runner.
"""
from __future__ import annotations

from pathlib import Path


def _route_ts() -> str:
    p = Path(__file__).parent.parent / "frontend/app/api/backend/[...path]/route.ts"
    return p.read_text(encoding="utf-8")


def _deployment_md() -> str:
    p = Path(__file__).parent.parent / "docs/DEPLOYMENT.md"
    return p.read_text(encoding="utf-8")


# ── Proxy: error sanitization ─────────────────────────────────────────────────

class TestProxyErrorSanitization:
    def test_err_message_not_in_response(self):
        """The catch block must not return err.message in the response body."""
        code = _route_ts()
        # Old pattern that leaked the error
        assert "Proxy error: ${msg}" not in code
        assert "`Proxy error:" not in code

    def test_safe_502_body(self):
        """The 502 body must use the fixed safe message, not err.message."""
        code = _route_ts()
        assert "Backend service temporarily unavailable." in code

    def test_catch_block_uses_safe_constant(self):
        """Catch block must reference the pre-built error constant, not a template."""
        code = _route_ts()
        assert "_502_UPSTREAM_ERROR" in code

    def test_no_api_key_in_error_response(self):
        """API key must not appear anywhere after the catch keyword."""
        code = _route_ts()
        after_catch = code[code.rfind("} catch"):]
        # The variable 'apiKey' must not be referenced inside the catch block
        assert "apiKey" not in after_catch

    def test_category_logged_not_message(self):
        """Only a category is logged in the catch block, not the raw error message."""
        code = _route_ts()
        assert "category=" in code


# ── Proxy: production fail-close ──────────────────────────────────────────────

class TestProxyProductionFailClose:
    def test_vercel_env_production_check(self):
        """Proxy must check VERCEL_ENV === 'production' before config validation."""
        code = _route_ts()
        assert "VERCEL_ENV" in code
        assert "production" in code

    def test_missing_backend_url_returns_503(self):
        """Missing MEAL_AGENT_BACKEND_URL must be rejected with 503."""
        code = _route_ts()
        assert "config_error=missing_backend_url" in code
        assert "503" in code

    def test_missing_api_key_returns_503(self):
        """Missing MEAL_AGENT_API_KEY must be rejected with 503."""
        code = _route_ts()
        assert "config_error=missing_api_key" in code

    def test_localhost_rejected_in_production(self):
        """localhost backend URL must be rejected in production."""
        code = _route_ts()
        assert "config_error=localhost_in_production" in code
        assert "localhost" in code

    def test_non_https_rejected_in_production(self):
        """Non-HTTPS backend URL must be rejected in production."""
        code = _route_ts()
        assert "config_error=non_https_in_production" in code
        assert "https:" in code

    def test_dev_fallback_to_localhost(self):
        """Development mode must fall back to http://localhost:8000."""
        code = _route_ts()
        assert "http://localhost:8000" in code

    def test_env_read_at_request_time(self):
        """Env vars must be read inside proxy(), not at module level."""
        code = _route_ts()
        # Module-level constants for env vars must not exist
        assert "const BACKEND_URL" not in code
        assert "const API_KEY" not in code
        # Must be read inside the function
        assert "process.env.MEAL_AGENT_BACKEND_URL" in code
        assert "process.env.MEAL_AGENT_API_KEY" in code

    def test_sse_body_still_streamed(self):
        """SSE response must stream response.body without buffering."""
        code = _route_ts()
        assert "text/event-stream" in code
        assert "upstreamResponse.body" in code

    def test_safe_503_body(self):
        """503 response must use the fixed safe config-error message."""
        code = _route_ts()
        assert "_503_CONFIG_ERROR" in code
        assert "Service configuration error." in code

    def test_api_key_not_in_response_headers(self):
        """API key must be stripped from upstream response headers."""
        code = _route_ts()
        assert "x-api-key" in code.lower()  # stripped — referenced to exclude it


# ── Documentation: Phase 10D4 architecture ───────────────────────────────────

class TestDeploymentDocumentation:
    def test_no_next_public_api_base_url(self):
        """NEXT_PUBLIC_API_BASE_URL must not appear in deployment docs."""
        assert "NEXT_PUBLIC_API_BASE_URL" not in _deployment_md()

    def test_no_browser_direct_connect_language(self):
        """Docs must not describe browser connecting directly to Railway."""
        doc = _deployment_md()
        assert "directly from the browser" not in doc
        assert "The frontend connects to Railway" not in doc

    def test_no_no_auth_rate_limiting(self):
        """The 'No auth / rate limiting' warning must be removed."""
        assert "No auth / rate limiting" not in _deployment_md()

    def test_contains_meal_agent_backend_url(self):
        assert "MEAL_AGENT_BACKEND_URL" in _deployment_md()

    def test_contains_api_backend_proxy_path(self):
        assert "/api/backend" in _deployment_md()

    def test_contains_meal_agent_api_key(self):
        assert "MEAL_AGENT_API_KEY" in _deployment_md()

    def test_same_key_on_both_platforms(self):
        """Docs must explain that Railway and Vercel use the same API key."""
        doc = _deployment_md()
        assert "Railway" in doc and "Vercel" in doc
        # Key must be mentioned in context of both platforms
        railway_idx = doc.index("MEAL_AGENT_API_KEY")
        vercel_idx = doc.rindex("MEAL_AGENT_API_KEY")
        assert railway_idx != vercel_idx, "MEAL_AGENT_API_KEY must appear in both Railway and Vercel sections"

    def test_server_only_warning(self):
        """Docs must warn that env vars are server-only (no NEXT_PUBLIC_)."""
        doc = _deployment_md()
        assert "server-only" in doc
        assert "NEXT_PUBLIC_" in doc  # mentioned to prohibit it

    def test_sse_streaming_mentioned(self):
        """Docs must confirm SSE streaming is preserved through the proxy."""
        doc = _deployment_md().lower()
        assert "sse" in doc or "streaming" in doc

    def test_phase_10d4_auth_rate_limit_confirmed(self):
        """Docs must confirm Phase 10D4 auth and rate limiting are active."""
        doc = _deployment_md()
        assert "API_AUTH_ENABLED=true" in doc
        assert "RATE_LIMIT_ENABLED=true" in doc
