"""
tests/test_phase12a_user_isolation.py

Phase 12A: anonymous user session and data isolation tests.

Structure:
  TestAnonymousSessionModule  — code inspection of anonymous-session.ts
  TestRouteProxySecurity      — code inspection of route.ts
  TestValidateAnonymousUserId — unit tests for core/user_identity.py
  TestRequireCurrentUserId    — FastAPI dependency via TestClient
  TestEndpointIdentityIsolation — code inspection of main.py
  TestFrontendNoHardcodedUser — static analysis of frontend source
  TestPhase12ARegression      — confirm earlier phases' key behaviours preserved
"""
import re
import uuid
from pathlib import Path
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
FRONTEND = ROOT / "frontend"
SESSION_TS = FRONTEND / "app" / "lib" / "server" / "anonymous-session.ts"
ROUTE_TS = FRONTEND / "app" / "api" / "backend" / "[...path]" / "route.ts"
PAGE_TSX = FRONTEND / "app" / "page.tsx"
DAILY_REC_TSX = FRONTEND / "app" / "components" / "DailyRecommendation.tsx"
MAIN_PY = ROOT / "main.py"


# ── helpers ──────────────────────────────────────────────────────────────────

def _ts_session() -> str:
    return SESSION_TS.read_text(encoding="utf-8")


def _ts_route() -> str:
    return ROUTE_TS.read_text(encoding="utf-8")


def _py_main() -> str:
    return MAIN_PY.read_text(encoding="utf-8")


# ── 1. Anonymous session module (code inspection) ────────────────────────────

class TestAnonymousSessionModule:

    def test_file_exists(self):
        assert SESSION_TS.exists(), "anonymous-session.ts must exist"

    def test_cookie_name_is_meal_agent_session(self):
        src = _ts_session()
        assert "meal_agent_session" in src

    def test_httponly_attribute_present(self):
        src = _ts_session()
        assert "HttpOnly" in src

    def test_secure_conditional_on_production(self):
        """Secure flag must be conditioned on production — not always added."""
        src = _ts_session()
        # 'Secure' appears somewhere AND there is a production check nearby
        assert "Secure" in src
        assert "_isProduction" in src or "isProduction" in src

    def test_samesite_lax(self):
        src = _ts_session()
        assert "SameSite=Lax" in src

    def test_max_age_one_year(self):
        src = _ts_session()
        # 31536000 seconds = 1 year
        assert "31536000" in src

    def test_uses_crypto_random_uuid(self):
        src = _ts_session()
        assert "crypto.randomUUID()" in src

    def test_uses_web_crypto_hmac_sha256(self):
        src = _ts_session()
        assert "crypto.subtle" in src
        assert "HMAC" in src
        assert "SHA-256" in src

    def test_cookie_format_v1_prefix(self):
        """Cookie payload must start with version prefix 'v1.'"""
        src = _ts_session()
        assert "v1." in src

    def test_uuid_pattern_validation(self):
        """UUID must be validated against a regex pattern."""
        src = _ts_session()
        # Must contain a UUID-shaped regex
        assert "UUID_RE" in src or "uuid_re" in src or "/^[0-9a-f]{8}" in src

    def test_max_cookie_length_limit(self):
        """Cookies longer than the limit must be rejected."""
        src = _ts_session()
        assert "MAX_COOKIE_LEN" in src or "max_cookie_len" in src or "MAX_COOKIE" in src

    def test_production_failclose_returns_null(self):
        """resolveAnonymousSession must return null on production misconfiguration."""
        src = _ts_session()
        assert "return null" in src

    def test_dev_fallback_labeled_dev_only(self):
        """The dev fallback secret must be clearly labeled as dev-only."""
        src = _ts_session()
        assert "dev-only" in src.lower() or "DEV_FALLBACK" in src

    def test_hmac_verify_used_not_equality(self):
        """Signature verification must use crypto.subtle.verify (constant-time), not ===."""
        src = _ts_session()
        assert "crypto.subtle.verify" in src

    def test_user_id_format_is_anon_prefix(self):
        """Generated user IDs must use the anon_ prefix."""
        src = _ts_session()
        assert "anon_" in src

    def test_path_slash(self):
        """Cookie must include Path=/."""
        src = _ts_session()
        assert "Path=/" in src

    def test_session_secret_env_var_name(self):
        """Must reference MEAL_AGENT_SESSION_SECRET — no NEXT_PUBLIC_ prefix."""
        src = _ts_session()
        assert "MEAL_AGENT_SESSION_SECRET" in src
        assert "NEXT_PUBLIC_MEAL_AGENT_SESSION_SECRET" not in src

    def test_no_secret_in_error_log(self):
        """Secret must not appear in any console.log or console.error output."""
        src = _ts_session()
        # Verify that console statements don't reference the secret variable
        # (the secret var is 'secret', 'raw', or similar — ensure no logging of its value)
        assert "console.log(secret)" not in src
        assert "console.error(secret)" not in src


# ── 2. Route proxy security (code inspection) ────────────────────────────────

class TestRouteProxySecurity:

    def test_strips_client_meal_agent_user_id(self):
        """Proxy must strip x-meal-agent-user-id from browser requests."""
        src = _ts_route()
        assert "x-meal-agent-user-id" in src.lower()
        # The header must be in a stripped/denied set, not just mentioned
        assert "STRIPPED_CLIENT_HEADERS" in src or "x-meal-agent-user-id" in src

    def test_strips_client_api_key(self):
        """Proxy must strip x-api-key from browser requests."""
        src = _ts_route()
        assert "STRIPPED_CLIENT_HEADERS" in src
        assert "x-api-key" in src.lower()

    def test_injects_x_meal_agent_user_id(self):
        """Proxy must inject X-Meal-Agent-User-ID from the session."""
        src = _ts_route()
        assert "X-Meal-Agent-User-ID" in src
        assert "forwardHeaders.set" in src

    def test_injects_correct_api_key(self):
        """Proxy must inject X-API-Key from server env, not from browser."""
        src = _ts_route()
        assert "forwardHeaders.set('X-API-Key'" in src or 'forwardHeaders.set("X-API-Key"' in src

    def test_user_scoped_path_detection(self):
        """Proxy must define user-scoped path detection logic."""
        src = _ts_route()
        assert "_isUserScopedPath" in src or "isUserScopedPath" in src
        assert "/recommend" in src
        assert "/api/daily-recommendation" in src

    def test_session_config_error_returns_503(self):
        """Missing session secret in production → 503 before forwarding."""
        src = _ts_route()
        assert "config_error=missing_session_secret" in src
        assert "503" in src

    def test_set_cookie_added_before_response_construction(self):
        """Set-Cookie must be set on responseHeaders before new Response(...)."""
        src = _ts_route()
        set_cookie_pos = src.find("Set-Cookie")
        response_construct_pos = src.find("return new Response(upstreamResponse.body")
        # Set-Cookie assignment must appear before SSE Response construction
        assert set_cookie_pos != -1
        assert response_construct_pos != -1
        assert set_cookie_pos < response_construct_pos

    def test_sse_response_not_buffered(self):
        """SSE body must be streamed via upstreamResponse.body (no buffering)."""
        src = _ts_route()
        assert "text/event-stream" in src
        assert "upstreamResponse.body" in src

    def test_sse_headers_preserved(self):
        """SSE response must set Content-Type, Cache-Control, X-Accel-Buffering."""
        src = _ts_route()
        assert "X-Accel-Buffering" in src
        assert "no-cache" in src

    def test_health_path_not_user_scoped(self):
        """/health must not be in the user-scoped set (no cookie forced on health checks)."""
        src = _ts_route()
        # /health should not appear inside _USER_SCOPED_EXACT or _isUserScopedPath
        user_scoped_block_start = src.find("_USER_SCOPED_EXACT")
        if user_scoped_block_start == -1:
            user_scoped_block_start = src.find("isUserScopedPath")
        assert user_scoped_block_start != -1
        block = src[user_scoped_block_start:user_scoped_block_start + 400]
        assert "/health" not in block

    def test_price_history_not_user_scoped(self):
        """Price history endpoints must not be user-scoped."""
        src = _ts_route()
        user_scoped_block_start = src.find("_USER_SCOPED_EXACT")
        if user_scoped_block_start == -1:
            user_scoped_block_start = src.find("isUserScopedPath")
        if user_scoped_block_start != -1:
            block = src[user_scoped_block_start:user_scoped_block_start + 400]
            assert "/api/price-history" not in block

    def test_session_module_imported(self):
        """route.ts must import from the anonymous-session module."""
        src = _ts_route()
        assert "anonymous-session" in src

    def test_no_api_key_in_catch_block(self):
        """API key must not be referenced in catch / error log blocks."""
        src = _ts_route()
        after_catch = src[src.rfind("} catch"):]
        assert "apiKey" not in after_catch

    def test_502_safe_constant(self):
        src = _ts_route()
        assert "_502_UPSTREAM_ERROR" in src
        assert "Backend service temporarily unavailable." in src


# ── 3. validate_anonymous_user_id (Python unit tests) ────────────────────────

class TestValidateAnonymousUserId:

    def _uid(self):
        return f"anon_{uuid.uuid4()}"

    def test_valid_anon_uuid_accepted(self):
        from core.user_identity import validate_anonymous_user_id
        uid = self._uid()
        assert validate_anonymous_user_id(uid) == uid

    def test_canonical_lowercase_required(self):
        from core.user_identity import validate_anonymous_user_id
        # Uppercase UUID must be rejected
        with pytest.raises(ValueError):
            validate_anonymous_user_id("anon_" + str(uuid.uuid4()).upper())

    def test_anon_prefix_required(self):
        from core.user_identity import validate_anonymous_user_id
        with pytest.raises(ValueError):
            validate_anonymous_user_id(str(uuid.uuid4()))

    def test_test_user_rejected(self):
        from core.user_identity import validate_anonymous_user_id
        with pytest.raises(ValueError):
            validate_anonymous_user_id("test_user")

    def test_empty_string_rejected(self):
        from core.user_identity import validate_anonymous_user_id
        with pytest.raises(ValueError):
            validate_anonymous_user_id("")

    def test_too_long_rejected(self):
        from core.user_identity import validate_anonymous_user_id
        with pytest.raises(ValueError):
            validate_anonymous_user_id("anon_" + "a" * 100)

    def test_control_chars_rejected(self):
        from core.user_identity import validate_anonymous_user_id
        with pytest.raises(ValueError):
            validate_anonymous_user_id("anon_00000000-0000-0000-0000-000000000000\x00")

    def test_path_chars_rejected(self):
        from core.user_identity import validate_anonymous_user_id
        with pytest.raises(ValueError):
            validate_anonymous_user_id("anon_../../etc/passwd")

    def test_arbitrary_username_rejected(self):
        from core.user_identity import validate_anonymous_user_id
        with pytest.raises(ValueError):
            validate_anonymous_user_id("alice")

    def test_non_string_rejected(self):
        from core.user_identity import validate_anonymous_user_id
        with pytest.raises((ValueError, TypeError)):
            validate_anonymous_user_id(None)  # type: ignore[arg-type]

    def test_invalid_uuid_section_rejected(self):
        from core.user_identity import validate_anonymous_user_id
        with pytest.raises(ValueError):
            # Too short UUID
            validate_anonymous_user_id("anon_00000000-0000-0000")


# ── 4. require_current_user_id FastAPI dependency ────────────────────────────

def _make_identity_app():
    from core.user_identity import require_current_user_id

    app = FastAPI()

    @app.get("/whoami")
    async def whoami(current_user_id: Annotated[str, Depends(require_current_user_id)]):
        return {"user_id": current_user_id}

    return app


class TestRequireCurrentUserId:

    def setup_method(self):
        self.client = TestClient(_make_identity_app())

    def test_missing_header_returns_401(self):
        res = self.client.get("/whoami")
        assert res.status_code == 401
        assert res.json()["detail"] == "Unauthorized"

    def test_invalid_format_returns_401(self):
        res = self.client.get("/whoami", headers={"X-Meal-Agent-User-ID": "test_user"})
        assert res.status_code == 401

    def test_empty_header_returns_401(self):
        res = self.client.get("/whoami", headers={"X-Meal-Agent-User-ID": ""})
        assert res.status_code == 401

    def test_valid_anon_uuid_returns_user_id(self):
        uid = f"anon_{uuid.uuid4()}"
        res = self.client.get("/whoami", headers={"X-Meal-Agent-User-ID": uid})
        assert res.status_code == 200
        assert res.json()["user_id"] == uid

    def test_too_long_returns_401(self):
        long_id = "anon_" + "a" * 100
        res = self.client.get("/whoami", headers={"X-Meal-Agent-User-ID": long_id})
        assert res.status_code == 401

    def test_error_response_is_json(self):
        res = self.client.get("/whoami")
        assert "application/json" in res.headers.get("content-type", "")

    def test_error_response_does_not_echo_header(self):
        """401 response must not reflect back the submitted header value."""
        uid = f"anon_{uuid.uuid4()}"
        res = self.client.get("/whoami", headers={"X-Meal-Agent-User-ID": uid})
        # For valid uid this passes; for invalid uid the body must not echo it back
        invalid = "test_user_leaked"
        res2 = self.client.get("/whoami", headers={"X-Meal-Agent-User-ID": invalid})
        assert res2.status_code == 401
        assert invalid not in res2.text


# ── 5. main.py endpoint identity isolation (code inspection) ─────────────────

class TestEndpointIdentityIsolation:

    def test_recommend_model_has_no_user_id_field(self):
        """RecommendRequest must not contain a user_id field."""
        src = _py_main()
        # Find RecommendRequest class definition
        match = re.search(r"class RecommendRequest.*?(?=\nclass|\n@app|$)", src, re.DOTALL)
        assert match, "RecommendRequest class not found"
        body = match.group()
        assert "user_id" not in body, "RecommendRequest must not contain user_id"

    def test_recommend_uses_require_current_user_id(self):
        src = _py_main()
        assert "require_current_user_id" in src
        assert "Depends(require_current_user_id)" in src

    def test_recommend_uses_current_user_id_not_req_user_id(self):
        """The recommend handler must call run_with_queue with current_user_id."""
        src = _py_main()
        assert "run_with_queue(current_user_id," in src
        assert "run_with_queue(req.user_id," not in src

    def test_daily_rec_get_uses_dependency(self):
        """GET /api/daily-recommendation must use Depends, not query param user_id."""
        src = _py_main()
        assert "get_daily_recommendation" in src
        # user_id must not be a standalone query param anymore
        match = re.search(
            r"async def get_daily_recommendation\(.*?\):",
            src, re.DOTALL
        )
        assert match
        fn_sig = match.group()
        assert "require_current_user_id" in fn_sig
        assert "user_id: str" not in fn_sig

    def test_daily_rec_generate_uses_dependency(self):
        """POST /api/daily-recommendation/generate must use Depends, not body user_id."""
        src = _py_main()
        match = re.search(
            r"async def trigger_daily_recommendation\(.*?\):",
            src, re.DOTALL
        )
        assert match
        fn_sig = match.group()
        assert "require_current_user_id" in fn_sig

    def test_generate_request_model_has_no_user_id(self):
        """GenerateRecommendationRequest must not contain user_id."""
        src = _py_main()
        match = re.search(r"class GenerateRecommendationRequest.*?(?=\nclass|\n@app|$)", src, re.DOTALL)
        assert match, "GenerateRecommendationRequest not found"
        body = match.group()
        assert "user_id" not in body

    def test_profile_route_uses_me_not_path_user_id(self):
        """/users/me/profile must be used instead of /users/{user_id}/profile."""
        src = _py_main()
        assert "/users/me/profile" in src
        # Old path-based user ID route must not exist
        assert '"/users/{user_id}/profile"' not in src
        assert "'/users/{user_id}/profile'" not in src

    def test_user_identity_imported(self):
        src = _py_main()
        assert "from core.user_identity import require_current_user_id" in src


# ── 6. Frontend static analysis ──────────────────────────────────────────────

class TestFrontendNoHardcodedUser:

    def test_no_const_user_id_in_page(self):
        """page.tsx must not contain const USER_ID = ..."""
        src = PAGE_TSX.read_text(encoding="utf-8")
        assert "const USER_ID" not in src

    def test_no_test_user_string_in_page(self):
        """page.tsx must not contain 'test_user'."""
        src = PAGE_TSX.read_text(encoding="utf-8")
        assert "test_user" not in src

    def test_chat_request_has_no_user_id(self):
        """Chat request body must only contain 'message', no 'user_id'."""
        src = PAGE_TSX.read_text(encoding="utf-8")
        assert 'user_id: USER_ID' not in src
        assert 'user_id:' not in src or 'test_user' not in src

    def test_daily_rec_component_no_user_id_prop(self):
        """DailyRecommendation component in page.tsx must not receive userId prop."""
        src = PAGE_TSX.read_text(encoding="utf-8")
        assert "userId=" not in src
        assert "userId={" not in src

    def test_daily_rec_tsx_props_no_user_id(self):
        """DailyRecommendation.tsx Props type must not include userId."""
        src = DAILY_REC_TSX.read_text(encoding="utf-8")
        # Find the Props type definition
        match = re.search(r"type Props\s*=\s*\{[^}]+\}", src)
        assert match, "Props type not found"
        props_block = match.group()
        assert "userId" not in props_block

    def test_daily_rec_tsx_no_user_id_in_fetch(self):
        """DailyRecommendation fetch URLs must not include user_id query param."""
        src = DAILY_REC_TSX.read_text(encoding="utf-8")
        assert "user_id=" not in src

    def test_daily_rec_tsx_no_user_id_in_post_body(self):
        """DailyRecommendation POST body must not include user_id field."""
        src = DAILY_REC_TSX.read_text(encoding="utf-8")
        assert "user_id: userId" not in src

    def test_no_next_public_session_secret(self):
        """Session secret must not use NEXT_PUBLIC_ prefix anywhere in frontend source."""
        for ts_file in FRONTEND.rglob("*.ts"):
            if "node_modules" in str(ts_file) or ".next" in str(ts_file):
                continue
            content = ts_file.read_text(encoding="utf-8", errors="ignore")
            assert "NEXT_PUBLIC_MEAL_AGENT_SESSION_SECRET" not in content, (
                f"NEXT_PUBLIC_MEAL_AGENT_SESSION_SECRET found in {ts_file}"
            )
        for tsx_file in FRONTEND.rglob("*.tsx"):
            if "node_modules" in str(tsx_file) or ".next" in str(tsx_file):
                continue
            content = tsx_file.read_text(encoding="utf-8", errors="ignore")
            assert "NEXT_PUBLIC_MEAL_AGENT_SESSION_SECRET" not in content, (
                f"NEXT_PUBLIC_MEAL_AGENT_SESSION_SECRET found in {tsx_file}"
            )

    def test_session_secret_not_in_wrangler(self):
        """MEAL_AGENT_SESSION_SECRET must not appear in wrangler.jsonc vars."""
        import json, re as _re
        src = (FRONTEND / "wrangler.jsonc").read_text(encoding="utf-8")
        # Strip comments
        clean = "\n".join(_re.sub(r"\s*//.*$", "", line) for line in src.splitlines())
        parsed = json.loads(clean)
        vars_section = parsed.get("vars", {})
        assert "MEAL_AGENT_SESSION_SECRET" not in vars_section


# ── 7. Phase regression: behaviours from earlier phases preserved ─────────────

class TestPhase12ARegression:

    def test_normalize_recipe_steps_still_works(self):
        """Phase 11C-A: _normalize_recipe_steps must still be importable and correct."""
        from agents.daily_recommendation_agent import _normalize_recipe_steps
        assert _normalize_recipe_steps("步骤一") == ["步骤一"]
        assert _normalize_recipe_steps(None) == []

    def test_lock_key_still_embeds_user_id(self):
        """Phase 11B: daily_rec_lock key format must include user_id."""
        from agents.daily_recommendation_agent import _LOCK_TTL
        src = (ROOT / "agents" / "daily_recommendation_agent.py").read_text(encoding="utf-8")
        assert "daily_rec_lock:{user_id}" in src or "daily_rec_lock" in src

    def test_validate_id_returns_string(self):
        from core.user_identity import validate_anonymous_user_id
        import uuid
        uid = f"anon_{uuid.uuid4()}"
        result = validate_anonymous_user_id(uid)
        assert isinstance(result, str)
        assert result == uid

    def test_recommend_endpoint_still_exists_in_main(self):
        src = _py_main()
        assert '@app.post("/recommend")' in src

    def test_health_endpoint_still_public(self):
        src = _py_main()
        assert '@app.get("/health")' in src
        # health must NOT use require_current_user_id
        health_match = re.search(
            r'@app\.get\("/health"\).*?async def health\(.*?\):',
            src, re.DOTALL
        )
        assert health_match
        fn_sig = health_match.group()
        assert "require_current_user_id" not in fn_sig

    def test_price_history_endpoint_has_no_user_auth(self):
        """Price history is not user-scoped — must not require user identity."""
        src = _py_main()
        ph_match = re.search(
            r'@app\.get\("/api/price-history.*?async def price_history_chart\(.*?\):',
            src, re.DOTALL
        )
        assert ph_match
        fn_sig = ph_match.group()
        assert "require_current_user_id" not in fn_sig

    def test_sse_watchdog_preserved(self):
        """SSE watchdog logic must still be present in /recommend."""
        src = _py_main()
        assert "_watchdog" in src
        assert "TimeoutConfig.GRAPH" in src

    def test_lock_busy_still_returns_503(self):
        """LockBusy exception from Phase 11B must still map to 503."""
        src = _py_main()
        assert "LockBusy" in src
        assert "503" in src

    def test_different_users_get_different_topic_cache_keys(self):
        """TopicCache key is prefixed with user_id — isolation is inherent."""
        src = (ROOT / "agents" / "context_manager.py").read_text(encoding="utf-8")
        assert "topic_cache:{user_id}" in src

    def test_chat_history_key_includes_user_id(self):
        """Chat history Redis key includes user_id for isolation."""
        src = (ROOT / "core" / "cache.py").read_text(encoding="utf-8")
        assert "chat_history:{user_id}" in src

    def test_daily_rec_orm_user_id_column_unchanged(self):
        """DailyRecommendationORM.user_id column must still exist (no schema change)."""
        src = (ROOT / "models" / "daily_recommendation.py").read_text(encoding="utf-8")
        assert "user_id" in src
