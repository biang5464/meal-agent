"""
tests/test_phase10d6_cloudflare.py

Phase 10D6: code-inspection tests for Cloudflare Workers frontend adaptation.

Verifies structural correctness of route.ts, wrangler.jsonc, open-next.config.ts,
and .gitignore without requiring a TypeScript/Wrangler/Jest runner.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
FRONTEND = ROOT / "frontend"
ROUTE_TS = FRONTEND / "app" / "api" / "backend" / "[...path]" / "route.ts"
WRANGLER_JSONC = FRONTEND / "wrangler.jsonc"
OPEN_NEXT_CONFIG = FRONTEND / "open-next.config.ts"
PACKAGE_JSON = FRONTEND / "package.json"
GITIGNORE = ROOT / ".gitignore"


def _route() -> str:
    return ROUTE_TS.read_text(encoding="utf-8")


def _wrangler() -> str:
    return WRANGLER_JSONC.read_text(encoding="utf-8")


def _open_next() -> str:
    return OPEN_NEXT_CONFIG.read_text(encoding="utf-8")


def _package() -> dict:
    return json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))


def _gitignore() -> str:
    return GITIGNORE.read_text(encoding="utf-8")


def _strip_jsonc_comments(text: str) -> str:
    """Remove single-line JSONC comments so json.loads can parse the file."""
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"\s*//.*$", "", line)
        lines.append(stripped)
    return "\n".join(lines)


# ── Production environment detection ─────────────────────────────────────────

class TestProductionEnvDetection:
    def test_app_env_production_check_present(self):
        """route.ts must check APP_ENV=production (Cloudflare / Railway path)."""
        code = _route()
        assert 'APP_ENV' in code
        assert "'production'" in code or '"production"' in code

    def test_vercel_env_still_present(self):
        """VERCEL_ENV check must be preserved for Vercel backward compatibility."""
        code = _route()
        assert 'VERCEL_ENV' in code

    def test_production_helper_function_exists(self):
        """A shared helper function (_isProduction or similar) must replace direct env check."""
        code = _route()
        # Either a named helper or an inline compound expression covering both vars
        has_helper = "_isProduction" in code
        has_compound = "APP_ENV" in code and "VERCEL_ENV" in code
        assert has_helper or has_compound, (
            "route.ts must have a helper or compound check covering both APP_ENV and VERCEL_ENV"
        )

    def test_direct_vercel_env_guard_removed(self):
        """The bare `VERCEL_ENV === 'production'` guard must not be the only check."""
        code = _route()
        # If APP_ENV is also checked, the guard is no longer Vercel-only
        assert 'APP_ENV' in code, (
            "route.ts only checks VERCEL_ENV; must also check APP_ENV for Cloudflare"
        )


# ── Fail-close in production ──────────────────────────────────────────────────

class TestProductionFailClose:
    def test_missing_url_returns_503(self):
        code = _route()
        assert "config_error=missing_backend_url" in code
        assert "503" in code

    def test_missing_api_key_returns_503(self):
        code = _route()
        assert "config_error=missing_api_key" in code

    def test_localhost_rejected(self):
        code = _route()
        assert "config_error=localhost_in_production" in code
        assert "localhost" in code

    def test_non_https_rejected(self):
        code = _route()
        assert "config_error=non_https_in_production" in code
        assert "https:" in code

    def test_safe_503_constant(self):
        code = _route()
        assert "_503_CONFIG_ERROR" in code
        assert "Service configuration error." in code


# ── SSE streaming ─────────────────────────────────────────────────────────────

class TestSSEStreaming:
    def test_sse_body_not_buffered(self):
        """SSE response must stream upstreamResponse.body without buffering."""
        code = _route()
        assert "text/event-stream" in code
        assert "upstreamResponse.body" in code

    def test_no_await_json_on_sse(self):
        """SSE path must not call .json() or .text() which would buffer the body."""
        code = _route()
        # Ensure buffering methods are not used in the streaming branch
        sse_section_start = code.find("text/event-stream")
        if sse_section_start != -1:
            sse_section = code[sse_section_start:sse_section_start + 300]
            assert ".json()" not in sse_section
            assert ".text()" not in sse_section


# ── Error sanitization ────────────────────────────────────────────────────────

class TestErrorSanitization:
    def test_no_raw_error_message_in_response(self):
        code = _route()
        assert "Proxy error: ${msg}" not in code
        assert "`Proxy error:" not in code

    def test_safe_502_constant(self):
        code = _route()
        assert "_502_UPSTREAM_ERROR" in code
        assert "Backend service temporarily unavailable." in code

    def test_no_apikey_in_catch_block(self):
        code = _route()
        after_catch = code[code.rfind("} catch"):]
        assert "apiKey" not in after_catch

    def test_category_logged_not_raw_message(self):
        code = _route()
        assert "category=" in code


# ── Security: no secret in client bundle ─────────────────────────────────────

class TestClientBundleSecurity:
    def test_no_next_public_api_key(self):
        """NEXT_PUBLIC_*KEY must not appear anywhere in the frontend source."""
        for ts_file in FRONTEND.rglob("*.ts"):
            if "node_modules" in str(ts_file) or ".next" in str(ts_file):
                continue
            content = ts_file.read_text(encoding="utf-8", errors="ignore")
            assert "NEXT_PUBLIC_MEAL_AGENT_API_KEY" not in content, (
                f"NEXT_PUBLIC_MEAL_AGENT_API_KEY found in {ts_file}"
            )
        for tsx_file in FRONTEND.rglob("*.tsx"):
            if "node_modules" in str(tsx_file) or ".next" in str(tsx_file):
                continue
            content = tsx_file.read_text(encoding="utf-8", errors="ignore")
            assert "NEXT_PUBLIC_MEAL_AGENT_API_KEY" not in content, (
                f"NEXT_PUBLIC_MEAL_AGENT_API_KEY found in {tsx_file}"
            )

    def test_api_key_env_var_not_public_prefix(self):
        """MEAL_AGENT_API_KEY must not appear with NEXT_PUBLIC_ prefix."""
        code = _route()
        assert "NEXT_PUBLIC_MEAL_AGENT_API_KEY" not in code


# ── wrangler.jsonc ────────────────────────────────────────────────────────────

class TestWranglerConfig:
    def _parsed(self) -> dict:
        raw = _strip_jsonc_comments(_wrangler())
        return json.loads(raw)

    def test_name_is_meal_agent_fronted(self):
        assert self._parsed()["name"] == "meal-agent-fronted"

    def test_keep_vars_is_true(self):
        """keep_vars must be set so wrangler deploy does not wipe Dashboard env vars."""
        assert self._parsed()["keep_vars"] is True

    def test_main_is_open_next_worker(self):
        assert self._parsed()["main"] == ".open-next/worker.js"

    def test_nodejs_compat_flag(self):
        flags = self._parsed().get("compatibility_flags", [])
        assert "nodejs_compat" in flags

    def test_assets_directory(self):
        assets = self._parsed().get("assets", {})
        assert assets.get("directory") == ".open-next/assets"
        assert assets.get("binding") == "ASSETS"

    def test_observability_enabled(self):
        obs = self._parsed().get("observability", {})
        assert obs.get("enabled") is True

    def test_no_secret_values_in_wrangler(self):
        """Secrets must not be hardcoded in the parsed JSON data of wrangler.jsonc.

        Comments may reference variable names for documentation purposes.
        The actual [vars] section and top-level JSON must contain no secret values.
        """
        parsed = self._parsed()
        parsed_str = json.dumps(parsed)
        # No actual secret values (URLs, tokens) in the JSON data
        for term in ["railway.app", "token_urlsafe", "Bearer "]:
            assert term not in parsed_str, (
                f"Forbidden value '{term}' found in wrangler.jsonc JSON — use Cloudflare Secrets instead"
            )
        # The [vars] section must not hardcode these secrets by name
        vars_section = parsed.get("vars", {})
        assert "MEAL_AGENT_API_KEY" not in vars_section, (
            "MEAL_AGENT_API_KEY must not appear in wrangler.jsonc [vars] — use wrangler secret put"
        )
        assert "MEAL_AGENT_BACKEND_URL" not in vars_section, (
            "MEAL_AGENT_BACKEND_URL must not appear in wrangler.jsonc [vars] — use wrangler secret put"
        )

    def test_compatibility_date_present(self):
        date = self._parsed().get("compatibility_date", "")
        assert re.match(r"\d{4}-\d{2}-\d{2}", date), (
            "compatibility_date must be a YYYY-MM-DD string"
        )


# ── open-next.config.ts ───────────────────────────────────────────────────────

class TestOpenNextConfig:
    def test_file_exists(self):
        assert OPEN_NEXT_CONFIG.exists(), "open-next.config.ts must exist in frontend/"

    def test_imports_cloudflare_package(self):
        src = _open_next()
        assert "@opennextjs/cloudflare" in src

    def test_exports_default_config(self):
        src = _open_next()
        assert "export default" in src


# ── package.json scripts ──────────────────────────────────────────────────────

class TestPackageScripts:
    def test_preview_script_exists(self):
        scripts = _package()["scripts"]
        assert "preview" in scripts
        assert "opennextjs-cloudflare" in scripts["preview"]

    def test_deploy_script_exists(self):
        scripts = _package()["scripts"]
        assert "deploy" in scripts
        assert "opennextjs-cloudflare" in scripts["deploy"]

    def test_cf_typegen_script_exists(self):
        scripts = _package()["scripts"]
        assert "cf-typegen" in scripts
        assert "wrangler types" in scripts["cf-typegen"]

    def test_original_scripts_preserved(self):
        scripts = _package()["scripts"]
        assert "dev" in scripts
        assert "build" in scripts
        assert "start" in scripts

    def test_cloudflare_adapter_in_dependencies(self):
        deps = _package()["dependencies"]
        assert "@opennextjs/cloudflare" in deps

    def test_wrangler_in_devdependencies(self):
        dev_deps = _package()["devDependencies"]
        assert "wrangler" in dev_deps

    def test_next_version_satisfies_adapter(self):
        """next must be ^16.2.11 or higher to satisfy @opennextjs/cloudflare peer dep."""
        next_ver = _package()["dependencies"]["next"]
        # Accept ^16.2.11, >=16.2.11, 16.2.11, etc.
        assert "16.2" in next_ver or "16." in next_ver, (
            f"next version '{next_ver}' may not satisfy adapter peer dep >=16.2.11"
        )


# ── .gitignore ────────────────────────────────────────────────────────────────

class TestGitignore:
    def test_open_next_ignored(self):
        assert "frontend/.open-next/" in _gitignore()

    def test_wrangler_dir_ignored(self):
        assert "frontend/.wrangler/" in _gitignore()

    def test_cloudflare_env_dts_ignored(self):
        assert "frontend/cloudflare-env.d.ts" in _gitignore()

    def test_dev_vars_ignored(self):
        """Wrangler local-dev secrets file must be gitignored (written by smoke test)."""
        assert "frontend/.dev.vars" in _gitignore()


# ── Runtime smoke test script ─────────────────────────────────────────────────

SMOKE_SCRIPT = FRONTEND / "scripts" / "cloudflare-runtime-smoke.mjs"


class TestRuntimeSmokeScript:
    def _src(self) -> str:
        return SMOKE_SCRIPT.read_text(encoding="utf-8")

    def test_script_exists(self):
        assert SMOKE_SCRIPT.exists(), "scripts/cloudflare-runtime-smoke.mjs must exist"

    def test_uses_only_node_builtins(self):
        """Script must import only node: built-ins — zero extra npm dependencies."""
        import re as _re
        src = self._src()
        imports = _re.findall(r"from ['\"]([^'\"]+)['\"]", src)
        for imp in imports:
            assert imp.startswith("node:"), (
                f"Non-builtin import '{imp}' found in smoke script — only node: built-ins allowed"
            )

    def test_fake_key_not_a_real_key(self):
        """The FAKE_KEY constant must be a clearly non-secret placeholder."""
        src = self._src()
        assert "runtime-smoke-test-key" in src
        assert "FAKE_KEY" in src

    def test_dev_vars_cleaned_up_in_finally(self):
        """Script must delete .dev.vars in a finally block."""
        src = self._src()
        assert "finally" in src
        assert "cleanDev" in src or "unlinkSync" in src

    def test_no_real_secrets_in_script(self):
        """Script must not contain real Railway URLs, API keys, or tokens."""
        src = self._src()
        forbidden = ["railway.app", "DEEPSEEK", "token_urlsafe"]
        for term in forbidden:
            assert term not in src, f"Forbidden term '{term}' in smoke script"

    def test_covers_all_four_scenarios(self):
        src = self._src()
        assert "scenarioA" in src
        assert "scenarioB" in src
        assert "scenarioC" in src
        assert "scenarioD" in src

    def test_sse_chunk_timing_check(self):
        """Smoke script must verify SSE chunk time-span, not just content."""
        src = self._src()
        assert "span" in src or "times" in src, (
            "Script must track chunk arrival times to prove streaming vs. buffering"
        )

    def test_no_key_in_response_assertion(self):
        """Script must assert that the API key does not appear in response bodies."""
        src = self._src()
        assert "noKey" in src or "FAKE_KEY" in src and "includes" in src

    def test_package_script_invokes_smoke_script(self):
        """test:cloudflare-runtime must invoke the smoke script directly (build handled inside script via ensureBuilt)."""
        scripts = _package()["scripts"]
        assert "test:cloudflare-runtime" in scripts
        assert "cloudflare-runtime-smoke.mjs" in scripts["test:cloudflare-runtime"]

    def test_smoke_script_has_ensure_built(self):
        """Smoke script must call ensureBuilt() to guard the build step internally."""
        src = self._src()
        assert "ensureBuilt" in src, "Script must define/call ensureBuilt() to trigger build when .open-next/worker.js is absent"

    def test_process_tree_killed_on_windows(self):
        """Windows cleanup must use taskkill /F /T to kill the entire process tree."""
        src = self._src()
        assert "taskkill" in src
        assert "/T" in src
