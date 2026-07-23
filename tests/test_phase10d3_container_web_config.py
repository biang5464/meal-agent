"""
tests/test_phase10d3_container_web_config.py

Phase 10D3 tests: CORS parsing, health endpoint, Dockerfile, .dockerignore,
and railway.toml. No real network connections, Docker daemon, or external
services are used.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── CORS: parse_allowed_origins ────────────────────────────────────────────────


class TestCorsParseAllowedOrigins:
    def test_development_no_vars_returns_localhost_defaults(self, monkeypatch):
        """No ALLOWED_ORIGINS + development → two localhost origins."""
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        monkeypatch.delenv("APP_ENV", raising=False)
        from core.web_config import parse_allowed_origins
        origins = parse_allowed_origins()
        assert "http://localhost:3000" in origins
        assert "http://127.0.0.1:3000" in origins
        assert len(origins) == 2

    def test_comma_separated_parsing(self, monkeypatch):
        """Comma-separated origins are parsed into a list."""
        raw = "http://localhost:3000,http://localhost:4000"
        from core.web_config import parse_allowed_origins
        origins = parse_allowed_origins(raw=raw, app_env="development")
        assert origins == ["http://localhost:3000", "http://localhost:4000"]

    def test_whitespace_stripped(self, monkeypatch):
        """Leading/trailing whitespace around each origin is stripped."""
        raw = "  http://localhost:3000  ,  http://localhost:4000  "
        from core.web_config import parse_allowed_origins
        origins = parse_allowed_origins(raw=raw, app_env="development")
        assert origins == ["http://localhost:3000", "http://localhost:4000"]

    def test_trailing_slash_removed(self):
        """Trailing slash is stripped from each origin."""
        from core.web_config import parse_allowed_origins
        origins = parse_allowed_origins(
            raw="https://app.vercel.app/", app_env="production"
        )
        assert origins == ["https://app.vercel.app"]

    def test_stable_dedup(self):
        """Duplicate origins are removed, first occurrence is kept."""
        raw = "http://localhost:3000,http://localhost:4000,http://localhost:3000"
        from core.web_config import parse_allowed_origins
        origins = parse_allowed_origins(raw=raw, app_env="development")
        assert origins.count("http://localhost:3000") == 1
        assert origins.index("http://localhost:3000") < origins.index("http://localhost:4000")

    def test_production_missing_origins_raises(self):
        """Production with no ALLOWED_ORIGINS raises ValueError."""
        from core.web_config import parse_allowed_origins
        with pytest.raises(ValueError, match="ALLOWED_ORIGINS"):
            parse_allowed_origins(raw=None, app_env="production")

    def test_production_wildcard_raises(self):
        """Production with ALLOWED_ORIGINS='*' raises ValueError."""
        from core.web_config import parse_allowed_origins
        with pytest.raises(ValueError, match=r"\*"):
            parse_allowed_origins(raw="*", app_env="production")

    def test_non_http_scheme_raises(self):
        """An origin without http:// or https:// scheme raises ValueError."""
        from core.web_config import parse_allowed_origins
        with pytest.raises(ValueError, match="scheme"):
            parse_allowed_origins(raw="ftp://example.com", app_env="development")

    def test_production_valid_vercel_origin(self):
        """Production with a valid https:// Vercel origin succeeds."""
        from core.web_config import parse_allowed_origins
        origins = parse_allowed_origins(
            raw="https://meal-agent.vercel.app", app_env="production"
        )
        assert origins == ["https://meal-agent.vercel.app"]

    def test_allow_credentials_is_false_in_middleware(self, monkeypatch):
        """main.py middleware does not set allow_credentials=True."""
        src = (_REPO_ROOT / "main.py").read_text(encoding="utf-8")
        assert "allow_credentials=True" not in src, (
            "CORSMiddleware must not use allow_credentials=True — "
            "no session/cookie auth is implemented"
        )


# ── Health endpoint ────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        """GET /health returns {"status": "ok", "service": "meal-agent"}."""
        from main import health
        result = await health()
        assert result["status"] == "ok"
        assert result["service"] == "meal-agent"

    @pytest.mark.asyncio
    async def test_health_minimal_structure(self):
        """Health response contains exactly the expected keys."""
        from main import health
        result = await health()
        assert set(result.keys()) == {"status", "service"}

    @pytest.mark.asyncio
    async def test_health_no_sensitive_values(self):
        """Health response must not expose paths, URLs, credentials, or config."""
        from main import health
        result = await health()
        resp_str = str(result).lower()
        forbidden = ("data/", "/app/", "redis", "mysql", "sqlite", "sk-",
                     "password", "api_key", "secret", "localhost")
        for token in forbidden:
            assert token not in resp_str, (
                f"Health response must not contain {token!r}"
            )

    def test_health_route_has_no_external_dependency_calls(self):
        """The health() function body must not reference external service calls."""
        src = (_REPO_ROOT / "main.py").read_text(encoding="utf-8")
        # Find the health function and check its body
        lines = src.splitlines()
        in_health = False
        health_body: list[str] = []
        for line in lines:
            if "async def health(" in line:
                in_health = True
                continue
            if in_health:
                if line.startswith("@") or (line.startswith("async def") or line.startswith("def")):
                    break
                health_body.append(line)
        body = "\n".join(health_body)
        forbidden_calls = ("redis", "mysql", "sqlite", "chroma", "deepseek",
                           "request", "httpx", "aiohttp")
        for call in forbidden_calls:
            assert call not in body.lower(), (
                f"health() must not call {call!r} — health check must be lightweight"
            )


# ── Dockerfile ────────────────────────────────────────────────────────────────


class TestDockerfile:
    def _src(self) -> str:
        return (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    def test_uses_python_313(self):
        """Dockerfile uses Python 3.13 base image."""
        assert "python:3.13" in self._src()

    def test_uses_port_env_var(self):
        """Dockerfile CMD uses ${PORT:-8000} not a hardcoded port."""
        src = self._src()
        assert "${PORT:-8000}" in src, "CMD must use ${PORT:-8000} to respect Railway $PORT"
        assert "8000" in src  # default fallback present

    def test_no_reload_flag(self):
        """Dockerfile must not use --reload (dev-only flag)."""
        assert "--reload" not in self._src()

    def test_single_worker(self):
        """Dockerfile uses exactly one Uvicorn worker."""
        assert "--workers 1" in self._src()

    def test_no_secret_values_in_dockerfile(self):
        """Dockerfile must not contain API keys or passwords."""
        src = self._src().lower()
        for forbidden in ("sk-", "password=", "api_key=", "secret="):
            assert forbidden not in src, (
                f"Dockerfile must not contain {forbidden!r}"
            )

    def test_no_chroma_init_in_build(self):
        """Dockerfile must not run Chroma bootstrap during build."""
        src = self._src()
        run_lines = [ln for ln in src.splitlines() if ln.strip().startswith("RUN")]
        for line in run_lines:
            assert "chroma" not in line.lower(), (
                f"Chroma must not be initialised during docker build: {line!r}"
            )
            assert "init_chroma" not in line, (
                f"Chroma must not be initialised during docker build: {line!r}"
            )


# ── .dockerignore ─────────────────────────────────────────────────────────────


class TestDockerignore:
    def _patterns(self) -> str:
        return (_REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    def _active_patterns(self) -> list[str]:
        """Return non-comment, non-empty lines from .dockerignore."""
        return [
            ln.strip()
            for ln in self._patterns().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    def test_excludes_dot_env(self):
        """.env is excluded from the build context."""
        patterns = self._patterns()
        assert ".env" in patterns

    def test_excludes_chroma_runtime_data(self):
        """data/chroma/ is excluded (goes on Railway Volume)."""
        patterns = self._patterns()
        assert "data/chroma" in patterns or "data/chroma/" in patterns

    def test_excludes_sqlite_db_files(self):
        """data/*.db files are excluded."""
        patterns = self._patterns()
        assert "data/*.db" in patterns

    def test_excludes_frontend(self):
        """frontend/ is excluded (deployed separately to Vercel)."""
        patterns = self._patterns()
        assert "frontend" in patterns or "frontend/" in patterns

    def test_excludes_tests_directory(self):
        """tests/ directory is excluded from production image."""
        patterns = self._patterns()
        assert "tests" in patterns or "tests/" in patterns

    def test_does_not_exclude_nutrition_seed(self):
        """data/nutrition/ must NOT be excluded — it's a committed seed."""
        active = self._active_patterns()
        for pattern in active:
            # Pattern should not be an exact match or blanket data/ exclude
            assert pattern != "data/nutrition" and pattern != "data/nutrition/", (
                f"data/nutrition must not be excluded by pattern {pattern!r}"
            )
            assert pattern != "data/" and pattern != "data", (
                "Blanket 'data/' exclusion would hide seed documents"
            )

    def test_does_not_exclude_food_safety_seed(self):
        """data/food_safety/ must NOT be excluded — it's a committed seed."""
        active = self._active_patterns()
        for pattern in active:
            assert pattern != "data/food_safety" and pattern != "data/food_safety/", (
                f"data/food_safety must not be excluded by pattern {pattern!r}"
            )


# ── railway.toml ──────────────────────────────────────────────────────────────


class TestRailwayToml:
    def _parsed(self) -> dict:
        raw = (_REPO_ROOT / "railway.toml").read_bytes()
        return tomllib.loads(raw.decode("utf-8"))

    def test_toml_is_parseable(self):
        """railway.toml must be valid TOML."""
        config = self._parsed()
        assert isinstance(config, dict)

    def test_builder_is_dockerfile(self):
        """build.builder must be 'DOCKERFILE'."""
        config = self._parsed()
        assert config["build"]["builder"] == "DOCKERFILE"

    def test_healthcheck_path_is_health(self):
        """deploy.healthcheckPath must point to /health."""
        config = self._parsed()
        assert config["deploy"]["healthcheckPath"] == "/health"

    def test_no_volume_configuration(self):
        """railway.toml must not declare Volume configuration."""
        config = self._parsed()
        config_str = str(config).lower()
        assert "volume" not in config_str, (
            "Volume must be configured in Railway Dashboard, not railway.toml"
        )

    def test_no_secret_values(self):
        """railway.toml must not contain API keys or passwords."""
        raw = (_REPO_ROOT / "railway.toml").read_text(encoding="utf-8").lower()
        for forbidden in ("sk-", "password", "api_key", "deepseek", "redis://", "mysql://"):
            assert forbidden not in raw, (
                f"railway.toml must not contain {forbidden!r}"
            )

    def test_no_replica_count(self):
        """railway.toml must not configure multiple replicas."""
        config = self._parsed()
        deploy = config.get("deploy", {})
        assert "numReplicas" not in deploy
        assert "replicas" not in deploy

    def test_no_start_command_override(self):
        """railway.toml must not override the Docker CMD via startCommand."""
        config = self._parsed()
        deploy = config.get("deploy", {})
        assert "startCommand" not in deploy, (
            "startCommand must not be set — the Docker CMD handles startup"
        )
