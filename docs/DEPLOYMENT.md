# Meal Agent Deployment Guide

## Architecture (Phase 10D4)

```
Browser
  └─→ Vercel /api/backend/*  (Next.js Route Handler — server-side proxy)
        └─→ Railway FastAPI   (X-API-Key injected server-side)
              ├─→ Railway Redis plugin   (rate limiting)
              ├─→ Railway MySQL plugin   (user profiles, price history)
              └─→ Railway Volume /app/runtime-data
                    ├─ Chroma (vector store)
                    ├─ SQLite (users.db, dead_letter.db)
                    └─ HuggingFace model cache
```

The browser never connects directly to Railway. All requests go through the
Vercel Route Handler at `/api/backend/[...path]`, which injects the API key
server-side and forwards the request to Railway. SSE streaming from `/recommend`
is passed through `response.body` without buffering.

---

## Backend: Railway

### Service setup

1. Create a new Railway project.
2. Add a **GitHub repo** service pointing at this repository.
3. Railway detects `railway.toml` → Dockerfile build is used automatically.

### Railway Volume

**Volume configuration must be done in the Railway Dashboard** — it cannot be
declared in `railway.toml`.

Steps in Railway Dashboard → Service → Volumes:
1. Create a volume.
2. Mount path: `/app/runtime-data`
3. **Do NOT** mount at `/app/data` — that would shadow the committed seed
   documents (`data/nutrition/`, `data/food_safety/`) baked into the Docker image.

> **Warning:** clicking "Wipe Volume" permanently deletes all persisted data
> (user profiles, conversation memory, price history). Only do this during a
> full reset.

> **Note:** Volume changes and environment variable updates only take effect
> after a new Railway deploy.

### Railway plugins

Add Redis and MySQL plugins from the Railway Dashboard. Their connection
variables are injected automatically:

- Redis: `REDIS_URL`
- MySQL: `MYSQLHOST`, `MYSQLPORT`, `MYSQLUSER`, `MYSQLPASSWORD`, `MYSQLDATABASE`
  (or `MYSQL_URL` / `DATABASE_URL`)

### Environment variables (Railway)

Set these in Railway Dashboard → Service → Variables. **Never commit real
values to this file.**

```
# App
APP_ENV=production
DEEPSEEK_API_KEY=<your-key>

# Authentication & rate limiting (Phase 10D4)
API_AUTH_ENABLED=true
MEAL_AGENT_API_KEY=<generate: python -c "import secrets; print(secrets.token_urlsafe(32))">
RATE_LIMIT_ENABLED=true
RATE_LIMIT_RECOMMEND_PER_MINUTE=10
RATE_LIMIT_READ_PER_MINUTE=60

# Redis & MySQL (set automatically by Railway plugins — shown for reference)
# REDIS_URL=redis://...
# MYSQL_URL=mysql://...

# Scheduler (keep true on the single Web instance)
RUN_SCHEDULER=true

# CORS — must match your actual Vercel domain exactly
ALLOWED_ORIGINS=https://<your-project>.vercel.app

# Runtime writable storage (Railway Volume)
CHROMA_PERSIST_DIR=/app/runtime-data/chroma
SQLITE_DB_PATH=/app/runtime-data/users.db
DEAD_LETTER_DB_PATH=/app/runtime-data/dead_letter.db
HF_HOME=/app/runtime-data/huggingface

# Read-only seed docs (baked into Docker image)
NUTRITION_DIR=/app/data/nutrition
FOOD_SAFETY_DIR=/app/data/food_safety
```

### Replica limit

Keep the service at **1 replica**. Multiple replicas would conflict over:
- The Railway Volume (SQLite WAL, Chroma write lock)
- APScheduler (would run duplicate cron jobs)

### First deploy note

The first deploy downloads the `BAAI/bge-small-zh` embedding model (~130 MB).
`healthcheckTimeout = 600` in `railway.toml` gives Railway 10 minutes to wait
before declaring the service unhealthy. Subsequent deploys use the cached model
from the Railway Volume (`HF_HOME=/app/runtime-data/huggingface`).

### Container user

The container currently runs as **root**. This is required because Railway
Volumes are mounted as root by default, and a non-root user cannot write to
`/app/runtime-data`.

---

## Frontend: Vercel

### Setup

1. Import the repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Add the environment variables below in Vercel Dashboard → Project Settings →
   Environment Variables. These are **server-only** — do not use the
   `NEXT_PUBLIC_` prefix, which would expose them to the browser bundle.
4. Deploy. After every change to these variables, trigger a new Vercel deploy
   for the new values to take effect.

### Environment variables (Vercel — server-only)

```
MEAL_AGENT_BACKEND_URL=https://<your-railway-domain>
MEAL_AGENT_API_KEY=<same secret as Railway MEAL_AGENT_API_KEY>
```

**Security rules:**
- Both variables are server-only. Never use `NEXT_PUBLIC_MEAL_AGENT_API_KEY`
  or any `NEXT_PUBLIC_*` variant for secrets.
- The API key is injected into the `X-API-Key` header inside the Vercel Route
  Handler and never returned to the browser or included in any response body.
- The browser communicates only with same-origin Vercel routes (`/api/backend/*`),
  never directly with the Railway domain.

### How the proxy works

The Next.js Route Handler at `frontend/app/api/backend/[...path]/route.ts`:
- Reads `MEAL_AGENT_BACKEND_URL` and `MEAL_AGENT_API_KEY` at request time
  (never at build time — the handler is always dynamic).
- Validates config in production: missing URL or key → 503; localhost URL → 503;
  non-HTTPS URL → 503.
- Strips hop-by-hop headers and injects `X-API-Key` before forwarding.
- For SSE responses (`/recommend`): streams `response.body` directly without
  buffering, preserving the streaming experience.
- Upstream errors return a fixed `{"detail": "Backend service temporarily
  unavailable."}` — internal error details, URLs, and keys are never reflected.

### Local development

Create `frontend/.env.local` (not committed):

```
MEAL_AGENT_BACKEND_URL=http://localhost:8000
MEAL_AGENT_API_KEY=
```

Leave `MEAL_AGENT_API_KEY` empty when `API_AUTH_ENABLED=false` on the local backend.

---

## Phase 10D4 security summary

| Layer | Mechanism |
|---|---|
| Auth | `X-API-Key` checked by `AuthMiddleware` (pure ASGI, no SSE buffering) |
| Rate limit | Redis atomic Lua INCR+EXPIRE; `/recommend` 10 req/min, read APIs 60 req/min |
| Key exposure | API key only in Railway env var + Vercel server env var; never in browser |
| IP privacy | Client IPs are SHA-256 hashed before use in Redis keys; raw IPs never stored |
| Proxy errors | Fixed generic message returned; internal details logged server-side only |
| Redis failure | Production + cost-bearing endpoint → 503 fail-close (LLM not called) |

---

## Current limitations

- **Single worker, single replica.** Horizontal scaling requires migrating to
  PostgreSQL + a distributed vector store.
- **Scheduler and Web share one process.** A Web/Worker split is a future
  improvement.
- **First startup is slow** while the embedding model downloads.
- **Volume must be configured before real data is written.** Without a Volume,
  runtime data is lost on every redeploy.
