# Meal Agent Deployment Guide

## Architecture

```
Vercel (Next.js frontend)
  └─→ Railway (FastAPI backend)
        ├─→ Railway Redis plugin
        ├─→ Railway MySQL plugin
        └─→ Railway Volume  /app/runtime-data
```

The frontend and backend are deployed independently. The frontend calls the
backend's SSE endpoint directly — no Vercel Function proxy is involved.

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
   documents (`data/nutrition/`, `data/food_safety/`) that are baked into the
   Docker image.

> **Warning:** clicking "Wipe Volume" permanently deletes all persisted data
> (user profiles, conversation memory, price history). Only do this during a
> full reset.

### Railway plugins

Add Redis and MySQL plugins from the Railway Dashboard. Their connection
variables are injected automatically:

- Redis: `REDIS_URL`
- MySQL: `MYSQLHOST`, `MYSQLPORT`, `MYSQLUSER`, `MYSQLPASSWORD`, `MYSQLDATABASE`
  (or `MYSQL_URL` / `DATABASE_URL`)

### Environment variables

Set these in Railway Dashboard → Service → Variables. **Never commit real
values to this file.**

```
# App
APP_ENV=production
DEEPSEEK_API_KEY=<your-key>

# Scheduler (keep true on the single Web instance)
RUN_SCHEDULER=true

# Runtime writable storage (Railway Volume)
CHROMA_PERSIST_DIR=/app/runtime-data/chroma
SQLITE_DB_PATH=/app/runtime-data/users.db
DEAD_LETTER_DB_PATH=/app/runtime-data/dead_letter.db
HF_HOME=/app/runtime-data/huggingface

# Read-only seed docs (baked into Docker image)
NUTRITION_DIR=/app/data/nutrition
FOOD_SAFETY_DIR=/app/data/food_safety

# CORS — set to your actual Vercel domain
ALLOWED_ORIGINS=https://<your-project>.vercel.app
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
`/app/runtime-data`. A future improvement is to use an entrypoint script that
`chown`s the mount point before dropping privileges.

---

## Frontend: Vercel

1. Import the repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Add environment variable:
   ```
   NEXT_PUBLIC_API_BASE_URL=https://<railway-domain>
   ```
4. Deploy. After every change to this variable, trigger a new Vercel deploy
   so the build-time substitution takes effect.

The frontend connects to Railway's SSE stream directly from the browser. No
Vercel serverless function proxy is used.

---

## Current limitations (Phase 10D3)

- **No auth / rate limiting.** Do not expose the Railway domain publicly until
  Phase 10D4 authentication is in place.
- **Scheduler and Web share one process.** A Web/Worker split is a future
  improvement.
- **Single worker, single replica.** Horizontal scaling requires migrating to
  PostgreSQL + a distributed vector store.
- **First startup is slow** while the embedding model downloads.
- **Volume must be configured before real data is written.** Without a Volume,
  runtime data (user profiles, Chroma embeddings) is lost on every redeploy.
