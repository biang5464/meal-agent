/**
 * frontend/scripts/cloudflare-runtime-smoke.mjs
 *
 * Cloudflare workerd runtime smoke test — Node.js built-ins only.
 *
 * For each scenario:
 *   1. Writes a temporary .dev.vars  (deleted in finally)
 *   2. Starts `wrangler dev --no-bundle` against .open-next/worker.js
 *   3. Fires real HTTP requests at http://127.0.0.1:WORKER_PORT
 *   4. Asserts status, headers, body, and streaming behaviour
 *   5. Kills the worker and deletes .dev.vars
 *
 * The FAKE_KEY is a safe placeholder — never a real secret.
 * No real keys, URLs, or external services are used.
 */

import http from 'node:http';
import { spawn } from 'node:child_process';
import { writeFileSync, unlinkSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import net from 'node:net';

// ─── Constants ────────────────────────────────────────────────────────────────

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(SCRIPT_DIR, '..');
const DEV_VARS = path.join(FRONTEND, '.dev.vars');
const MOCK_PORT = 19001;
const WORKER_PORT = 8789;
const FAKE_KEY = 'runtime-smoke-test-key'; // NOT a real key

let mockServer = null;
let workerProc = null;
let passed = 0, failed = 0;
const failures = [];

// ─── Logging ──────────────────────────────────────────────────────────────────

const ok = msg => { console.log(`  ✓ ${msg}`); passed++; };
const ko = (msg, hint = '') => {
  const full = hint ? `${msg} (${hint})` : msg;
  console.error(`  ✗ ${full}`);
  failed++;
  failures.push(full);
};
const chk = (cond, msg, hint = '') => (cond ? ok : ko)(msg, hint || '');
// API key must never appear in any response body
const noKey = (text, label) =>
  chk(!String(text).includes(FAKE_KEY), label, 'API key leaked into output');

const delay = ms => new Promise(r => setTimeout(r, ms));

// ─── Process & file cleanup ───────────────────────────────────────────────────

async function killWorker() {
  if (!workerProc || workerProc.exitCode !== null) { workerProc = null; return; }
  const pid = workerProc.pid;
  workerProc = null;
  await new Promise(r => {
    if (process.platform === 'win32') {
      // Kill entire process tree so wrangler child processes also die
      const k = spawn('taskkill', ['/PID', String(pid), '/F', '/T'], { stdio: 'ignore' });
      k.on('close', r);
      k.on('error', r);
    } else {
      try { process.kill(-pid, 'SIGKILL'); } catch {}
      setTimeout(r, 600);
    }
  });
  await delay(700); // wait for OS to release the port
}

function cleanDev() {
  try { if (existsSync(DEV_VARS)) unlinkSync(DEV_VARS); } catch {}
}

// ─── Mock backend ─────────────────────────────────────────────────────────────

// Tracks the X-API-Key the Worker forwarded to us on the most recent request
let proxyApiKey = null;

function startMock() {
  return new Promise(resolve => {
    mockServer = http.createServer((req, res) => {
      proxyApiKey = req.headers['x-api-key'] ?? null;

      // GET /health
      if (req.method === 'GET' && req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ status: 'ok', service: 'mock-backend' }));
      }

      // POST /echo — mirrors the request body
      if (req.method === 'POST' && req.url === '/echo') {
        const bufs = [];
        req.on('data', c => bufs.push(c));
        req.on('end', () => {
          let body = {};
          try { body = JSON.parse(Buffer.concat(bufs).toString()); } catch {}
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ received: body }));
        });
        return;
      }

      // POST /recommend — SSE with two distinct chunks separated by 60 ms each
      if (req.method === 'POST' && req.url === '/recommend') {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'X-Accel-Buffering': 'no',
        });
        res.write('data: first\n\n');
        setTimeout(() => {
          res.write('data: second\n\n');
          setTimeout(() => { res.write('data: [DONE]\n\n'); res.end(); }, 60);
        }, 60);
        return;
      }

      res.writeHead(404);
      res.end();
    });
    mockServer.listen(MOCK_PORT, '127.0.0.1', resolve);
  });
}

// ─── Worker lifecycle ─────────────────────────────────────────────────────────

function writeVars(vars) {
  // Filter out null/undefined so omitted vars are truly absent in the worker env
  const text = Object.entries(vars)
    .filter(([, v]) => v != null)
    .map(([k, v]) => `${k}=${v}`)
    .join('\n') + '\n';
  writeFileSync(DEV_VARS, text, 'utf8');
}

async function startWorker(vars) {
  writeVars(vars);

  // wrangler.jsonc specifies main: ".open-next/worker.js"
  // No --no-bundle: wrangler runs esbuild to inline the relative ./cloudflare/*.js
  // imports that the OpenNext template marks as "@ts-expect-error: Will be resolved by wrangler build".
  // .dev.vars is read automatically by wrangler dev for local secrets/vars.
  const cmd = `npx wrangler dev --port ${WORKER_PORT}`;
  const log = [];
  let exitedEarly = false;

  workerProc = spawn(cmd, {
    shell: true, cwd: FRONTEND,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' },
  });
  workerProc.on('exit', () => { exitedEarly = true; });

  // Wait for wrangler to signal readiness (output contains "Ready" or port number)
  // OR fall back to TCP port open after 45 s
  const ready = await new Promise(resolve => {
    const deadline = setTimeout(() => resolve(false), 45000);
    const check = chunk => {
      const s = chunk.toString();
      log.push(s);
      if (/ready|listening|\b8789\b/i.test(s)) {
        clearTimeout(deadline);
        resolve(true);
      }
    };
    workerProc.stdout.on('data', check);
    workerProc.stderr.on('data', check);
    // Also poll TCP as a fallback (some wrangler versions don't print "Ready")
    const poll = setInterval(async () => {
      if (exitedEarly) { clearInterval(poll); clearTimeout(deadline); resolve(false); return; }
      const open = await new Promise(r => {
        const s = net.createConnection(WORKER_PORT, '127.0.0.1');
        s.on('connect', () => { s.destroy(); r(true); });
        s.on('error', () => r(false));
      });
      if (open) { clearInterval(poll); clearTimeout(deadline); resolve(true); }
    }, 500);
  });

  if (!ready || exitedEarly) {
    console.error('Worker failed to start. Last output:\n' + log.join('').slice(-3000));
    throw new Error(exitedEarly
      ? 'wrangler dev process exited unexpectedly'
      : `wrangler dev did not bind on port ${WORKER_PORT} within 45 s`);
  }

  // Extra grace period for workerd to fully compile the worker after reporting ready
  await delay(2500);
}

// req() with automatic retry on ECONNRESET/ECONNREFUSED (wrangler may still be warming up)
async function reqRetry(method, urlPath, body, extraHeaders = {}, maxRetries = 4) {
  let lastErr;
  for (let i = 0; i <= maxRetries; i++) {
    try {
      return await req(method, urlPath, body, extraHeaders);
    } catch (e) {
      lastErr = e;
      if ((e.code === 'ECONNRESET' || e.code === 'ECONNREFUSED') && i < maxRetries) {
        await delay(800 * (i + 1));
      } else break;
    }
  }
  throw lastErr;
}

// sseReq() with automatic retry on ECONNRESET/ECONNREFUSED
async function sseReqRetry(urlPath, body, maxRetries = 3) {
  let lastErr;
  for (let i = 0; i <= maxRetries; i++) {
    try {
      return await sseReq(urlPath, body);
    } catch (e) {
      lastErr = e;
      if ((e.code === 'ECONNRESET' || e.code === 'ECONNREFUSED') && i < maxRetries) {
        await delay(800 * (i + 1));
      } else break;
    }
  }
  throw lastErr;
}

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

function req(method, urlPath, body, extraHeaders = {}) {
  return new Promise((resolve, reject) => {
    const bodyBuf = body != null ? Buffer.from(JSON.stringify(body)) : null;
    const r = http.request(
      {
        hostname: '127.0.0.1', port: WORKER_PORT, path: urlPath, method,
        headers: bodyBuf
          ? { 'Content-Type': 'application/json', 'Content-Length': bodyBuf.length, ...extraHeaders }
          : extraHeaders,
        timeout: 12000,
      },
      res => {
        const chunks = [];
        res.on('data', c => chunks.push(c));
        res.on('end', () => resolve({
          status: res.statusCode,
          headers: res.headers,
          body: Buffer.concat(chunks).toString(),
        }));
      }
    );
    r.on('error', reject);
    r.on('timeout', () => { r.destroy(); reject(new Error('request timeout')); });
    if (bodyBuf) r.end(bodyBuf); else r.end();
  });
}

// Streams an SSE response; records each data event and its arrival time.
function sseReq(urlPath, body) {
  return new Promise((resolve, reject) => {
    const bodyBuf = Buffer.from(JSON.stringify(body));
    const chunks = [], times = [];
    const r = http.request(
      {
        hostname: '127.0.0.1', port: WORKER_PORT, path: urlPath, method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': bodyBuf.length,
          'Accept': 'text/event-stream',
        },
        timeout: 20000,
      },
      res => {
        const t0 = Date.now();
        let full = '';
        res.on('data', c => {
          chunks.push(c.toString());
          times.push(Date.now() - t0);
          full += c;
        });
        res.on('end', () => resolve({
          status: res.statusCode,
          headers: res.headers,
          chunks, times, body: full,
        }));
      }
    );
    r.on('error', reject);
    r.on('timeout', () => { r.destroy(); reject(new Error('SSE timeout')); });
    r.end(bodyBuf);
  });
}

// ─── Scenarios ────────────────────────────────────────────────────────────────

async function scenarioA() {
  console.log('\n── A: Dev mode — health proxy, POST body, SSE streaming ──');

  await startWorker({
    APP_ENV: 'development',
    MEAL_AGENT_BACKEND_URL: `http://127.0.0.1:${MOCK_PORT}`,
    MEAL_AGENT_API_KEY: FAKE_KEY,
  });

  // A1: GET /health — basic proxy + API key injection
  console.log('  [A1] GET /api/backend/health');
  proxyApiKey = null;
  const h = await reqRetry('GET', '/api/backend/health');
  chk(h.status === 200, 'status 200');
  chk((h.headers['content-type'] || '').includes('application/json'), 'content-type: application/json');
  const hj = (() => { try { return JSON.parse(h.body); } catch { return {}; } })();
  chk(hj.status === 'ok', 'body.status = ok');
  chk(hj.service === 'mock-backend', 'body.service = mock-backend');
  chk(proxyApiKey === FAKE_KEY, 'Worker forwarded X-API-Key to upstream');
  noKey(h.body, 'health response does not expose API key');

  // A2: POST /echo — POST body forwarding + duplex:'half' compatibility in workerd
  console.log('  [A2] POST /api/backend/echo  (POST body + duplex compatibility)');
  const e = await reqRetry('POST', '/api/backend/echo', { probe: 'duplex-smoke', n: 42 });
  chk(e.status === 200, 'status 200');
  // duplex failure manifests as a TypeError; verify it does not appear
  chk(!e.body.toLowerCase().includes('typeerror'), 'no TypeError (duplex: half safe in workerd)');
  const ej = (() => { try { return JSON.parse(e.body); } catch { return {}; } })();
  chk(ej.received?.probe === 'duplex-smoke', 'POST body forwarded correctly');
  chk(ej.received?.n === 42, 'numeric field preserved');
  noKey(e.body, 'echo response does not expose API key');

  // A3: SSE streaming — verify chunks arrive incrementally, not buffered
  console.log('  [A3] POST /api/backend/recommend  (SSE streaming)');
  const s = await sseReqRetry('/api/backend/recommend', { q: 'smoke' });
  chk(s.status === 200, 'status 200');
  chk((s.headers['content-type'] || '').includes('text/event-stream'), 'content-type: text/event-stream');
  chk(s.body.includes('data: first'), 'body contains "data: first"');
  chk(s.body.includes('data: second'), 'body contains "data: second"');
  chk(s.body.includes('data: [DONE]'), 'body contains "data: [DONE]"');
  const doneCount = (s.body.match(/\[DONE\]/g) || []).length;
  chk(doneCount === 1, `[DONE] appears exactly once (got ${doneCount})`);
  // Streaming proof: at least 2 separate data events received
  chk(s.chunks.length >= 2, `${s.chunks.length} network chunks received (≥2 = streamed not buffered)`);
  // Time-span proof: last chunk arrived ≥50ms after first (mock sends each chunk 60ms apart)
  const span = s.times.length >= 2 ? s.times[s.times.length - 1] - s.times[0] : 0;
  chk(span >= 50, `chunk time-span ${span}ms ≥ 50ms (confirms response not buffered)`);
}

async function scenarioB() {
  console.log('\n── B: Production — missing BACKEND_URL → 503 ──');
  // Only APP_ENV is set; MEAL_AGENT_BACKEND_URL and MEAL_AGENT_API_KEY are absent
  await startWorker({ APP_ENV: 'production' });
  const r = await reqRetry('GET', '/api/backend/health');
  chk(r.status === 503, `status 503 (got ${r.status})`);
  chk(!r.body.includes('localhost'), 'no "localhost" in body');
  chk(!r.body.includes('127.0.0.1'), 'no IP address in body');
  chk(!r.body.includes('http://'), 'no internal URL in body');
  noKey(r.body, 'no API key in body');
  chk(!r.body.includes('stack') && !r.body.includes('Error:'), 'no stack trace in body');
  chk(r.body.includes('configuration') || r.body.includes('Service'), 'safe config-error message returned');
}

async function scenarioC() {
  console.log('\n── C: Production — missing API Key → 503 ──');
  await startWorker({ APP_ENV: 'production', MEAL_AGENT_BACKEND_URL: 'https://example.invalid' });
  const r = await reqRetry('GET', '/api/backend/health');
  chk(r.status === 503, `status 503 (got ${r.status})`);
  chk(!r.body.includes('example.invalid'), 'backend URL not exposed in body');
  noKey(r.body, 'no API key in body');
  chk(r.body.includes('configuration') || r.body.includes('Service'), 'safe error message returned');
}

async function scenarioD() {
  console.log('\n── D: Production — non-HTTPS backend URL → 503 ──');
  await startWorker({
    APP_ENV: 'production',
    MEAL_AGENT_BACKEND_URL: 'http://example.invalid', // HTTP, not HTTPS
    MEAL_AGENT_API_KEY: FAKE_KEY,
  });
  const r = await reqRetry('GET', '/api/backend/health');
  chk(r.status === 503, `status 503 (got ${r.status})`);
  chk(!r.body.includes('example.invalid'), 'backend URL not exposed in body');
  noKey(r.body, 'no API key in body');
}

// ─── Build guard ──────────────────────────────────────────────────────────────

async function ensureBuilt() {
  const workerJs = path.join(FRONTEND, '.open-next', 'worker.js');
  if (existsSync(workerJs)) {
    console.log('  Using existing .open-next/ build.\n');
    return;
  }
  // Build if .open-next/worker.js is absent.
  // We do NOT delete any existing .open-next/ first — on Windows the
  // opennextjs-cloudflare build's own rmSync handles cleanup only when needed.
  console.log('  .open-next/worker.js not found — running opennextjs-cloudflare build...');
  await new Promise((resolve, reject) => {
    const b = spawn('npx opennextjs-cloudflare build', {
      shell: true, cwd: FRONTEND, stdio: 'inherit',
    });
    b.on('close', code =>
      code === 0 ? resolve() : reject(new Error(`opennextjs-cloudflare build exited ${code}`))
    );
    b.on('error', reject);
  });
  console.log();
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log('=== Cloudflare Worker Runtime Smoke Test ===');
  console.log(`  mock backend : http://127.0.0.1:${MOCK_PORT}`);
  console.log(`  worker port  : http://127.0.0.1:${WORKER_PORT}`);
  console.log(`  .dev.vars    : ${DEV_VARS}  (deleted after each scenario)\n`);

  await ensureBuilt();

  try {
    await startMock();

    for (const scenario of [scenarioA, scenarioB, scenarioC, scenarioD]) {
      try {
        await scenario();
      } catch (err) {
        ko(`${scenario.name} threw an unexpected error`, err.message);
      } finally {
        await killWorker();
        cleanDev();
        await delay(800); // let the port be released before next scenario
      }
    }
  } finally {
    // Belt-and-suspenders cleanup
    await killWorker();
    cleanDev();
    if (mockServer) { mockServer.close(); mockServer = null; }
  }

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  if (failures.length) {
    console.error('\nFailures:');
    failures.forEach(f => console.error(`  – ${f}`));
  }

  process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
  console.error('\nFatal error:', err.message);
  killWorker().finally(() => {
    cleanDev();
    if (mockServer) mockServer.close();
    process.exit(1);
  });
});
