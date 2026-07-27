/**
 * Catch-all server-side proxy: /api/backend/[...path] → Railway backend.
 *
 * Security design:
 * - MEAL_AGENT_API_KEY is a server-only env var (no NEXT_PUBLIC_ prefix).
 * - The key is injected into the X-API-Key header server-side and never
 *   returned to the browser or included in any response.
 * - Client-supplied X-Meal-Agent-User-ID and X-API-Key headers are stripped
 *   unconditionally; the proxy re-injects both from server state.
 * - For user-scoped paths, an anonymous session is resolved from
 *   MEAL_AGENT_SESSION_SECRET (HttpOnly HMAC cookie). Missing secret in
 *   production → 503 fail-close before any upstream request.
 * - Client IP is hashed by the backend; raw IP is forwarded only to the
 *   backend as X-Client-IP for rate-limit bucketing, never stored.
 * - SSE responses (/recommend) are streamed via response.body — no buffering.
 *   Set-Cookie is added to response headers before the Response is created,
 *   so it works for both streaming and non-streaming responses.
 * - Hop-by-hop headers are stripped from both directions.
 */

import { NextRequest } from 'next/server';
import { resolveAnonymousSession } from '../../../lib/server/anonymous-session';

// Headers that must not be forwarded (hop-by-hop)
const HOP_BY_HOP = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailers',
  'transfer-encoding',
  'upgrade',
  // Next.js internal
  'host',
]);

// Browser-supplied identity headers the proxy must strip unconditionally.
// The proxy re-injects correct values from server state (API key, session cookie).
const STRIPPED_CLIENT_HEADERS = new Set([
  'x-meal-agent-user-id',
  'x-api-key',
]);

// Paths that require an authenticated anonymous session.
// Non-scoped paths (health, price-history, tracked-terms) are forwarded without
// a session so missing MEAL_AGENT_SESSION_SECRET does not block status checks.
const _USER_SCOPED_EXACT = new Set([
  '/recommend',
  '/api/daily-recommendation',
  '/api/daily-recommendation/generate',
]);
const _USER_SCOPED_PREFIXES = ['/users/'];

function _isUserScopedPath(path: string): boolean {
  if (_USER_SCOPED_EXACT.has(path)) return true;
  if (path.startsWith('/api/daily-recommendation/')) return true;
  return _USER_SCOPED_PREFIXES.some(prefix => path.startsWith(prefix));
}

type Context = { params: Promise<{ path: string[] }> };

/**
 * True when running in a production deployment.
 * Checks APP_ENV (Cloudflare Workers / Railway) and VERCEL_ENV (Vercel) so the
 * same route handler works on both platforms without platform-specific branches.
 */
function _isProduction(): boolean {
  return (
    process.env.APP_ENV === 'production' ||
    process.env.VERCEL_ENV === 'production'
  );
}

const _503_CONFIG_ERROR = JSON.stringify({ detail: 'Service configuration error.' });
const _502_UPSTREAM_ERROR = JSON.stringify({ detail: 'Backend service temporarily unavailable.' });
const _JSON_HEADERS = { 'content-type': 'application/json' };

/** Validate production config at request time. Returns a 503 Response on failure, null on success. */
function _checkProductionConfig(rawBackendUrl: string, apiKey: string): Response | null {
  if (!rawBackendUrl) {
    console.error('[proxy] config_error=missing_backend_url');
    return new Response(_503_CONFIG_ERROR, { status: 503, headers: _JSON_HEADERS });
  }
  if (!apiKey) {
    console.error('[proxy] config_error=missing_api_key');
    return new Response(_503_CONFIG_ERROR, { status: 503, headers: _JSON_HEADERS });
  }
  let parsed: URL;
  try {
    parsed = new URL(rawBackendUrl);
  } catch {
    console.error('[proxy] config_error=invalid_backend_url');
    return new Response(_503_CONFIG_ERROR, { status: 503, headers: _JSON_HEADERS });
  }
  const host = parsed.hostname;
  if (host === 'localhost' || host === '127.0.0.1' || host === '::1') {
    console.error('[proxy] config_error=localhost_in_production');
    return new Response(_503_CONFIG_ERROR, { status: 503, headers: _JSON_HEADERS });
  }
  if (parsed.protocol !== 'https:') {
    console.error('[proxy] config_error=non_https_in_production');
    return new Response(_503_CONFIG_ERROR, { status: 503, headers: _JSON_HEADERS });
  }
  return null;
}

/** Forward a request to the Railway backend and return the response. */
async function proxy(request: NextRequest, context: Context): Promise<Response> {
  const { path } = await context.params;
  const backendPath = '/' + path.join('/');

  // Read env at request time — Route Handler is always dynamic (never statically bundled)
  const rawBackendUrl = (process.env.MEAL_AGENT_BACKEND_URL ?? '').replace(/\/+$/, '');
  const apiKey = process.env.MEAL_AGENT_API_KEY ?? '';

  // Production fail-close: reject before making any upstream request
  if (_isProduction()) {
    const configError = _checkProductionConfig(rawBackendUrl, apiKey);
    if (configError) return configError;
  }

  const backendBaseUrl = rawBackendUrl || 'http://localhost:8000';

  // Preserve query string
  const search = request.nextUrl.search;
  const upstreamUrl = `${backendBaseUrl}${backendPath}${search}`;

  // Extract client IP for backend rate-limit bucketing (never logged here)
  const xForwardedFor = request.headers.get('x-forwarded-for') ?? '';
  const clientIp = xForwardedFor.split(',')[0]?.trim() ?? request.headers.get('x-real-ip') ?? '';

  // Build forwarded headers — strip hop-by-hop AND client-supplied identity headers.
  // X-API-Key and X-Meal-Agent-User-ID are re-injected below from server state.
  const forwardHeaders = new Headers();
  request.headers.forEach((value, key) => {
    const lower = key.toLowerCase();
    if (!HOP_BY_HOP.has(lower) && !STRIPPED_CLIENT_HEADERS.has(lower)) {
      forwardHeaders.set(key, value);
    }
  });
  if (apiKey) {
    forwardHeaders.set('X-API-Key', apiKey);
  }
  if (clientIp) {
    forwardHeaders.set('X-Client-IP', clientIp);
  }

  // Resolve anonymous session for user-scoped paths and inject identity header
  let setCookieHeader: string | undefined;
  if (_isUserScopedPath(backendPath)) {
    const session = await resolveAnonymousSession(request);
    if (session === null) {
      // Production misconfiguration: MEAL_AGENT_SESSION_SECRET missing or invalid
      console.error('[proxy] config_error=missing_session_secret');
      return new Response(_503_CONFIG_ERROR, { status: 503, headers: _JSON_HEADERS });
    }
    forwardHeaders.set('X-Meal-Agent-User-ID', session.userId);
    setCookieHeader = session.setCookieHeader;
  }

  // Forward body for methods that carry one
  const hasBody = !['GET', 'HEAD', 'DELETE'].includes(request.method.toUpperCase());

  let upstreamResponse: Response;
  try {
    upstreamResponse = await fetch(upstreamUrl, {
      method: request.method,
      headers: forwardHeaders,
      body: hasBody ? request.body : undefined,
      // @ts-expect-error — duplex required for streaming request bodies in Node.js
      duplex: 'half',
      signal: request.signal,
    });
  } catch (err: unknown) {
    // Log safe error category only — no keys, URLs, or raw error messages
    const category = err instanceof TypeError ? 'network' : 'internal';
    console.error(`[proxy] upstream_error category=${category} path=${backendPath}`);
    return new Response(_502_UPSTREAM_ERROR, {
      status: 502,
      headers: _JSON_HEADERS,
    });
  }

  // Build safe response headers — strip hop-by-hop, never reflect API key
  const responseHeaders = new Headers();
  upstreamResponse.headers.forEach((value, key) => {
    const lower = key.toLowerCase();
    if (!HOP_BY_HOP.has(lower) && lower !== 'x-api-key') {
      responseHeaders.set(key, value);
    }
  });

  // Attach Set-Cookie when a new anonymous session was created.
  // This is set on responseHeaders before the Response is constructed, so it is
  // included in both streaming SSE responses and regular JSON responses without
  // buffering the body.
  if (setCookieHeader) {
    responseHeaders.set('Set-Cookie', setCookieHeader);
  }

  const contentType = upstreamResponse.headers.get('content-type') ?? '';

  // SSE: stream body directly without buffering
  if (contentType.includes('text/event-stream')) {
    responseHeaders.set('Content-Type', 'text/event-stream');
    responseHeaders.set('Cache-Control', 'no-cache');
    responseHeaders.set('X-Accel-Buffering', 'no');
    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers: responseHeaders,
    });
  }

  // All other responses: pass body through directly
  return new Response(upstreamResponse.body, {
    status: upstreamResponse.status,
    headers: responseHeaders,
  });
}

export async function GET(request: NextRequest, context: Context) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: Context) {
  return proxy(request, context);
}

export async function PUT(request: NextRequest, context: Context) {
  return proxy(request, context);
}

export async function PATCH(request: NextRequest, context: Context) {
  return proxy(request, context);
}

export async function DELETE(request: NextRequest, context: Context) {
  return proxy(request, context);
}
