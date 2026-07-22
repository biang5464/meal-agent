/**
 * Catch-all server-side proxy: /api/backend/[...path] → Railway backend.
 *
 * Security design:
 * - MEAL_AGENT_API_KEY is a server-only env var (no NEXT_PUBLIC_ prefix).
 * - The key is injected into the X-API-Key header server-side and never
 *   returned to the browser or included in any response.
 * - Client IP is hashed by the backend; raw IP is forwarded only to the
 *   backend as X-Client-IP for rate-limit bucketing, never stored.
 * - SSE responses (/recommend) are streamed via response.body — no buffering.
 * - Hop-by-hop headers are stripped from both directions.
 */

import { NextRequest } from 'next/server';

// Server-only environment variables (not NEXT_PUBLIC_)
const BACKEND_URL = (process.env.MEAL_AGENT_BACKEND_URL ?? 'http://localhost:8000').replace(/\/+$/, '');
const API_KEY = process.env.MEAL_AGENT_API_KEY ?? '';

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

type Context = { params: Promise<{ path: string[] }> };

/** Forward a request to the Railway backend and return the response. */
async function proxy(request: NextRequest, context: Context): Promise<Response> {
  const { path } = await context.params;
  const backendPath = '/' + path.join('/');

  // Preserve query string
  const search = request.nextUrl.search;
  const upstreamUrl = `${BACKEND_URL}${backendPath}${search}`;

  // Extract client IP for backend rate-limit bucketing (never logged here)
  const xForwardedFor = request.headers.get('x-forwarded-for') ?? '';
  const clientIp = xForwardedFor.split(',')[0]?.trim() ?? request.headers.get('x-real-ip') ?? '';

  // Build forwarded headers — strip hop-by-hop, inject auth and client IP
  const forwardHeaders = new Headers();
  request.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) {
      forwardHeaders.set(key, value);
    }
  });
  if (API_KEY) {
    forwardHeaders.set('X-API-Key', API_KEY);
  }
  if (clientIp) {
    forwardHeaders.set('X-Client-IP', clientIp);
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
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'upstream error';
    return new Response(JSON.stringify({ detail: `Proxy error: ${msg}` }), {
      status: 502,
      headers: { 'content-type': 'application/json' },
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
