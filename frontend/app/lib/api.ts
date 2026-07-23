/**
 * All backend calls go through the same-origin Next.js Server Route Handler
 * at /api/backend/[...path]. The Route Handler injects the server-side
 * MEAL_AGENT_API_KEY and proxies requests to Railway.
 *
 * No NEXT_PUBLIC_ backend URL or API key is exposed to the browser.
 */

const PROXY_BASE = '/api/backend';

/** Build a proxied URL from a backend path (leading slash optional). */
export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${PROXY_BASE}${normalizedPath}`;
}
