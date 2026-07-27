/**
 * Server-only anonymous session management.
 *
 * Cookie:   meal_agent_session=v1.<uuid>.<hmac-sha256-base64url>
 * User ID:  anon_<uuid>
 *
 * SECURITY: Do NOT import this file from any Client Component.
 * It reads MEAL_AGENT_SESSION_SECRET from server-side env vars and must
 * never be bundled into the browser.
 */

import type { NextRequest } from 'next/server';

export const COOKIE_NAME = 'meal_agent_session';

// Hard limit on cookie value length — rejects obviously malformed inputs
const MAX_COOKIE_LEN = 200;

// RFC 4122 canonical UUID pattern (lowercase hex with dashes)
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

// Placeholder values that must not be used as a real session secret
const _WEAK_SECRETS = new Set([
  '',
  'changeme',
  'placeholder',
  '...',
  'your-secret-here',
  'dev-only-meal-agent-session-secret-2026',
]);

// Dev-only hardcoded fallback — clearly NOT suitable for production use.
// Sessions created with this secret are not secure; all dev browsers share the
// same signing key. Never set MEAL_AGENT_SESSION_SECRET to this value in prod.
const _DEV_FALLBACK_SECRET = 'dev-only-meal-agent-session-secret-2026';

export type AnonymousSession = {
  userId: string;
  /** Present only when a new cookie must be issued (new or replacement session). */
  setCookieHeader?: string;
};

function _isProduction(): boolean {
  return (
    process.env.APP_ENV === 'production' ||
    process.env.VERCEL_ENV === 'production'
  );
}

/**
 * Returns the signing secret, or null if the config is unusable in production.
 * In development, falls back to a clearly-labeled dev constant.
 */
function _getSecret(): string | null {
  const raw = (process.env.MEAL_AGENT_SESSION_SECRET ?? '').trim();
  if (raw && !_WEAK_SECRETS.has(raw.toLowerCase())) {
    return raw;
  }
  if (!_isProduction()) {
    return _DEV_FALLBACK_SECRET;
  }
  // Production with no valid secret → fail-close (caller returns 503)
  return null;
}

async function _importHmacKey(secret: string, usages: KeyUsage[]): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    usages,
  );
}

function _toBase64Url(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function _fromBase64Url(s: string): Uint8Array<ArrayBuffer> {
  const padded = s.replace(/-/g, '+').replace(/_/g, '/');
  const pad = (4 - (padded.length % 4)) % 4;
  const binary = atob(padded + '='.repeat(pad));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function _signPayload(secret: string, payload: string): Promise<string> {
  const key = await _importHmacKey(secret, ['sign']);
  const sig = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(payload));
  return _toBase64Url(sig);
}

/** Constant-time HMAC verification via crypto.subtle.verify. */
async function _verifyPayload(
  secret: string,
  payload: string,
  sigB64url: string,
): Promise<boolean> {
  try {
    const key = await _importHmacKey(secret, ['verify']);
    const sigBytes = _fromBase64Url(sigB64url);
    return crypto.subtle.verify('HMAC', key, sigBytes, new TextEncoder().encode(payload));
  } catch {
    return false;
  }
}

function _buildSetCookieHeader(value: string, isProduction: boolean): string {
  const parts = [
    `${COOKIE_NAME}=${value}`,
    'Path=/',
    'Max-Age=31536000',
    'HttpOnly',
    'SameSite=Lax',
  ];
  if (isProduction) parts.push('Secure');
  return parts.join('; ');
}

async function _createSession(
  secret: string,
  isProduction: boolean,
): Promise<AnonymousSession> {
  const uuid = crypto.randomUUID();
  const payload = `v1.${uuid}`;
  const sig = await _signPayload(secret, payload);
  const cookieValue = `${payload}.${sig}`;
  return {
    userId: `anon_${uuid}`,
    setCookieHeader: _buildSetCookieHeader(cookieValue, isProduction),
  };
}

/**
 * Try to parse and verify an existing cookie value.
 * Returns the userId string on success, or null on any failure (tampered, malformed, expired).
 * Invalid inputs always produce null — never a thrown exception.
 */
async function _parseSession(rawCookie: string, secret: string): Promise<string | null> {
  if (!rawCookie || rawCookie.length > MAX_COOKIE_LEN) return null;

  // Cookie format: v1.<uuid>.<sig>
  // Use lastIndexOf to cleanly split signature (base64url has no dots)
  const lastDot = rawCookie.lastIndexOf('.');
  if (lastDot === -1) return null;

  const payload = rawCookie.slice(0, lastDot);
  const sig = rawCookie.slice(lastDot + 1);

  if (!payload.startsWith('v1.')) return null;
  const uuid = payload.slice(3);
  if (!UUID_RE.test(uuid)) return null;
  if (!sig) return null;

  const valid = await _verifyPayload(secret, payload, sig);
  if (!valid) return null;

  return `anon_${uuid}`;
}

/**
 * Resolve (or create) the anonymous session for a request.
 *
 * Returns:
 *   - { userId }                          — valid existing session found
 *   - { userId, setCookieHeader }         — new session created; caller MUST send Set-Cookie
 *   - null                                — production misconfiguration (session secret missing);
 *                                           caller MUST return a 503 and NOT forward the request
 *
 * No secrets, cookies, or user IDs are written to any log.
 */
export async function resolveAnonymousSession(
  request: NextRequest,
): Promise<AnonymousSession | null> {
  const secret = _getSecret();
  if (secret === null) {
    // Production fail-close: MEAL_AGENT_SESSION_SECRET missing or invalid
    return null;
  }

  const isProduction = _isProduction();
  const rawCookie = request.cookies.get(COOKIE_NAME)?.value ?? '';

  const userId = await _parseSession(rawCookie, secret);
  if (userId !== null) {
    return { userId };
  }

  // Missing, malformed, or tampered cookie — issue a fresh session
  return _createSession(secret, isProduction);
}
