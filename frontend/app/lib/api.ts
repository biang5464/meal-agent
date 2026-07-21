const rawApiBase =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/** Backend base URL with any trailing slashes stripped. */
export const API_BASE_URL = rawApiBase.replace(/\/+$/, "");

/** Build a full backend URL from a path (leading slash optional). */
export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}
