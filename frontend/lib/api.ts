export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

export function authHeaders(token?: string): HeadersInit {
  const headers: HeadersInit = { };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

export async function apiFetch(path: string, options: RequestInit & { token?: string } = {}) {
  const { token, ...rest } = options;
  const url = `${API_BASE_URL}${path}`;
  const headers = new Headers(rest.headers || {});
  const auth = authHeaders(token);
  Object.entries(auth).forEach(([k, v]) => headers.set(k, String(v)));
  return fetch(url, { ...rest, headers });
}
