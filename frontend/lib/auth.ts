import { API_BASE_URL } from "./api";

export type RegisterPayload = {
  email: string;
  password: string;
  full_name: string;
};
// API functions
export type TokenResponse = {
  access_token: string;
  token_type: string;
  expires_in: number;
};

export async function loginUser(email: string, password: string): Promise<TokenResponse> {
  const body = new URLSearchParams();
  body.set('username', email);
  body.set('password', password);

  const res = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
  
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || 'Login failed');
  }
  
  return res.json();
}

export async function registerUser(payload: RegisterPayload): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || 'Registration failed');
  }
  
  return res.json();
}

// Helper function to get auth headers
export function getAuthHeaders(token: string) {
  return {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };
}
