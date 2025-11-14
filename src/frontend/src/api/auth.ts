const API_BASE = ""; // package.json proxy -> http://localhost:8000

export type AuthResponse = {
  access_token: string;
  token_type: string;
  username: string;
};

export async function register(
  email: string,
  username: string,
  password: string
): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, username, password }),
  });
  if (!res.ok) throw new Error((await res.text()) || "Registration failed");
  const data = (await res.json()) as AuthResponse;
  localStorage.setItem("token", data.access_token);
  localStorage.setItem("email", email);
  localStorage.setItem("username", data.username);
  return data;
}

export async function login(
  email: string,
  password: string
): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error((await res.text()) || "Login failed");
  const data = (await res.json()) as AuthResponse;
  localStorage.setItem("token", data.access_token);
  localStorage.setItem("email", email);
  localStorage.setItem("username", data.username);
  return data;
}

export function isAuthed() {
  return Boolean(localStorage.getItem("token"));
}

export function getUserEmail(): string | null {
  return localStorage.getItem("email");
}

export function getUserUsername(): string | null {
  return localStorage.getItem("username");
}

export function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("email");
  localStorage.removeItem("username");
}

export function getToken(): string | null {
  return localStorage.getItem("token");
}
