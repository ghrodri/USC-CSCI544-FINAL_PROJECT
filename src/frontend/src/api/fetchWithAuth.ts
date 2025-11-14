export async function fetchWithAuth(
  input: string | URL,
  init: RequestInit = {}
) {
  const headers = new Headers(init.headers || {});
  const token = localStorage.getItem("token");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(input, { ...init, headers });
}
