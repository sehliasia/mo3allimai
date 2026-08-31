export const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000/api'
export type AuthUser = { id: number; full_name: string; email: string; role: 'teacher' | 'admin'; is_active: boolean }
type ApiError = { detail?: string }
export class AuthApiError extends Error {
  constructor(message: string, public readonly status?: number) { super(message) }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response
  try { response = await fetch(`${API_URL}${path}`, { ...options, headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers } }) }
  catch { throw new AuthApiError('تعذر الاتصال بالخادم') }
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as ApiError
    const messages: Record<string, string> = { 'Email already registered': 'هذا البريد الإلكتروني مستخدم بالفعل', 'Account is inactive': 'الحساب غير مفعل', 'Invalid email or password': 'البريد الإلكتروني أو كلمة المرور غير صحيحة' }
    throw new AuthApiError(messages[body.detail ?? ''] ?? 'يرجى التحقق من البيانات المدخلة', response.status)
  }
  return response.json() as Promise<T>
}

export function register(data: { full_name: string; email: string; password: string }) { return request<{ message: string; user: AuthUser }>('/auth/register', { method: 'POST', body: JSON.stringify(data) }) }
export function login(data: { email: string; password: string }) { return request<{ access_token: string; token_type: string; user: AuthUser }>('/auth/login', { method: 'POST', body: JSON.stringify(data) }) }
export function getCurrentUser(token: string) { return request<AuthUser>('/auth/me', { headers: { Authorization: `Bearer ${token}` } }) }
export function saveToken(token: string) { localStorage.setItem('mo3allimai_token', token) }
export function getStoredToken() { return localStorage.getItem('mo3allimai_token') }
export function clearSession() { localStorage.removeItem('mo3allimai_token'); localStorage.removeItem('mo3allimai_user') }
