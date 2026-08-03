const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000/api'
export type AuthUser = { id: number; full_name: string; email: string; role: 'teacher' | 'admin'; is_active: boolean }
type ApiError = { detail?: string }
async function request<T>(path: string, options: RequestInit): Promise<T> {
  let response: Response
  try { response = await fetch(`${API_URL}${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } }) }
  catch { throw new Error('تعذر الاتصال بالخادم') }
  if (!response.ok) { const body = await response.json().catch(() => ({})) as ApiError; throw new Error(body.detail === 'Email already registered' ? 'هذا البريد الإلكتروني مستخدم بالفعل' : body.detail === 'Account is inactive' ? 'الحساب غير مفعل' : body.detail === 'Invalid email or password' ? 'البريد الإلكتروني أو كلمة المرور غير صحيحة' : 'يرجى التحقق من البيانات المدخلة') }
  return response.json() as Promise<T>
}
export function register(data: { full_name: string; email: string; password: string }) { return request<{ message: string; user: AuthUser }>('/auth/register', { method: 'POST', body: JSON.stringify(data) }) }
export function login(data: { email: string; password: string }) { return request<{ access_token: string; token_type: string; user: AuthUser }>('/auth/login', { method: 'POST', body: JSON.stringify(data) }) }
export function saveSession(token: string, user: AuthUser) { localStorage.setItem('mo3allimai_token', token); localStorage.setItem('mo3allimai_user', JSON.stringify(user)) }
export function getStoredUser(): AuthUser | null { try { return JSON.parse(localStorage.getItem('mo3allimai_user') ?? 'null') } catch { return null } }
export function logout() { localStorage.removeItem('mo3allimai_token'); localStorage.removeItem('mo3allimai_user') }
