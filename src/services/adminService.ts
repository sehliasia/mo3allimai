import { getStoredToken, type AuthUser } from './authService'
const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000/api'
async function adminRequest<T>(path: string, options: RequestInit = {}) { const token = getStoredToken(); const response = await fetch(`${API_URL}${path}`, { ...options, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...options.headers } }); if (!response.ok) throw new Error('تعذر تنفيذ الطلب. يرجى المحاولة مرة أخرى.'); return response.json() as Promise<T> }
export type TeachersPage = { items: AuthUser[]; page: number; page_size: number; total: number; total_pages: number }
export const getTeachers = (params = '') => adminRequest<TeachersPage>(`/admin/teachers${params}`)
export const getTeacherById = (id: number) => adminRequest<AuthUser>(`/admin/teachers/${id}`)
export const updateTeacherStatus = (id: number, is_active: boolean) => adminRequest<AuthUser>(`/admin/teachers/${id}/status`, { method: 'PATCH', body: JSON.stringify({ is_active }) })
export const deleteTeacher = (id: number) => adminRequest<AuthUser>(`/admin/teachers/${id}`, { method: 'DELETE' })
export const getAdminStatistics = () => adminRequest<{ total_teachers: number; active_teachers: number; inactive_teachers: number; new_teachers_this_month: number }>('/admin/statistics')
