import { AuthApiError, getStoredToken } from './authService'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000/api'

export type TeacherDocumentStatus = 'pending' | 'processing' | 'ready' | 'failed'
export type TeacherDocumentStage = 'uploaded' | 'extracting' | 'chunking' | 'embedding' | 'indexing' | 'completed' | 'failed' | 'indexed' | 'error' | null
export type TeacherLibraryDocument = {
  id: number; kind: 'document'; title: string; original_filename: string; mime_type: string; file_size: number
  status: TeacherDocumentStatus; processing_stage: TeacherDocumentStage; processing_error: string | null; created_at: string
}
export type TeacherSavedResource = { id: number; kind: 'creation'; resource_type: string; title: string; cefr_level: string | null; theme: string | null; content: Record<string, unknown>; created_at: string; updated_at: string }
export type TeacherLibraryItem = TeacherLibraryDocument | Omit<TeacherSavedResource, 'content'>

export class TeacherLibraryApiError extends Error {
  constructor(message: string, readonly status: number) { super(message) }
}

export async function teacherRequest<T>(path: string, options: RequestInit = {}, fallbackError = 'La demande liée au document a échoué.') {
  const token = getStoredToken()
  if (!token) throw new AuthApiError('Authentication is required', 401)
  const headers = new Headers(options.headers)
  headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(`${API_URL}${path}`, { ...options, headers })
  if (response.status === 401) throw new AuthApiError('Session expired', 401)
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: unknown } | null
    throw new TeacherLibraryApiError(typeof body?.detail === 'string' ? body.detail : fallbackError, response.status)
  }
  return response.status === 204 ? undefined as T : response.json() as Promise<T>
}

export async function getTeacherLibrary() { const response = await teacherRequest<{ items?: TeacherLibraryItem[] }>('/teacher/library'); return Array.isArray(response.items) ? response.items : [] }
export async function uploadTeacherDocument(file: File) { const body = new FormData(); body.append('file', file); return teacherRequest<TeacherLibraryDocument>('/teacher/library/documents', { method: 'POST', body }) }
export const deleteTeacherLibraryItem = (item: TeacherLibraryItem) => teacherRequest<void>(`/teacher/library/${item.kind === 'document' ? 'documents' : 'resources'}/${item.id}`, { method: 'DELETE' })
export function saveTeacherResource(data: { resource_type: 'lesson-plan' | 'lesson' | 'exercises' | 'exam' | 'assessment' | 'activity' | 'summary'; title: string; cefr_level: string; theme: string; content: Record<string, unknown> }) { return teacherRequest<TeacherSavedResource>('/teacher/library/resources', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }) }
export function updateTeacherResource(id: number, data: { resource_type: 'lesson-plan' | 'lesson' | 'exercises' | 'exam' | 'assessment' | 'activity' | 'summary'; title: string; cefr_level: string; theme: string; content: Record<string, unknown> }) { return teacherRequest<TeacherSavedResource>(`/teacher/library/resources/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }) }
export const getTeacherResource = (id: number) => teacherRequest<TeacherSavedResource>(`/teacher/library/resources/${id}`)

