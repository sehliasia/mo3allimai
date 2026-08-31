import { AuthApiError, getStoredToken, type AuthUser } from './authService'
const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000/api'
async function adminRequest<T>(path: string, options: RequestInit = {}) {
  const token = getStoredToken()
  if (!token) throw new AuthApiError('Authentication is required', 401)
  const response = await fetch(`${API_URL}${path}`, { ...options, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...options.headers } })
  if (response.status === 401) throw new AuthApiError('Session expired', 401)
  if (!response.ok) throw new Error('تعذر تنفيذ الطلب. يرجى المحاولة مرة أخرى.')
  return response.json() as Promise<T>
}
export type TeachersPage = { items: AuthUser[]; page: number; page_size: number; total: number; total_pages: number }
export const getTeachers = (params = '') => adminRequest<TeachersPage>(`/admin/teachers${params}`)
export const getTeacherById = (id: number) => adminRequest<AuthUser>(`/admin/teachers/${id}`)
export const updateTeacherStatus = (id: number, is_active: boolean) => adminRequest<AuthUser>(`/admin/teachers/${id}/status`, { method: 'PATCH', body: JSON.stringify({ is_active }) })
export const deleteTeacher = (id: number) => adminRequest<AuthUser>(`/admin/teachers/${id}`, { method: 'DELETE' })
export const getAdminStatistics = () => adminRequest<{ total_teachers: number; active_teachers: number; inactive_teachers: number; new_teachers_this_month: number }>('/admin/statistics')
export type KnowledgePreflight = { status: 'complete' | 'partial' | 'failed'; pages_total: number | null; pages_analyzed: number | null; analysis_failed_pages: number | null; native_good_pages: number | null; native_borderline_pages: number | null; native_bad_pages: number | null; ocr_candidate_page_count: number | null; ocr_required_page_ratio: number | null; recommended_strategy: 'native_only' | 'native_with_targeted_ocr' | 'ocr_heavy' | null; estimated_complexity: 'low' | 'medium' | 'high' | null; analyzed_at: string | null }
export type IngestionSummary = { quality_status: 'complete' | 'partial' | 'failed'; pages_total: number; pages_quarantined_count: number; quarantined_page_numbers: number[]; chunks_valid: number; chunks_quarantined_count: number; tables_count?: number; pictures_count?: number; ocr_pages_count?: number; ocr_failures_count: number; late_repairs_attempted?: number; late_repairs_accepted?: number; late_repairs_rejected?: number; warnings_count: number; completed_at?: string; failed_page_ratio?: number; max_failed_page_ratio?: number; failure_reason?: 'failed_page_ratio_exceeded' }
export type KnowledgeDocument = { id: number; title: string; document_type: string | null; language: string | null; cefr_level: string | null; skill: string | null; source: string | null; description: string | null; original_filename: string; mime_type: string; file_size: number; status: 'pending' | 'processing' | 'ready' | 'partial' | 'indexed' | 'failed'; created_at: string; active_jobs: KnowledgeProcessingJob[]; ingestion_summary?: IngestionSummary | null; preflight: KnowledgePreflight | null }
export async function getKnowledgeDocuments(): Promise<{ items: KnowledgeDocument[] }> {
  const response = await adminRequest<{ items?: KnowledgeDocument[] }>('/admin/knowledge-documents')
  const items = Array.isArray(response.items) ? response.items : []
  return { items: items.map(document => ({ ...document, active_jobs: Array.isArray(document.active_jobs) ? document.active_jobs : [], ingestion_summary: document.ingestion_summary && typeof document.ingestion_summary === 'object' ? document.ingestion_summary : null })) }
}
export const runKnowledgePreflight = (id: number) => adminRequest<KnowledgePreflight & { document_id: number }>(`/admin/knowledge-documents/${id}/preflight`, { method: 'POST' })
export type KnowledgeProcessingJob = { id: number; document_id: number; job_type: 'preflight' | 'ingestion'; status: 'pending' | 'processing' | 'completed' | 'failed'; stage: string; attempts: number; created_at: string; started_at: string | null; completed_at: string | null; error_message: string | null }
export type KnowledgeQueueResponse = { queued: number; skipped: number; jobs: KnowledgeProcessingJob[]; skipped_documents: Array<{ document_id: number; reason: string }> }
export const enqueueKnowledgePreflight = (document_ids: number[]) => adminRequest<KnowledgeQueueResponse>('/admin/knowledge-documents/preflight-selected', { method: 'POST', body: JSON.stringify({ document_ids }) })
export const enqueueKnowledgeProcessing = (document_ids: number[]) => adminRequest<KnowledgeQueueResponse>('/admin/knowledge-documents/process-selected', { method: 'POST', body: JSON.stringify({ document_ids }) })
export async function getKnowledgeProcessingJobs(documentIds: number[]): Promise<KnowledgeProcessingJob[]> {
  const response = await adminRequest<{ items?: KnowledgeProcessingJob[] }>(`/admin/knowledge-processing-jobs?active_only=true&${documentIds.map(id => `document_ids=${id}`).join('&')}`)
  return Array.isArray(response.items) ? response.items : []
}
export type KnowledgePreview = {
  document_id: number; pages: number; items: number; chunks_count: number; parsing_duration_ms: number; chunking_duration_ms: number
  quality_status?: 'complete' | 'partial' | 'failed'; warnings?: Array<{ page?: number; reason?: string }>
  statistics: Record<string, number>; extraction?: { native_pages_count?: number; ocr_pages_count?: number; per_page_quality?: Array<{ page: number; extraction_mode: string; quality_score: number; languages: string[] }> }
  chunks: Array<{ chunk_index: number; page_start: number | null; page_end: number | null; section: string | null; headings: string[]; content_type: string; token_count: number; text_preview: string; image_ids?: string[] }>
  images?: Array<{ image_id: string; page: number; bbox?: unknown; caption?: string | null; nearby_text?: string | null; associated_chunk_ids: string[] }>
}
export const getKnowledgePreview = (id: number) => adminRequest<KnowledgePreview>(`/admin/knowledge-documents/${id}/parse-preview`, { method: 'POST' })
export type KnowledgeSegment = { id: string; chunk_index: number; page_start: number | null; page_end: number | null; content_type: string; extraction_mode: string | null; token_count: number; headings: string[]; content: string }
export type KnowledgeSegments = { availability: 'available' | 'legacy_debug' | 'not_persisted' | 'unavailable'; items: KnowledgeSegment[]; total: number; page: number; page_size: number }
export const getKnowledgeSegments = (id: number, page = 1, sourcePage?: number) => adminRequest<KnowledgeSegments>(`/admin/knowledge-documents/${id}/segments?page=${page}&page_size=20${sourcePage ? `&source_page=${sourcePage}` : ''}`)
export async function uploadKnowledgeDocument(file: File) {
  const token = getStoredToken()
  const body = new FormData()
  body.append('file', file)
  const response = await fetch(`${API_URL}/admin/knowledge-documents`, { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body })
  if (!response.ok) throw new Error('Knowledge document upload failed')
  return response.json() as Promise<KnowledgeDocument>
}
