import { API_URL, AuthApiError, getStoredToken } from './authService'

export type AssistantCEFRLevel = 'PRE-A1' | 'A1' | 'A2' | 'A2+' | 'B1' | 'B1+' | 'B2' | 'B2+' | 'C1' | 'C2'
export type AssistantSkill = 'listening' | 'reading' | 'speaking' | 'writing'
export type AssistantAnswerLanguage = 'ar' | 'fr' | 'en' | 'es'

export type AssistantChatRequest = {
  message: string
  conversation_id?: number | null
  cefr_level?: AssistantCEFRLevel | null
  skills?: AssistantSkill[]
  language?: AssistantAnswerLanguage
  topic?: string | null
  objective?: string | null
  top_k?: number
  mode?: 'knowledge_base' | 'user_documents'
  document_ids?: number[]
}

export type AssistantChatSource = {
  source_type: 'cefr_structured' | 'pedagogical_resource' | 'personal_document'
  document_id: number
  document_title: string | null
  page_start: number | null
  page_end: number | null
  cefr_scale: string | null
}

export type AssistantChatDiagnostics = {
  requested_cefr_level: string | null
  output_language: string
  retrieved_count: number
  selected_count: number
  source_count: number
  requires_vision_count: number
  warnings: string[]
  provider_model: string | null
  finish_reason: string | null
  history_messages_used: number
  history_chars_used: number
}

export type AssistantChatResponse = {
  conversation_id: number
  user_message_id: number
  assistant_message_id: number
  answer: string
  sources: AssistantChatSource[]
  diagnostics: AssistantChatDiagnostics
}

export type AssistantStreamEvent =
  | { type: 'start'; conversation_id: number; message_id: number }
  | { type: 'delta'; content: string }
  | { type: 'done' | 'stopped'; assistant_message_id?: number; sources?: AssistantChatSource[]; diagnostics?: AssistantChatDiagnostics }
  | { type: 'error'; message: string }

export type AssistantConversationSummary = {
  id: number
  title: string
  created_at: string
  updated_at: string
  archived_at: string | null
  message_count: number
}

export type AssistantConversationListResponse = {
  items: AssistantConversationSummary[]
  total: number
  limit: number
  offset: number
}

export type AssistantHistorySource = {
  source_type: 'cefr_structured' | 'pedagogical_resource'
  document_id: number | null
  document_title: string | null
  page_start: number | null
  page_end: number | null
  descriptor_scale: string | null
}

export type AssistantHistoryMessage = {
  id: number
  role: 'USER' | 'ASSISTANT'
  content: string
  created_at: string
  sources: AssistantHistorySource[]
}

export type AssistantConversationDetail = {
  id: number
  title: string
  created_at: string
  updated_at: string
  archived_at: string | null
  messages: AssistantHistoryMessage[]
}

export type AssistantConversationListParams = {
  limit?: number
  offset?: number
  archived?: boolean
}

export class AssistantApiError extends AuthApiError {
  constructor(message: string, status?: number, public readonly code?: string) {
    super(message, status)
    this.name = 'AssistantApiError'
  }
}

type AssistantErrorPayload = {
  detail?: string | { code?: string; message?: string }
}

function normalizeAssistantError(status: number, payload: AssistantErrorPayload): AssistantApiError {
  const detail = payload.detail
  const code = typeof detail === 'object' && detail !== null && typeof detail.code === 'string' ? detail.code : undefined
  const message = typeof detail === 'object' && detail !== null && typeof detail.message === 'string'
    ? detail.message
    : typeof detail === 'string'
      ? detail
      : status === 401
        ? 'Authentication is required.'
        : status === 403
          ? 'Access is forbidden.'
          : status === 404
            ? 'Conversation is unavailable.'
            : status === 422
              ? 'The assistant request is invalid.'
              : status === 503
                ? 'The assistant service is temporarily unavailable.'
                : 'The assistant request failed.'
  return new AssistantApiError(message, status, code)
}

async function assistantRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getStoredToken()
  if (!token) throw new AssistantApiError('Authentication is required.', 401)

  let response: Response
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...options,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    })
  } catch {
    throw new AssistantApiError('The assistant service is unreachable.')
  }

  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as AssistantErrorPayload
    throw normalizeAssistantError(response.status, payload)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function chatPayload(request: AssistantChatRequest): AssistantChatRequest {
  const { message, conversation_id, cefr_level, skills, language, topic, objective, top_k, mode, document_ids } = request
  return {
    message,
    ...(conversation_id == null ? {} : { conversation_id }),
    ...(cefr_level == null ? {} : { cefr_level }),
    ...(skills === undefined ? {} : { skills }),
    ...(language === undefined ? {} : { language }),
    ...(topic == null ? {} : { topic }),
    ...(objective == null ? {} : { objective }),
    ...(top_k === undefined ? {} : { top_k }),
    ...(mode === undefined ? {} : { mode }),
    ...(document_ids === undefined || document_ids.length === 0 ? {} : { document_ids }),
  }
}

export function sendAssistantMessage(request: AssistantChatRequest): Promise<AssistantChatResponse> {
  return assistantRequest<AssistantChatResponse>('/assistant/chat', {
    method: 'POST',
    body: JSON.stringify(chatPayload(request)),
  })
}

export async function streamAssistantMessage(
  request: AssistantChatRequest, signal: AbortSignal, onEvent: (event: AssistantStreamEvent) => void,
): Promise<void> {
  const token = getStoredToken()
  if (!token) throw new AssistantApiError('Authentication is required.', 401)
  const response = await fetch(`${API_URL}/assistant/chat/stream`, {
    method: 'POST', signal,
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
    body: JSON.stringify(chatPayload(request)),
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as AssistantErrorPayload
    throw normalizeAssistantError(response.status, payload)
  }
  if (!response.body) throw new AssistantApiError('The assistant stream is unavailable.')
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffered = ''
  while (true) {
    const { done, value } = await reader.read()
    buffered += decoder.decode(value, { stream: !done })
    const lines = buffered.split('\n'); buffered = lines.pop() ?? ''
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line) as AssistantStreamEvent)
    if (done) break
  }
  if (buffered.trim()) onEvent(JSON.parse(buffered) as AssistantStreamEvent)
}

export function listAssistantConversations(
  params: AssistantConversationListParams = {},
): Promise<AssistantConversationListResponse> {
  const search = new URLSearchParams()
  if (params.limit !== undefined) search.set('limit', String(params.limit))
  if (params.offset !== undefined) search.set('offset', String(params.offset))
  if (params.archived !== undefined) search.set('archived', String(params.archived))
  const suffix = search.size ? `?${search.toString()}` : ''
  return assistantRequest<AssistantConversationListResponse>(`/assistant/conversations${suffix}`)
}

export function getAssistantConversation(conversationId: number): Promise<AssistantConversationDetail> {
  return assistantRequest<AssistantConversationDetail>(`/assistant/conversations/${conversationId}`)
}

export function deleteAssistantConversation(conversationId: number): Promise<void> {
  return assistantRequest<void>(`/assistant/conversations/${conversationId}`, { method: 'DELETE' })
}

export function regenerateAssistantMessage(conversationId: number, messageId: number, message: string): Promise<AssistantChatResponse> {
  return assistantRequest<AssistantChatResponse>(`/assistant/conversations/${conversationId}/messages/${messageId}/regenerate`, {
    method: 'PATCH', body: JSON.stringify({ message }),
  })
}

export function archiveAssistantConversation(conversationId: number): Promise<AssistantConversationDetail> {
  return assistantRequest<AssistantConversationDetail>(`/assistant/conversations/${conversationId}/archive`, { method: 'PATCH' })
}

export function restoreAssistantConversation(conversationId: number): Promise<AssistantConversationDetail> {
  return assistantRequest<AssistantConversationDetail>(`/assistant/conversations/${conversationId}/restore`, { method: 'PATCH' })
}
