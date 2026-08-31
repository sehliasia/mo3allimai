import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AssistantApiError,
  archiveAssistantConversation,
  deleteAssistantConversation,
  getAssistantConversation,
  listAssistantConversations,
  restoreAssistantConversation,
  regenerateAssistantMessage,
  streamAssistantMessage,
  type AssistantChatDiagnostics,
  type AssistantChatRequest,
  type AssistantChatSource,
  type AssistantConversationDetail,
  type AssistantConversationListParams,
  type AssistantConversationSummary,
  type AssistantHistorySource,
} from '../services/assistantService'

export type AssistantUiSource = {
  source_type: AssistantChatSource['source_type']
  document_id: number | null
  document_title: string | null
  page_start: number | null
  page_end: number | null
  descriptor_scale: string | null
}

export type AssistantUiMessage = {
  id: number | string
  role: 'USER' | 'ASSISTANT'
  content: string
  created_at?: string
  sources: AssistantUiSource[]
  status: 'pending' | 'thinking' | 'streaming' | 'stopped' | 'sent' | 'error'
}

export type AssistantChatError = {
  status?: number
  code?: string
  kind: 'authentication' | 'forbidden' | 'not_found' | 'validation' | 'unavailable' | 'persistence' | 'unknown'
  message: string
}

export type SendAssistantMessageInput = Omit<AssistantChatRequest, 'conversation_id'>

function normalizeSource(source: AssistantChatSource | AssistantHistorySource): AssistantUiSource {
  return {
    source_type: source.source_type,
    document_id: source.document_id,
    document_title: source.document_title,
    page_start: source.page_start,
    page_end: source.page_end,
    descriptor_scale: 'cefr_scale' in source ? source.cefr_scale : source.descriptor_scale,
  }
}

function historyMessages(detail: AssistantConversationDetail): AssistantUiMessage[] {
  return detail.messages.map(message => ({
    id: message.id,
    role: message.role,
    content: message.content,
    created_at: message.created_at,
    sources: message.sources.map(normalizeSource),
    status: 'sent',
  }))
}

function normalizeError(error: unknown): AssistantChatError {
  if (error instanceof AssistantApiError) {
    const kind = error.status === 401 ? 'authentication'
      : error.status === 403 ? 'forbidden'
        : error.status === 404 ? 'not_found'
          : error.status === 422 ? 'validation'
            : error.status === 503 ? 'unavailable'
              : error.code === 'ASSISTANT_PERSISTENCE_ERROR' ? 'persistence'
                : 'unknown'
    return { status: error.status, code: error.code, kind, message: error.message }
  }
  return { kind: 'unknown', message: 'The assistant request failed.' }
}

export function useAssistantChat(initialPagination: AssistantConversationListParams = {}) {
  const initialPaginationRef = useRef(initialPagination)
  const [conversations, setConversations] = useState<AssistantConversationSummary[]>([])
  const [conversationTotal, setConversationTotal] = useState(0)
  const [conversationLimit, setConversationLimit] = useState(initialPagination.limit ?? 20)
  const [conversationOffset, setConversationOffset] = useState(initialPagination.offset ?? 0)
  const [showArchived, setShowArchived] = useState(false)
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null)
  const [activeConversation, setActiveConversation] = useState<AssistantConversationDetail | null>(null)
  const [messages, setMessages] = useState<AssistantUiMessage[]>([])
  const [isLoadingConversations, setIsLoadingConversations] = useState(false)
  const [isLoadingConversation, setIsLoadingConversation] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [deletingConversationId, setDeletingConversationId] = useState<number | null>(null)
  const [error, setError] = useState<AssistantChatError | null>(null)
  const [lastDiagnostics, setLastDiagnostics] = useState<AssistantChatDiagnostics | null>(null)
  const listRequestVersion = useRef(0)
  const conversationRequestVersion = useRef(0)
  const sendInFlight = useRef(false)
  const streamController = useRef<AbortController | null>(null)
  const temporaryMessageSequence = useRef(0)
  const paginationRef = useRef({ limit: initialPagination.limit ?? 20, offset: initialPagination.offset ?? 0 })

  const clearError = useCallback(() => setError(null), [])

  const loadConversations = useCallback(async (params: AssistantConversationListParams = {}) => {
    const requestVersion = ++listRequestVersion.current
    const limit = params.limit ?? paginationRef.current.limit
    const offset = params.offset ?? paginationRef.current.offset
    setIsLoadingConversations(true)
    try {
      const response = await listAssistantConversations({ limit, offset, archived: params.archived ?? showArchived })
      if (requestVersion !== listRequestVersion.current) return
      setConversations(response.items)
      setConversationTotal(response.total)
      setConversationLimit(response.limit)
      setConversationOffset(response.offset)
      paginationRef.current = { limit: response.limit, offset: response.offset }
    } catch (requestError) {
      if (requestVersion === listRequestVersion.current) setError(normalizeError(requestError))
    } finally {
      if (requestVersion === listRequestVersion.current) setIsLoadingConversations(false)
    }
  }, [showArchived])

  useEffect(() => { void loadConversations(initialPaginationRef.current) }, [loadConversations])

  const selectConversation = useCallback(async (conversationId: number) => {
    const requestVersion = ++conversationRequestVersion.current
    setActiveConversationId(conversationId)
    setIsLoadingConversation(true)
    setError(null)
    try {
      const detail = await getAssistantConversation(conversationId)
      if (requestVersion !== conversationRequestVersion.current) return
      setActiveConversation(detail)
      setMessages(historyMessages(detail))
      setLastDiagnostics(null)
    } catch (requestError) {
      if (requestVersion !== conversationRequestVersion.current) return
      setActiveConversationId(null)
      setActiveConversation(null)
      setMessages([])
      setError(normalizeError(requestError))
    } finally {
      if (requestVersion === conversationRequestVersion.current) setIsLoadingConversation(false)
    }
  }, [])

  const startNewConversation = useCallback(() => {
    conversationRequestVersion.current += 1
    setActiveConversationId(null)
    setActiveConversation(null)
    setMessages([])
    setLastDiagnostics(null)
    setError(null)
  }, [])

  const setArchivedView = useCallback((archived: boolean) => {
    startNewConversation()
    setShowArchived(archived)
  }, [startNewConversation])

  const sendMessage = useCallback(async (input: SendAssistantMessageInput): Promise<boolean> => {
    const content = input.message.trim()
    if (!content || sendInFlight.current || isLoadingConversation) return false

    sendInFlight.current = true
    setIsSending(true)
    setError(null)
    const temporaryId = `local-user-${++temporaryMessageSequence.current}`
    const temporaryAssistantId = `local-assistant-${temporaryMessageSequence.current}`
    const conversationIdAtSend = activeConversationId
    const conversationViewVersion = conversationRequestVersion.current
    setMessages(current => [...current, {
      id: temporaryId,
      role: 'USER',
      content,
      sources: [],
      status: 'pending',
    }, { id: temporaryAssistantId, role: 'ASSISTANT', content: '', sources: [], status: 'thinking' }])
    const controller = new AbortController(); streamController.current = controller

    try {
      await streamAssistantMessage({
        ...input,
        message: content,
        ...(conversationIdAtSend === null ? {} : { conversation_id: conversationIdAtSend }),
      }, controller.signal, event => {
        if (conversationViewVersion !== conversationRequestVersion.current) return
        if (event.type === 'start') {
          setActiveConversationId(event.conversation_id)
          setMessages(current => current.map(message => message.id === temporaryId ? { ...message, id: event.message_id, status: 'sent' } : message))
        } else if (event.type === 'delta') {
          setMessages(current => current.map(message => message.id === temporaryAssistantId ? { ...message, content: message.content + event.content, status: 'streaming' } : message))
        } else if (event.type === 'done' || event.type === 'stopped') {
          setMessages(current => current.flatMap(message => {
            if (message.id !== temporaryAssistantId) return [message]
            // The server intentionally persists no empty assistant response.
            if (!message.content && !event.assistant_message_id) return []
            return [{
              ...message, id: event.assistant_message_id ?? message.id, sources: event.sources?.map(normalizeSource) ?? [],
              status: event.type === 'stopped' ? 'stopped' : 'sent',
            }]
          }))
          if (event.diagnostics) setLastDiagnostics(event.diagnostics)
        } else if (event.type === 'error') {
          setError({ kind: 'unavailable', message: event.message })
          setMessages(current => current.map(message => message.id === temporaryAssistantId ? { ...message, status: message.content ? 'stopped' : 'error' } : message))
        }
      })
      void loadConversations()
      return true
    } catch (requestError) {
      if (conversationViewVersion === conversationRequestVersion.current) {
        const aborted = requestError instanceof DOMException && requestError.name === 'AbortError'
        setMessages(current => current.flatMap(message => {
          if (message.id !== temporaryAssistantId) return [message]
          if (message.content) return [{ ...message, status: 'stopped' }]
          return aborted ? [] : [{ ...message, status: 'error' }]
        }))
        if (!aborted) setError(normalizeError(requestError))
      }
      return false
    } finally {
      if (streamController.current === controller) streamController.current = null
      sendInFlight.current = false
      setIsSending(false)
    }
  }, [activeConversationId, isLoadingConversation, loadConversations])

  const stopGenerating = useCallback(() => streamController.current?.abort(), [])
  useEffect(() => () => streamController.current?.abort(), [])

  const deleteConversation = useCallback(async (conversationId: number): Promise<boolean> => {
    if (deletingConversationId !== null) return false
    setDeletingConversationId(conversationId)
    setError(null)
    try {
      await deleteAssistantConversation(conversationId)
      setConversations(current => current.filter(conversation => conversation.id !== conversationId))
      setConversationTotal(current => Math.max(0, current - 1))
      if (activeConversationId === conversationId) startNewConversation()
      return true
    } catch (requestError) {
      setError(normalizeError(requestError))
      return false
    } finally {
      setDeletingConversationId(null)
    }
  }, [activeConversationId, deletingConversationId, startNewConversation])

  const regenerateLatestUserMessage = useCallback(async (messageId: number, content: string): Promise<boolean> => {
    if (activeConversationId === null || sendInFlight.current) return false
    sendInFlight.current = true; setIsSending(true); setError(null)
    try {
      await regenerateAssistantMessage(activeConversationId, messageId, content)
      await selectConversation(activeConversationId)
      void loadConversations()
      return true
    } catch (requestError) { setError(normalizeError(requestError)); return false }
    finally { sendInFlight.current = false; setIsSending(false) }
  }, [activeConversationId, loadConversations, selectConversation])

  const archiveConversation = useCallback(async (conversationId: number): Promise<boolean> => {
    try {
      await archiveAssistantConversation(conversationId)
      setConversations(current => current.filter(conversation => conversation.id !== conversationId))
      setConversationTotal(current => Math.max(0, current - 1))
      if (activeConversationId === conversationId) startNewConversation()
      return true
    } catch (requestError) {
      setError(normalizeError(requestError))
      return false
    }
  }, [activeConversationId, startNewConversation])

  const restoreConversation = useCallback(async (conversationId: number): Promise<boolean> => {
    try {
      await restoreAssistantConversation(conversationId)
      setConversations(current => current.filter(conversation => conversation.id !== conversationId))
      setConversationTotal(current => Math.max(0, current - 1))
      if (activeConversationId === conversationId) startNewConversation()
      return true
    } catch (requestError) {
      setError(normalizeError(requestError))
      return false
    }
  }, [activeConversationId, startNewConversation])

  return {
    conversations,
    conversationTotal,
    conversationLimit,
    conversationOffset,
    showArchived,
    activeConversationId,
    activeConversation,
    messages,
    isLoadingConversations,
    isLoadingConversation,
    isSending,
    deletingConversationId,
    error,
    lastDiagnostics,
    loadConversations,
    selectConversation,
    startNewConversation,
    setArchivedView,
    sendMessage,
    stopGenerating,
    deleteConversation,
    archiveConversation,
    restoreConversation,
    regenerateLatestUserMessage,
    clearError,
  }
}
