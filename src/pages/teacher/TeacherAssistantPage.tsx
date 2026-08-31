import {
  Archive,
  BookOpenText,
  Bot,
  Check,
  ChevronDown,
  Copy,
  MoreHorizontal,
  Pencil,
  Plus,
  RotateCcw,
  Sparkles,
  Trash2,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { AssistantMarkdown } from '../../components/teacher/AssistantMarkdown'
import { TeacherAssistantComposer } from '../../components/teacher/TeacherAssistantComposer'
import { TeacherPageHeader } from '../../components/teacher/TeacherPageHeader'
import { useAssistantChat, type AssistantUiMessage } from '../../hooks/useAssistantChat'
import type { ChatDocumentAttachment } from '../../components/teacher/TeacherAssistantDocuments'

function formatConversationDate(value: string, locale: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'short', year: 'numeric' }).format(date)
}

export function TeacherAssistantPage() {
  const { t, i18n } = useTranslation('teacher')
  const chat = useAssistantChat()
  const [value, setValue] = useState('')
  const [editing, setEditing] = useState<AssistantUiMessage | null>(null)
  const [copied, setCopied] = useState<string | number | null>(null)
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [openMenuId, setOpenMenuId] = useState<number | null>(null)
  const [deleteTargetId, setDeleteTargetId] = useState<number | null>(null)
  const [selectedDocuments, setSelectedDocuments] = useState<ChatDocumentAttachment[]>([])
  const [attachmentError, setAttachmentError] = useState<string | null>(null)
  const bottom = useRef<HTMLDivElement>(null)
  const lastUser = [...chat.messages].reverse().find(message => message.role === 'USER')

  useEffect(() => { bottom.current?.scrollIntoView({ block: 'end' }) }, [chat.messages, chat.isSending])
  useEffect(() => {
    const closeMenu = () => setOpenMenuId(null)
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') closeMenu() }
    window.addEventListener('click', closeMenu)
    window.addEventListener('keydown', closeOnEscape)
    return () => { window.removeEventListener('click', closeMenu); window.removeEventListener('keydown', closeOnEscape) }
  }, [])

  const copy = async (message: AssistantUiMessage) => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(message.id)
      window.setTimeout(() => setCopied(null), 1400)
    } catch { /* Clipboard permission is non-fatal. */ }
  }

  const send = async () => {
    const content = value.trim()
    if (!content || chat.isSending) return
    if (editing && typeof editing.id === 'number') {
      if (await chat.regenerateLatestUserMessage(editing.id, content)) { setEditing(null); setValue('') }
      return
    }
    const pending = selectedDocuments.filter(item => item.status !== 'ready')
    if (pending.length) {
      setAttachmentError(t('assistant.documentNotReady'))
      return
    }
    setAttachmentError(null)
    setValue('')
    const readyDocumentIds = selectedDocuments.filter((item): item is Extract<ChatDocumentAttachment, { id: number }> => typeof item.id === 'number').map(item => item.id)
    if (!await chat.sendMessage({ message: content, mode: readyDocumentIds.length ? 'user_documents' : 'knowledge_base', ...(readyDocumentIds.length ? { document_ids: readyDocumentIds } : {}) })) setValue(content)
  }

  const startNewConversation = () => {
    if (chat.showArchived) chat.setArchivedView(false)
    chat.startNewConversation()
    setEditing(null)
    setValue('')
  }

  const confirmDelete = async () => {
    if (deleteTargetId === null) return
    if (await chat.deleteConversation(deleteTargetId)) setDeleteTargetId(null)
  }

  const conversationTitle = chat.showArchived ? t('assistant.archives') : t('assistant.conversations')

  return <div className="space-y-4 sm:space-y-5">
    <TeacherPageHeader emphasis title={t('assistant.title')} description={t('assistant.description')} />

    <section className="grid min-h-[calc(100vh-13rem)] gap-4 lg:grid-cols-[17rem_minmax(0,1fr)]">
      <aside className="flex min-h-64 flex-col rounded-2xl border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900">
        <div className="flex items-center justify-between gap-2 px-2 pb-3">
          <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">{conversationTitle}</h2>
          <button type="button" onClick={startNewConversation} className="inline-flex size-8 items-center justify-center rounded-lg text-emerald-700 transition hover:bg-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 dark:text-emerald-300 dark:hover:bg-emerald-950/50" aria-label={t('assistant.newConversation')} title={t('assistant.newConversation')}>
            <Plus className="size-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
          {chat.isLoadingConversations && <p className="px-2 py-3 text-xs text-slate-500">{t('assistant.loadingConversations')}</p>}
          {!chat.isLoadingConversations && chat.conversations.length === 0 && <p className="px-2 py-3 text-xs leading-5 text-slate-500">{t(chat.showArchived ? 'assistant.noArchivedConversations' : 'assistant.noConversations')}</p>}
          {chat.conversations.map(item => <div key={item.id} className="group relative">
            <button type="button" onClick={() => void chat.selectConversation(item.id)} className={`block w-full rounded-xl px-3 py-2 pe-10 text-start transition ${chat.activeConversationId === item.id ? 'bg-emerald-50 text-emerald-950 dark:bg-emerald-950/45 dark:text-emerald-50' : 'text-slate-700 hover:bg-slate-50 dark:text-slate-200 dark:hover:bg-slate-800'}`}>
              <span dir="auto" className="block truncate text-sm font-medium">{item.title}</span>
              <span className="mt-0.5 block text-xs text-slate-500 dark:text-slate-400">{formatConversationDate(item.updated_at, i18n.language)}</span>
            </button>
            <button type="button" onClick={event => { event.stopPropagation(); setOpenMenuId(current => current === item.id ? null : item.id) }} className="absolute end-2 top-2 inline-flex size-7 items-center justify-center rounded-lg text-slate-500 transition hover:bg-white hover:text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 dark:hover:bg-slate-700 dark:hover:text-slate-100" aria-label={t('assistant.conversationActions')} aria-haspopup="menu" aria-expanded={openMenuId === item.id}>
              <MoreHorizontal className="size-4" />
            </button>
            {openMenuId === item.id && <div role="menu" onClick={event => event.stopPropagation()} className="absolute end-2 top-10 z-20 min-w-36 rounded-xl border border-slate-200 bg-white p-1 shadow-lg shadow-slate-950/10 dark:border-slate-700 dark:bg-slate-800">
              {chat.showArchived
                ? <button type="button" role="menuitem" onClick={() => { setOpenMenuId(null); void chat.restoreConversation(item.id) }} className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-start text-xs font-medium text-slate-700 hover:bg-slate-50 dark:text-slate-100 dark:hover:bg-slate-700"><RotateCcw className="size-3.5" />{t('assistant.restore')}</button>
                : <button type="button" role="menuitem" onClick={() => { setOpenMenuId(null); void chat.archiveConversation(item.id) }} className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-start text-xs font-medium text-slate-700 hover:bg-slate-50 dark:text-slate-100 dark:hover:bg-slate-700"><Archive className="size-3.5" />{t('assistant.archive')}</button>}
              <button type="button" role="menuitem" onClick={() => { setOpenMenuId(null); setDeleteTargetId(item.id) }} className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-start text-xs font-medium text-rose-700 hover:bg-rose-50 dark:text-rose-300 dark:hover:bg-rose-950/40"><Trash2 className="size-3.5" />{t('assistant.delete')}</button>
            </div>}
          </div>)}
        </div>

        <div className="mt-3 border-t border-slate-100 pt-3 dark:border-slate-800">
          <button type="button" onClick={() => chat.setArchivedView(!chat.showArchived)} className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-start text-sm font-medium text-slate-600 transition hover:bg-slate-50 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white">
            <Archive className="size-4" />
            {chat.showArchived ? t('assistant.activeConversations') : t('assistant.archives')}
          </button>
        </div>
      </aside>

      <div className="flex min-h-[calc(100vh-13rem)] flex-col overflow-hidden rounded-2xl bg-white shadow-md shadow-slate-950/5 dark:bg-slate-900">
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-7">
          {chat.messages.length === 0
            ? <div className="flex min-h-[280px] flex-col items-center justify-center text-center"><Sparkles className="size-6 text-emerald-700 dark:text-emerald-300" /><h2 className="mt-4 font-bold text-slate-900 dark:text-slate-100">{t('assistant.emptyTitle')}</h2></div>
            : <div className="mx-auto max-w-4xl space-y-5">
              {chat.messages.map(message => {
                const isUser = message.role === 'USER'
                const canEdit = isUser && lastUser?.id === message.id && !chat.isSending
                const isExpanded = Boolean(expanded[String(message.id)])
                return <div key={message.id} className={isUser ? 'ms-auto max-w-[80%]' : 'max-w-[88%]'}>
                  <div className={isUser ? 'rounded-2xl rounded-se-none border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm leading-6 text-emerald-950 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-50' : 'rounded-2xl rounded-ss-none border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-700 dark:border-slate-800 dark:bg-slate-800/60 dark:text-slate-200'}>
                    {isUser
                      ? <p dir="auto" className="whitespace-pre-wrap">{message.content}</p>
                      : message.status === 'thinking' && !message.content
                        ? <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400"><Bot className="size-4 animate-pulse" />{t('assistant.thinking')}</div>
                        : <AssistantMarkdown content={message.content} />}
                  </div>
                  <div className={`mt-1.5 flex gap-1 ${isUser ? 'justify-end' : 'justify-start'}`}>
                    {(isUser || message.content.trim()) && <button type="button" onClick={() => void copy(message)} className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800" aria-label={t('assistant.copyMessage')}>
                      {copied === message.id ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}{copied === message.id ? t('assistant.copied') : t('assistant.copy')}
                    </button>}
                    {canEdit && <button type="button" onClick={() => { setEditing(message); setValue(message.content) }} className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800" aria-label={t('assistant.editMessage')}><Pencil className="size-3.5" />{t('assistant.edit')}</button>}
                  </div>
                  {!isUser && message.sources.length > 0 && <div className="mt-2">
                    <button type="button" onClick={() => setExpanded(current => ({ ...current, [String(message.id)]: !isExpanded }))} aria-expanded={isExpanded} className="inline-flex items-center gap-1 text-xs font-semibold text-slate-600 dark:text-slate-300"><BookOpenText className="size-3.5" />{t('assistant.sourcesCount', { count: message.sources.length })}<ChevronDown className={isExpanded ? 'size-3.5 rotate-180' : 'size-3.5'} /></button>
                    {isExpanded && <div className="mt-2 space-y-1">{message.sources.map((source, index) => <div key={index} className="rounded-lg bg-white px-3 py-2 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300"><span dir="auto" className="font-semibold">{source.document_title || t('assistant.untitledSource')}</span>{source.page_start !== null && <span className="ms-1 text-slate-500 dark:text-slate-400">· p. {source.page_start}{source.page_end && source.page_end !== source.page_start ? `–${source.page_end}` : ''}</span>}</div>)}</div>}
                  </div>}
                </div>
              })}
              <div ref={bottom} />
            </div>}
        </div>
        <div className="border-t border-slate-100 bg-slate-50/60 p-3 dark:border-slate-800 dark:bg-slate-900">
          {editing && <div className="mb-2 flex items-center justify-between text-xs font-semibold text-emerald-800 dark:text-emerald-300"><span>{t('assistant.editingMessage')}</span><button type="button" onClick={() => { setEditing(null); setValue('') }}>{t('assistant.cancel')}</button></div>}
          <TeacherAssistantComposer value={value} error={attachmentError || (chat.error ? t(`assistant.errors.${chat.error.kind}`) : null)} isSending={chat.isSending} onChange={setValue} onSend={() => void send()} onStop={chat.stopGenerating} selectedDocuments={selectedDocuments} onDocumentsChange={setSelectedDocuments} />
          {editing && <button type="button" onClick={() => void send()} disabled={chat.isSending || !value.trim()} className="mt-2 rounded-lg px-2 py-1 text-xs font-semibold text-emerald-800 hover:bg-emerald-100 dark:text-emerald-300 dark:hover:bg-emerald-950/50">{t('assistant.saveAndResend')}</button>}
        </div>
      </div>
    </section>

    {deleteTargetId !== null && <div role="dialog" aria-modal="true" aria-labelledby="assistant-delete-title" className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4">
      <div className="w-full max-w-sm rounded-2xl bg-white p-5 shadow-xl dark:bg-slate-900">
        <h2 id="assistant-delete-title" className="text-base font-bold text-slate-900 dark:text-slate-100">{t('assistant.deleteTitle')}</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">{t('assistant.deleteDescription')}</p>
        <div className="mt-5 flex justify-end gap-2">
          <button type="button" onClick={() => setDeleteTargetId(null)} disabled={chat.deletingConversationId !== null} className="rounded-lg px-3 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800">{t('assistant.cancel')}</button>
          <button type="button" onClick={() => void confirmDelete()} disabled={chat.deletingConversationId !== null} className="rounded-lg bg-rose-600 px-3 py-2 text-sm font-semibold text-white hover:bg-rose-700 disabled:opacity-60">{chat.deletingConversationId !== null ? t('assistant.deleting') : t('assistant.delete')}</button>
        </div>
      </div>
    </div>}
  </div>
}
