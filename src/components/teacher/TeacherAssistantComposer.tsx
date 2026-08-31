import { Send, Square } from 'lucide-react'
import { useCallback, useEffect, useRef, type Dispatch, type SetStateAction } from 'react'
import { useTranslation } from 'react-i18next'
import { SelectedAssistantDocuments, TeacherAssistantDocuments, type ChatDocumentAttachment } from './TeacherAssistantDocuments'
import type { TeacherLibraryDocument } from '../../services/teacherLibraryService'

type Props = { value: string; error: string | null; isSending: boolean; onChange: (value: string) => void; onSend: () => void; onStop: () => void; selectedDocuments: ChatDocumentAttachment[]; onDocumentsChange: Dispatch<SetStateAction<ChatDocumentAttachment[]>> }

export function TeacherAssistantComposer({ value, error, isSending, onChange, onSend, onStop, selectedDocuments, onDocumentsChange }: Props) {
  const { t } = useTranslation('teacher')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  useEffect(() => { const textarea = textareaRef.current; if (!textarea) return; textarea.style.height = '0px'; textarea.style.height = `${Math.min(textarea.scrollHeight, 210)}px` }, [value])
  const submit = () => { if (isSending) onStop(); else if (value.trim()) onSend() }
  const refreshDocuments = useCallback((updated: TeacherLibraryDocument[]) => {
    if (!updated.length) return
    const byId = new Map(updated.map(item => [item.id, item]))
    onDocumentsChange(current => current.map(item => typeof item.id === 'number' ? byId.get(item.id) ?? item : item))
  }, [onDocumentsChange])
  return <div className="rounded-2xl border border-slate-200 bg-white p-3 transition focus-within:border-emerald-600 focus-within:ring-2 focus-within:ring-emerald-100 sm:p-4">
    <SelectedAssistantDocuments documents={selectedDocuments} onRemove={id => onDocumentsChange(current => current.filter(item => item.id !== id))} onRefresh={refreshDocuments} />
    <label className="sr-only" htmlFor="teacher-assistant-message">{t('assistant.composer')}</label><textarea ref={textareaRef} id="teacher-assistant-message" rows={1} value={value} onChange={event => onChange(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !isSending) { event.preventDefault(); onSend() } }} placeholder={t('assistant.composer')} aria-describedby={error ? 'teacher-assistant-error' : undefined} className="block min-h-[58px] max-h-[210px] w-full resize-none bg-transparent px-2 py-2 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400" />
    <div className="mt-2 flex items-center justify-between gap-2"><TeacherAssistantDocuments selected={selectedDocuments} onChange={onDocumentsChange} disabled={isSending} /><button type="button" onClick={submit} disabled={!isSending && !value.trim()} className="inline-flex h-10 items-center gap-2 rounded-xl bg-[#065F46] px-4 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-700 focus-visible:ring-offset-2">{isSending ? <Square className="size-3.5 fill-current" /> : <Send className="size-4" />}<span>{isSending ? t('assistant.stop') : t('assistant.send')}</span></button></div>
    {error && <p id="teacher-assistant-error" role="alert" className="mt-2 text-sm text-rose-700">{error}</p>}
  </div>
}
