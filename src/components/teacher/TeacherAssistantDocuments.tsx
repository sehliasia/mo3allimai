import { CheckCircle2, CircleAlert, FileText, LoaderCircle, Paperclip, X } from 'lucide-react'
import { useEffect, useRef, type Dispatch, type SetStateAction } from 'react'
import { useTranslation } from 'react-i18next'
import { getTeacherLibrary, uploadTeacherDocument, type TeacherLibraryDocument } from '../../services/teacherLibraryService'

export type ChatDocumentAttachment = TeacherLibraryDocument | { id: `upload-${string}`; kind: 'document'; original_filename: string; file_size: number; status: 'uploading'; processing_stage: 'uploaded'; processing_error: null }
type Props = { selected: ChatDocumentAttachment[]; onChange: Dispatch<SetStateAction<ChatDocumentAttachment[]>>; disabled: boolean }
const accepted = '.pdf,.docx,.txt'

export function TeacherAssistantDocuments({ selected, onChange, disabled }: Props) {
  const { t } = useTranslation('teacherLibrary')
  const input = useRef<HTMLInputElement>(null)
  const importFiles = async (files: FileList | null) => {
    if (!files?.length) return
    const sourceFiles = Array.from(files)
    const temporary = sourceFiles.map(file => ({ id: `upload-${crypto.randomUUID()}` as const, kind: 'document' as const, original_filename: file.name, file_size: file.size, status: 'uploading' as const, processing_stage: 'uploaded' as const, processing_error: null }))
    onChange(current => [...current, ...temporary])
    await Promise.all(temporary.map(async (attachment, index) => {
      try {
        const document = await uploadTeacherDocument(sourceFiles[index])
        onChange(current => current.map(item => item.id === attachment.id ? document : item))
      } catch (error) {
        const message = error instanceof Error ? error.message : t('uploadFailed')
        onChange(current => current.map(item => item.id === attachment.id ? { ...item, status: 'failed' as const, processing_stage: 'failed' as const, processing_error: message } : item))
      }
    }))
  }
  return <><input ref={input} type="file" multiple accept={accepted} className="hidden" onChange={event => { void importFiles(event.target.files); event.target.value = '' }} /><button type="button" disabled={disabled} onClick={() => input.current?.click()} className="inline-flex h-10 items-center gap-2 rounded-xl px-3 text-sm font-semibold text-slate-600 transition hover:bg-slate-100 disabled:opacity-50 dark:text-slate-300 dark:hover:bg-slate-800"><Paperclip className="size-4" />{t('chatDocuments')}</button></>
}

export function SelectedAssistantDocuments({ documents, onRemove, onRefresh }: { documents: ChatDocumentAttachment[]; onRemove: (id: number | string) => void; onRefresh: (documents: TeacherLibraryDocument[]) => void }) {
  const { t } = useTranslation('teacherLibrary')
  const pendingIds = documents.filter(item => item.status === 'pending' || item.status === 'processing').map(item => item.id)
  const requestInFlight = useRef(false)
  useEffect(() => {
    if (!pendingIds.length) return
    const refresh = async () => { if (requestInFlight.current) return; requestInFlight.current = true; try { const items = await getTeacherLibrary(); onRefresh(items.filter((item): item is TeacherLibraryDocument => item.kind === 'document' && pendingIds.includes(item.id))) } catch { /* Keep the last confirmed backend state. */ } finally { requestInFlight.current = false } }
    void refresh(); const timer = window.setInterval(() => void refresh(), 2500); return () => window.clearInterval(timer)
  }, [onRefresh, pendingIds.join(',')])
  if (!documents.length) return null
  return <div className="flex flex-wrap gap-1.5 px-1 pb-2">{documents.map(item => {
    const indexed = item.status === 'ready'; const failed = item.status === 'failed'; const progress = item.status === 'uploading' ? 'uploading' : item.processing_stage
    const label = failed ? item.processing_error || t('uploadFailed') : indexed ? t('status.indexed') : progress ? t(`status.${progress}`) : t('status.processing')
    const tone = failed ? 'border-rose-200 bg-rose-50 text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/35 dark:text-rose-100' : indexed ? 'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900/70 dark:bg-emerald-950/45 dark:text-emerald-100' : 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/70 dark:bg-amber-950/35 dark:text-amber-100'
    return <div key={item.id} dir="auto" className={`inline-flex max-w-full items-center gap-1.5 rounded-xl border px-2 py-1.5 text-xs ${tone}`}><FileText className="size-3.5 shrink-0" /><span className="max-w-44 truncate font-medium">{item.original_filename}</span><span className="text-[11px] opacity-85">· {label}</span>{indexed ? <CheckCircle2 className="size-3.5 shrink-0" /> : failed ? <CircleAlert className="size-3.5 shrink-0" /> : <LoaderCircle className="size-3.5 shrink-0 animate-spin" />}<button type="button" onClick={() => onRemove(item.id)} aria-label={t('removeAttachment', { filename: item.original_filename })} className="rounded p-0.5 hover:bg-black/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-current"><X className="size-3.5" /></button></div>
  })}</div>
}
