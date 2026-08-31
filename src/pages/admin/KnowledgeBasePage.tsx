import { Check, Eye, FileText, LoaderCircle, MoreHorizontal, Plus, Search, UploadCloud, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { enqueueKnowledgePreflight, enqueueKnowledgeProcessing, getKnowledgeDocuments, getKnowledgePreview, getKnowledgeProcessingJobs, runKnowledgePreflight, type KnowledgeDocument, type KnowledgePreview, uploadKnowledgeDocument } from '../../services/adminService'
import { AuthApiError } from '../../services/authService'
import { KnowledgePreviewModal } from '../../components/admin/KnowledgePreviewModal'
import { useAuth } from '../../contexts/AuthContext'

const MAX_FILE_SIZE = 25 * 1024 * 1024
const MAX_FILES_PER_BATCH = 20
const MAX_CONCURRENT_UPLOADS = 3

type UploadStatus = 'ready' | 'uploading' | 'success' | 'error'
type UploadItem = { id: string; file: File; status: UploadStatus; error?: string; retryable: boolean }

const fileKey = (file: File) => `${file.name}-${file.size}-${file.lastModified}`
const isPdf = (file: File) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')

export function KnowledgeBasePage() {
  const { t } = useTranslation('admin')
  const navigate = useNavigate()
  const { logout } = useAuth()
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([])
  const [open, setOpen] = useState(false)
  const [items, setItems] = useState<UploadItem[]>([])
  const [batchActive, setBatchActive] = useState(false)
  const [limitMessage, setLimitMessage] = useState('')
  const [actionsOpenId, setActionsOpenId] = useState<number | null>(null)
  const [selectedDocument, setSelectedDocument] = useState<KnowledgeDocument | null>(null)
  const [previewDocument, setPreviewDocument] = useState<KnowledgeDocument | null>(null)
  const [preview, setPreview] = useState<KnowledgePreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState('')
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const [loadingDocuments, setLoadingDocuments] = useState(true)
  const [documentsError, setDocumentsError] = useState('')
  const [analyzingId, setAnalyzingId] = useState<number | null>(null)
  const [notice, setNotice] = useState('')
  const [queueLoading, setQueueLoading] = useState<'preflight' | 'ingestion' | null>(null)
  const [pollDocumentIds, setPollDocumentIds] = useState<number[]>([])
  const fileInput = useRef<HTMLInputElement>(null)
  const actionsMenuRef = useRef<HTMLDivElement>(null)

  const load = async () => {
    setLoadingDocuments(true); setDocumentsError('')
    try { setDocuments((await getKnowledgeDocuments()).items) } catch { setDocumentsError(t('knowledge.phase3.loadError')) } finally { setLoadingDocuments(false) }
  }

  const close = () => {
    if (batchActive) return
    setOpen(false)
    setItems([])
    setLimitMessage('')
  }

  useEffect(() => {
    void load()
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') { setActionsOpenId(null); if (previewDocument) { setPreviewDocument(null); setPreview(null); setPreviewError('') } else if (selectedDocument) setSelectedDocument(null); else if (!batchActive) close() } }
    const closeOnOutsideClick = (event: MouseEvent) => { if (actionsMenuRef.current && !actionsMenuRef.current.contains(event.target as Node)) setActionsOpenId(null) }
    document.addEventListener('keydown', closeOnEscape)
    document.addEventListener('mousedown', closeOnOutsideClick)
    return () => { document.removeEventListener('keydown', closeOnEscape); document.removeEventListener('mousedown', closeOnOutsideClick) }
  }, [batchActive, previewDocument, selectedDocument])

  const addFiles = (files: FileList | File[]) => {
    const currentKeys = new Set(items.map(item => fileKey(item.file)))
    let available = MAX_FILES_PER_BATCH - items.filter(item => item.status !== 'error' || item.retryable).length
    const incoming = Array.from(files).filter(file => {
      const key = fileKey(file)
      if (currentKeys.has(key)) return false
      currentKeys.add(key)
      return true
    })
    let limitExceeded = false
    const accepted = incoming.flatMap(file => {
      if (!isPdf(file)) return [{ id: crypto.randomUUID(), file, status: 'error' as const, error: t('knowledge.batch.invalidPdf'), retryable: false }]
      if (!file.size) return [{ id: crypto.randomUUID(), file, status: 'error' as const, error: t('knowledge.batch.emptyFile'), retryable: false }]
      if (file.size > MAX_FILE_SIZE) return [{ id: crypto.randomUUID(), file, status: 'error' as const, error: t('knowledge.batch.fileTooLarge'), retryable: false }]
      if (available-- <= 0) { limitExceeded = true; return [] }
      return [{ id: crypto.randomUUID(), file, status: 'ready' as const, retryable: true }]
    })
    setLimitMessage(limitExceeded ? t('knowledge.batch.limitReached', { count: MAX_FILES_PER_BATCH }) : '')
    setItems(current => [...current, ...accepted])
  }

  const updateItem = (id: string, patch: Partial<UploadItem>) => setItems(current => current.map(item => item.id === id ? { ...item, ...patch } : item))
  const removeItem = (id: string) => { if (!batchActive) setItems(current => current.filter(item => item.id !== id)) }
  const formatSize = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} ${t('common:megabytes')}`
  const formatDate = (value: string) => new Date(value).toLocaleDateString()
  const toggleDocument = (id: number) => setSelectedIds(current => { const next = new Set(current); next.has(id) ? next.delete(id) : next.add(id); return next })
  const toggleVisible = () => setSelectedIds(current => current.size === documents.length ? new Set() : new Set(documents.map(document => document.id)))
  const strategy = (value: string | null) => value ? t(`knowledge.phase3.${value === 'native_only' ? 'nativeOnly' : value === 'native_with_targeted_ocr' ? 'targetedOcr' : 'ocrHeavy'}`) : '—'
  const preflightLabel = (document: KnowledgeDocument) => !document.preflight ? t('knowledge.phase3.notAnalyzed') : document.preflight.status === 'complete' ? t('knowledge.phase3.analysisComplete') : document.preflight.status === 'partial' ? t('knowledge.phase3.analysisPartial') : t('knowledge.phase3.analysisFailed')
  const runPreflight = async (document: KnowledgeDocument) => { setActionsOpenId(null); setAnalyzingId(document.id); setNotice(''); try { await runKnowledgePreflight(document.id); await load(); setNotice(t('knowledge.phase3.analysisSuccess')) } catch { setNotice(t('knowledge.phase3.analysisError')) } finally { setAnalyzingId(null) } }
  const handleAuthenticationError = (error: unknown) => {
    if (error instanceof AuthApiError && error.status === 401) { logout(); navigate('/login'); return true }
    return false
  }
  const enqueueSelection = async (type: 'preflight' | 'ingestion') => {
    const documentIds = [...selectedIds]
    if (!documentIds.length || queueLoading) return
    setQueueLoading(type); setNotice('')
    try {
      const result = type === 'preflight' ? await enqueueKnowledgePreflight(documentIds) : await enqueueKnowledgeProcessing(documentIds)
      setNotice(t('knowledge.phase3.queueResult', { queued: result.queued, skipped: result.skipped }))
      setSelectedIds(new Set())
      setPollDocumentIds(documentIds)
    } catch (error) { if (!handleAuthenticationError(error)) setNotice(t('knowledge.phase3.queueError')) } finally { setQueueLoading(null) }
  }

  useEffect(() => {
    if (!pollDocumentIds.length) return
    let cancelled = false
    const poll = async () => {
      try {
        const items = await getKnowledgeProcessingJobs(pollDocumentIds)
        if (cancelled) return
        if (!items.some(job => job.status === 'pending' || job.status === 'processing')) { setPollDocumentIds([]); await load() }
      } catch (error) { if (!cancelled) { if (!handleAuthenticationError(error)) setPollDocumentIds([]) } }
    }
    void poll()
    const interval = window.setInterval(() => void poll(), 5000)
    return () => { cancelled = true; window.clearInterval(interval) }
  }, [pollDocumentIds])

  const upload = async (requestedItems: UploadItem[]) => {
    const queued = requestedItems.filter(item => item.status === 'ready' || (item.status === 'error' && item.retryable))
    if (!queued.length || batchActive) return
    setBatchActive(true)
    queued.forEach(item => updateItem(item.id, { status: 'uploading', error: undefined }))
    let nextIndex = 0
    let successCount = 0
    const worker = async () => {
      while (nextIndex < queued.length) {
        const item = queued[nextIndex++]
        try {
          await uploadKnowledgeDocument(item.file)
          successCount += 1
          updateItem(item.id, { status: 'success', retryable: false })
        } catch {
          updateItem(item.id, { status: 'error', error: t('knowledge.uploadError'), retryable: true })
        }
      }
    }
    await Promise.all(Array.from({ length: Math.min(MAX_CONCURRENT_UPLOADS, queued.length) }, worker))
    if (successCount) await load()
    setBatchActive(false)
  }

  const readyItems = items.filter(item => item.status === 'ready')
  const failedItems = items.filter(item => item.status === 'error')
  const successCount = items.filter(item => item.status === 'success').length
  const progressTotal = items.filter(item => item.status !== 'error').length
  const statusText = (item: UploadItem) => item.status === 'ready' ? t('knowledge.batch.ready') : item.status === 'uploading' ? t('knowledge.batch.uploading') : item.status === 'success' ? t('knowledge.batch.uploaded') : item.error
  const renderDocumentCell = (document: KnowledgeDocument) => <div className="flex min-w-0 items-center gap-3"><span className="inline-flex size-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-700"><FileText className="size-5" /></span><div className="min-w-0"><p dir="auto" title={document.title} className="truncate font-semibold text-slate-900">{document.title}</p><p dir="auto" title={document.original_filename} className="mt-0.5 truncate text-sm text-slate-500">{document.original_filename}</p></div></div>
  const statusBadge = (status: KnowledgeDocument['status']) => {
    const styles = { pending: 'bg-slate-100 text-slate-700', processing: 'bg-amber-50 text-amber-700', ready: 'bg-emerald-50 text-emerald-700', partial: 'bg-amber-50 text-amber-700', indexed: 'bg-emerald-50 text-emerald-700', failed: 'bg-rose-50 text-rose-700' }
    const label = status === 'ready' ? t('knowledge.phase3.analysisComplete') : status === 'partial' ? t('knowledge.phase3.readyWithWarnings') : t(`knowledge.details.statuses.${status}`)
    return <span className={`inline-flex h-8 items-center justify-center rounded-full px-3 text-xs font-semibold ${styles[status]}`}>{label}</span>
  }
  const displayedStatus = (document: KnowledgeDocument) => {
    const ingestionJob = (document.active_jobs ?? []).find(job => job.job_type === 'ingestion')
    return ingestionJob?.status === 'processing' ? 'processing' : ingestionJob?.status === 'pending' ? 'pending' : document.status
  }
  const openPreview = async (document: KnowledgeDocument) => { setPreviewDocument(document); setPreview(null); setPreviewError(''); setPreviewLoading(true); setActionsOpenId(null); try { setPreview(await getKnowledgePreview(document.id)) } catch { setPreviewError(t('knowledge.details.previewError')) } finally { setPreviewLoading(false) } }
  const actionMenu = (document: KnowledgeDocument) => <div ref={actionsOpenId === document.id ? actionsMenuRef : undefined} className="relative inline-flex"><button type="button" onClick={() => setActionsOpenId(current => current === document.id ? null : document.id)} aria-label={t('common:actions')} aria-haspopup="menu" aria-expanded={actionsOpenId === document.id} className="inline-flex size-10 cursor-pointer items-center justify-center rounded-xl text-slate-500 transition-colors hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-emerald-600"><MoreHorizontal className="size-5" /></button>{actionsOpenId === document.id && <div role="menu" onMouseDown={event => event.stopPropagation()} className="pointer-events-auto absolute end-0 top-11 z-20 w-52 rounded-xl border border-slate-200 bg-white p-1.5 text-start shadow-lg"><button type="button" role="menuitem" onClick={event => { event.stopPropagation(); navigate(`/admin/knowledge-base/${document.id}`); setActionsOpenId(null) }} className="flex w-full cursor-pointer items-center gap-2 rounded-lg px-3 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-emerald-600"><FileText className="size-4 text-emerald-700" />{t('knowledge.details.view')}</button><button type="button" role="menuitem" disabled={analyzingId === document.id} onClick={event => { event.stopPropagation(); void runPreflight(document) }} className="flex w-full cursor-pointer items-center gap-2 rounded-lg px-3 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-emerald-600 disabled:cursor-not-allowed disabled:opacity-60">{analyzingId === document.id ? <LoaderCircle className="size-4 animate-spin text-emerald-700" /> : <Check className="size-4 text-emerald-700" />}{t('knowledge.phase3.analyze')}</button><button type="button" role="menuitem" disabled={previewLoading} onClick={event => { event.stopPropagation(); void openPreview(document) }} className="flex w-full cursor-pointer items-center gap-2 rounded-lg px-3 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-emerald-600 disabled:cursor-not-allowed disabled:opacity-60"><Eye className="size-4 text-emerald-700" />{t('knowledge.details.preview')}</button></div>}</div>

  return <div className="space-y-6">
    <div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-xl font-extrabold text-slate-900">{t('page.knowledge.title')}</h2><p className="mt-1 text-sm text-slate-500">{t('page.knowledge.description')}</p></div><button type="button" onClick={() => setOpen(true)} className="inline-flex items-center gap-2 rounded-xl bg-emerald-700 px-4 py-2.5 text-sm font-bold text-white hover:bg-emerald-800 focus-visible:ring-2 focus-visible:ring-emerald-600"><Plus className="size-4" />{t('knowledge.batch.add')}</button></div>
    {notice && <p role="status" className="rounded-xl bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-800">{notice}</p>}
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">{(['all', 'pending', 'processing', 'indexed', 'failed'] as const).map(key => <div key={key} className="rounded-2xl border border-slate-200 bg-white p-4"><p className="text-sm text-slate-500">{t(key === 'all' ? 'knowledge.all' : `knowledge.details.statuses.${key}`)}</p><p className="mt-2 text-2xl font-extrabold text-slate-900">{key === 'all' ? documents.length : documents.filter(document => document.status === key).length}</p></div>)}</div>
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"><label className="relative block max-w-xl"><Search className="pointer-events-none absolute start-3 top-1/2 size-4 -translate-y-1/2 text-slate-400" /><input aria-label={t('knowledge.search')} placeholder={t('knowledge.search')} className="w-full rounded-xl border border-slate-200 bg-slate-50/50 py-2.5 ps-10 pe-3 text-sm outline-none transition focus:border-emerald-600 focus:bg-white focus:ring-4 focus:ring-emerald-100" /></label></div>
    {loadingDocuments ? <div className="grid gap-3"><div className="h-16 animate-pulse rounded-2xl bg-slate-200" /><div className="h-16 animate-pulse rounded-2xl bg-slate-100" /></div> : documentsError ? <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-rose-200 bg-rose-50 p-5 text-rose-800"><span>{documentsError}</span><button type="button" onClick={() => void load()} className="rounded-lg border border-rose-300 px-3 py-2 text-sm font-bold focus-visible:ring-2 focus-visible:ring-rose-600">{t('dashboard.retry')}</button></div> : documents.length === 0 ? <div className="rounded-2xl border border-slate-200 bg-white px-5 py-14 text-center"><FileText className="mx-auto size-10 text-emerald-700" /><h3 className="mt-4 font-extrabold">{t('knowledge.emptyTitle')}</h3><p className="mx-auto mt-2 max-w-lg text-sm text-slate-500">{t('knowledge.emptyDescription')}</p></div> : <>
      {selectedIds.size > 0 && <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm"><strong>{t('knowledge.phase3.selected', { count: selectedIds.size })}</strong><button type="button" disabled={queueLoading !== null} onClick={() => void enqueueSelection('preflight')} className="rounded-lg bg-emerald-700 px-3 py-2 font-bold text-white disabled:opacity-60">{queueLoading === 'preflight' ? <LoaderCircle className="size-4 animate-spin" /> : null}{t('knowledge.phase3.analyze')}</button><button type="button" disabled={queueLoading !== null} onClick={() => void enqueueSelection('ingestion')} className="rounded-lg border border-emerald-300 px-3 py-2 font-bold text-emerald-800 disabled:opacity-60">{queueLoading === 'ingestion' ? <LoaderCircle className="size-4 animate-spin" /> : null}{t('knowledge.phase3.process')}</button>{pollDocumentIds.length > 0 && <span className="text-slate-600">{t('knowledge.phase3.polling')}</span>}</div>}
      <div className="hidden overflow-x-auto rounded-2xl border border-slate-200 bg-white shadow-sm md:block"><table className="w-full min-w-[1180px] text-sm"><thead className="border-b border-slate-200 bg-slate-50 text-slate-700"><tr><th className="px-3 py-4"><input type="checkbox" checked={selectedIds.size === documents.length} onChange={toggleVisible} aria-label={t('knowledge.phase3.selectAll')} /></th><th className="px-4 text-start">{t('knowledge.document')}</th><th className="px-3 text-center">{t('knowledge.phase3.pages')}</th><th className="px-3 text-center">{t('knowledge.phase3.preflight')}</th><th className="px-3 text-center">{t('knowledge.phase3.ocrCandidates')}</th><th className="px-3 text-center">{t('knowledge.phase3.strategy')}</th><th className="px-3 text-center">{t('knowledge.status')}</th><th className="px-3 text-center">{t('knowledge.addedAt')}</th><th className="px-3 text-center">{t('common:actions')}</th></tr></thead><tbody>{documents.map(document => <tr key={document.id} className="border-b border-slate-100 hover:bg-slate-50"><td className="px-3 py-4 text-center"><input type="checkbox" checked={selectedIds.has(document.id)} onChange={() => toggleDocument(document.id)} aria-label={t('knowledge.phase3.select')} /></td><td className="px-4 py-4">{renderDocumentCell(document)}</td><td className="px-3 py-4 text-center">{document.preflight?.pages_total ?? '—'}</td><td className="px-3 py-4 text-center">{preflightLabel(document)}</td><td className="px-3 py-4 text-center">{document.preflight?.ocr_required_page_ratio == null ? '—' : `${Math.round(document.preflight.ocr_required_page_ratio * 100)}%`}</td><td className="px-3 py-4 text-center">{strategy(document.preflight?.recommended_strategy ?? null)}</td><td className="px-3 py-4 text-center">{statusBadge(displayedStatus(document))}</td><td className="whitespace-nowrap px-3 py-4 text-center">{formatDate(document.created_at)}</td><td className="px-3 py-4 text-center">{actionMenu(document)}</td></tr>)}</tbody></table></div>
      <div className="space-y-3 md:hidden">{documents.map(document => <article key={document.id} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"><div className="flex items-start gap-3">{renderDocumentCell(document)}<div className="shrink-0">{actionMenu(document)}</div></div><div className="mt-4 flex items-center justify-between gap-3 border-t border-slate-100 pt-3"><div>{statusBadge(displayedStatus(document))}</div><time className="whitespace-nowrap text-sm font-medium text-slate-700">{formatDate(document.created_at)}</time></div></article>)}</div>
    </>}
    {selectedDocument && <div role="dialog" aria-modal="true" aria-label={t('knowledge.details.title')} className="fixed inset-0 z-[70] flex items-center justify-center bg-slate-950/40 p-4"><div className="w-full max-w-lg rounded-2xl bg-white shadow-2xl"><div className="flex items-center justify-between border-b border-slate-100 px-5 py-4 sm:px-6"><h2 className="text-xl font-extrabold text-slate-900">{t('knowledge.details.title')}</h2><button type="button" onClick={() => setSelectedDocument(null)} aria-label={t('common:close')} className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-emerald-600"><X className="size-5" /></button></div><dl className="space-y-5 px-5 py-6 sm:px-6"><div><dt className="text-sm font-medium text-slate-500">{t('knowledge.details.documentTitle')}</dt><dd dir="auto" className="mt-1 break-words font-semibold text-slate-900">{selectedDocument.title}</dd></div><div><dt className="text-sm font-medium text-slate-500">{t('knowledge.details.filename')}</dt><dd dir="auto" className="mt-1 break-all text-slate-800">{selectedDocument.original_filename}</dd></div><div className="grid grid-cols-1 gap-5 sm:grid-cols-2"><div><dt className="text-sm font-medium text-slate-500">{t('knowledge.details.fileSize')}</dt><dd className="mt-1 font-semibold text-slate-900">{formatSize(selectedDocument.file_size)}</dd></div><div><dt className="text-sm font-medium text-slate-500">{t('knowledge.details.status')}</dt><dd className="mt-2">{statusBadge(selectedDocument.status)}</dd></div></div><div><dt className="text-sm font-medium text-slate-500">{t('knowledge.details.createdAt')}</dt><dd className="mt-1 font-semibold text-slate-900">{formatDate(selectedDocument.created_at)}</dd></div></dl><div className="flex justify-end border-t border-slate-100 px-5 py-4 sm:px-6"><button type="button" onClick={() => setSelectedDocument(null)} className="rounded-xl bg-emerald-700 px-4 py-2.5 text-sm font-bold text-white hover:bg-emerald-800 focus-visible:ring-2 focus-visible:ring-emerald-600">{t('common:close')}</button></div></div></div>}
    {previewDocument && <KnowledgePreviewModal document={previewDocument} preview={preview} loading={previewLoading} error={previewError} onClose={() => { setPreviewDocument(null); setPreview(null); setPreviewError('') }} />}
    {open && <div role="dialog" aria-modal="true" aria-label={t('knowledge.batch.title')} className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-950/40 p-3 sm:p-4"><div className="flex max-h-[calc(100dvh-1.5rem)] w-full max-w-3xl flex-col rounded-2xl bg-white shadow-2xl"><div className="flex items-center justify-between border-b border-slate-100 px-5 py-4 sm:px-7"><h2 className="text-xl font-extrabold text-slate-900">{t('knowledge.batch.title')}</h2><button type="button" disabled={batchActive} onClick={close} aria-label={t('common:close')} className="rounded-lg p-2 hover:bg-slate-100 disabled:opacity-50"><X className="size-5" /></button></div><div className="min-h-0 overflow-y-auto px-5 py-5 sm:px-7">
      <input ref={fileInput} type="file" multiple accept="application/pdf,.pdf" className="sr-only" onChange={event => { addFiles(event.target.files ?? []); event.currentTarget.value = '' }} />
      <button type="button" onClick={() => fileInput.current?.click()} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); addFiles(event.dataTransfer.files) }} className={`flex w-full flex-col items-center rounded-2xl border-2 border-dashed px-6 text-center transition focus-visible:ring-2 focus-visible:ring-emerald-600 ${items.length ? 'border-slate-200 py-5 hover:border-emerald-400 hover:bg-emerald-50/40' : 'border-slate-200 py-10 hover:border-emerald-400 hover:bg-emerald-50/40'}`}><UploadCloud className="size-9 text-emerald-700" /><p className="mt-3 font-extrabold text-slate-900">{items.length ? t('knowledge.batch.addMore') : t('knowledge.batch.dropzone')}</p><p className="mt-1 text-sm text-slate-500">{items.length ? t('knowledge.batch.browse') : t('knowledge.batch.dropSubtitle')}</p><small className="mt-3 text-slate-500">{t('knowledge.batch.requirements')}</small></button>
      {limitMessage && <p role="alert" className="mt-3 rounded-xl bg-amber-50 p-3 text-sm text-amber-800">{limitMessage}</p>}
      {items.length > 0 && <section className="mt-5"><h3 className="font-bold text-slate-900">{t('knowledge.batch.selected', { count: items.length })}</h3><div className="mt-3 max-h-72 divide-y divide-slate-100 overflow-y-auto rounded-xl border border-slate-200">{items.map(item => <div key={item.id} className="flex items-center gap-3 p-3"><FileText className={`size-8 shrink-0 ${item.status === 'error' ? 'text-rose-600' : item.status === 'success' ? 'text-emerald-700' : 'text-slate-500'}`} /><div className="min-w-0 flex-1"><p className="truncate text-sm font-bold text-slate-900">{item.file.name}</p><p className={`mt-0.5 text-xs ${item.status === 'error' ? 'text-rose-700' : 'text-slate-500'}`}>{formatSize(item.file.size)} · {statusText(item)}</p></div>{item.status === 'uploading' ? <LoaderCircle className="size-5 animate-spin text-emerald-700" /> : item.status === 'success' ? <Check className="size-5 text-emerald-700" /> : item.status === 'error' && item.retryable ? <button type="button" disabled={batchActive} onClick={() => void upload([item])} className="rounded-lg px-2 py-1 text-xs font-bold text-emerald-700 hover:bg-emerald-50">{t('knowledge.batch.retry')}</button> : null}{item.status !== 'success' && item.status !== 'uploading' && <button type="button" disabled={batchActive} onClick={() => removeItem(item.id)} aria-label={t('knowledge.batch.remove')} className="rounded-lg p-1.5 text-slate-500 hover:bg-rose-50 hover:text-rose-700"><X className="size-4" /></button>}</div>)}</div></section>}
      {!batchActive && successCount > 0 && <p role="status" className="mt-4 rounded-xl bg-emerald-50 p-3 text-sm font-medium text-emerald-800">{t('knowledge.batch.successSummary', { count: successCount })}{failedItems.length ? ` · ${t('knowledge.batch.failedSummary', { count: failedItems.length })}` : ''}</p>}
      {batchActive && <p role="status" className="mt-4 text-sm font-medium text-slate-600">{t('knowledge.batch.progress', { success: successCount, total: progressTotal })}</p>}
    </div><div className="flex flex-wrap justify-end gap-3 border-t border-slate-100 px-5 py-4 sm:px-7"><button type="button" disabled={batchActive} onClick={close} className="rounded-xl px-4 py-2.5 text-sm font-bold hover:bg-slate-100 disabled:opacity-50">{successCount > 0 ? t('common:close') : t('common:cancel')}</button>{failedItems.some(item => item.retryable) && !batchActive && <button type="button" onClick={() => void upload(failedItems)} className="rounded-xl border border-emerald-200 px-4 py-2.5 text-sm font-bold text-emerald-700 hover:bg-emerald-50">{t('knowledge.batch.retryErrors')}</button>}<button type="button" disabled={!readyItems.length || batchActive} onClick={() => void upload(readyItems)} className="rounded-xl bg-emerald-700 px-4 py-2.5 text-sm font-bold text-white disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-500">{batchActive ? t('knowledge.batch.uploading') : t('knowledge.batch.importButton', { count: readyItems.length })}</button></div></div></div>}
  </div>
}
