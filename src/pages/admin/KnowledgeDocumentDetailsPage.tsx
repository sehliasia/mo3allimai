import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { getKnowledgeDocuments, getKnowledgeSegments, type KnowledgeDocument, type KnowledgeSegments } from '../../services/adminService'

const dash = (value: string | number | null | undefined) => value ?? '—'
const percent = (value: number | undefined) => value == null ? '—' : `${(value * 100).toFixed(2)}%`

export function KnowledgeDocumentDetailsPage() {
  const { documentId } = useParams()
  const { t } = useTranslation('admin')
  const [document, setDocument] = useState<KnowledgeDocument | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState(false)
  const [tab, setTab] = useState<'information' | 'preflight' | 'extraction' | 'chunks' | 'images'>('information')
  const [segments, setSegments] = useState<KnowledgeSegments | null>(null)
  const [segmentPage, setSegmentPage] = useState(1)
  const [sourcePage, setSourcePage] = useState('')
  const strategy = (value: string | null | undefined) => value === 'native_only' ? t('knowledge.phase3.nativeOnly') : value === 'native_with_targeted_ocr' ? t('knowledge.phase3.targetedOcr') : value === 'ocr_heavy' ? t('knowledge.phase3.ocrHeavy') : '—'
  const complexity = (value: 'low' | 'medium' | 'high' | null | undefined) => value ? t(`knowledge.phase3.${value}`) : '—'

  useEffect(() => {
    const id = Number(documentId)
    if (!Number.isInteger(id) || id <= 0) { setDocument(null); setLoadError(true); setLoading(false); return }
    setLoading(true); setLoadError(false)
    void getKnowledgeDocuments().then(result => setDocument(result.items.find(item => item.id === id) ?? null)).catch(() => setLoadError(true)).finally(() => setLoading(false))
  }, [documentId])
  useEffect(() => { const id = Number(documentId); if (tab !== 'chunks' || !Number.isInteger(id)) return; void getKnowledgeSegments(id, segmentPage, Number(sourcePage) || undefined).then(setSegments).catch(() => setSegments(null)) }, [tab, documentId, segmentPage, sourcePage])

  if (loading) return <div className="h-40 animate-pulse rounded-2xl bg-slate-200" />
  if (loadError) return <p role="alert" className="rounded-2xl bg-rose-50 p-5 text-rose-800">{t('knowledge.phase3.loadError')}</p>
  if (!document) return <p className="rounded-2xl bg-rose-50 p-5 text-rose-800">{t('knowledge.phase3.unavailable')}</p>

  const preflight = document.preflight
  const summary = document.ingestion_summary
  const tabs = ['information', 'preflight', 'extraction', 'chunks', 'images'] as const
  const extractionCards = summary ? [
    [t('knowledge.phase3.pages'), summary.pages_total],
    [t('knowledge.phase3.quarantinedPages'), summary.pages_quarantined_count],
    [t('knowledge.phase3.validChunks'), summary.chunks_valid],
    [t('knowledge.phase3.discardedChunks'), summary.chunks_quarantined_count],
    [t('knowledge.phase3.ocrUsed'), summary.ocr_pages_count],
    [t('knowledge.phase3.lateRepairs'), `${summary.late_repairs_accepted ?? 0}/${summary.late_repairs_attempted ?? 0}`],
    [t('knowledge.phase3.warnings'), summary.warnings_count],
    ...(summary.quality_status === 'failed' ? [[t('knowledge.phase3.failedPageRatio'), percent(summary.failed_page_ratio)], [t('knowledge.phase3.maximumAllowed'), percent(summary.max_failed_page_ratio)]] : []),
  ] : []

  return <div className="space-y-6">
    <div><h2 dir="auto" className="text-xl font-extrabold text-slate-900">{document.title}</h2><p dir="auto" className="mt-1 text-sm text-slate-500">{document.original_filename}</p></div>
    <div className="flex gap-2 overflow-x-auto border-b border-slate-200">{tabs.map(item => <button key={item} type="button" onClick={() => setTab(item)} className={`shrink-0 border-b-2 px-3 py-3 text-sm font-bold ${tab === item ? 'border-emerald-700 text-emerald-700' : 'border-transparent text-slate-500'}`}>{t(`knowledge.phase3.${item}`)}</button>)}</div>
    {tab === 'information' && <dl className="grid gap-4 sm:grid-cols-2"><div><dt className="text-sm text-slate-500">{t('knowledge.details.filename')}</dt><dd dir="auto" className="font-semibold">{document.original_filename}</dd></div><div><dt className="text-sm text-slate-500">{t('knowledge.details.status')}</dt><dd className="font-semibold">{document.status === 'partial' ? t('knowledge.phase3.readyWithWarnings') : document.status === 'failed' ? t('knowledge.phase3.analysisFailed') : document.status}</dd></div></dl>}
    {tab === 'preflight' && <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{[[t('knowledge.phase3.pages'), preflight?.pages_total], [t('knowledge.phase3.ocrCandidates'), preflight?.ocr_candidate_page_count], [t('knowledge.phase3.strategy'), strategy(preflight?.recommended_strategy)], [t('knowledge.phase3.complexity'), complexity(preflight?.estimated_complexity)], [t('knowledge.phase3.analysisComplete'), preflight?.pages_analyzed], [t('knowledge.phase3.analysisFailed'), preflight?.analysis_failed_pages]].map(([label, value]) => <div key={String(label)} className="rounded-xl border border-slate-200 bg-white p-4"><p className="text-sm text-slate-500">{label}</p><p className="mt-1 font-extrabold text-slate-900">{dash(value as string | number | null)}</p></div>)}</div>}
    {tab === 'extraction' && (summary ? <><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{extractionCards.map(([label, value]) => <div key={String(label)} className="rounded-xl border border-slate-200 bg-white p-4"><p className="text-sm text-slate-500">{label}</p><p className="mt-1 font-extrabold text-slate-900">{dash(value as string | number | null)}</p></div>)}</div>{summary.quality_status === 'failed' && <p role="alert" className="rounded-xl bg-rose-50 p-4 text-sm font-semibold text-rose-800">{summary.failure_reason === 'failed_page_ratio_exceeded' ? t('knowledge.phase3.failedPageRatioExceeded') : t('knowledge.phase3.analysisFailed')}</p>}{summary.quarantined_page_numbers.length > 0 && <details className={`rounded-xl p-4 text-sm ${summary.quality_status === 'failed' ? 'bg-rose-50 text-rose-900' : 'bg-amber-50 text-amber-900'}`}><summary className="cursor-pointer font-bold">{t('knowledge.phase3.quarantinedPages')}</summary><p className="mt-2">{summary.quarantined_page_numbers.join(', ')}</p></details>}</> : <p className="rounded-xl bg-slate-50 p-5 text-sm text-slate-600">{t('knowledge.phase3.unavailable')}</p>)}
    {tab === 'chunks' && <div className="space-y-4"><div className="flex flex-wrap gap-2"><input value={sourcePage} onChange={event => { setSegmentPage(1); setSourcePage(event.target.value) }} type="number" min="1" placeholder="Page PDF" className="h-10 rounded-xl border border-slate-200 px-3 text-sm" /></div>{segments === null ? <p className="rounded-xl bg-slate-50 p-5 text-sm text-slate-600">{t('knowledge.phase3.unavailable')}</p> : !['available', 'legacy_debug'].includes(segments.availability) ? <p className="rounded-xl bg-slate-50 p-5 text-sm text-slate-600">{segments.availability === 'not_persisted' ? "Les segments de cette ingestion ne sont pas disponibles pour prévisualisation." : t('knowledge.phase3.unavailable')}</p> : <>{segments.availability === 'legacy_debug' && <p className="rounded-xl bg-amber-50 p-3 text-sm text-amber-800">Segments issus d’un export de diagnostic historique.</p>}{segments.items.map(segment => <article key={segment.id} className="rounded-xl border border-slate-200 bg-white p-4"><div className="flex flex-wrap gap-2 text-xs font-semibold text-slate-500"><span>#{segment.chunk_index + 1}</span><span>Page {segment.page_start ?? '—'}</span><span>{segment.content_type}</span><span>{segment.token_count} tokens</span></div><p dir="auto" className="mt-3 whitespace-pre-wrap break-words text-sm leading-6 text-slate-800">{segment.content}</p></article>)}<div className="flex justify-between"><button disabled={segmentPage===1} onClick={()=>setSegmentPage(value=>value-1)}>←</button><span className="text-sm">{segments.total}</span><button disabled={segmentPage * segments.page_size >= segments.total} onClick={()=>setSegmentPage(value=>value+1)}>→</button></div></>}</div>}
    {tab === 'images' && <p className="rounded-xl bg-slate-50 p-5 text-sm text-slate-600">{t('knowledge.phase3.unavailable')}</p>}
  </div>
}
