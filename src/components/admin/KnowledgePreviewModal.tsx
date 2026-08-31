import { AlertTriangle, ChevronDown, FileImage, LoaderCircle, X } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { KnowledgeDocument, KnowledgePreview } from '../../services/adminService'

type Props = {
  document: KnowledgeDocument
  preview: KnowledgePreview | null
  loading: boolean
  error: string
  onClose: () => void
}

const dash = (value: string | number | null | undefined) => value ?? '—'
const duration = (milliseconds: number | undefined) => milliseconds === undefined ? '—' : `${(milliseconds / 1000).toFixed(1)} s`

export function KnowledgePreviewModal({ document, preview, loading, error, onClose }: Props) {
  const { t } = useTranslation('admin')
  const [expandedChunk, setExpandedChunk] = useState<number | null>(null)
  const qualityStyle = preview?.quality_status === 'complete' ? 'bg-emerald-50 text-emerald-700' : preview?.quality_status === 'partial' ? 'bg-amber-50 text-amber-800' : 'bg-rose-50 text-rose-700'
  const qualityLabel = preview?.quality_status ? t(`knowledge.details.qualityStatuses.${preview.quality_status}`) : '—'

  return <div role="dialog" aria-modal="true" aria-label={t('knowledge.details.preview')} className="fixed inset-0 z-[80] flex items-center justify-center bg-slate-950/40 p-3 sm:p-5">
    <div className="flex max-h-[calc(100dvh-1.5rem)] w-full max-w-5xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl">
      <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4 sm:px-6">
        <div className="min-w-0"><h2 className="truncate text-xl font-extrabold text-slate-900">{t('knowledge.details.preview')}</h2><p dir="auto" className="mt-0.5 truncate text-sm text-slate-500">{document.original_filename}</p></div>
        <button type="button" onClick={onClose} aria-label={t('common:close')} className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-emerald-600"><X className="size-5" /></button>
      </div>
      <div className="min-h-0 overflow-y-auto px-5 py-5 sm:px-6">
        {loading && <div role="status" className="flex min-h-56 flex-col items-center justify-center text-center"><LoaderCircle className="size-9 animate-spin text-emerald-700" /><p className="mt-4 font-bold text-slate-900">{t('knowledge.details.processing')}</p><p className="mt-1 text-sm text-slate-500">{t('knowledge.details.processingHint')}</p></div>}
        {!loading && error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">{error}</div>}
        {!loading && preview && <div className="space-y-6">
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[[t('knowledge.details.pages'), preview.pages], [t('knowledge.details.chunks'), preview.chunks_count], [t('knowledge.details.processingDuration'), duration((preview.parsing_duration_ms ?? 0) + (preview.chunking_duration_ms ?? 0))], [t('knowledge.details.quality'), qualityLabel]].map(([label, value]) => <div key={String(label)} className="rounded-xl border border-slate-200 bg-slate-50/70 p-3"><p className="text-xs font-medium text-slate-500">{label}</p><p className={`mt-1 text-lg font-extrabold ${label === t('knowledge.details.quality') ? qualityStyle : 'text-slate-900'}`}>{value}</p></div>)}
          </section>
          {preview.extraction && <section><h3 className="text-base font-extrabold text-slate-900">{t('knowledge.details.extraction')}</h3><div className="mt-3 overflow-x-auto rounded-xl border border-slate-200"><table className="w-full min-w-[560px] text-sm"><thead className="bg-slate-50 text-start text-slate-600"><tr><th className="px-4 py-3">{t('knowledge.details.page')}</th><th className="px-4 py-3">{t('knowledge.details.mode')}</th><th className="px-4 py-3">{t('knowledge.details.quality')}</th><th className="px-4 py-3">{t('knowledge.details.languages')}</th></tr></thead><tbody>{preview.extraction.per_page_quality?.map(item => <tr key={item.page} className="border-t border-slate-100"><td className="px-4 py-3 font-semibold">{item.page}</td><td className="px-4 py-3"><span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700">{item.extraction_mode === 'full_page_ocr' ? t('knowledge.details.ocr') : t('knowledge.details.native')}</span></td><td className="px-4 py-3">{item.quality_score.toFixed(2)}</td><td className="px-4 py-3">{item.languages?.join(', ') || '—'}</td></tr>)}</tbody></table></div></section>}
          {preview.warnings?.length ? <section className="rounded-xl border border-amber-200 bg-amber-50 p-4"><div className="flex items-center gap-2 font-bold text-amber-900"><AlertTriangle className="size-4" />{t('knowledge.details.warnings')}</div><ul className="mt-2 space-y-1 text-sm text-amber-800">{preview.warnings.map((warning, index) => <li key={`${warning.page}-${index}`}>{warning.page ? `${t('knowledge.details.page')} ${warning.page}: ` : ''}{warning.reason}</li>)}</ul></section> : null}
          <section><h3 className="text-base font-extrabold text-slate-900">{t('knowledge.details.chunks')}</h3><div className="mt-3 space-y-2">{preview.chunks.map(chunk => <article key={chunk.chunk_index} className="rounded-xl border border-slate-200"><button type="button" onClick={() => setExpandedChunk(current => current === chunk.chunk_index ? null : chunk.chunk_index)} aria-expanded={expandedChunk === chunk.chunk_index} className="flex w-full items-center gap-3 p-4 text-start hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-emerald-600"><ChevronDown className={`size-4 shrink-0 transition-transform ${expandedChunk === chunk.chunk_index ? 'rotate-180' : ''}`} /><span className="font-bold text-slate-900">#{chunk.chunk_index}</span><span className="text-sm text-slate-500">{t('knowledge.details.page')} {dash(chunk.page_start)}–{dash(chunk.page_end)}</span><span className="ms-auto text-xs font-semibold text-slate-600">{chunk.token_count} {t('knowledge.details.tokens')}</span></button>{expandedChunk === chunk.chunk_index && <div className="border-t border-slate-100 px-4 py-3 text-sm"><div className="flex flex-wrap gap-2 text-xs text-slate-600"><span>{chunk.content_type}</span>{chunk.headings?.map(heading => <span key={heading} dir="auto" className="rounded bg-slate-100 px-2 py-1">{heading}</span>)}{chunk.image_ids?.length ? <span className="rounded bg-emerald-50 px-2 py-1 text-emerald-800">{t('knowledge.details.imageCount', { count: chunk.image_ids.length })}</span> : null}</div><p dir="auto" className="mt-3 whitespace-pre-wrap text-slate-700">{chunk.text_preview}</p></div>}</article>)}</div></section>
          <section><h3 className="text-base font-extrabold text-slate-900">{t('knowledge.details.images')}</h3>{preview.images?.length ? <div className="mt-3 grid gap-3 sm:grid-cols-2">{preview.images.map(image => <article key={image.image_id} className="rounded-xl border border-slate-200 p-4"><div className="flex items-center gap-2 font-bold text-slate-900"><FileImage className="size-5 text-emerald-700" />{t('knowledge.details.page')} {image.page}</div>{image.caption && <p dir="auto" className="mt-3 text-sm text-slate-700">{image.caption}</p>}{image.nearby_text && <p dir="auto" className="mt-2 line-clamp-3 text-sm text-slate-500">{image.nearby_text}</p>}<p className="mt-3 text-xs text-slate-500">{t('knowledge.details.associatedChunks', { count: image.associated_chunk_ids.length })}</p></article>)}</div> : <p className="mt-3 rounded-xl bg-slate-50 p-4 text-sm text-slate-500">{t('knowledge.details.noImages')}</p>}</section>
        </div>}
      </div>
    </div>
  </div>
}
