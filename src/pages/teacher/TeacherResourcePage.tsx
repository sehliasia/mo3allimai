import { Clock3, FileText, Library, LoaderCircle, Plus, Sparkles, Trash2, Upload } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { TFunction } from 'i18next'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { LanguageSwitcher } from '../../components/common/LanguageSwitcher'
import { TeacherPageHeader } from '../../components/teacher/TeacherPageHeader'
import { useAuth } from '../../contexts/AuthContext'
import { deleteTeacherLibraryItem, getTeacherLibrary, uploadTeacherDocument, type TeacherLibraryItem, type TeacherSavedResource } from '../../services/teacherLibraryService'
import { getTeacherHistory, type TeacherActivity } from '../../services/teacherHistoryService'

type ResourcePage = 'library' | 'history' | 'settings'

export function TeacherResourcePage({ page }: { page: ResourcePage }) {
  const { t, i18n } = useTranslation('teacher'); const { t: libraryT } = useTranslation('teacherLibrary'); const { t: commonT } = useTranslation('common'); const { user } = useAuth(); const navigate = useNavigate(); const isRTL = i18n.resolvedLanguage === 'ar'
  const [items, setItems] = useState<TeacherLibraryItem[]>([]); const [activities, setActivities] = useState<TeacherActivity[]>([]); const [loading, setLoading] = useState(page !== 'settings'); const [error, setError] = useState(false); const [query, setQuery] = useState(''); const [file, setFile] = useState<File | null>(null); const [uploadOpen, setUploadOpen] = useState(false); const [uploading, setUploading] = useState(false); const [selected, setSelected] = useState<TeacherLibraryItem | null>(null); const inputRef = useRef<HTMLInputElement>(null)
  const load = async () => { setLoading(true); setError(false); try { if (page === 'library') setItems(await getTeacherLibrary()); else if (page === 'history') setActivities(await getTeacherHistory()) } catch { setError(true) } finally { setLoading(false) } }
  useEffect(() => { if (page !== 'settings') void load() }, [page])
  if (page === 'settings') return <div className="space-y-7"><TeacherPageHeader title={t('settings.title')} description={t('settings.description')} /><div className="grid gap-5 lg:grid-cols-2"><section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><h2 className="text-lg font-bold text-slate-900">{t('settings.profile')}</h2><dl className="mt-5 space-y-4 text-sm"><div><dt className="text-slate-500">{t('settings.displayName')}</dt><dd className="mt-1 font-medium text-slate-900">{user?.full_name || t('settings.notAvailable')}</dd></div><div><dt className="text-slate-500">{t('settings.email')}</dt><dd dir="ltr" className="mt-1 font-medium text-slate-900">{user?.email || t('settings.notAvailable')}</dd></div></dl></section><section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><h2 className="text-lg font-bold text-slate-900">{t('settings.interface')}</h2><p className="mt-2 text-sm leading-6 text-slate-500">{t('settings.interfaceDescription')}</p><div className="mt-5"><LanguageSwitcher /></div></section></div></div>
  const matches = (item: TeacherLibraryItem) => `${item.title} ${item.kind === 'document' ? item.original_filename : item.theme || ''}`.toLowerCase().includes(query.toLowerCase())
  const documents = items.filter(item => item.kind === 'document' && matches(item))
  const creations = items.filter(item => item.kind === 'creation' && matches(item))
  const upload = async () => { if (!file || uploading) return; if (!['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'].includes(file.type) || file.size > 10 * 1024 * 1024) { setError(true); return }; setUploading(true); try { await uploadTeacherDocument(file); setFile(null); setUploadOpen(false); await load() } catch { setError(true) } finally { setUploading(false) } }
  if (page === 'history') return <div className="space-y-6"><TeacherPageHeader title={t('history.title')} description={t('history.description')} />{loading ? <LoaderCircle className="animate-spin text-emerald-700" /> : error ? <button onClick={() => void load()} className="rounded-xl border bg-white px-4 py-3">{t('common.retry', 'Réessayer')}</button> : activities.length ? <div className="divide-y rounded-2xl border border-slate-200 bg-white">{activities.map(activity => <div key={activity.id} className="flex gap-3 p-4"><Clock3 className="mt-1 size-5 text-emerald-700" /><div><p className="font-semibold text-slate-800">{activity.activity_type === 'document_uploaded' ? t('history.documentUploaded', 'Document importé') : t('history.resourceSaved', 'Ressource sauvegardée')}</p><p className="text-sm text-slate-600">{activity.title}</p><p className="mt-1 text-xs text-slate-400">{new Date(activity.created_at).toLocaleDateString()}</p></div></div>)}</div> : <Empty icon={Clock3} title={t('history.emptyTitle')} description={t('history.emptyDescription')} />}</div>
  return <div className="space-y-6">
    <TeacherPageHeader title={t('library.title')} description={t('library.description')} action={<button onClick={() => setUploadOpen(true)} className="inline-flex h-11 items-center gap-2 rounded-xl bg-[#065F46] px-4 text-sm font-semibold text-white"><Plus className="size-4" />{libraryT('importDocument')}</button>} />
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"><input dir={isRTL ? 'rtl' : 'ltr'} value={query} onChange={event => setQuery(event.target.value)} placeholder={libraryT('searchPlaceholder')} className="h-11 w-full rounded-xl border px-3 text-start text-sm outline-none focus:ring-2 focus:ring-emerald-100" /></div>
    {loading ? <LoaderCircle className="animate-spin text-emerald-700" /> : error ? <button onClick={() => void load()} className="rounded-xl border bg-white px-4 py-3">{commonT('retry')}</button> : (documents.length === 0 && creations.length === 0) ? <Empty icon={Library} title={libraryT('emptyTitle')} description={libraryT('emptyDescription')} /> : <>
      <LibrarySection libraryT={libraryT} title={libraryT('myDocuments')} icon={FileText} total={documents.length} items={documents} onSelect={setSelected} />
      <LibrarySection libraryT={libraryT} title={libraryT('myCreations')} icon={Sparkles} total={creations.length} items={creations} onSelect={setSelected} onOpen={item => navigate(`/teacher/tools/${item.resource_type === 'lesson-plan' ? 'lesson-plan' : item.resource_type === 'activity' ? 'activity' : item.resource_type === 'course' ? 'lesson' : item.resource_type === 'exercises' ? 'exercises' : 'lesson-plan'}?resourceId=${item.id}`)} />
    </>}
    {uploadOpen && <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/40 p-4"><div className="w-full max-w-lg rounded-2xl bg-white p-6"><h2 className="text-lg font-bold">{libraryT('importDocument')}</h2><button onClick={() => inputRef.current?.click()} className="mt-5 flex w-full flex-col items-center rounded-xl border-2 border-dashed p-8 text-slate-600"><Upload className="size-7 text-emerald-700" />{file ? <span className="mt-3">{file.name}</span> : <span className="mt-3">{libraryT('dropDocument')}</span>}<span className="mt-1 text-xs">PDF, DOCX, TXT · 10 MB max</span></button><input ref={inputRef} type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={event => setFile(event.target.files?.[0] || null)} /><div className="mt-5 flex justify-end gap-2"><button onClick={() => setUploadOpen(false)} className="rounded-xl border px-4 py-2">{commonT('cancel')}</button><button disabled={!file || uploading} onClick={() => void upload()} className="rounded-xl bg-[#065F46] px-4 py-2 text-white disabled:bg-slate-300">{uploading ? '…' : libraryT('importDocument')}</button></div></div></div>}
    {selected && <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/40 p-4"><div className="rounded-2xl bg-white p-6"><p className="font-bold">{libraryT('deleteConfirm')}</p><div className="mt-4 flex justify-end gap-2"><button onClick={() => setSelected(null)}>{commonT('cancel')}</button><button onClick={async () => { await deleteTeacherLibraryItem(selected); setSelected(null); void load() }} className="text-rose-600">{libraryT('delete')}</button></div></div></div>}
  </div>
}

function LibrarySection({ title, icon: Icon, total, items, libraryT, onSelect, onOpen }: {
  title: string; icon: typeof FileText; total: number; items: TeacherLibraryItem[]; libraryT: TFunction;
  onSelect: (item: TeacherLibraryItem) => void; onOpen?: (item: TeacherSavedResource) => void;
}) {
  if (items.length === 0) return null
  return <section className="space-y-3">
    <div className="flex items-center justify-between gap-3">
      <h2 className="flex items-center gap-2 text-lg font-bold text-slate-900"><span className="flex size-8 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700"><Icon className="size-4" /></span>{title}</h2>
      <span className="text-xs font-medium text-slate-400">{total} {total > 1 ? total : ''}</span>
    </div>
    {items.length ? <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">{items.map(item => <LibraryCard key={`${item.kind}-${item.id}`} item={item} libraryT={libraryT} onDelete={() => onSelect(item)} onOpen={onOpen} />)}</div> : <p className="rounded-xl border border-dashed border-slate-300 p-4 text-center text-sm text-slate-400">{libraryT('emptyTitle')}</p>}
  </section>
}

function LibraryCard({ item, libraryT, onDelete, onOpen }: { item: TeacherLibraryItem; libraryT: TFunction; onDelete: () => void; onOpen?: (item: TeacherSavedResource) => void }) {
  return <article className="rounded-2xl border border-slate-200 bg-white p-5 text-start shadow-sm"><FileText className="size-6 text-emerald-700" /><h2 dir="auto" className="mt-4 font-semibold text-slate-900">{item.kind === 'document' ? item.original_filename : item.title}</h2>{item.kind === 'document' ? <p dir="auto" className="mt-1 text-sm text-slate-500">{item.mime_type.split('/').pop()?.toUpperCase()} · {(item.file_size / 1024 / 1024).toFixed(1)} MB</p> : <p dir="auto" className="mt-1 text-sm text-slate-500">{item.resource_type === 'lesson-plan' ? '📚 Fiche pédagogique' : item.resource_type === 'activity' ? '🎯 Activité' : item.resource_type === 'course' ? '📚 Cours' : item.resource_type === 'exercises' ? '📝 Exercices' : item.resource_type}{item.cefr_level ? ` · ${item.cefr_level}` : ''}</p>}{item.kind === 'creation' && <><p className="mt-1 text-xs text-slate-400">Créée le : {new Date(item.created_at).toLocaleDateString()}<br/>Modifiée le : {new Date(item.updated_at).toLocaleDateString()}</p><button onClick={() => onOpen?.(item)} className="mt-4 me-3 text-sm font-semibold text-emerald-700">Ouvrir</button></>}<button onClick={onDelete} className="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-rose-600"><Trash2 className="size-4" />{libraryT('delete')}</button></article>
}

function Empty({ icon: Icon, title, description }: { icon: typeof Library; title: string; description: string }) {
  return <section className="rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-10 text-center"><Icon className="mx-auto size-8 text-slate-400" /><h2 className="mt-4 font-bold">{title}</h2><p className="mt-2 text-sm text-slate-500">{description}</p></section>
}
