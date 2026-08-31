import { Search, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { TeacherPageHeader } from '../../components/teacher/TeacherPageHeader'
import { TeacherToolCard } from '../../components/teacher/TeacherToolCard'
import { teacherTools } from '../../teacherTools'

type Category = 'all' | typeof teacherTools[number]['category']

export function TeacherToolsPage() {
  const { t } = useTranslation('teacher')
  const { t: toolT } = useTranslation('teacherToolsUi')
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState<Category>('all')
  const categories: Category[] = ['all', 'preparation', 'evaluation', 'activities', 'assistant']
  const normalizedQuery = query.trim().toLocaleLowerCase()
  const visibleTools = teacherTools.filter(tool =>
    (category === 'all' || tool.category === category)
    && (!normalizedQuery || t(`tools.items.${tool.title}`).toLocaleLowerCase().includes(normalizedQuery)),
  )

  return <div className="space-y-6">
    <div className="flex items-start gap-3"><span className="mt-1 flex size-9 items-center justify-center rounded-xl bg-emerald-50 text-emerald-700"><Sparkles className="size-4" /></span><TeacherPageHeader title={t('tools.title')} description={toolT('subtitle')} /></div>
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <label className="relative block max-w-2xl"><Search className="pointer-events-none absolute inset-y-0 my-auto size-5 text-slate-400 start-4" aria-hidden="true" /><input value={query} onChange={event => setQuery(event.target.value)} placeholder={t('tools.search')} className="h-12 w-full rounded-xl border border-slate-200 bg-white ps-11 pe-4 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100" /></label>
      <div className="mt-3 flex flex-wrap gap-2">{categories.map(item => <button key={item} type="button" onClick={() => setCategory(item)} className={`h-9 rounded-full border px-3.5 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-700 ${category === item ? 'border-[#065F46] bg-[#065F46] text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-emerald-200 hover:bg-emerald-50 hover:text-emerald-800'}`}>{t(`tools.categories.${item}`)}</button>)}</div>
    </div>
    {visibleTools.length ? <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3">{visibleTools.map(tool => <TeacherToolCard key={tool.id} tool={tool} />)}</div> : <div className="rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-10 text-center"><Search className="mx-auto size-5 text-slate-400" /><p className="mt-3 font-semibold text-slate-800">{t('tools.noResults')}</p><p className="mt-1 text-sm text-slate-500">{t('tools.noResultsDescription')}</p></div>}
  </div>
}
