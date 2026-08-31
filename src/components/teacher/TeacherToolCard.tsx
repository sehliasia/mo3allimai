import { ArrowLeft, ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { teacherTools } from '../../teacherTools'

export function TeacherToolCard({ tool }: { tool: typeof teacherTools[number] }) {
  const { t, i18n } = useTranslation('teacher')
  const { t: toolT } = useTranslation('teacherToolsUi')
  const Icon = tool.icon
  const Arrow = i18n.resolvedLanguage === 'ar' ? ArrowLeft : ArrowRight
  return <Link to={tool.route} className="teacher-tool-card group flex h-full min-h-[230px] flex-col rounded-2xl border border-slate-200 bg-white p-5 text-start shadow-sm transition duration-200 hover:-translate-y-0.5 hover:border-emerald-300 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-700 focus-visible:ring-offset-2"><div className="flex items-start justify-between gap-3"><span className="teacher-tool-icon flex size-12 items-center justify-center rounded-xl bg-emerald-50 text-emerald-700 transition group-hover:bg-emerald-100"><Icon className="size-5" /></span><span className="teacher-tool-badge rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600">{toolT(`categories.${tool.category}`)}</span></div><h3 className="mt-5 text-lg font-semibold text-slate-950">{t(`tools.items.${tool.title}`)}</h3><p className="mt-2 text-sm leading-6 text-slate-600">{t(`tools.items.${tool.description}`)}</p><span className="teacher-tool-cta mt-auto flex items-center gap-2 pt-6 text-sm font-semibold text-emerald-800"><span>{toolT(`actions.${tool.title}`)}</span><Arrow className="size-4 transition-transform group-hover:translate-x-1 rtl:group-hover:-translate-x-1" /></span></Link>
}
