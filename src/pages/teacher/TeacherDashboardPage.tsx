import { Plus, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { TeacherPageHeader } from '../../components/teacher/TeacherPageHeader'
import { TeacherToolCard } from '../../components/teacher/TeacherToolCard'
import { useAuth } from '../../contexts/AuthContext'
import { teacherTools } from '../../teacherTools'

export function TeacherDashboardPage() {
  const { user } = useAuth()
  const { t } = useTranslation('teacher')
  return (
    <div className="space-y-9">
      <section className="rounded-2xl bg-gradient-to-br from-emerald-50 to-white p-6 shadow-sm sm:p-8">
        <Sparkles className="size-5 text-[#C89B3C]" aria-hidden="true" />
        <div className="teacher-hero-actions mt-4 flex flex-wrap items-end justify-between gap-5">
          <TeacherPageHeader title={t('dashboard.welcome', { name: user?.full_name ?? '' })} description={t('dashboard.intro')} />
          <Link to="/teacher/tools" className="inline-flex h-11 items-center gap-2 rounded-xl bg-[#065F46] px-4 text-sm font-semibold text-white transition hover:bg-emerald-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-700 focus-visible:ring-offset-2"><Plus className="size-4" />{t('dashboard.create')}</Link>
        </div>
      </section>
      <section>
        <TeacherPageHeader title={t('dashboard.tools')} description={t('dashboard.toolsDescription')} />
        <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">{teacherTools.map(tool => <TeacherToolCard key={tool.id} tool={tool} />)}</div>
      </section>
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-bold text-slate-900">{t('dashboard.recent')}</h2>
        <p className="mt-1 text-sm text-slate-500">{t('dashboard.recentDescription')}</p>
        <div className="mt-5 rounded-xl border border-dashed border-slate-200 bg-slate-50 px-5 py-8 text-center"><p className="font-semibold text-slate-700">{t('dashboard.emptyTitle')}</p><p className="mt-1 text-sm text-slate-500">{t('dashboard.emptyDescription')}</p></div>
      </section>
    </div>
  )
}
