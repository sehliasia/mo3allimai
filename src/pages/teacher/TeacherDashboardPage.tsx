import { Bot, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../../contexts/AuthContext'

const toolKeys = ['lessonPlan', 'lesson', 'test', 'exercises', 'assessment', 'activity', 'summary', 'assistant'] as const

export function TeacherDashboardPage() {
  const { user } = useAuth()
  const { t } = useTranslation('teacher')
  return <div className="space-y-8"><section className="rounded-2xl border border-emerald-100 bg-emerald-50 p-7 text-start"><Sparkles className="text-emerald-700" /><h1 className="mt-3 text-3xl font-extrabold">{t('dashboard.welcome', { name: user?.full_name ?? '' })}</h1><p className="mt-2 text-slate-600">{t('dashboard.intro')}</p></section><section><h2 className="text-start text-2xl font-extrabold">{t('dashboard.tools')}</h2><div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{toolKeys.map(key => <Link key={key} to="/teacher/tools" className="rounded-2xl border bg-white p-5 text-start shadow-sm transition hover:-translate-y-1"><Bot className="text-emerald-700" /><h3 className="mt-3 font-bold">{t(`dashboard.toolsList.${key}`)}</h3><p className="mt-2 text-sm text-slate-500">{t('dashboard.toolDescription')}</p></Link>)}</div></section><section className="rounded-2xl border bg-white p-6 text-start"><h2 className="font-extrabold">{t('dashboard.assistantTitle')}</h2><p className="mt-2 text-slate-500">{t('dashboard.assistantDescription')}</p><Link to="/teacher/assistant" className="mt-4 inline-block rounded-xl bg-emerald-700 px-5 py-3 text-white">{t('dashboard.startChat')}</Link></section></div>
}
