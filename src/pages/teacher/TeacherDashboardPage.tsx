import { ArrowLeft, ArrowRight, BookOpen, ClipboardList, FileCheck, Shapes, Sparkles } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { TeacherPageHeader } from '../../components/teacher/TeacherPageHeader'
import { TeacherToolCard } from '../../components/teacher/TeacherToolCard'
import { useAuth } from '../../contexts/AuthContext'
import { teacherTools } from '../../teacherTools'
import { getTeacherLibrary, type TeacherLibraryItem, type TeacherSavedResource } from '../../services/teacherLibraryService'

const RESOURCE_STATS: Record<string, string> = {
  course: 'dashboard.courses',
  'lesson-plan': 'dashboard.lessonPlans',
  activity: 'dashboard.activities',
  exam: 'dashboard.evaluations',
  assessment: 'dashboard.evaluations',
}

export function TeacherDashboardPage() {
  const { user } = useAuth()
  const { t, i18n } = useTranslation('teacher')
  const isRTL = i18n.resolvedLanguage === 'ar'
  const Arrow = isRTL ? ArrowLeft : ArrowRight

  const [items, setItems] = useState<TeacherLibraryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(false)
    try {
      setItems(await getTeacherLibrary())
    } catch {
      setError(true)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const resources = useMemo(() => items.filter((item): item is TeacherSavedResource => item.kind === 'creation'), [items])
  const stats = useMemo(() => {
    const counters: Record<string, number> = {}
    for (const resource of resources) {
      const key = RESOURCE_STATS[resource.resource_type] ?? 'other'
      counters[key] = (counters[key] ?? 0) + 1
      counters.total = (counters.total ?? 0) + 1
    }
    return {
      courses: counters['dashboard.courses'] ?? 0,
      lessonPlans: counters['dashboard.lessonPlans'] ?? 0,
      activities: counters['dashboard.activities'] ?? 0,
      evaluations: counters['dashboard.evaluations'] ?? 0,
      total: counters.total ?? 0,
    }
  }, [resources])

  const recent = useMemo(
    () => [...resources].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()).slice(0, 5),
    [resources],
  )

  return (
    <div className="space-y-10">
      <section className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-emerald-600 via-emerald-700 to-teal-800 p-6 text-white shadow-sm sm:p-8">
        <Sparkles className="absolute end-6 top-6 size-6 text-amber-300/80" aria-hidden="true" />
        <div className="max-w-2xl">
          <p className="text-sm font-semibold uppercase tracking-wide text-emerald-100/90">{t('dashboard.overview')}</p>
          <h1 className="mt-2 text-3xl font-extrabold tracking-tight sm:text-4xl">{t('dashboard.welcome', { name: user?.full_name?.split(' ')[0] || t('dashboard.overview') })}</h1>
          <p className="mt-3 max-w-xl text-sm leading-6 text-emerald-50/90 sm:text-base">{t('dashboard.intro')}</p>
        </div>
      </section>

      <StatsSection loading={loading} error={error} onRetry={() => void load()} stats={stats} />

      <section>
        <TeacherPageHeader title={t('dashboard.tools')} description={t('dashboard.toolsDescription')} />
        <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">{teacherTools.map(tool => <TeacherToolCard key={tool.id} tool={tool} />)}</div>
      </section>

      <RecentSection loading={loading} error={error} onRetry={() => void load()} recent={recent} />
    </div>
  )
}

function StatsSection({ loading, error, onRetry, stats }: { loading: boolean; error: boolean; onRetry: () => void; stats: { courses: number; lessonPlans: number; activities: number; evaluations: number; total: number } }) {
  const { t } = useTranslation('teacher')
  if (loading) {
    return (
      <section>
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-44" />
        </div>
        <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {[0, 1, 2, 3].map(index => <Skeleton key={index} className="h-36 rounded-2xl" />)}
        </div>
      </section>
    )
  }
  if (error) {
    return (
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-slate-900">{t('dashboard.errorTitle')}</h2>
            <p className="mt-1 text-sm text-slate-500">{t('dashboard.errorDescription')}</p>
          </div>
          <button type="button" onClick={onRetry} className="inline-flex h-10 items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 text-sm font-semibold text-emerald-800 transition hover:border-emerald-300 hover:bg-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-700">{t('dashboard.retry')}</button>
        </div>
      </section>
    )
  }
  const cards: { labelKey: string; value: number; icon: typeof BookOpen; tint: string }[] = [
    { labelKey: 'dashboard.courses', value: stats.courses, icon: BookOpen, tint: 'bg-emerald-50 text-emerald-700' },
    { labelKey: 'dashboard.lessonPlans', value: stats.lessonPlans, icon: ClipboardList, tint: 'bg-amber-50 text-amber-700' },
    { labelKey: 'dashboard.activities', value: stats.activities, icon: Shapes, tint: 'bg-sky-50 text-sky-700' },
    { labelKey: 'dashboard.evaluations', value: stats.evaluations, icon: FileCheck, tint: 'bg-rose-50 text-rose-700' },
  ]
  return (
    <section>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <TeacherPageHeader title={t('dashboard.overview')} description={t('dashboard.statistics')} />
        <p className="text-sm font-medium text-slate-500">{t('dashboard.total')} · {stats.total}</p>
      </div>
      <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {cards.map(card => (
          <article key={card.labelKey} className="flex flex-col rounded-2xl border border-slate-200 bg-white p-5 text-start shadow-sm transition duration-200 hover:-translate-y-0.5 hover:border-emerald-200 hover:shadow-md">
            <span className={`flex size-12 items-center justify-center rounded-xl ${card.tint}`}><card.icon className="size-5" /></span>
            <h3 className="mt-4 text-sm font-semibold text-slate-500">{t(card.labelKey)}</h3>
            <p className="mt-1 text-4xl font-extrabold tracking-tight text-slate-900">{card.value}</p>
            <p className="mt-2 text-xs text-slate-400">{t('dashboard.createdCount', { count: card.value })}</p>
          </article>
        ))}
      </div>
    </section>
  )
}

function RecentSection({ loading, error, onRetry, recent }: { loading: boolean; error: boolean; onRetry: () => void; recent: TeacherSavedResource[] }) {
  const { t, i18n } = useTranslation('teacher')
  const isRTL = i18n.resolvedLanguage === 'ar'
  const Arrow = isRTL ? ArrowLeft : ArrowRight
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <TeacherPageHeader title={t('dashboard.recentActivities')} description={t('dashboard.recentActivitiesDescription')} />
        <Link to="/teacher/library" className="inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold text-emerald-800 transition hover:bg-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-700"><span>{t('dashboard.viewLibrary')}</span><Arrow className="size-4" /></Link>
      </div>
      <div className="mt-5">
        {loading ? (
          <div className="space-y-3">{[0, 1, 2].map(index => <Skeleton key={index} className="h-16 rounded-xl" />)}</div>
        ) : error ? (
          <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-5 py-6 text-center">
            <p className="font-semibold text-slate-700">{t('dashboard.errorTitle')}</p>
            <button type="button" onClick={onRetry} className="mt-3 inline-flex h-10 items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 text-sm font-semibold text-emerald-800 transition hover:border-emerald-300 hover:bg-emerald-50">{t('dashboard.retry')}</button>
          </div>
        ) : recent.length ? (
          <ul className="divide-y divide-slate-100 overflow-hidden rounded-xl border border-slate-200">
            {recent.map(resource => <RecentRow key={resource.id} resource={resource} />)}
          </ul>
        ) : (
          <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-5 py-8 text-center">
            <span className="mx-auto flex size-12 items-center justify-center rounded-xl bg-emerald-50 text-emerald-700"><BookOpen className="size-5" /></span>
            <h3 className="mt-4 font-semibold text-slate-800">{t('dashboard.recentEmptyTitle')}</h3>
            <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">{t('dashboard.recentEmptyDescription')}</p>
            <Link to="/teacher/tools/lesson" className="mt-4 inline-flex h-10 items-center gap-2 rounded-xl bg-[#065F46] px-4 text-sm font-semibold text-white transition hover:bg-emerald-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-700"><span>{t('dashboard.createCourse')}</span><Arrow className="size-4" /></Link>
          </div>
        )}
      </div>
    </section>
  )
}

function RecentRow({ resource }: { resource: TeacherSavedResource }) {
  const { t, i18n } = useTranslation('teacher')
  const isRTL = i18n.resolvedLanguage === 'ar'
  const Arrow = isRTL ? ArrowLeft : ArrowRight
  const targetRoute = resource.resource_type === 'lesson-plan'
    ? '/teacher/tools/lesson-plan'
    : resource.resource_type === 'activity'
      ? '/teacher/tools/activity'
      : '/teacher/tools/lesson'
  const Icon = resource.resource_type === 'lesson-plan'
    ? ClipboardList
    : resource.resource_type === 'activity'
      ? Shapes
      : resource.resource_type === 'exam' || resource.resource_type === 'assessment'
        ? FileCheck
        : BookOpen
  const typeLabel = resource.resource_type === 'lesson-plan'
    ? t('dashboard.lessonPlans')
    : resource.resource_type === 'activity'
      ? t('dashboard.activities')
      : resource.resource_type === 'exam' || resource.resource_type === 'assessment'
        ? t('dashboard.evaluations')
        : t('dashboard.courses')
  return (
    <li>
      <Link to={`${targetRoute}?resourceId=${resource.id}`} className="group flex items-center gap-4 p-4 text-start transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-emerald-700 hover:bg-slate-50">
        <span className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-700"><Icon className="size-5" /></span>
        <span className="min-w-0 flex-1">
          <span dir="auto" className="block truncate font-semibold text-slate-900">{resource.title}</span>
          <span className="mt-0.5 block truncate text-sm text-slate-500">
            {typeLabel}{resource.cefr_level ? ` · ${resource.cefr_level}` : ''}{resource.theme ? ` · ${resource.theme}` : ''}
          </span>
        </span>
        <span className="shrink-0 text-xs font-medium text-slate-400">{formatRelativeTime(resource.created_at, t)}</span>
        <Arrow className="shrink-0 size-4 text-slate-300 transition group-hover:text-emerald-700" />
      </Link>
    </li>
  )
}

function formatRelativeTime(date: string, t: (key: string, options?: Record<string, unknown>) => string): string {
  const diffSeconds = Math.max(0, (Date.now() - new Date(date).getTime()) / 1000)
  const minutes = Math.floor(diffSeconds / 60)
  if (minutes < 1) return t('dashboard.justNow')
  if (minutes < 60) return t('dashboard.minutesAgo', { count: minutes })
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return t('dashboard.hoursAgo', { count: hours })
  const days = Math.floor(hours / 24)
  if (days === 1) return t('dashboard.yesterday')
  return t('dashboard.daysAgo', { count: days })
}

function Skeleton({ className }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-slate-200/70 ${className ?? ''}`} aria-hidden="true" />
}
