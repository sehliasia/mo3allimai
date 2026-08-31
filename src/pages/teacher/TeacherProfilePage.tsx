import { UserRound } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { TeacherPageHeader } from '../../components/teacher/TeacherPageHeader'
import { useAuth } from '../../contexts/AuthContext'

export function TeacherProfilePage() {
  const { user } = useAuth()
  const { t } = useTranslation('teacherProfile')
  const initials = user?.full_name?.trim().split(/\s+/).slice(0, 2).map(part => part[0]).join('').toUpperCase() || 'M'
  return <div className="space-y-7"><TeacherPageHeader title={t('title')} description={t('description')} /><section className="max-w-2xl rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><div className="flex items-center gap-4"><span className="flex size-14 items-center justify-center rounded-2xl bg-emerald-100 text-lg font-bold text-[#065F46]">{initials}</span><div><h2 className="text-lg font-bold text-slate-900">{user?.full_name || t('teacher')}</h2><p dir="ltr" className="mt-1 text-sm text-slate-500">{user?.email}</p></div></div><div className="mt-7 border-t border-slate-100 pt-5"><h3 className="flex items-center gap-2 font-semibold text-slate-900"><UserRound className="size-4 text-emerald-700" />{t('personalInformation')}</h3><dl className="mt-5 grid gap-5 sm:grid-cols-2"><div><dt className="text-sm text-slate-500">{t('name')}</dt><dd className="mt-1 font-medium text-slate-900">{user?.full_name || t('notAvailable')}</dd></div><div><dt className="text-sm text-slate-500">{t('email')}</dt><dd dir="ltr" className="mt-1 font-medium text-slate-900">{user?.email || t('notAvailable')}</dd></div><div><dt className="text-sm text-slate-500">{t('role')}</dt><dd className="mt-1 font-medium text-slate-900">{t('teacher')}</dd></div></dl></div></section></div>
}
