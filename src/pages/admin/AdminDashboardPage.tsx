import { UserCheck, UserPlus, Users, UserX } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { StatCard } from '../../components/admin/StatCard'
import { getAdminStatistics } from '../../services/adminService'

export function AdminDashboardPage() {
  const { t } = useTranslation('admin')
  const [stats, setStats] = useState<{ total_teachers: number; active_teachers: number; inactive_teachers: number; new_teachers_this_month: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)
  const load = () => { setLoading(true); setError(false); getAdminStatistics().then(setStats).catch(() => setError(true)).finally(() => setLoading(false)) }
  useEffect(load, [])
  return <section className="space-y-6">{error ? <button onClick={load} className="rounded-xl bg-emerald-700 px-5 py-3 text-white">{t('dashboard.retry')}</button> : <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4"><StatCard title={t('dashboard.total')} value={stats?.total_teachers} description={t('dashboard.totalDescription')} icon={Users} variant="navy" isLoading={loading} /><StatCard title={t('dashboard.active')} value={stats?.active_teachers} description={t('dashboard.activeDescription')} icon={UserCheck} variant="emerald" isLoading={loading} /><StatCard title={t('dashboard.suspended')} value={stats?.inactive_teachers} description={t('dashboard.suspendedDescription')} icon={UserX} variant="amber" isLoading={loading} /><StatCard title={t('dashboard.new')} value={stats?.new_teachers_this_month} description={t('dashboard.newDescription')} icon={UserPlus} variant="gold" isLoading={loading} /></div>}</section>
}
