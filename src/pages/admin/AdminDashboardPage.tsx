import { UserCheck, UserPlus, Users, UserX } from 'lucide-react'
import { useEffect, useState } from 'react'
import { StatCard } from '../../components/admin/StatCard'
import { getAdminStatistics } from '../../services/adminService'

export function AdminDashboardPage() {
  const [stats, setStats] = useState<{ total_teachers: number; active_teachers: number; inactive_teachers: number; new_teachers_this_month: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)
  const load = () => { setLoading(true); setError(false); getAdminStatistics().then(setStats).catch(() => setError(true)).finally(() => setLoading(false)) }
  useEffect(load, [])
  return <section className="space-y-6" dir="rtl">{error ? <button onClick={load} className="rounded-xl bg-emerald-700 px-5 py-3 text-white">إعادة المحاولة</button> : <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4"><StatCard title="إجمالي الأساتذة" value={stats?.total_teachers} description="جميع حسابات الأساتذة" icon={Users} variant="navy" isLoading={loading} /><StatCard title="الحسابات النشطة" value={stats?.active_teachers} description="حسابات متاحة حالياً" icon={UserCheck} variant="emerald" isLoading={loading} /><StatCard title="الحسابات الموقوفة" value={stats?.inactive_teachers} description="حسابات غير مفعلة" icon={UserX} variant="amber" isLoading={loading} /><StatCard title="التسجيلات الجديدة هذا الشهر" value={stats?.new_teachers_this_month} description="منذ بداية الشهر" icon={UserPlus} variant="gold" isLoading={loading} /></div>}</section>
}
