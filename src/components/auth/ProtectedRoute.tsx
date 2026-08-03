import type { ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { useAuth } from '../../contexts/AuthContext'

export function ProtectedRoute({ children, allowedRoles }: { children: ReactNode; allowedRoles?: Array<'teacher' | 'admin'> }) {
  const { user, isAuthenticated, isLoading, serviceError, loadCurrentUser } = useAuth(); const location = useLocation()
  if (isLoading) return <main className="grid min-h-screen place-items-center bg-slate-50"><Loader2 className="size-7 animate-spin text-emerald-700" aria-label="جارٍ التحقق من الجلسة" /></main>
  if (serviceError) return <main dir="rtl" className="grid min-h-screen place-items-center bg-slate-50 p-4"><section className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 text-center shadow-sm"><p className="text-slate-700">{serviceError}</p><button onClick={() => void loadCurrentUser()} className="mt-5 rounded-xl bg-[#065F46] px-5 py-3 font-bold text-white hover:bg-emerald-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700">إعادة المحاولة</button></section></main>
  if (!isAuthenticated || !user) return <Navigate to="/login" replace state={{ from: location.pathname }} />
  if (allowedRoles && !allowedRoles.includes(user.role)) return <Navigate to={user.role === 'admin' ? '/admin/dashboard' : '/teacher/dashboard'} replace />
  return <>{children}</>
}
