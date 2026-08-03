import { LayoutDashboard, Menu, Settings, Users } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import logo1 from '../assets/logo1.png'
import { useAuth } from '../contexts/AuthContext'

const navigation = [
  { to: '/admin/dashboard', label: 'لوحة التحكم', icon: LayoutDashboard },
  { to: '/admin/teachers', label: 'إدارة الأساتذة', icon: Users },
]

const pageMeta: Record<string, [string, string]> = {
  '/admin/dashboard': ['لوحة التحكم', 'نظرة شاملة على حسابات الأساتذة ونشاط المنصة'],
  '/admin/teachers': ['إدارة الأساتذة', 'إدارة حسابات الأساتذة ومتابعة حالتهم داخل المنصة'],
  '/admin/settings': ['إعدادات الحساب', 'إدارة معلومات الحساب والأمان'],
}

export function AdminLayout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [profileOpen, setProfileOpen] = useState(false)
  const profileRef = useRef<HTMLDivElement>(null)
  const [title, description] = pageMeta[pathname] ?? pageMeta['/admin/dashboard']

  useEffect(() => {
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) setProfileOpen(false)
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setProfileOpen(false); setDrawerOpen(false) }
    }
    document.addEventListener('mousedown', closeOnOutsideClick)
    document.addEventListener('keydown', closeOnEscape)
    return () => { document.removeEventListener('mousedown', closeOnOutsideClick); document.removeEventListener('keydown', closeOnEscape) }
  }, [])

  const goToSettings = () => { setProfileOpen(false); setDrawerOpen(false); navigate('/admin/settings') }
  const handleLogout = () => { setProfileOpen(false); setDrawerOpen(false); logout(); navigate('/login') }

  const sidebar = (
    <div className="flex h-full w-[280px] flex-col bg-[#0F172A] p-5 text-white">
      <div className="flex items-center justify-center px-5 py-6">
        <img src={logo1} alt="Mo3allimAI" className="h-auto max-h-16 w-auto max-w-[200px] object-contain" />
      </div>

      <nav className="mt-8 flex-1 space-y-2" aria-label="التنقل الإداري">
        {navigation.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} onClick={() => setDrawerOpen(false)} className={({ isActive }) => `flex items-center gap-3 rounded-xl px-4 py-3 font-semibold transition duration-200 ${isActive ? 'bg-emerald-700 text-white shadow-md shadow-emerald-950/20' : 'text-slate-300 hover:bg-white/10 hover:text-white'}`}>
            <Icon className="size-5" aria-hidden="true" />{label}
          </NavLink>
        ))}
      </nav>

      <div ref={profileRef} className="relative border-t border-white/10 pt-4">
        <button type="button" aria-expanded={profileOpen} aria-label="فتح قائمة الحساب" onClick={() => setProfileOpen((open) => !open)} className="flex w-full items-center gap-3 rounded-xl p-2 text-right transition hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-400">
          <span className="grid size-10 shrink-0 place-items-center rounded-full bg-emerald-100 font-bold text-emerald-800">{user?.full_name?.charAt(0)}</span>
          <span className="min-w-0 flex-1"><strong className="block truncate text-sm">{user?.full_name}</strong><small className="block truncate text-slate-400">{user?.email}</small></span>
          <Settings className="size-4 text-slate-300" aria-hidden="true" />
        </button>
        {profileOpen && <div className="absolute bottom-full right-0 z-50 mb-3 w-64 rounded-xl bg-white p-2 text-slate-800 shadow-xl"><button onClick={goToSettings} className="w-full rounded-lg p-2 text-right hover:bg-slate-50">الملف الشخصي</button><button onClick={goToSettings} className="w-full rounded-lg p-2 text-right hover:bg-slate-50">إعدادات الحساب</button><button onClick={handleLogout} className="w-full rounded-lg p-2 text-right text-rose-700 hover:bg-rose-50">تسجيل الخروج</button></div>}
      </div>
    </div>
  )

  return <div dir="rtl" className="min-h-screen bg-slate-50"><aside className="fixed inset-y-0 right-0 z-40 hidden w-[280px] lg:flex">{sidebar}</aside>{drawerOpen && <><button onClick={() => setDrawerOpen(false)} aria-label="إغلاق القائمة" className="fixed inset-0 z-40 bg-slate-950/40 lg:hidden" /><aside className="fixed inset-y-0 right-0 z-50 lg:hidden">{sidebar}</aside></>}<div className="min-h-screen lg:mr-[280px]"><header className="sticky top-0 z-30 border-b border-slate-200 bg-white px-4 py-4 sm:px-6"><div className="flex items-center gap-3"><button onClick={() => setDrawerOpen(true)} className="lg:hidden" aria-label="فتح القائمة"><Menu /></button><div><h1 className="font-extrabold text-slate-900">{title}</h1><p className="text-sm text-slate-500">{description}</p></div></div></header><main className="px-4 py-6 sm:px-6 lg:px-8"><div className="mx-auto w-full max-w-7xl"><Outlet /></div></main></div></div>
}
