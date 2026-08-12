import { Database, LayoutDashboard, Menu, Settings, Users } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import logo1 from '../assets/logo1.png'
import { LanguageSwitcher } from '../components/common/LanguageSwitcher'
import { useAuth } from '../contexts/AuthContext'

const links = [
  { to: '/admin/dashboard', labelKey: 'sidebar.dashboard', icon: LayoutDashboard },
  { to: '/admin/teachers', labelKey: 'sidebar.teachers', icon: Users },
  { to: '/admin/knowledge-base', labelKey: 'sidebar.knowledge', icon: Database },
] as const

const pageKeys: Record<string, 'dashboard' | 'teachers' | 'knowledge' | 'settings'> = {
  '/admin/dashboard': 'dashboard',
  '/admin/teachers': 'teachers',
  '/admin/knowledge-base': 'knowledge',
  '/admin/settings': 'settings',
}

export function AdminLayout() {
  const { user, logout } = useAuth()
  const { t, i18n } = useTranslation('admin')
  const navigate = useNavigate()
  const location = useLocation()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [profileOpen, setProfileOpen] = useState(false)
  const profileRef = useRef<HTMLDivElement>(null)
  const isRTL = i18n.resolvedLanguage === 'ar'
  const pageKey = pageKeys[location.pathname] ?? 'dashboard'
  const sidebarSide = isRTL ? 'right-0 border-l' : 'left-0 border-r'
  const contentOffset = isRTL ? 'lg:mr-[280px]' : 'lg:ml-[280px]'
  const popoverSide = isRTL ? 'right-0' : 'left-0'

  useEffect(() => {
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) setProfileOpen(false)
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setProfileOpen(false); setDrawerOpen(false) }
    }
    document.addEventListener('mousedown', closeOnOutsideClick)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('mousedown', closeOnOutsideClick)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [])

  const goToSettings = () => { setProfileOpen(false); setDrawerOpen(false); navigate('/admin/settings') }
  const signOut = () => { setProfileOpen(false); setDrawerOpen(false); logout(); navigate('/login') }

  const sidebar = (
    <div className="flex h-full w-[280px] flex-col bg-white">
      <div dir="ltr" className="flex items-center justify-center gap-3 border-b border-slate-200 px-5 py-5">
        <img src={logo1} alt="Mo3allimAI" className="h-12 w-12 shrink-0 object-contain" />
        <div className="min-w-0 text-left"><div className="whitespace-nowrap text-2xl font-extrabold tracking-tight"><span className="text-[#065F46]">Mo3allim</span><span className="text-[#C89B3C]">AI</span></div><p className="mt-0.5 text-center text-xs font-medium text-slate-500">{t('brand.portal')}</p></div>
      </div>
      <nav className="flex-1 space-y-2 p-4">
        {links.map(({ to, labelKey, icon: Icon }) => <NavLink key={to} to={to} onClick={() => setDrawerOpen(false)} className={({ isActive }) => `flex items-center gap-3 rounded-xl px-4 py-3 text-start transition duration-200 ${isActive ? 'bg-emerald-600 text-white shadow-sm' : 'text-slate-700 hover:bg-slate-100 hover:text-slate-950'}`}><Icon className="size-5" />{t(labelKey)}</NavLink>)}
      </nav>
      <div ref={profileRef} className="relative border-t border-slate-200 p-4">
        <button type="button" onClick={() => setProfileOpen(open => !open)} aria-expanded={profileOpen} className="flex w-full items-center gap-3 rounded-xl p-2 text-start hover:bg-slate-100">
          <span className="grid size-10 place-items-center rounded-full bg-emerald-100 text-emerald-800">{user?.full_name?.[0]}</span><span className="min-w-0 flex-1"><b className="block truncate text-sm text-slate-900">{user?.full_name}</b><small className="block truncate text-xs text-slate-500">{user?.email}</small></span><Settings className="size-4 text-slate-500" />
        </button>
        {profileOpen && <div className={`absolute bottom-full z-50 mb-3 w-64 rounded-xl border border-slate-200 bg-white p-2 shadow-xl ${popoverSide}`}><button type="button" onClick={goToSettings} className="w-full rounded-lg p-2 text-start text-slate-700 hover:bg-slate-50">{t('profile.profile')}</button><button type="button" onClick={goToSettings} className="w-full rounded-lg p-2 text-start text-slate-700 hover:bg-slate-50">{t('profile.settings')}</button><button type="button" onClick={signOut} className="w-full rounded-lg p-2 text-start text-red-600 hover:bg-red-50">{t('profile.logout')}</button></div>}
      </div>
    </div>
  )

  return <div className="min-h-screen bg-slate-50">
    <aside className={`fixed inset-y-0 z-40 hidden w-[280px] border-slate-200 bg-white lg:flex ${sidebarSide}`}>{sidebar}</aside>
    {drawerOpen && <><button type="button" onClick={() => setDrawerOpen(false)} aria-label={t('menu.open')} className="fixed inset-0 z-40 bg-slate-950/40 lg:hidden" /><aside className={`fixed inset-y-0 z-50 lg:hidden ${sidebarSide}`}>{sidebar}</aside></>}
    <div className={`min-h-screen ${contentOffset}`}><header className="sticky top-0 z-30 flex items-center justify-between gap-4 border-b bg-white px-4 py-4"><div><button type="button" onClick={() => setDrawerOpen(true)} className="lg:hidden" aria-label={t('menu.open')}><Menu /></button><h1 className="text-start font-extrabold text-slate-900">{t(`page.${pageKey}.title`)}</h1><p className="text-start text-sm text-slate-500">{t(`page.${pageKey}.description`)}</p></div><LanguageSwitcher /></header><main className="px-4 py-6"><div className="mx-auto max-w-7xl"><Outlet /></div></main></div>
  </div>
}
