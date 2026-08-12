import { Bot, History, LayoutDashboard, Library, Menu, Settings, WandSparkles } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import logo1 from '../assets/logo1.png'
import { LanguageSwitcher } from '../components/common/LanguageSwitcher'
import { useAuth } from '../contexts/AuthContext'

const links = [
  ['/teacher/dashboard', 'sidebar.dashboard', LayoutDashboard],
  ['/teacher/assistant', 'sidebar.assistant', Bot],
  ['/teacher/tools', 'sidebar.aiTools', WandSparkles],
  ['/teacher/library', 'sidebar.library', Library],
  ['/teacher/history', 'sidebar.history', History],
  ['/teacher/settings', 'sidebar.settings', Settings],
] as const

export function TeacherLayout() {
  const { user, logout } = useAuth()
  const { t, i18n } = useTranslation('teacher')
  const navigate = useNavigate()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const isRTL = i18n.resolvedLanguage === 'ar'
  const sidebarSide = isRTL ? 'right-0 border-l' : 'left-0 border-r'
  const contentOffset = isRTL ? 'lg:mr-[280px]' : 'lg:ml-[280px]'
  const signOut = () => { logout(); navigate('/login') }

  const sidebar = (
    <div className="flex h-full w-[280px] flex-col bg-white">
      <div dir="ltr" className="flex items-center gap-3 border-b p-5"><img src={logo1} alt="Mo3allimAI" className="size-10 object-contain" /><div><b className="text-[#065F46]">Mo3allim<span className="text-[#C89B3C]">AI</span></b><p className="text-xs text-slate-500">{t('brand.space')}</p></div></div>
      <nav className="flex-1 space-y-2 p-4">{links.map(([to, labelKey, Icon]) => <NavLink key={to} to={to} onClick={() => setDrawerOpen(false)} className={({ isActive }) => `flex items-center gap-3 rounded-xl px-4 py-3 text-start ${isActive ? 'bg-emerald-600 text-white' : 'text-slate-600 hover:bg-slate-100'}`}><Icon className="size-5" />{t(labelKey)}</NavLink>)}</nav>
      <button type="button" onClick={signOut} className="border-t p-4 text-start text-slate-700">{user?.full_name} · {t('auth.logout')}</button>
    </div>
  )

  return <div className="min-h-screen bg-slate-50"><aside className={`fixed inset-y-0 z-40 hidden w-[280px] border-slate-200 bg-white lg:block ${sidebarSide}`}>{sidebar}</aside>{drawerOpen && <><button type="button" onClick={() => setDrawerOpen(false)} aria-label={t('menu.open')} className="fixed inset-0 z-40 bg-slate-950/40 lg:hidden" /><aside className={`fixed inset-y-0 z-50 lg:hidden ${sidebarSide}`}>{sidebar}</aside></>}<div className={contentOffset}><header className="flex items-center justify-between gap-4 border-b bg-white p-4"><button type="button" onClick={() => setDrawerOpen(true)} className="lg:hidden" aria-label={t('menu.open')}><Menu /></button><LanguageSwitcher /></header><main className="mx-auto max-w-7xl p-5"><Outlet /></main></div></div>
}
