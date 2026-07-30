import { AnimatePresence, motion } from 'framer-motion'
import { Menu, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import logo1 from '../../assets/logo1.png'
import { Container } from '../common/Container'

interface HeaderLink { label: string; href: string }

const headerLinks: HeaderLink[] = [
  { label: 'الرئيسية', href: '#home' },
  { label: 'لماذا Mo3allimAI؟', href: '#teacher-problem' },
  { label: 'المميزات', href: '#المميزات' },
  { label: 'كيف تعمل', href: '#how-it-works' },
  { label: 'تواصل معنا', href: '#تواصل-معنا' },
]

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const [activeHref, setActiveHref] = useState('#home')
  const handleNavigation = (href: string) => { setActiveHref(href); setIsOpen(false) }

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') setIsOpen(false) }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [])

  return (
    <header className="sticky top-0 z-50 h-[76px] border-b border-[#E5E7EB] bg-white/95 font-['Cairo'] backdrop-blur-xl">
      <Container className="h-full px-6">
        <div dir="rtl" className="relative flex h-full items-center justify-between gap-6">
        <div className="order-3 hidden shrink-0 items-center xl:flex" aria-label="حساب المستخدم">
          <Link to="/login" dir="rtl" className="inline-flex h-11 items-center justify-center rounded-xl border border-[#059669] bg-white px-6 text-sm font-semibold text-[#059669] transition-all duration-200 hover:bg-[#059669] hover:text-white hover:shadow-md hover:shadow-emerald-900/15 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:ring-offset-4">تسجيل الدخول</Link>
        </div>

        <nav dir="rtl" className="order-2 hidden min-w-0 xl:flex xl:flex-1 xl:justify-center" aria-label="التنقل الرئيسي">
          <ul className="flex items-center gap-8 whitespace-nowrap">
            {headerLinks.map((link) => {
              const isActive = activeHref === link.href
              return <li key={link.href} className="relative"><a href={link.href} onClick={() => handleNavigation(link.href)} className={`relative block py-3 text-base font-semibold leading-6 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:ring-offset-4 ${isActive ? 'text-emerald-700' : 'text-slate-600 hover:text-emerald-700'}`}>{link.label}{isActive && <motion.span layoutId="header-active-link" className="absolute inset-x-0 -bottom-0.5 h-0.5 rounded-full bg-emerald-700" transition={{ type: 'spring', stiffness: 380, damping: 30 }} />}</a></li>
            })}
          </ul>
        </nav>

        <a href="/" dir="ltr" aria-label="Mo3allimAI" className="order-1 flex shrink-0 items-center gap-1.5 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:ring-offset-4 sm:gap-2">
          <img src={logo1} alt="Mo3allimAI" className="h-10 w-auto shrink-0 object-contain sm:h-11 lg:h-12" />
          <h1 className="m-0 whitespace-nowrap text-[clamp(1.25rem,6vw,1.875rem)] font-extrabold leading-none tracking-[-0.03em]"><span className="text-[#065F46]">Mo3allim</span><span className="text-[#C89B3C]">AI</span></h1>
        </a>

          <button onClick={() => setIsOpen((open) => !open)} className="order-2 ml-auto rounded-xl p-2.5 text-slate-700 transition hover:bg-emerald-50 hover:text-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:ring-offset-4 xl:hidden" aria-label={isOpen ? 'إغلاق القائمة' : 'فتح القائمة'} aria-expanded={isOpen} aria-controls="mobile-navigation">
          {isOpen ? <X className="size-6" /> : <Menu className="size-6" />}
          </button>
        </div>
      </Container>

      <AnimatePresence>
        {isOpen && <motion.nav id="mobile-navigation" aria-label="التنقل عبر الهاتف" initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.24, ease: 'easeOut' }} className="absolute inset-x-0 top-full w-full border-b border-slate-100 bg-white p-4 shadow-xl shadow-slate-900/10 sm:left-auto sm:right-4 sm:max-w-sm sm:rounded-b-2xl sm:border sm:p-5 xl:hidden">
          <ul className="space-y-1">{headerLinks.map((link) => <li key={link.href}><a href={link.href} onClick={() => handleNavigation(link.href)} className={`block rounded-xl px-4 py-3 text-lg font-semibold transition-colors ${activeHref === link.href ? 'bg-emerald-50 text-emerald-700' : 'text-slate-700 hover:bg-slate-50 hover:text-emerald-700'}`}>{link.label}</a></li>)}</ul>
          <div className="mt-5 border-t border-slate-100 pt-5"><Link to="/login" onClick={() => setIsOpen(false)} className="flex h-11 items-center justify-center rounded-xl border border-[#059669] bg-white px-6 text-sm font-semibold text-[#059669] transition-all duration-200 hover:bg-[#059669] hover:text-white hover:shadow-md hover:shadow-emerald-900/15">تسجيل الدخول</Link></div>
        </motion.nav>}
      </AnimatePresence>
    </header>
  )
}
