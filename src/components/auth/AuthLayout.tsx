import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { AuthCard } from './AuthCard'
import { AuthLogo } from './AuthLogo'

interface AuthLayoutProps {
  title: string
  description: string
  children: ReactNode
}

export function AuthLayout({ title, description, children }: AuthLayoutProps) {
  return <main dir="rtl" lang="ar" className="relative flex min-h-[100dvh] items-center justify-center overflow-hidden bg-[#F8FAFC] px-4 py-8 font-['Cairo'] sm:px-6">
    <div className="pointer-events-none absolute left-1/2 top-1/2 size-[34rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,rgba(16,185,129,0.12)_0%,rgba(236,253,245,0)_62%)]" aria-hidden="true" />
    <Link to="/" aria-label="العودة إلى الصفحة الرئيسية" className="group absolute right-4 top-4 z-10 inline-flex min-h-11 min-w-11 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white/85 px-3 text-sm font-bold text-emerald-700 shadow-[0_3px_10px_rgba(15,23,42,0.04)] backdrop-blur-sm transition-[background-color,border-color,box-shadow] duration-200 hover:border-emerald-200 hover:bg-emerald-50 hover:shadow-[0_5px_14px_rgba(6,95,70,0.08)] focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-700 focus-visible:ring-offset-2 sm:right-6 sm:top-6">
      <ArrowRight className="size-5 transition-transform duration-200 group-hover:translate-x-0.5" aria-hidden="true" />
      <span className="hidden sm:inline">العودة إلى الرئيسية</span>
    </Link>
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: 'easeOut' }} className="relative flex w-full flex-col items-center">
      <div className="mb-7 text-center"><AuthLogo /><p className="mt-3 text-sm font-medium text-slate-500 sm:text-base">{description}</p></div>
      <AuthCard><h1 className="text-right text-2xl font-extrabold tracking-[-0.02em] text-[#0F172A]">{title}</h1>{children}</AuthCard>
    </motion.div>
  </main>
}
