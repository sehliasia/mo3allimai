import type { ReactNode } from 'react'

export function AuthCard({ children }: { children: ReactNode }) {
  return <section className="w-full max-w-[420px] rounded-[22px] border border-slate-200/95 bg-white p-5 shadow-[0_12px_35px_rgba(15,23,42,0.09),0_3px_10px_rgba(15,23,42,0.04)] transition-[transform,box-shadow,border-color] duration-[220ms] ease-out hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-[0_16px_42px_rgba(15,23,42,0.12),0_5px_14px_rgba(15,23,42,0.05)] sm:p-8">{children}</section>
}
