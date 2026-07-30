import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '../../utils/cn'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> { children: ReactNode; variant?: 'primary' | 'secondary' | 'ghost'; className?: string }

export function Button({ children, variant = 'primary', className, ...props }: ButtonProps) {
  const variants = { primary: 'bg-emerald-700 text-white shadow-lg shadow-emerald-900/10 hover:bg-emerald-800', secondary: 'border border-slate-200 bg-white text-slate-800 hover:border-emerald-200 hover:bg-emerald-50', ghost: 'text-slate-600 hover:bg-slate-100' }
  return <button className={cn('inline-flex max-w-full items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-bold transition duration-200 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:ring-offset-2', variants[variant], className)} {...props}>{children}</button>
}
