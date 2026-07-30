import type { InputHTMLAttributes, ReactNode } from 'react'
import { cn } from '../../utils/cn'

interface FormInputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string
  error?: string
  icon?: ReactNode
}

export function FormInput({ id, label, error, icon, className, ...props }: FormInputProps) {
  return <div><label htmlFor={id} className="mb-2 block text-sm font-semibold text-slate-700">{label}</label><div className="group relative">{icon && <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 transition-colors duration-200 group-focus-within:text-emerald-700" aria-hidden="true">{icon}</span>}<input id={id} className={cn('h-[52px] w-full rounded-xl border bg-slate-50 px-4 text-right text-sm text-slate-900 outline-none transition-[border-color,background-color,box-shadow] duration-[200ms] placeholder:text-slate-400 hover:border-slate-300 hover:bg-white focus:border-emerald-700 focus:bg-white focus:shadow-[0_0_0_4px_rgba(16,185,129,0.12)] disabled:cursor-not-allowed disabled:bg-slate-100', icon && 'pr-12', error ? 'border-rose-300 focus:border-rose-500 focus:shadow-[0_0_0_4px_rgba(251,113,133,0.12)]' : 'border-slate-200', className)} aria-invalid={Boolean(error)} aria-describedby={error ? `${id}-error` : undefined} {...props} /></div>{error && <p id={`${id}-error`} className="mt-1.5 text-xs font-medium text-rose-600">{error}</p>}</div>
}
