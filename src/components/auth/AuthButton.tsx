import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { Button } from '../ui/Button'

export function AuthButton({ children, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode }) {
  return <Button {...props} className="h-[52px] w-full rounded-xl bg-[#065F46] text-base shadow-[0_8px_20px_rgba(6,95,70,0.2)] transition-[transform,background-color,box-shadow] duration-[200ms] hover:-translate-y-0.5 hover:bg-emerald-800 hover:shadow-[0_12px_24px_rgba(6,95,70,0.24)] focus-visible:ring-emerald-700 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:translate-y-0">{children}</Button>
}
