import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '../../utils/cn'
interface CardProps extends HTMLAttributes<HTMLDivElement> { children: ReactNode }
export function Card({ children, className, ...props }: CardProps) { return <div className={cn('min-w-0 rounded-2xl border border-slate-100 bg-white shadow-sm', className)} {...props}>{children}</div> }
