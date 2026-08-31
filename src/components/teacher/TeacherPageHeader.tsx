import type { ReactNode } from 'react'

export function TeacherPageHeader({ title, description, action, emphasis = false }: { title: string; description: string; action?: ReactNode; emphasis?: boolean }) {
  return <header className="teacher-page-header flex flex-wrap items-end justify-between gap-4"><div className="teacher-page-heading"><h1 className={`${emphasis ? 'text-3xl leading-tight sm:text-4xl' : 'text-2xl sm:text-3xl'} font-extrabold tracking-tight text-slate-900`}>{title}</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500 sm:text-base">{description}</p></div>{action && <div className="teacher-page-header-action">{action}</div>}</header>
}
