import { motion } from 'framer-motion'
import type { Step } from '../../types/landing'

export function StepCard({ step }: { step: Step }) {
  const Icon = step.icon

  return (
    <motion.article className="relative z-10 h-full" whileHover={{ y: -4 }} transition={{ duration: 0.28, ease: 'easeOut' }}>
      <div className="flex h-full min-h-[210px] flex-col rounded-[24px] border border-emerald-100/90 bg-white/85 p-6 text-right shadow-[0_10px_28px_rgba(15,23,42,0.045)] transition-[border-color,box-shadow] duration-300 hover:border-emerald-200 hover:shadow-[0_18px_38px_rgba(15,23,42,0.1)]">
        <div className="flex items-start justify-between">
          <span className="grid size-12 place-items-center rounded-2xl bg-emerald-50 text-emerald-700 ring-1 ring-emerald-100"><Icon className="size-5" aria-hidden="true" /></span>
          <span className="text-4xl font-black leading-none tracking-[-0.08em] text-emerald-600">{step.number}</span>
        </div>
        <h3 className="mt-8 text-lg font-extrabold text-slate-900">{step.title}</h3>
        <p className="mt-2 text-sm leading-6 text-slate-600">{step.description}</p>
      </div>
    </motion.article>
  )
}
