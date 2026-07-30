import { motion } from 'framer-motion'
import type { Feature } from '../../types/landing'
import { Card } from '../ui/Card'

export function FeatureCard({ feature }: { feature: Feature }) {
  const Icon = feature.icon

  return (
    <motion.div className="group min-w-0" whileHover={{ y: -6 }} transition={{ duration: 0.3, ease: 'easeOut' }}>
      <Card className="flex h-full min-h-[208px] flex-col items-center justify-center rounded-[22px] border-slate-200 bg-white p-7 text-center shadow-[0_8px_24px_rgba(15,23,42,0.045)] transition-[border-color,box-shadow] duration-300 group-hover:border-emerald-100 group-hover:shadow-[0_20px_40px_rgba(15,23,42,0.1)]">
        <span className={`grid size-16 place-items-center rounded-full transition-transform duration-300 group-hover:scale-110 ${feature.accent}`}><Icon className="size-8" aria-hidden="true" /></span>
        <div className="mt-6 w-full"><h3 className="text-[1.35rem] font-extrabold leading-7 text-slate-900">{feature.title}</h3><p className="mt-2 text-sm leading-6 text-slate-500">{feature.description}</p></div>
      </Card>
    </motion.div>
  )
}
