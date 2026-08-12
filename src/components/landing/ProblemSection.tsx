import { CalendarClock, Clock3, Files } from 'lucide-react'
import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import problemImage from '../../assets/problem.png'
import { Button } from '../ui/Button'
import { Container } from '../common/Container'
import { Section } from '../common/Section'

const fadeUp = { hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }

const problems = [{ key: 'preparation', icon: Clock3, accent: 'bg-emerald-50 text-emerald-700' }, { key: 'tools', icon: Files, accent: 'bg-sky-50 text-sky-700' }, { key: 'pressure', icon: CalendarClock, accent: 'bg-amber-50 text-[#A16207]' }] as const

function scrollToFeatures() {
  document.querySelector('#teacher-problem + section')?.scrollIntoView({ behavior: 'smooth' })
}

export function ProblemSection() {
  const { t } = useTranslation('home')
  return <Section id="teacher-problem" aria-labelledby="teacher-problem-title" className="relative overflow-hidden bg-[#FCFDFD] py-12 font-['Cairo'] sm:py-16 lg:py-20">
    <div className="pointer-events-none absolute -right-32 top-24 -z-10 size-80 rounded-full bg-emerald-100/45 blur-3xl" aria-hidden="true" />
    <Container className="relative max-w-[1200px]">
      <div className="grid items-center gap-10 min-[900px]:grid-cols-[minmax(0,1.05fr)_minmax(0,.95fr)] min-[900px]:gap-14">
        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.2 }} variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.08 } } }} className="order-1 text-start min-[900px]:col-start-2">
          <motion.p variants={fadeUp} className="inline-flex rounded-full bg-emerald-50 px-4 py-2 text-sm font-bold text-emerald-700 ring-1 ring-emerald-100">{t('problem.badge')}</motion.p>
          <motion.h2 id="teacher-problem-title" variants={fadeUp} className="mt-4 max-w-xl text-[clamp(2rem,4vw,3.1rem)] font-extrabold leading-[1.2] tracking-[-0.03em] text-[#0F172A]">{t('problem.title')}</motion.h2>
          <motion.p variants={fadeUp} className="mt-5 max-w-xl text-base leading-8 text-slate-600 sm:text-lg">{t('problem.description')}</motion.p>

          <motion.div variants={fadeUp} className="mt-7 grid gap-3 sm:mt-8">
            {problems.map((problem) => {
              const Icon = problem.icon
              return <motion.article key={problem.key} whileHover={{ y: -2 }} transition={{ duration: 0.2, ease: 'easeOut' }} className="flex items-start gap-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_6px_16px_rgba(15,23,42,0.03)] transition-shadow duration-200 hover:shadow-[0_10px_22px_rgba(15,23,42,0.07)]">
                <span className={`grid size-10 shrink-0 place-items-center rounded-xl ${problem.accent}`}><Icon className="size-5" aria-hidden="true" /></span>
                <span><h3 className="font-extrabold text-slate-900">{t(`problem.items.${problem.key}.title`)}</h3><p className="mt-1 text-sm leading-6 text-slate-600">{t(`problem.items.${problem.key}.description`)}</p></span>
              </motion.article>
            })}
          </motion.div>

          <motion.div variants={fadeUp} className="mt-6">
           
          </motion.div>
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.97 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true, amount: 0.2 }} transition={{ duration: 0.45, ease: 'easeOut' }} className="relative order-2 mx-auto w-full max-w-[580px] min-[900px]:col-start-1 min-[900px]:row-start-1">
          <div className="pointer-events-none absolute inset-8 -z-10 rounded-[28px] bg-emerald-200/45 blur-3xl" aria-hidden="true" />
          <div className="overflow-hidden rounded-[22px] border border-slate-200 shadow-[0_18px_42px_rgba(15,23,42,0.11)]"><img src={problemImage} alt={t('problem.imageAlt')} loading="lazy" className="block aspect-square w-full object-cover" /></div>
        </motion.div>
      </div>
    </Container>
  </Section>
}
