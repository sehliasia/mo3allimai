import { motion } from 'framer-motion'
import { howItWorksContent, steps } from '../../constants/steps'
import { Container } from '../common/Container'
import { Section } from '../common/Section'
import { StepCard } from './StepCard'
import { stagger } from '../../lib/motion'

export function HowItWorksSection() {
  return (
    <Section id="how-it-works" className="bg-[#F4FBF7] py-12 sm:py-16 lg:py-20">
      <Container>
        <div dir="rtl" className="mx-auto max-w-3xl text-center">
          <h2 className="text-[clamp(2rem,5vw,3rem)] font-extrabold leading-tight tracking-tight text-slate-900">{howItWorksContent.title}</h2>
          <p className="mx-auto mt-4 max-w-2xl text-base leading-8 text-slate-600 sm:text-lg">{howItWorksContent.description}</p>
        </div>

        <div dir="rtl" className="relative mt-10 sm:mt-12">
          <div className="absolute bottom-12 right-10 top-12 w-px bg-gradient-to-b from-emerald-200 via-emerald-300 to-emerald-200 sm:hidden" aria-hidden="true" />
          <div className="absolute left-[12.5%] right-[12.5%] top-14 hidden h-px bg-gradient-to-l from-emerald-200 via-emerald-400 to-emerald-200 lg:block" aria-hidden="true" />
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.15 }} variants={stagger} className="grid auto-rows-fr gap-5 sm:grid-cols-2 sm:gap-6 lg:grid-cols-4 lg:gap-7">
            {steps.map((step) => <StepCard step={step} key={step.number} />)}
          </motion.div>
        </div>
      </Container>
    </Section>
  )
}
