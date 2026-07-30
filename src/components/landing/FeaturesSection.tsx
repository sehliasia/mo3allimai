import { motion } from 'framer-motion'
import { features } from '../../constants/features'
import { Container } from '../common/Container'
import { Section } from '../common/Section'
import { FeatureCard } from './FeatureCard'
import { stagger } from '../../lib/motion'

export function FeaturesSection() {
  return (
    <Section id="المميزات" className="bg-slate-50 py-12 sm:py-16 lg:py-20">
      <Container>
        <div dir="rtl" className="mx-auto max-w-2xl text-center">
          <h2 className="text-[clamp(2rem,5vw,3rem)] font-extrabold leading-tight tracking-tight text-slate-900">أدوات Mo3allimAI</h2>
          <p className="mx-auto mt-4 text-base leading-8 text-slate-600 sm:text-lg">كل ما يحتاجه معلم اللغة العربية في منصة واحدة.</p>
        </div>

        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.15 }} variants={stagger} className="mt-10 grid auto-rows-fr gap-5 sm:grid-cols-2 sm:gap-6 lg:mt-12 lg:grid-cols-4 lg:gap-7">
          {features.map((feature) => <FeatureCard feature={feature} key={feature.title} />)}
        </motion.div>
      </Container>
    </Section>
  )
}
