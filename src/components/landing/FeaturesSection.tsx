import { motion } from 'framer-motion'
import { features } from '../../constants/features'
import { Container } from '../common/Container'
import { Section } from '../common/Section'
import { FeatureCard } from './FeatureCard'
import { stagger } from '../../lib/motion'
import { useTranslation } from 'react-i18next'

export function FeaturesSection() {
  const { t } = useTranslation('home')
  const featureKeys = ['assistant', 'lessons', 'activities', 'assessments', 'year', 'resources', 'mindMaps', 'content'] as const
  return (
    <Section id="المميزات" className="bg-slate-50 py-12 sm:py-16 lg:py-20">
      <Container>
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-[clamp(2rem,5vw,3rem)] font-extrabold leading-tight tracking-tight text-slate-900">{t('features.title')}</h2>
          <p className="mx-auto mt-4 text-base leading-8 text-slate-600 sm:text-lg">{t('features.description')}</p>
        </div>

        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.15 }} variants={stagger} className="mt-10 grid auto-rows-fr gap-5 sm:grid-cols-2 sm:gap-6 lg:mt-12 lg:grid-cols-4 lg:gap-7">
          {features.map((feature, index) => {
            const key = featureKeys[index]
            return <FeatureCard feature={{ ...feature, title: t(`features.items.${key}.title`), description: t(`features.items.${key}.description`) }} key={key} />
          })}
        </motion.div>
      </Container>
    </Section>
  )
}
