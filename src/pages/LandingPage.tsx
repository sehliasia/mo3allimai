import { FeaturesSection } from '../components/landing/FeaturesSection'
import { HeroSection } from '../components/landing/HeroSection'
import { HowItWorksSection } from '../components/landing/HowItWorksSection'
import { ProblemSection } from '../components/landing/ProblemSection'
import { PublicLayout } from '../layouts/PublicLayout'

export function LandingPage() {
  return <PublicLayout><HeroSection/><ProblemSection/><FeaturesSection/><HowItWorksSection/></PublicLayout>
}
