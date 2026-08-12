import { Github, Instagram, Linkedin, Mail, MapPin } from 'lucide-react'
import logo1 from '../../assets/logo1.png'
import { Container } from '../common/Container'
import { useTranslation } from 'react-i18next'

const socialLinks = [
  { label: 'Instagram', href: '#instagram', icon: Instagram },
  { label: 'LinkedIn', href: '#linkedin', icon: Linkedin },
  { label: 'GitHub', href: '#github', icon: Github },
]

const currentYear = new Date().getFullYear()

export function Footer() {
  const { t } = useTranslation('home')
  const quickLinks = [{ label: t('nav.home'), href: '#home' }, { label: t('nav.about'), href: '#teacher-problem' }, { label: t('nav.features'), href: '#المميزات' }, { label: t('nav.howItWorks'), href: '#how-it-works' }, { label: t('nav.contact'), href: '#تواصل-معنا' }]
  return (
    <footer id="تواصل-معنا" className="border-t border-[#D1FAE5] bg-gradient-to-b from-white to-[#F0FDF4] font-['Cairo']">
      <Container className="py-14 sm:py-16">
        <div className="grid gap-8 sm:grid-cols-2 sm:gap-x-10 sm:gap-y-10 lg:grid-cols-4 lg:gap-8">
          <div>
            <a href="#home" dir="ltr" className="inline-flex shrink-0 items-center gap-1.5 transition-opacity hover:opacity-80 sm:gap-2" aria-label="Mo3allimAI">
              <img src={logo1} alt="Mo3allimAI" className="h-10 w-auto shrink-0 object-contain sm:h-11 lg:h-12" />
              <span className="whitespace-nowrap text-[clamp(1.25rem,6vw,1.875rem)] font-extrabold leading-none tracking-[-0.03em]"><span className="text-[#065F46]">Mo3allim</span><span className="text-[#C89B3C]">AI</span></span>
            </a>
            <p className="mt-5 max-w-sm text-sm leading-7 text-[#475569]">{t('footer.description')}</p>
          </div>

          <div>
            <h3 className="text-base font-bold text-[#059669]">{t('footer.quickLinks')}</h3>
            <ul className="mt-5 space-y-3">
              {quickLinks.map((link) => (
                <li key={link.href}>
                  <a href={link.href} className="inline-flex text-sm font-medium text-[#475569] transition-all duration-200 hover:translate-x-[-2px] hover:text-[#059669]">
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="text-base font-bold text-[#059669]">{t('footer.contact')}</h3>
            <ul className="mt-5 space-y-4 text-sm text-[#475569]">
              <li><a href="mailto:contact@mo3allimai.com" className="inline-flex items-center gap-2.5 transition-colors duration-200 hover:text-[#059669]"><Mail className="size-4 text-[#059669]" />contact@mo3allimai.com</a></li>
              <li className="inline-flex items-center gap-2.5"><MapPin className="size-4 text-[#059669]" />{t('footer.country')}</li>
            </ul>
          </div>

          <div>
            <h3 className="text-base font-bold text-[#059669]">{t('footer.follow')}</h3>
            <div className="mt-5 flex items-center gap-3" dir="ltr">
              {socialLinks.map(({ label, href, icon: Icon }) => (
                <a key={label} href={href} aria-label={label} className="inline-flex size-10 items-center justify-center rounded-xl border border-[#D1D5DB] bg-white text-[#475569] transition-all duration-200 hover:-translate-y-0.5 hover:border-[#059669] hover:bg-[#059669] hover:text-white hover:shadow-md hover:shadow-emerald-900/15">
                  <Icon className="size-5" />
                </a>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-10 border-t border-[#D1FAE5] pt-6 text-center sm:mt-12">
          <p className="text-sm text-[#64748B]">© {currentYear} Mo3allimAI. {t('footer.copyright')}</p>
        </div>
      </Container>
    </footer>
  )
}
