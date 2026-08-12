import { Check, ChevronDown, Globe } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { SUPPORTED_LANGUAGES, type AppLanguage } from '../../i18n/languages'

export function LanguageSwitcher() {
  const { i18n } = useTranslation()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const activeLanguage = SUPPORTED_LANGUAGES.find(language => language.code === i18n.resolvedLanguage) ?? SUPPORTED_LANGUAGES[0]

  useEffect(() => {
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false)
    }
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', closeOnOutsideClick)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('mousedown', closeOnOutsideClick)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [])

  return <div ref={ref} className="relative">
    <button type="button" onClick={() => setOpen(value => !value)} aria-label="Change interface language" aria-haspopup="menu" aria-expanded={open} className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 focus-visible:outline-2 focus-visible:outline-emerald-600">
      <Globe className="size-4" aria-hidden="true" />{activeLanguage.label}<ChevronDown className="size-4" aria-hidden="true" />
    </button>
    {open && <div role="menu" className="absolute start-0 z-50 mt-2 w-40 rounded-xl border border-slate-200 bg-white p-1 shadow-lg">
      {SUPPORTED_LANGUAGES.map(language => <button type="button" key={language.code} role="menuitem" onClick={() => { void i18n.changeLanguage(language.code as AppLanguage); setOpen(false) }} className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-start hover:bg-emerald-50">
        {language.label}{activeLanguage.code === language.code && <Check className="size-4 text-emerald-700" aria-hidden="true" />}
      </button>)}
    </div>}
  </div>
}
