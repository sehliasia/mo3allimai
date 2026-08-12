import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import arAdmin from '../locales/ar/admin.json'
import arAuth from '../locales/ar/auth.json'
import arCommon from '../locales/ar/common.json'
import arHome from '../locales/ar/home.json'
import arTeacher from '../locales/ar/teacher.json'
import enAdmin from '../locales/en/admin.json'
import enAuth from '../locales/en/auth.json'
import enCommon from '../locales/en/common.json'
import enHome from '../locales/en/home.json'
import enTeacher from '../locales/en/teacher.json'
import esAdmin from '../locales/es/admin.json'
import esAuth from '../locales/es/auth.json'
import esCommon from '../locales/es/common.json'
import esHome from '../locales/es/home.json'
import esTeacher from '../locales/es/teacher.json'
import frAdmin from '../locales/fr/admin.json'
import frAuth from '../locales/fr/auth.json'
import frCommon from '../locales/fr/common.json'
import frHome from '../locales/fr/home.json'
import frTeacher from '../locales/fr/teacher.json'
import { isSupportedLanguage, type AppLanguage } from './languages'

const storageKey = 'mo3allimai_language'
const storedLanguage = localStorage.getItem(storageKey)
const browserLanguage = navigator.language.slice(0, 2)
const initialLanguage: AppLanguage = isSupportedLanguage(storedLanguage)
  ? storedLanguage
  : isSupportedLanguage(browserLanguage)
    ? browserLanguage
    : 'ar'

export function applyLanguage(language: AppLanguage) {
  document.documentElement.lang = language
  document.documentElement.dir = language === 'ar' ? 'rtl' : 'ltr'
  localStorage.setItem(storageKey, language)
}

i18n
  .use(initReactI18next)
  .init({
    resources: {
      ar: { common: arCommon, home: arHome, auth: arAuth, admin: arAdmin, teacher: arTeacher },
      fr: { common: frCommon, home: frHome, auth: frAuth, admin: frAdmin, teacher: frTeacher },
      en: { common: enCommon, home: enHome, auth: enAuth, admin: enAdmin, teacher: enTeacher },
      es: { common: esCommon, home: esHome, auth: esAuth, admin: esAdmin, teacher: esTeacher },
    },
    lng: initialLanguage,
    fallbackLng: 'ar',
    ns: ['common', 'home', 'auth', 'admin', 'teacher'],
    defaultNS: 'common',
    interpolation: { escapeValue: false },
  })

applyLanguage(initialLanguage)
i18n.on('languageChanged', language => {
  if (isSupportedLanguage(language)) applyLanguage(language)
})

export default i18n
