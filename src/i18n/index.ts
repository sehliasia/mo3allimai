import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import arAdmin from '../locales/ar/admin.json'
import arAuth from '../locales/ar/auth.json'
import arCommon from '../locales/ar/common.json'
import arHome from '../locales/ar/home.json'
import arTeacher from '../locales/ar/teacher.json'
import arTeacherAssistant from '../locales/ar/teacher-assistant.json'
import arTeacherToolsUi from '../locales/ar/teacher-tools-ui.json'
import arTeacherProfile from '../locales/ar/teacher-profile.json'
import arTeacherLibrary from '../locales/ar/teacher-library.json'
import arKnowledgeUpload from '../locales/ar/knowledge-upload.json'
import arKnowledgeDetails from '../locales/ar/knowledge-details.json'
import arKnowledgePhase3 from '../locales/ar/knowledge-phase3.json'
import enAdmin from '../locales/en/admin.json'
import enAuth from '../locales/en/auth.json'
import enCommon from '../locales/en/common.json'
import enHome from '../locales/en/home.json'
import enTeacher from '../locales/en/teacher.json'
import enTeacherAssistant from '../locales/en/teacher-assistant.json'
import enTeacherToolsUi from '../locales/en/teacher-tools-ui.json'
import enTeacherProfile from '../locales/en/teacher-profile.json'
import enTeacherLibrary from '../locales/en/teacher-library.json'
import enKnowledgeUpload from '../locales/en/knowledge-upload.json'
import enKnowledgeDetails from '../locales/en/knowledge-details.json'
import enKnowledgePhase3 from '../locales/en/knowledge-phase3.json'
import esAdmin from '../locales/es/admin.json'
import esAuth from '../locales/es/auth.json'
import esCommon from '../locales/es/common.json'
import esHome from '../locales/es/home.json'
import esTeacher from '../locales/es/teacher.json'
import esTeacherAssistant from '../locales/es/teacher-assistant.json'
import esTeacherToolsUi from '../locales/es/teacher-tools-ui.json'
import esTeacherProfile from '../locales/es/teacher-profile.json'
import esTeacherLibrary from '../locales/es/teacher-library.json'
import esKnowledgeUpload from '../locales/es/knowledge-upload.json'
import esKnowledgeDetails from '../locales/es/knowledge-details.json'
import esKnowledgePhase3 from '../locales/es/knowledge-phase3.json'
import frAdmin from '../locales/fr/admin.json'
import frAuth from '../locales/fr/auth.json'
import frCommon from '../locales/fr/common.json'
import frHome from '../locales/fr/home.json'
import frTeacher from '../locales/fr/teacher.json'
import frTeacherAssistant from '../locales/fr/teacher-assistant.json'
import frTeacherToolsUi from '../locales/fr/teacher-tools-ui.json'
import frTeacherProfile from '../locales/fr/teacher-profile.json'
import frTeacherLibrary from '../locales/fr/teacher-library.json'
import frKnowledgeUpload from '../locales/fr/knowledge-upload.json'
import frKnowledgeDetails from '../locales/fr/knowledge-details.json'
import frKnowledgePhase3 from '../locales/fr/knowledge-phase3.json'
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
      ar: { common: arCommon, home: arHome, auth: arAuth, admin: { ...arAdmin, knowledge: { ...arAdmin.knowledge, batch: arKnowledgeUpload, details: arKnowledgeDetails, phase3: arKnowledgePhase3 } }, teacher: { ...arTeacher, assistant: { ...arTeacher.assistant, ...arTeacherAssistant }, profile: { ...arTeacher.profile, ...arTeacherProfile } }, teacherToolsUi: arTeacherToolsUi, teacherProfile: arTeacherProfile },
      fr: { common: frCommon, home: frHome, auth: frAuth, admin: { ...frAdmin, knowledge: { ...frAdmin.knowledge, batch: frKnowledgeUpload, details: frKnowledgeDetails, phase3: frKnowledgePhase3 } }, teacher: { ...frTeacher, assistant: { ...frTeacher.assistant, ...frTeacherAssistant }, profile: { ...frTeacher.profile, ...frTeacherProfile } }, teacherToolsUi: frTeacherToolsUi, teacherProfile: frTeacherProfile },
      en: { common: enCommon, home: enHome, auth: enAuth, admin: { ...enAdmin, knowledge: { ...enAdmin.knowledge, batch: enKnowledgeUpload, details: enKnowledgeDetails, phase3: enKnowledgePhase3 } }, teacher: { ...enTeacher, assistant: { ...enTeacher.assistant, ...enTeacherAssistant }, profile: { ...enTeacher.profile, ...enTeacherProfile } }, teacherToolsUi: enTeacherToolsUi, teacherProfile: enTeacherProfile },
      es: { common: esCommon, home: esHome, auth: esAuth, admin: { ...esAdmin, knowledge: { ...esAdmin.knowledge, batch: esKnowledgeUpload, details: esKnowledgeDetails, phase3: esKnowledgePhase3 } }, teacher: { ...esTeacher, assistant: { ...esTeacher.assistant, ...esTeacherAssistant }, profile: { ...esTeacher.profile, ...esTeacherProfile } }, teacherToolsUi: esTeacherToolsUi, teacherProfile: esTeacherProfile },
    },
    lng: initialLanguage,
    fallbackLng: 'ar',
    ns: ['common', 'home', 'auth', 'admin', 'teacher'],
    defaultNS: 'common',
    interpolation: { escapeValue: false },
  })

i18n.addResourceBundle('ar', 'teacherLibrary', arTeacherLibrary)
i18n.addResourceBundle('fr', 'teacherLibrary', frTeacherLibrary)
i18n.addResourceBundle('en', 'teacherLibrary', enTeacherLibrary)
i18n.addResourceBundle('es', 'teacherLibrary', esTeacherLibrary)

applyLanguage(initialLanguage)
i18n.on('languageChanged', language => {
  if (isSupportedLanguage(language)) applyLanguage(language)
})

export default i18n
