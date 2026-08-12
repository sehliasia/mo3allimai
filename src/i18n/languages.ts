export type AppLanguage = 'ar' | 'fr' | 'en' | 'es'
export const SUPPORTED_LANGUAGES = [{ code: 'ar', label: 'العربية', direction: 'rtl' }, { code: 'fr', label: 'Français', direction: 'ltr' }, { code: 'en', label: 'English', direction: 'ltr' }, { code: 'es', label: 'Español', direction: 'ltr' }] as const
export const isSupportedLanguage = (value: string | null): value is AppLanguage => SUPPORTED_LANGUAGES.some(language => language.code === value)
export const getLanguageDirection = (language: AppLanguage) => language === 'ar' ? 'rtl' : 'ltr'
