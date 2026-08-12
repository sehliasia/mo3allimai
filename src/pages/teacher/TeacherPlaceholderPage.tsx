import { Construction } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export function TeacherPlaceholderPage({ page }: { page: 'assistant' | 'tools' | 'library' | 'history' | 'settings' }) {
  const { t } = useTranslation('teacher')
  return <section className="rounded-2xl border border-slate-200 bg-white px-6 py-14 text-center shadow-sm"><Construction className="mx-auto size-10 text-emerald-700" /><h1 className="mt-4 text-2xl font-extrabold text-slate-900">{t(`pages.${page}.title`)}</h1><h2 className="mt-2 font-bold text-slate-700">{t('empty.title')}</h2><p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">{t('empty.description')}</p></section>
}
