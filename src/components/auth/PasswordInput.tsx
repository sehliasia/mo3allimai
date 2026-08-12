import { Eye, EyeOff, LockKeyhole } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { FormInput } from './FormInput'

interface PasswordInputProps { id: string; label: string; value: string; onChange: (value: string) => void; error?: string; autoComplete?: string; disabled?: boolean }

export function PasswordInput({ id, label, value, onChange, error, autoComplete, disabled }: PasswordInputProps) {
  const { t } = useTranslation('auth')
  const [visible, setVisible] = useState(false)
  return <div className="relative"><FormInput id={id} label={label} type={visible ? 'text' : 'password'} value={value} onChange={(event) => onChange(event.target.value)} error={error} autoComplete={autoComplete} disabled={disabled} icon={<LockKeyhole className="size-4" />} className="ps-12" /><button type="button" disabled={disabled} onClick={() => setVisible((current) => !current)} aria-label={visible ? t('fields.hidePassword') : t('fields.showPassword')} className="absolute start-3 top-9 grid size-7 place-items-center rounded-md text-slate-500 transition-colors duration-200 hover:text-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"><>{visible ? <EyeOff className="size-4" /> : <Eye className="size-4" />}</></button></div>
}
