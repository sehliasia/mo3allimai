import { useState } from 'react'
import { Loader2, Mail, UserRound } from 'lucide-react'
import { Link } from 'react-router-dom'
import { AuthButton } from '../components/auth/AuthButton'
import { AuthLayout } from '../components/auth/AuthLayout'
import { FormInput } from '../components/auth/FormInput'
import { PasswordInput } from '../components/auth/PasswordInput'
import { isValidEmail } from '../utils/validation'

type RegisterErrors = Partial<Record<'name' | 'email' | 'password' | 'confirmPassword' | 'terms', string>>
export function RegisterPage() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [acceptedTerms, setAcceptedTerms] = useState(false)
  const [errors, setErrors] = useState<RegisterErrors>({})
  const [isSubmitting, setIsSubmitting] = useState(false)

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const nextErrors: RegisterErrors = {}
    if (!name.trim()) nextErrors.name = 'الاسم الكامل مطلوب.'
    if (!isValidEmail(email)) nextErrors.email = 'يرجى إدخال بريد إلكتروني صالح.'
    if (password.length < 6) nextErrors.password = 'يجب أن تتكون كلمة المرور من 6 أحرف على الأقل.'
    if (password !== confirmPassword) nextErrors.confirmPassword = 'تأكيد كلمة المرور غير مطابق.'
    if (!acceptedTerms) nextErrors.terms = 'يرجى الموافقة على الشروط وسياسة الخصوصية.'
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) return
    setIsSubmitting(true)
    window.setTimeout(() => setIsSubmitting(false), 450)
  }

  return <AuthLayout title="إنشاء حساب" description="أنشئ حسابك وابدأ استخدام Mo3allimAI.">
    <form onSubmit={handleSubmit} noValidate className="mt-8 space-y-4">
      <FormInput id="register-name" label="الاسم الكامل" value={name} onChange={(event) => setName(event.target.value)} autoComplete="name" placeholder="أدخل اسمك الكامل" error={errors.name} icon={<UserRound className="size-4" />} disabled={isSubmitting} />
      <FormInput id="register-email" label="البريد الإلكتروني" type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" placeholder="example@email.com" error={errors.email} icon={<Mail className="size-4" />} disabled={isSubmitting} />
      <PasswordInput id="register-password" label="كلمة المرور" value={password} onChange={setPassword} autoComplete="new-password" error={errors.password} disabled={isSubmitting} />
      <PasswordInput id="register-confirm-password" label="تأكيد كلمة المرور" value={confirmPassword} onChange={setConfirmPassword} autoComplete="new-password" error={errors.confirmPassword} disabled={isSubmitting} />
      <div><label className="flex cursor-pointer items-start gap-2 text-sm leading-6 text-slate-600"><input type="checkbox" checked={acceptedTerms} disabled={isSubmitting} onChange={(event) => setAcceptedTerms(event.target.checked)} className="mt-1 size-4 rounded border-slate-300 text-emerald-700 focus:ring-emerald-600 disabled:cursor-not-allowed" />أوافق على الشروط وسياسة الخصوصية</label>{errors.terms && <p className="mt-1.5 text-xs font-medium text-rose-600">{errors.terms}</p>}</div>
      <AuthButton type="submit" disabled={isSubmitting}>{isSubmitting ? <><Loader2 className="size-5 animate-spin" aria-hidden="true" />جارٍ إنشاء الحساب...</> : 'إنشاء الحساب'}</AuthButton>
    </form>
    <p className="mt-7 text-center text-sm text-slate-600">لديك حساب بالفعل؟ <Link to="/login" className="font-bold text-emerald-700 transition-colors duration-200 hover:text-emerald-800 hover:underline focus:outline-none focus:ring-2 focus:ring-emerald-600">تسجيل الدخول</Link></p>
  </AuthLayout>
}
