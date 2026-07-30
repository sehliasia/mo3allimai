import { Loader2, Mail } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { AuthButton } from '../components/auth/AuthButton'
import { AuthLayout } from '../components/auth/AuthLayout'
import { FormInput } from '../components/auth/FormInput'
import { PasswordInput } from '../components/auth/PasswordInput'
import { isValidEmail } from '../utils/validation'

type LoginErrors = Partial<Record<'email' | 'password', string>>

export function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [errors, setErrors] = useState<LoginErrors>({})
  const [isSubmitting, setIsSubmitting] = useState(false)

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const nextErrors: LoginErrors = {}
    if (!isValidEmail(email)) nextErrors.email = 'يرجى إدخال بريد إلكتروني صحيح.'
    if (!password) nextErrors.password = 'كلمة المرور مطلوبة.'
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) return
    setIsSubmitting(true)
    window.setTimeout(() => setIsSubmitting(false), 450)
  }

  return <AuthLayout title="تسجيل الدخول" description="مساعدك الذكي لإعداد المحتوى التعليمي">
    <form onSubmit={handleSubmit} noValidate className="mt-7 space-y-5">
      <FormInput id="login-email" label="البريد الإلكتروني" type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" placeholder="example@email.com" error={errors.email} icon={<Mail className="size-4" />} disabled={isSubmitting} />
      <PasswordInput id="login-password" label="كلمة المرور" value={password} onChange={setPassword} autoComplete="current-password" error={errors.password} disabled={isSubmitting} />
      <AuthButton type="submit" disabled={isSubmitting}>{isSubmitting ? <><Loader2 className="size-5 animate-spin" aria-hidden="true" />جارٍ تسجيل الدخول...</> : 'تسجيل الدخول'}</AuthButton>
    </form>
    <p className="mt-6 text-center text-sm text-slate-500">ليس لديك حساب؟ <Link to="/register" className="font-bold text-emerald-700 transition-colors duration-200 hover:text-emerald-800 hover:underline focus:outline-none focus:ring-2 focus:ring-emerald-600">إنشاء حساب</Link></p>
  </AuthLayout>
}
