import { Loader2, Mail } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AuthButton } from '../components/auth/AuthButton'
import { AuthLayout } from '../components/auth/AuthLayout'
import { FormInput } from '../components/auth/FormInput'
import { PasswordInput } from '../components/auth/PasswordInput'
import { login, saveSession } from '../services/authService'
import { isValidEmail } from '../utils/validation'
type LoginErrors = Partial<Record<'email' | 'password' | 'form', string>>
export function LoginPage() {
  const [email, setEmail] = useState(''); const [password, setPassword] = useState(''); const [errors, setErrors] = useState<LoginErrors>({}); const [isSubmitting, setIsSubmitting] = useState(false); const navigate = useNavigate()
  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault(); const next: LoginErrors = {}; if (!isValidEmail(email)) next.email = 'يرجى إدخال بريد إلكتروني صحيح.'; if (!password) next.password = 'كلمة المرور مطلوبة.'; setErrors(next); if (Object.keys(next).length) return
    setIsSubmitting(true); try { const result = await login({ email, password }); saveSession(result.access_token, result.user); navigate(result.user.role === 'admin' ? '/admin/dashboard' : '/teacher/dashboard') } catch (error) { setErrors({ form: error instanceof Error ? error.message : 'تعذر الاتصال بالخادم' }) } finally { setIsSubmitting(false) }
  }
  return <AuthLayout title="تسجيل الدخول" description="مساعدك الذكي لإعداد المحتوى التعليمي"><form onSubmit={handleSubmit} noValidate className="mt-7 space-y-5">{errors.form && <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{errors.form}</p>}<FormInput id="login-email" label="البريد الإلكتروني" type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" placeholder="example@email.com" error={errors.email} icon={<Mail className="size-4" />} disabled={isSubmitting} /><PasswordInput id="login-password" label="كلمة المرور" value={password} onChange={setPassword} autoComplete="current-password" error={errors.password} disabled={isSubmitting} /><AuthButton type="submit" disabled={isSubmitting}>{isSubmitting ? <><Loader2 className="size-5 animate-spin" aria-hidden="true" />جارٍ تسجيل الدخول...</> : 'تسجيل الدخول'}</AuthButton></form><p className="mt-6 text-center text-sm text-slate-500">ليس لديك حساب؟ <Link to="/register" className="font-bold text-emerald-700 hover:underline">إنشاء حساب</Link></p></AuthLayout>
}
