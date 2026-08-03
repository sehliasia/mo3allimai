import { Loader2, Mail, UserRound } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AuthButton } from '../components/auth/AuthButton'
import { AuthLayout } from '../components/auth/AuthLayout'
import { FormInput } from '../components/auth/FormInput'
import { PasswordInput } from '../components/auth/PasswordInput'
import { register } from '../services/authService'
import { isValidEmail } from '../utils/validation'

export function RegisterPage() {
  const [name, setName] = useState(''); const [email, setEmail] = useState(''); const [password, setPassword] = useState(''); const [confirm, setConfirm] = useState(''); const [accepted, setAccepted] = useState(false); const [message, setMessage] = useState(''); const [loading, setLoading] = useState(false); const navigate = useNavigate()
  async function submit(event: React.FormEvent<HTMLFormElement>) { event.preventDefault(); if (!name.trim() || !isValidEmail(email) || password.length < 6 || password !== confirm || !accepted) { setMessage('يرجى التحقق من البيانات المدخلة'); return }; setLoading(true); try { await register({ full_name: name, email, password }); navigate('/login', { state: { success: 'تم إنشاء الحساب بنجاح' } }) } catch (error) { setMessage(error instanceof Error ? error.message : 'تعذر الاتصال بالخادم') } finally { setLoading(false) } }
  return <AuthLayout title="إنشاء حساب" description="أنشئ حسابك وابدأ استخدام Mo3allimAI."><form onSubmit={submit} noValidate className="mt-8 space-y-4">{message && <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{message}</p>}<FormInput id="register-name" label="الاسم الكامل" value={name} onChange={(event) => setName(event.target.value)} icon={<UserRound className="size-4" />} disabled={loading} /><FormInput id="register-email" label="البريد الإلكتروني" type="email" value={email} onChange={(event) => setEmail(event.target.value)} icon={<Mail className="size-4" />} disabled={loading} /><PasswordInput id="register-password" label="كلمة المرور" value={password} onChange={setPassword} disabled={loading} /><PasswordInput id="register-confirm" label="تأكيد كلمة المرور" value={confirm} onChange={setConfirm} disabled={loading} /><label className="flex gap-2 text-sm text-slate-600"><input type="checkbox" checked={accepted} onChange={(event) => setAccepted(event.target.checked)} />أوافق على الشروط وسياسة الخصوصية</label><AuthButton type="submit" disabled={loading}>{loading ? <><Loader2 className="size-5 animate-spin" />جارٍ إنشاء الحساب...</> : 'إنشاء الحساب'}</AuthButton></form><p className="mt-7 text-center text-sm text-slate-600">لديك حساب بالفعل؟ <Link to="/login" className="font-bold text-emerald-700 hover:underline">تسجيل الدخول</Link></p></AuthLayout>
}
