import { CheckCircle2, PauseCircle, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ConfirmationModal } from '../../components/admin/ConfirmationModal'
import { deleteTeacher, getTeachers, updateTeacherStatus, type TeachersPage } from '../../services/adminService'

type Action = 'suspend' | 'activate' | 'delete'

export function AdminTeachersPage() {
  const { t } = useTranslation('admin')
  const [data, setData] = useState<TeachersPage | null>(null)
  const [search, setSearch] = useState('')
  const [filter] = useState('')
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState<TeachersPage['items'][number] | null>(null)
  const [action, setAction] = useState<Action | null>(null)
  const [loading, setLoading] = useState(false)
  const [notice, setNotice] = useState('')

  const load = () => {
    const query = new URLSearchParams({ page: String(page), page_size: '10', sort_order: 'desc' })
    if (search) query.set('search', search)
    if (filter) query.set('is_active', filter)
    getTeachers(`?${query}`).then(setData).catch(() => setNotice(t('teachers.error')))
  }
  useEffect(() => { const timer = setTimeout(load, 300); return () => clearTimeout(timer) }, [search, filter, page])

  const confirm = async () => {
    if (!selected || !action) return
    setLoading(true)
    try {
      if (action === 'delete') await deleteTeacher(selected.id)
      else await updateTeacherStatus(selected.id, action === 'activate')
      setNotice(action === 'delete' ? t('teachers.delete') : action === 'activate' ? t('teachers.activate') : t('teachers.suspend'))
      setAction(null); setSelected(null); load()
    } catch { setNotice(t('teachers.error')) } finally { setLoading(false) }
  }

  const actionLabel = action === 'delete' ? t('teachers.delete') : action === 'activate' ? t('teachers.activate') : t('teachers.suspend')
  const modalTitle = action === 'delete' ? t('teachers.deleteTitle') : action === 'activate' ? t('teachers.activateTitle') : t('teachers.suspendTitle')

  return <div>
    <h2 className="text-start text-3xl font-extrabold text-slate-950">{t('teachers.title')}</h2>
    {notice && <p className="mt-4 rounded-xl bg-emerald-50 p-3 text-emerald-800">{notice}</p>}
    <input value={search} onChange={event => { setSearch(event.target.value); setPage(1) }} placeholder={t('teachers.search')} className="mt-6 block w-full max-w-[420px] rounded-xl border border-slate-200 bg-white px-4 py-3 text-start text-sm shadow-sm outline-none transition focus:border-emerald-600 focus:ring-4 focus:ring-emerald-100" />
    <div className="mt-7 overflow-x-auto rounded-2xl border border-slate-200 bg-white shadow-sm">
      <table className="w-full min-w-[900px] table-fixed text-start">
        <colgroup><col className="w-1/4" /><col className="w-[30%]" /><col className="w-[18%]" /><col className="w-[27%]" /></colgroup>
        <thead className="bg-slate-50/70 text-sm font-semibold text-slate-700"><tr><th className="px-5 py-4">{t('teachers.teacher')}</th><th className="px-5 py-4">{t('teachers.email')}</th><th className="px-5 py-4">{t('teachers.status')}</th><th className="px-5 py-4">{t('teachers.actions')}</th></tr></thead>
        <tbody>{data?.items.map(teacher => <tr key={teacher.id} className="border-t border-slate-100 transition-colors hover:bg-slate-50/70">
          <td className="px-5 py-5"><div className="flex min-w-0 items-center gap-3"><span className="grid size-11 shrink-0 place-items-center rounded-full bg-emerald-100 font-bold text-emerald-800">{teacher.full_name[0]}</span><span className="truncate font-medium text-slate-900" title={teacher.full_name}>{teacher.full_name}</span></div></td>
          <td className="min-w-0 px-5 py-5 text-slate-600"><span dir="ltr" title={teacher.email} className="block truncate">{teacher.email}</span></td>
          <td className="px-5 py-5"><span className={`inline-flex min-w-24 items-center justify-center rounded-full px-3 py-1.5 text-sm font-medium ${teacher.is_active ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}>{teacher.is_active ? t('teachers.active') : t('teachers.suspended')}</span></td>
          <td className="px-5 py-5"><div className="flex items-center gap-2"><button type="button" disabled={loading} onClick={() => { setSelected(teacher); setAction(teacher.is_active ? 'suspend' : 'activate') }} className={`inline-flex h-11 min-w-[124px] items-center justify-center gap-2 rounded-xl border px-3 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${teacher.is_active ? 'border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100' : 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100'}`}>{teacher.is_active ? <PauseCircle className="size-4 shrink-0" /> : <CheckCircle2 className="size-4 shrink-0" />}{teacher.is_active ? t('teachers.suspend') : t('teachers.activate')}</button><button type="button" disabled={loading} onClick={() => { setSelected(teacher); setAction('delete') }} className="inline-flex h-11 min-w-[104px] items-center justify-center gap-2 rounded-xl border border-red-200 bg-red-50 px-3 text-sm font-medium text-red-700 transition hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-50"><Trash2 className="size-4 shrink-0" />{t('teachers.delete')}</button></div></td>
        </tr>)}</tbody>
      </table>
    </div>
    <ConfirmationModal isOpen={!!action} title={modalTitle} description={t('teachers.confirmDescription', { name: selected?.full_name ?? '' })} confirmLabel={actionLabel} cancelLabel={t('teachers.cancel')} variant={action === 'delete' ? 'danger' : action === 'activate' ? 'success' : 'warning'} isLoading={loading} onConfirm={confirm} onCancel={() => { setAction(null); setSelected(null) }} />
  </div>
}
