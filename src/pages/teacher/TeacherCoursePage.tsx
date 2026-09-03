import { Download, LoaderCircle, Pencil, Save, Sparkles } from 'lucide-react'
import { type ReactNode, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { TeacherPageHeader } from '../../components/teacher/TeacherPageHeader'
import { generateCourse, type Course, type CourseExercise, type CourseInput } from '../../services/courseService'
import { getTeacherResource, saveTeacherResource, updateTeacherResource } from '../../services/teacherLibraryService'
import { TeacherLibraryApiError } from '../../services/teacherLibraryService'
import { courseTemplate, openPrintWindow } from '../../services/printService'

const LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
const DURATIONS = [30, 45, 60, 90, 120, 180]
const OBJECTIVE_CHOICES = ['Présenter les membres de sa famille', 'Se présenter et parler de soi', 'Décrire son école et sa journée', 'Découvrir la culture marocaine', 'Développer l’expression orale', 'Développer la compréhension écrite', 'Travailler la grammaire', 'Développer l’argumentation']
const COMPETENCE_CHOICES = ['Compréhension orale', 'Compréhension écrite', 'Expression orale', 'Expression écrite', 'Interaction orale', 'Vocabulaire', 'Grammaire', 'Culture']

type ObjectiveMode = 'predefined' | 'custom' | 'none'

const initial: CourseInput = { level: 'A1', theme: '', objective: OBJECTIVE_CHOICES[0], skills: ['Expression orale'], duration_minutes: 45, audience: '', learner_count: undefined, language: 'ar', special_instructions: '' }

export function TeacherCoursePage() {
  const [searchParams] = useSearchParams()
  const [form, setForm] = useState<CourseInput>(initial)
  const [objectiveMode, setObjectiveMode] = useState<ObjectiveMode>('predefined')
  const [course, setCourse] = useState<Course | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [editing, setEditing] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [savedId, setSavedId] = useState<number | null>(null)

  useEffect(() => {
    const id = Number(searchParams.get('resourceId'))
    if (!Number.isInteger(id) || id < 1) return
    void getTeacherResource(id)
      .then(resource => {
        if (resource.resource_type !== 'course') throw new Error()
        setCourse(resource.content as unknown as Course)
        setSavedId(resource.id)
        setNotice('Cours chargé depuis Mes créations.')
        setEditing(false)
      })
      .catch(() => setError('Impossible d’ouvrir ce cours.'))
  }, [searchParams])

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (loading) return
    setLoading(true)
    setError('')
    const payload = { ...form, audience: form.audience?.trim() ? form.audience : undefined, learner_count: form.learner_count || undefined, objective: objectiveMode === 'none' ? undefined : (form.objective?.trim() ? form.objective : undefined) }
    try {
      setCourse(await generateCourse(payload))
      setSavedId(null)
      setNotice('')
      setEditing(false)
    } catch (e) {
      setError(e instanceof TeacherLibraryApiError && e.status === 429
        ? 'Le service IA est temporairement très sollicité. Veuillez patienter quelques secondes puis réessayer.'
        : (e instanceof Error ? e.message : 'La génération a échoué.'))
    } finally {
      setLoading(false)
    }
  }

  const saveCourse = async () => {
    if (!course || saving) return
    setSaving(true)
    setError('')
    try {
      const data = { resource_type: 'course' as const, title: course.title, cefr_level: course.level, theme: course.theme, content: course as unknown as Record<string, unknown> }
      const resource = savedId ? await updateTeacherResource(savedId, data) : await saveTeacherResource(data)
      setSavedId(resource.id)
      setNotice('✓ Cours enregistré dans Mes créations')
      setEditing(false)
    } catch {
      setError('Impossible d’enregistrer le cours. Veuillez réessayer.')
    } finally {
      setSaving(false)
    }
  }

  const printPdf = () => {
    if (!course) return
    const base = course.title.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-|-$/g, '').toLowerCase() || 'sans-titre'
    if (!openPrintWindow(`cours-${base}-${course.level}.pdf`, courseTemplate(course))) {
      setError('Impossible d’ouvrir la fenêtre d’impression. Veuillez autoriser les fenêtres pop-up pour Mo3allimAI.')
    }
  }

  return (
    <div className="space-y-6">
      <TeacherPageHeader title="Générateur de cours" description="Créez un contenu pédagogique structuré, adapté à votre classe et à votre niveau CECRL." />
      <div className="grid gap-6 xl:grid-cols-2">
        <form onSubmit={submit} className="rounded-2xl border bg-white p-6 shadow-sm">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Niveau">
              <select value={form.level} onChange={e => setForm({ ...form, level: e.target.value })}>
                {LEVELS.map(x => <option key={x}>{x}</option>)}
              </select>
            </Field>
            <Field label="Durée (minutes)">
              <select value={form.duration_minutes} onChange={e => setForm({ ...form, duration_minutes: Number(e.target.value) })}>
                {DURATIONS.map(x => <option key={x} value={x}>{x}</option>)}
              </select>
            </Field>
          </div>
          <Field label="Thème"><input required value={form.theme} onChange={e => setForm({ ...form, theme: e.target.value })} placeholder="Ex. La famille, Le voyage, الثقافة المغربية" /></Field>
          <Field label="Objectif pédagogique">
            <div className="flex flex-wrap gap-1.5">
              <ModeButton active={objectiveMode === 'predefined'} onClick={() => { setObjectiveMode('predefined'); setForm({ ...form, objective: OBJECTIVE_CHOICES[0] }) }}>Prédéfini</ModeButton>
              <ModeButton active={objectiveMode === 'custom'} onClick={() => { setObjectiveMode('custom'); setForm({ ...form, objective: '' }) }}>Personnalisé</ModeButton>
              <ModeButton active={objectiveMode === 'none'} onClick={() => setObjectiveMode('none')}>Aucun (l’IA définit)</ModeButton>
            </div>
            {objectiveMode === 'predefined' && (
              <select value={form.objective ?? ''} onChange={e => setForm({ ...form, objective: e.target.value })}>
                {OBJECTIVE_CHOICES.map(x => <option key={x}>{x}</option>)}
              </select>
            )}
            {objectiveMode === 'custom' && (
              <textarea rows={2} placeholder="Ex. Apprendre à décrire son trajet quotidien…" value={form.objective ?? ''} onChange={e => setForm({ ...form, objective: e.target.value })} />
            )}
            {objectiveMode === 'none' && (
              <p className="text-xs text-slate-500">L’IA définira un objectif pédagogique pertinent pour votre thème et votre niveau.</p>
            )}
          </Field>
          <Field label="Compétence">
            <select value={form.skills[0]} onChange={e => setForm({ ...form, skills: [e.target.value] })}>
              {COMPETENCE_CHOICES.map(x => <option key={x}>{x}</option>)}
            </select>
          </Field>
          <Field label="Public / âge"><input value={form.audience ?? ''} onChange={e => setForm({ ...form, audience: e.target.value })} placeholder="Ex. Débutants, enfants 6-8 ans" /></Field>
          <Field label="Langue de production">
            <select value={form.language} onChange={e => setForm({ ...form, language: e.target.value as CourseInput['language'] })}>
              <option value="ar">Arabe</option>
              <option value="fr">Français</option>
              <option value="en">Anglais</option>
              <option value="es">Espagnol</option>
            </select>
          </Field>
          <Field label="Instructions supplémentaires"><textarea rows={3} value={form.special_instructions ?? ''} onChange={e => setForm({ ...form, special_instructions: e.target.value })} placeholder="Ex. Privilégier le vocabulaire concret, inclure un dialogue…" /></Field>
          <button disabled={loading} className="mt-5 inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-[#065F46] px-4 font-semibold text-white disabled:bg-slate-400">
            {loading ? <LoaderCircle className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
            {loading ? 'Recherche des ressources et génération…' : 'Générer le cours'}
          </button>
          {error && <p className="text-sm text-rose-600">{error}</p>}
        </form>

        <div className="space-y-4">
          {course && (
            <div className="flex w-full flex-wrap gap-2">
              <button onClick={() => setEditing(true)} className="inline-flex h-10 items-center gap-2 rounded-xl border border-slate-300 px-4 text-sm font-semibold text-slate-700"><Pencil className="size-4" />Modifier</button>
              <button disabled={saving} onClick={() => void saveCourse()} className="inline-flex h-10 items-center gap-2 rounded-xl border border-slate-300 px-4 text-sm font-semibold text-slate-700"><Save className="size-4" />{saving ? '…' : 'Enregistrer'}</button>
              <button onClick={printPdf} className="inline-flex h-10 items-center gap-2 rounded-xl bg-[#065F46] px-4 text-sm font-semibold text-white"><Download className="size-4" />Télécharger PDF</button>
            </div>
          )}
          {notice && <p className="text-sm text-emerald-700">{notice}</p>}
          {course && editing && (
            <div className="rounded-2xl border bg-white p-5">
              <p className="mb-2 font-semibold">Modifier le titre</p>
              <input value={course.title} onChange={e => setCourse({ ...course, title: e.target.value })} className="w-full rounded-lg border px-3 py-2" />
              <p className="mb-2 mt-4 font-semibold">Introduction</p>
              <textarea rows={3} value={course.introduction ?? ''} onChange={e => setCourse({ ...course, introduction: e.target.value })} className="w-full rounded-lg border p-3" />
              <div className="mt-4 flex gap-2"><button onClick={() => setEditing(false)} className="rounded-xl bg-[#065F46] px-4 py-2 text-sm text-white">Terminer la modification</button></div>
            </div>
          )}
          {course && !editing ? <CourseView course={course} /> : !course ? (
            <div className="rounded-2xl border border-dashed bg-white px-6 py-12 text-center text-slate-400">Le contenu pédagogique généré apparaîtra ici.</div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return <label className="mt-4 block text-sm font-medium text-slate-700 first:mt-0">{label}<span className="mt-1 block [&_input]:h-10 [&_input]:w-full [&_input]:rounded-lg [&_input]:border [&_input]:px-3 [&_select]:h-10 [&_select]:w-full [&_select]:rounded-lg [&_select]:border [&_select]:px-3 [&_textarea]:w-full [&_textarea]:rounded-lg [&_textarea]:border [&_textarea]:p-3">{children}</span></label>
}

function ModeButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return <button type="button" onClick={onClick} className={"rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors " + (active ? "border-[#065F46] bg-emerald-50 text-[#065F46]" : "border-slate-200 text-slate-600 hover:border-slate-300")}>{children}</button>
}

function CourseView({ course }: { course: Course }) {
  const grammar = course.grammar ?? []
  const content = course.content ?? []
  const comprehension = course.comprehension ?? []
  const guided = course.guided_practice ?? []
  const communicative = course.communicative_practice ?? []
  const production = course.production ?? []
  const summary = Array.isArray(course.summary) ? course.summary : (course.summary ? [String(course.summary)] : [])
  const dialogue = course.dialogue && typeof course.dialogue === 'object' ? course.dialogue : null
  return (
    <article className="space-y-5 text-sm">
      <h2 className="text-xl font-bold">📚 {course.title}</h2>
      <Section title="Informations générales">
        <p><b>Niveau :</b> {course.level} · <b>Thème :</b> {course.theme} · <b>Durée :</b> {course.duration} min</p>
        <p><b>Compétences :</b> {(course.skills ?? []).join(', ')}</p>
      </Section>
      <Section title="🎯 Objectifs"><List items={course.objectives ?? []} /></Section>
      <Section title="📌 Introduction"><p style={{ whiteSpace: 'pre-wrap' }}>{course.introduction ?? ''}</p></Section>
      <Section title="🧠 Vocabulaire"><List items={course.vocabulary ?? []} /></Section>
      {(course.expressions ?? []).length > 0 && <Section title="💬 Expressions"><List items={course.expressions ?? []} /></Section>}
      {grammar.length > 0 && (
        <Section title="📐 Grammaire">
          {grammar.map((g, i) => (
            <div key={i} className="border-t py-2">
              <b>{g.title}</b>
              <p style={{ whiteSpace: 'pre-wrap' }}>{g.body}</p>
              {(g.examples ?? []).map((ex, j) => <p key={j} dir="auto" className="text-slate-600">{ex.title} — {ex.body}</p>)}
            </div>
          ))}
        </Section>
      )}
      <Section title="📖 Contenu du cours">
        {content.map((b, i) => (
          <div key={i} className="border-t py-2">
            <b>{b.title}</b>
            <p style={{ whiteSpace: 'pre-wrap' }}>{b.body}</p>
            {(b.examples ?? []).map((ex, j) => <p key={j} dir="auto" className="text-slate-600">{ex.title} — {ex.body}</p>)}
          </div>
        ))}
      </Section>
      {dialogue && (
        <Section title="🗣️ Dialogue">
          {dialogue.context ? <p><em>{dialogue.context}</em></p> : null}
          {dialogue.lines.map((line, i) => <p key={i} dir="auto" style={{ whiteSpace: 'pre-wrap' }}>{line}</p>)}
        </Section>
      )}
      {comprehension.length > 0 && <Section title="❓ Compréhension"><Exercises items={comprehension} /></Section>}
      {guided.length > 0 && <Section title="✏️ Pratique guidée"><Exercises items={guided} /></Section>}
      {communicative.length > 0 && <Section title="💬 Pratique communicative"><Exercises items={communicative} /></Section>}
      {production.length > 0 && <Section title="🎨 Production"><Exercises items={production} /></Section>}
      <Section title="📌 Synthèse"><List items={summary} /></Section>
      {course.homework && <Section title="🏠 Devoir"><p style={{ whiteSpace: 'pre-wrap' }}>{course.homework}</p></Section>}
    </article>
  )
}

function Exercises({ items }: { items: CourseExercise[] }) {
  return (
    <div>
      {items.map((ex, i) => (
        <div key={i} className="border-t py-2">
          <b>{ex.title}</b>
          {ex.instructions ? <p style={{ whiteSpace: 'pre-wrap' }}>{ex.instructions}</p> : null}
          {ex.example ? <p dir="auto" className="text-slate-600">{ex.example.title} — {ex.example.body}</p> : null}
        </div>
      ))}
    </div>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section><h3 className="mb-2 font-bold">{title}</h3>{children}</section>
}

function List({ items }: { items: string[] }) {
  return <ul className="list-disc ps-5">{items.map((item, i) => <li key={i} dir="auto">{item}</li>)}</ul>
}
