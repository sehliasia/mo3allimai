import { LoaderCircle, Save, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { useParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { sendAssistantMessage, type AssistantCEFRLevel } from '../../services/assistantService'
import { saveTeacherResource } from '../../services/teacherLibraryService'

const labels: Record<string, { title: string; description: string; instruction: string; defaultTitle: string }> = {
  exercises: {
    title: 'Générateur d’exercices',
    description: 'Créez des exercices adaptés au niveau et au thème choisis, avec corrigé.',
    instruction: 'Génère une série d’exercices de langue arabe adaptés au niveau demandé. Pour chaque exercice, donne la consigne, les questions, puis un corrigé clair. Varie les types d’exercices et reste strictement au niveau indiqué.',
    defaultTitle: 'Exercices de langue arabe',
  },
  assessment: {
    title: 'Générateur d’évaluation',
    description: 'Préparez une évaluation structurée avec critères, barème et corrigé.',
    instruction: 'Construis une évaluation de langue arabe adaptée au niveau demandé. Inclue les compétences évaluées, les exercices, un barème explicite, les critères de réussite et un corrigé. Ne dépasse pas le niveau indiqué.',
    defaultTitle: 'Évaluation de langue arabe',
  },
  exam: {
    title: 'Générateur d’examen',
    description: 'Construisez un examen complet avec différentes parties et corrigé.',
    instruction: 'Construis un examen complet de langue arabe adapté au niveau demandé. Organise-le en parties, précise le nombre de points par partie, les consignes et fournis un corrigé détaillé.',
    defaultTitle: 'Examen de langue arabe',
  },
}

export function TeacherAiToolPage() {
  const { toolId } = useParams()
  const config = toolId ? labels[toolId] : undefined
  const [level, setLevel] = useState('A1')
  const [theme, setTheme] = useState('')
  const [count, setCount] = useState(5)
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  if (!config || !toolId) return null

  const generate = async (event: React.FormEvent) => {
    event.preventDefault()
    setLoading(true); setError(''); setNotice('')
    try {
      const result = await sendAssistantMessage({
        message: `${config.instruction}\n\nNiveau CECRL : ${level}\nThème : ${theme || 'général'}\nNombre souhaité : ${count}\nRéponds en français pour les explications et en arabe pour les exercices destinés aux apprenants.`,
        cefr_level: level as AssistantCEFRLevel,
        language: 'fr',
        topic: theme || undefined,
        objective: config.title,
        top_k: 8,
        mode: 'knowledge_base',
      })
      setContent(result.answer)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'La génération a échoué.')
    } finally { setLoading(false) }
  }

  const save = async () => {
    if (!content || saving) return
    setSaving(true); setError(''); setNotice('')
    try {
      await saveTeacherResource({ resource_type: toolId as 'exercises', title: `${config.defaultTitle} — ${theme || level}`, cefr_level: level, theme: theme || level, content: { generated_content: content, tool: toolId } })
      setNotice('✓ Ressource enregistrée dans Mes créations.')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Impossible d’enregistrer la ressource.')
    } finally { setSaving(false) }
  }

  return <div className="space-y-6">
    <header><h1 className="text-2xl font-bold text-slate-900">{config.title}</h1><p className="mt-1 text-sm text-slate-500">{config.description}</p></header>
    <div className="grid gap-6 xl:grid-cols-2">
      <form onSubmit={generate} className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <label className="block text-sm font-medium text-slate-700">Niveau<select value={level} onChange={e => setLevel(e.target.value)} className="mt-1 h-11 w-full rounded-xl border border-slate-200 px-3">{['A1','A2','B1','B2','C1','C2'].map(x => <option key={x}>{x}</option>)}</select></label>
        <label className="mt-4 block text-sm font-medium text-slate-700">Thème<input value={theme} onChange={e => setTheme(e.target.value)} placeholder="Ex. La famille, les couleurs..." className="mt-1 h-11 w-full rounded-xl border border-slate-200 px-3" /></label>
        <label className="mt-4 block text-sm font-medium text-slate-700">Nombre de questions / activités<input type="number" min="1" max="30" value={count} onChange={e => setCount(Number(e.target.value))} className="mt-1 h-11 w-full rounded-xl border border-slate-200 px-3" /></label>
        <button disabled={loading} className="mt-6 inline-flex h-11 items-center gap-2 rounded-xl bg-[#065F46] px-4 font-semibold text-white disabled:bg-slate-400">{loading ? <LoaderCircle className="size-4 animate-spin"/> : <Sparkles className="size-4"/>}{loading ? 'Génération…' : 'Générer'}</button>
        {error && <p className="mt-3 text-sm text-rose-700">{error}</p>}
      </form>
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        {content ? <><div className="mb-4 flex justify-end"><button onClick={() => void save()} disabled={saving} className="inline-flex items-center gap-2 rounded-xl bg-[#065F46] px-4 py-2 text-sm font-semibold text-white disabled:bg-slate-400"><Save className="size-4"/>{saving ? 'Enregistrement…' : 'Enregistrer'}</button></div>{notice && <p className="mb-3 text-sm text-emerald-700">{notice}</p>}<article className="prose prose-slate max-w-none text-sm"><ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown></article></> : <p className="text-sm text-slate-500">Le contenu généré apparaîtra ici.</p>}
      </section>
    </div>
  </div>
}

