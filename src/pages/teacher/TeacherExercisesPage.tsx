import {
  Download, LoaderCircle, Pencil, RotateCcw, Save,
  Sparkles, Wand2,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { TeacherPageHeader } from '../../components/teacher/TeacherPageHeader'
import { cn } from '../../utils/cn'
import {
  adaptExercise, generateExercises,
  type ExerciseAdaptIn, type ExerciseInput, type ExerciseItem,
  type ExercisePlan, type ExerciseSet,
} from '../../services/exerciseService'
import {
  getTeacherLibrary, getTeacherResource, saveTeacherResource, updateTeacherResource,
  type TeacherLibraryItem,
} from '../../services/teacherLibraryService'
import { exerciseTemplate, openPrintWindow } from '../../services/printService'

const LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'] as const
type Level = (typeof LEVELS)[number]

const SUGGESTIONS = [
  { icon: '👨‍👩‍👧', label: 'Famille', prompt: 'Crée 6 exercices sur la famille.' },
  { icon: '🏫', label: 'École', prompt: 'Crée 5 exercices sur la vie de l’école.' },
  { icon: '✈️', label: 'Voyage', prompt: 'Crée 5 exercices B1 sur le voyage pour travailler l’expression écrite.' },
  { icon: '📚', label: 'Vocabulaire', prompt: 'Crée 6 exercices de vocabulaire.' },
  { icon: '🔤', label: 'Grammaire', prompt: 'Crée 5 exercices A1 sur هذا و هذه.' },
  { icon: '💬', label: 'Communication', prompt: 'Crée 6 exercices sur la communication orale.' },
]

const STATUS_LABEL: Record<string, string> = {
  kb_original: 'Tiré de vos documents',
  adapted_from_kb: 'Adapté de vos documents par l’IA',
  ai_generated: 'Créé par l’IA',
}

const LEVEL_BADGE: Record<string, string> = {
  A1: 'text-emerald-700 ring-emerald-200', A2: 'text-teal-700 ring-teal-200',
  B1: 'text-sky-700 ring-sky-200', B2: 'text-indigo-700 ring-indigo-200',
  C1: 'text-violet-700 ring-violet-200', C2: 'text-fuchsia-700 ring-fuchsia-200',
}

/* =====================================================================
   Natural-language prompt parser
   The teacher writes plain text; the parser extracts the structured
   ExerciseInput. Heuristics only — fine-grained understanding is left to
   the generation engine on the backend.
   ===================================================================== */

type ParsedRequest = {
  level: Level | null
  count: number | null
  exercise_type: string | null
  skills: string[]
  theme: string
  objective: string
  language: ExerciseInput['language']
}

const COUNT_RE = /(\d+)\s*(?:exercices?|تمارين|تمرينات?|exercises|preguntas|ejercicios)/i

const TYPE_RULES: { re: RegExp; type: string }[] = [
  { re: /\b(qcm|choix multiples?|اختيار من متعدد|متعدد الخيارات|multiple choice)\b/i, type: 'QCM' },
  { re: /\b(vrai\s*ou\s*faux|صحيح أو خاطئ|صح أو خطأ|true\s*or\s*false)\b/i, type: 'Vrai ou faux' },
  { re: /\b(compl[êe]ter|texte[s]? \u00e0 trous|أكمل|املأ الفراغ|fill\s+in\s+the\s+blanks?|complete)\b/i, type: 'Compléter les phrases' },
  { re: /\b(relier|associer|associ[aà]tion|وصّل|صل|match)\b/i, type: 'Relier les mots' },
  { re: /\b(re[ée]ordonner|remettre dans l’?ordre|رتّب|رتب|reorder|put\s+in\s+order)\b/i, type: 'Remettre en ordre' },
  { re: /\b(transform[aà]tion|حول|حوّل|transform)\b/i, type: 'Transformation' },
  { re: /\b(question[s]?\s*ouvertes?|أسئلة مفتوحة|open[- ]?ended)\b/i, type: 'Question ouverte' },
  { re: /\b(production\s*écrite|rédaction|تعبير كتبي|التعبير الكتابي|written\s*production|essay)\b/i, type: 'Production écrite' },
  { re: /\b(production\s*orale|تعبير شفهي|التعبير الشفهي|oral\s*production)\b/i, type: 'Question ouverte' },
]

const SKILL_RULES: { re: RegExp; skill: string }[] = [
  { re: /\b(compr[ée]hension\s*(?:écrite|orale)|فهم|استيعاب|reading comprehension|listening comprehension)\b/i, skill: 'Compréhension écrite' },
  { re: /\b(production\s*(?:écrite|orale)|تعبير كتابي|تعبير شفهي|written\s*production|oral\s*production)\b/i, skill: 'Production écrite' },
  { re: /\b(communication\s*orale|محادثة|تكلم|communication|speaking)\b/i, skill: 'Production orale' },
  { re: /\b(interaction|تفاعل)\b/i, skill: 'Interaction' },
  { re: /\b(vocabulaire|مفردات|المفردات|vocabulary)\b/i, skill: 'Vocabulaire' },
  { re: /\b(grammaire|قواعد|النحو|قواعد اللغة|grammar)\b/i, skill: 'Grammaire' },
]

const LANGUAGE_RULES: { re: RegExp; lang: ExerciseInput['language'] }[] = [
  { re: /\b(en\s*français|en\s*francais)\b/i, lang: 'fr' },
  { re: /\b(in\s*english)\b/i, lang: 'en' },
  { re: /\b(en\s*español|en\s*espa\w+ol)\b/i, lang: 'es' },
]

const CEFR_RE = /\b(A1|A2|B1|B2|C1|C2)\b/i
const ARABIC_LEVEL_RE = /(?:مستوى|المستوى)\s*([أبج][12])/

const LEVEL_LETTER: Record<string, string> = { 'أ': 'A', 'ب': 'B', 'ج': 'C' }

const THEME_NOISE = /^(?:cr[ée]e[s]?|cr[ée]es?|fais|fais-moi|fais moi|g[ée]n[èe]re[s]?|donne[s]?|pr[ée]pare[s]?|construis|écris|propose[s]?|s'il te plaît|sil te plait|stp|merci|de|des|un|une|sur|à|au|aux|pour|avec|en|et|le|la|les|du|de la)\b/i

const THEME_STOP = /\b(?:travailler|exercices?|activit[ée]s?|qcm|niveau?|et\s|ou\s|pour\s|sur\s|en\s|avec\s|le\s|la\s|les\s|du\s|des\s|de\s|à\s|au\s|aux\s)\b/i

function detectLevel(text: string): Level | null {
  const arabic = ARABIC_LEVEL_RE.exec(text)
  if (arabic) {
    const letter = LEVEL_LETTER[arabic[1][0]]
    if (letter) {
      const candidate = `${letter}${arabic[1][1]}`
      if ((LEVELS as readonly string[]).includes(candidate)) return candidate as Level
    }
  }
  const match = CEFR_RE.exec(text)
  if (match) return match[1].toUpperCase() as Level
  return null
}

function detectType(text: string): string | null {
  for (const rule of TYPE_RULES) if (rule.re.test(text)) return rule.type
  return null
}

function detectSkills(text: string): string[] {
  const found: string[] = []
  for (const rule of SKILL_RULES) if (rule.re.test(text) && !found.includes(rule.skill)) found.push(rule.skill)
  return found
}

function detectLanguage(text: string): ExerciseInput['language'] {
  for (const rule of LANGUAGE_RULES) if (rule.re.test(text)) return rule.lang
  return 'ar'
}

function cleanTheme(text: string): string {
  let cleaned = text
    .replace(COUNT_RE, ' ')
    .replace(CEFR_RE, ' ')
    .replace(ARABIC_LEVEL_RE, ' ')
    .replace(/\bexercices?\b/gi, ' ')
    // Remove type + skill tokens so only the topic remains
    .replace(/qcm|choix multiples|vrai ou faux|compléter|relier|associer|reordonner|remettre dans l\'ordre|transformation|questions ouvertes|production écrite|production orale|vocabulaire|grammaire|compréhension écrite|compréhension orale|communication orale|interaction|rédaction/gi, ' ')
  // Cut at "pour" / "en" / "sur" when followed by a new instruction clause is risky;
  // instead strip leading verbs and prepositions, then keep the first meaningful chunk.
  cleaned = cleaned.replace(THEME_NOISE, ' ').trim()
  // Cut everything after a clause that introduces a purpose, to isolate the topic.
  const cut = cleaned.search(/\s+(?:pour|en vue de|afin d?e|de sorte que)\s+/i)
  if (cut > 0) cleaned = cleaned.slice(0, cut)
  cleaned = cleaned.replace(/\s+/g, ' ').trim()
  if (!cleaned) return text.trim().slice(0, 60)
  // Keep the first 5 words as a sane title length fallback.
  const words = cleaned.split(/\s+/)
  return words.slice(0, 5).join(' ') || cleaned
}

function parsePrompt(raw: string): ParsedRequest {
  const text = raw.trim()
  const countMatch = COUNT_RE.exec(text)
  const type = detectType(text)
  const skills = detectSkills(text)
  return {
    level: detectLevel(text),
    count: countMatch ? Number(countMatch[1]) : null,
    exercise_type: type,
    skills,
    theme: cleanTheme(text),
    objective: '',
    language: detectLanguage(text),
  }
}

/* ===================================================================== */

export function TeacherExercisesPage() {
  const [searchParams] = useSearchParams()
  const [prompt, setPrompt] = useState('')
  // 'auto' means: let the prompt decide; fall back to A1.
  const [level, setLevel] = useState<Level | 'auto'>('auto')
  const [setResult, setSetResult] = useState<ExerciseSet | null>(null)
  const [loading, setLoading] = useState(false)
  const [adaptingId, setAdaptingId] = useState<number | null>(null)
  const [saving, setSaving] = useState(false)
  const [editing, setEditing] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [savedId, setSavedId] = useState<number | null>(null)
  const [recent, setRecent] = useState<TeacherLibraryItem[]>([])
  const [shownCorrections, setShownCorrections] = useState<number[]>([])

  useEffect(() => {
    const id = Number(searchParams.get('resourceId'))
    if (!Number.isInteger(id) || id < 1) return
    void getTeacherResource(id)
      .then(resource => {
        if (resource.resource_type !== 'exercises') throw new Error()
        const content = resource.content as unknown as ExerciseSet
        setSetResult(content)
        setSavedId(resource.id)
        setEditing(false)
        setNotice('Exercices chargés depuis Mes créations.')
      })
      .catch(() => setError('Impossible d’ouvrir ces exercices.'))
  }, [searchParams])

  useEffect(() => {
    void getTeacherLibrary()
      .then(items => setRecent(items.filter(i => i.kind === 'creation' && i.resource_type === 'exercises').slice(0, 6)))
      .catch(() => setRecent([]))
  }, [savedId])

  const toggleCorrection = (index: number) =>
    setShownCorrections(current => current.includes(index) ? current.filter(i => i !== index) : [...current, index])

  const buildPayload = (): ExerciseInput => {
    const parsed = parsePrompt(prompt)
    // Explicit level in the text always wins; otherwise the selector; else A1.
    const resolvedLevel = parsed.level ?? (level === 'auto' ? 'A1' : level)
    return {
      level: resolvedLevel,
      theme: parsed.theme,
      objective: parsed.objective || null,
      skills: parsed.skills.length ? parsed.skills : ['Vocabulaire'],
      exercise_type: parsed.exercise_type ?? 'auto',
      count: parsed.count ? Math.min(Math.max(parsed.count, 1), 20) : 8,
      language: parsed.language,
      adapt_with_ai: true,
      special_instructions: null,
    }
  }

  const submit = async (event?: FormEvent) => {
    event?.preventDefault()
    if (loading) return
    const payload = buildPayload()
    if (!payload.theme.trim()) { setError('Décrivez votre besoin pour lancer la génération.'); return }
    setLoading(true); setError(''); setNotice('')
    try {
      const result = await generateExercises(payload)
      if (!result.exercises.length) {
        setError('Aucun exercice n’a pu être généré pour cette demande.')
        setSetResult(null)
      } else {
        setSetResult(result)
        setSavedId(null); setEditing(false); setShownCorrections([])
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'La génération a échoué.')
      setSetResult(null)
    } finally { setLoading(false) }
  }

  const adaptOne = async (index: number, instructions?: string) => {
    const item = setResult?.exercises[index]
    if (!item || adaptingId !== null) return
    setAdaptingId(index); setError('')
    try {
      const payload: ExerciseAdaptIn = {
        source: item, target_level: setResult?.level ?? 'A1', language: 'ar',
        instructions: instructions && instructions.trim() ? instructions.trim() : null,
      }
      const adapted = await adaptExercise(payload)
      setSetResult(current => current ? {
        ...current,
        exercises: current.exercises.map((ex, exIndex) => exIndex === index ? adapted : ex),
      } : current)
      setNotice('L’exercice a été adapté avec l’IA.')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'L’adaptation a échoué.')
    } finally { setAdaptingId(null) }
  }

  const saveSet = async () => {
    if (!setResult || saving) return
    setSaving(true); setError('')
    try {
      const data = {
        resource_type: 'exercises' as const, title: setResult.title,
        cefr_level: setResult.level, theme: setResult.theme,
        content: setResult as unknown as Record<string, unknown>,
      }
      const resource = savedId ? await updateTeacherResource(savedId, data) : await saveTeacherResource(data)
      setSavedId(resource.id)
      setNotice('✓ Exercices enregistrés dans Mes créations')
      setEditing(false)
    } catch {
      setError('Impossible d’enregistrer les exercices. Veuillez réessayer.')
    } finally { setSaving(false) }
  }

  const updateExercise = (index: number, field: 'title' | 'prompt' | 'answer_expectation', value: string) =>
    setSetResult(current => current ? {
      ...current,
      exercises: current.exercises.map((ex, exIndex) => exIndex === index ? { ...ex, [field]: value } : ex),
    } : current)

  const printPdf = () => {
    if (!setResult) return
    const base = slugify(setResult.title)
    if (!openPrintWindow(`exercices-${base}-${setResult.level}.pdf`, exerciseTemplate(setResult))) {
      setError('Impossible d’ouvrir la fenêtre d’impression. Veuillez autoriser les fenêtres pop-up pour Mo3allimAI.')
    }
  }

  const metaParts = useMemo(() => {
    if (!setResult || !setResult.exercises.length) return ''
    const parts: string[] = []
    if (setResult.level) parts.push(`Niveau ${setResult.level}`)
    if (setResult.theme?.trim()) parts.push(setResult.theme)
    if (setResult.skills?.length) parts.push(setResult.skills.join(', '))
    parts.push(`${setResult.exercises.length} exercice${setResult.exercises.length > 1 ? 's' : ''}`)
    return parts.join(' · ')
  }, [setResult])

  return (
    <div className="mx-auto w-full max-w-2xl space-y-8">
      <TeacherPageHeader title="Exercices IA" description="Créez des exercices adaptés à vos apprenants, en langage naturel." />

      <form onSubmit={submit} className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
        <div className="p-5 sm:p-6">
          <h2 className="text-lg font-bold text-slate-900">Que souhaitez-vous créer ?</h2>
          <p className="mt-1 text-sm text-slate-500">Écrivez votre demande simplement, l’IA s’occupe du reste.</p>

          <label htmlFor="exercise-prompt" className="mt-4 block">
            <textarea
              id="exercise-prompt"
              rows={5}
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="Exemple : Crée 6 exercices A1 sur la famille pour travailler le vocabulaire et la grammaire."
              className="w-full resize-y rounded-2xl border border-slate-300 bg-white p-4 text-base leading-relaxed text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
            />
          </label>

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <span className="text-sm font-medium text-slate-600">Niveau :</span>
            <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Niveau">
              <button type="button" role="radio" aria-checked={level === 'auto'} onClick={() => setLevel('auto')}
                className={cn('inline-flex h-9 min-w-11 items-center justify-center rounded-lg border px-2.5 text-sm font-semibold transition-colors', level === 'auto' ? 'border-[#065F46] bg-[#065F46] text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300')}>Auto</button>
              {LEVELS.map(lvl => (
                <button key={lvl} type="button" role="radio" aria-checked={level === lvl} onClick={() => setLevel(lvl)}
                  className={cn('inline-flex h-9 min-w-11 items-center justify-center rounded-lg border px-2.5 text-sm font-semibold transition-colors', level === lvl ? 'border-[#065F46] bg-[#065F46] text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300')}>{lvl}</button>
              ))}
            </div>
          </div>

          <div className="mt-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Suggestions</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {SUGGESTIONS.map(s => (
                <button key={s.label} type="button" onClick={() => setPrompt(s.prompt)}
                  className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:border-emerald-200 hover:bg-emerald-50 hover:text-emerald-800">
                  <span aria-hidden>{s.icon}</span>{s.label}
                </button>
              ))}
            </div>
          </div>

          <button type="submit" disabled={loading || !prompt.trim()}
            className="mt-6 inline-flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-[#065F46] px-4 text-sm font-bold text-white shadow-sm transition hover:bg-emerald-800 disabled:opacity-50 disabled:shadow-none">
            {loading ? <LoaderCircle className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
            {loading ? 'Génération en cours…' : '✨ Générer'}
          </button>
          {error && <p role="alert" className="mt-3 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{error}</p>}
        </div>
      </form>

      {loading && <LoadingPanel />}

      {!loading && setResult && setResult.exercises.length > 0 && (
        <section className="space-y-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="flex items-center gap-2 text-xl font-bold text-slate-900">Exercices générés <Wand2 className="size-5 text-emerald-600" /></h2>
              <p className="mt-1 text-xs text-slate-500">{metaParts}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <ActionButton onClick={() => setEditing(e => !e)} icon={Pencil} label={editing ? 'Fermer l’édition' : 'Modifier'} />
              <ActionButton onClick={() => void submit()} disabled={loading} icon={RotateCcw} label="Régénérer" />
              <ActionButton onClick={() => void saveSet()} disabled={saving} icon={saving ? LoaderCircle : Save} label={saving ? '…' : 'Enregistrer'} loading={saving} />
              <ActionButton onClick={printPdf} icon={Download} label="Imprimer / PDF" primary />
            </div>
          </div>

          <div className="space-y-4">
            {setResult.plan && <PlanPanel plan={setResult.plan} />}
            {setResult.exercises.map((ex, i) => (
              <ExerciseCard key={`${ex.title}-${i}`} index={i} ex={ex} editing={editing} adapting={adaptingId === i} onUpdate={updateExercise} onAdapt={adaptOne} correctionVisible={shownCorrections.includes(i)} onToggleCorrection={toggleCorrection} />
            ))}
          </div>
        </section>
      )}

      {!loading && !setResult && (
        <section className="rounded-2xl border border-dashed border-slate-300 bg-white p-8 text-center">
          <Sparkles className="mx-auto size-8 text-emerald-300" />
          <p className="mt-3 text-sm text-slate-500">Vos exercices apparaîtront ici, prêts à être relus, imprimés ou enregistrés.</p>
        </section>
      )}

      {recent.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-sm font-bold text-slate-900">Mes exercices récents</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {recent.map(item => <RecentCard key={`${item.id}-${item.kind}`} item={item} />)}
          </div>
        </section>
      )}

      {notice && !loading && <p className="rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">{notice}</p>}
    </div>
  )
}

function slugify(text: string) {
  return (text || 'sans-titre').normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-|-$/g, '').toLowerCase()
}

function LoadingPanel() {
  const STEPS = [
    'Analyse de votre demande',
    'Recherche de ressources pédagogiques (RAG)',
    'Planification de la progression (objectifs, distribution)',
    'Génération des exercices par l’IA',
    'Validation pédagogique et anti-doublons',
  ]
  return (
    <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm" role="status" aria-live="polite">
      <div className="text-center">
        <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-full bg-emerald-50"><LoaderCircle className="size-6 animate-spin text-emerald-600" /></div>
        <p className="mt-2 text-sm font-semibold text-slate-800">Génération en cours…</p>
      </div>
      <ol className="mx-auto max-w-sm space-y-2 text-left text-sm">
        {STEPS.map((step, i) => (
          <li key={step} className="flex items-center gap-2.5 text-slate-500">
            <span className={cn('inline-flex size-5 shrink-0 items-center justify-center rounded-full border text-[10px] font-bold', i === 0 ? 'animate-pulse border-emerald-400 bg-emerald-50 text-emerald-700' : 'border-slate-200 text-slate-400')}>{i + 1}</span>
            {step}
          </li>
        ))}
      </ol>
      <div className="mx-auto h-1.5 w-52 overflow-hidden rounded-full bg-slate-100"><div className="h-full w-1/2 animate-pulse rounded-full bg-emerald-500" /></div>
    </div>
  )
}

type IconProps = { className?: string }
type IconComponent = (props: IconProps) => React.ReactNode

const DIST_LABEL: Record<string, string> = {
  recognition: 'Reconnaissance', qcm: 'QCM', true_false: 'Vrai ou faux',
  matching: 'Relier', complete: 'Compléter', ordering: 'Remettre en ordre',
  grammar_transformation: 'Transformation', reading_comprehension: 'Compréhension écrite',
  open_question: 'Question ouverte', writing: 'Production écrite',
}

function PlanPanel({ plan }: { plan: ExercisePlan }) {
  const objectives = plan.learning_objectives?.length ? plan.learning_objectives : (plan.objective ? [plan.objective] : [])
  const vocab = plan.target_vocabulary?.slice(0, 12) ?? []
  const grammar = plan.target_grammar?.slice(0, 6) ?? []
  const distribution = plan.exercise_distribution ?? []
  return (
    <section className="rounded-2xl border border-emerald-100 bg-emerald-50/40 p-5 shadow-sm">
      <h3 className="flex items-center gap-2 text-sm font-bold text-slate-900">Pourquoi ces exercices ? <span className="text-xs font-normal text-slate-500">plan pédagogique</span></h3>
      {plan.objective && <p className="mt-2 text-sm text-slate-700">{plan.objective}</p>}
      <div className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
        {objectives.length > 0 && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">Objectifs d’apprentissage</p>
            <ul className="mt-1 list-inside list-disc space-y-1 text-slate-600">{objectives.map((o, i) => <li key={i}>{o}</li>)}</ul>
          </div>
        )}
        {distribution.length > 0 && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">Distribution</p>
            <ul className="mt-1 flex flex-wrap gap-1.5">{distribution.map((t, i) => <li key={i} className="rounded-full bg-white px-2.5 py-0.5 text-xs font-medium text-slate-700 ring-1 ring-emerald-200">{DIST_LABEL[t] ?? t}</li>)}</ul>
          </div>
        )}
        {vocab.length > 0 && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">Vocabulaire cible</p>
            <p className="mt-1 text-slate-600">{vocab.join(' · ')}</p>
          </div>
        )}
        {grammar.length > 0 && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">Grammaire cible</p>
            <p className="mt-1 text-slate-600">{grammar.join(' · ')}</p>
          </div>
        )}
      </div>
      {plan.rationale && <p className="mt-3 rounded-xl bg-white/70 p-3 text-xs leading-relaxed text-slate-500">{plan.rationale}</p>}
    </section>
  )
}

function ActionButton({ onClick, disabled, icon: Icon, label, primary, loading }: { onClick: () => void; disabled?: boolean; icon: IconComponent; label: string; primary?: boolean; loading?: boolean }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled}
      className={cn('inline-flex h-9 items-center gap-1.5 rounded-xl px-3 text-xs font-semibold transition', primary ? 'bg-[#065F46] text-white hover:bg-emerald-800 disabled:opacity-50' : 'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50')}>
      <Icon className={cn('size-4', loading && 'animate-spin')} />{label}
    </button>
  )
}

/* =====================================================================
   Typed exercise rendering
   The backend now returns structured fields (options / is_true / pairs)
   per exercise. These renderers surface that structure for each type,
   falling back to a clean textual view on older exercises that only
   carry `prompt`. Renders depend on exercise_type (never hardcoded items).
   ===================================================================== */

type ExerciseRenderKey =
  | 'qcm' | 'true_false' | 'matching' | 'ordering' | 'complete'
  | 'reading_comprehension' | 'writing' | 'grammar_transformation' | 'open_question'
  | 'recovery' | 'generic'

/** Map any backend id / legacy French/English label to a canonical key. */
function normalizeExerciseType(raw: string | undefined): ExerciseRenderKey {
  const t = (raw || '').trim().toLowerCase()
  if (!t) return 'generic'
  if (t.includes('qcm') || t.includes('choix multiple') || t.includes('multiple choice') || t.includes('متعدد')) return 'qcm'
  if (t.includes('vrai') || t.includes('faux') || t.includes('true') || t.includes('false') || t.includes('صحيح')) return 'true_false'
  if (t.includes('relier') || t.includes('associ') || t.includes('matching') || t.includes('match') || t.includes('وصل')) return 'matching'
  if (t.includes('ordre') || t.includes('reorder') || t.includes('ordering') || t.includes('رتب')) return 'ordering'
  if (t.includes('compléter') || t.includes('complete') || t.includes('trou')) return 'complete'
  if (t.includes('lecture') || t.includes('reading') || t.includes('compréhension écrite') || t.includes('written comprehension')) return 'reading_comprehension'
  if (t.includes('production orale') || t.includes('expression orale') || t.includes('orale') || t.includes('oral') || t.includes('speak')) return 'open_question'
  // Written production / ambiguous production → guided writing.
  if (t.includes('écrit') || t.includes('écrite') || t.includes('writing') || t.includes('rédaction') || t.includes('production') || t.includes('تعبير')) return 'writing'
  if (t.includes('transformation') || t.includes('transform') || t.includes('حول')) return 'grammar_transformation'
  if (t.includes('ouverte') || t.includes('open') || t.includes('question') || t.includes('سؤال')) return 'open_question'
  if (t === 'recovery' || t === 'recognition' || t.includes('reconnaissance') || t.includes('recogn')) return 'recovery'
  return 'generic'
}

function TypedExerciseContent({ ex }: { ex: ExerciseItem }) {
  const key = normalizeExerciseType(ex.exercise_type)
  const hasStructured =
    (ex.options && ex.options.length > 0) ||
    ex.is_true != null ||
    (ex.pairs && ex.pairs.length > 0)

  // Types whose display relies on structured fields. On an older exercise that
  // only carries `prompt`, fall back to a clean textual view.
  const structureDependent: ExerciseRenderKey[] = ['qcm', 'recovery', 'true_false', 'matching', 'ordering']
  if (structureDependent.includes(key) && !hasStructured) {
    return <p className="whitespace-pre-wrap text-sm text-slate-500">{ex.prompt}</p>
  }

  switch (key) {
    case 'qcm':
    case 'recovery':
      return (
        <ul className="space-y-1.5">
          {(ex.options ?? []).map((opt, oi) => (
            <li key={oi} className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2">
              <span className="inline-flex size-5 shrink-0 items-center justify-center rounded-full border border-slate-300 text-[10px] font-bold text-slate-500">{String.fromCharCode(65 + oi)}</span>
              <span className="text-slate-800">{opt}</span>
            </li>
          ))}
        </ul>
      )
    case 'true_false':
      return (
        <div className="flex flex-wrap gap-2">
          <span className={cn('rounded-lg border px-4 py-1.5 text-xs font-semibold', ex.is_true === true ? 'border-emerald-300 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-white text-slate-600')}>Vrai</span>
          <span className={cn('rounded-lg border px-4 py-1.5 text-xs font-semibold', ex.is_true === false ? 'border-rose-300 bg-rose-50 text-rose-700' : 'border-slate-200 bg-white text-slate-600')}>Faux</span>
        </div>
      )
    case 'matching':
      return (
        <ul className="space-y-1.5">
          {(ex.pairs ?? []).map((pair, pi) => (
            <li key={pi} className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-slate-800">
              <span className="min-w-0 flex-1">{pair.left || '—'}</span>
              <span className="text-slate-300">↔</span>
              <span className="min-w-0 flex-1 text-right">{pair.right || '—'}</span>
            </li>
          ))}
        </ul>
      )
    case 'ordering':
      return (
        <ol className="space-y-1.5">
          {(ex.options ?? []).map((opt, oi) => (
            <li key={oi} className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2">
              <span className="text-xs font-semibold text-slate-400">{oi + 1}.</span>
              <span className="text-slate-800">{opt}</span>
            </li>
          ))}
        </ol>
      )
    case 'reading_comprehension':
      return (
        <div className="space-y-2 rounded-lg bg-slate-50 p-3">
          {ex.context ? <p className="whitespace-pre-wrap text-sm text-slate-700">{ex.context}</p> : null}
          {ex.prompt ? <p className="whitespace-pre-wrap text-sm text-slate-800">{ex.prompt}</p> : null}
        </div>
      )
    case 'complete':
    case 'grammar_transformation':
      return (
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <p className="whitespace-pre-wrap text-sm text-slate-800">{ex.prompt}</p>
          <div className="mt-2 inline-block min-w-40 rounded-lg border-b-2 border-dashed border-emerald-400 bg-emerald-50/50 px-6 py-2" aria-label="Zone à compléter" />
        </div>
      )
    case 'writing':
    case 'open_question':
      return (
        <div className="rounded-lg border border-dashed border-slate-300 bg-white p-3">
          <p className="whitespace-pre-wrap text-sm text-slate-800">{ex.prompt}</p>
          <div className="mt-3 h-24 rounded-lg border border-slate-200 bg-slate-50/50" aria-label="Zone de rédaction" />
        </div>
      )
    default:
      return <p className="whitespace-pre-wrap text-sm text-slate-800">{ex.prompt}</p>
  }
}

function ExerciseCard({ index, ex, editing, adapting, onUpdate, onAdapt, correctionVisible, onToggleCorrection }: {
  index: number; ex: ExerciseItem; editing: boolean; adapting: boolean;
  onUpdate: (i: number, f: 'title' | 'prompt' | 'answer_expectation', v: string) => void;
  onAdapt: (i: number, instructions?: string) => void; correctionVisible: boolean; onToggleCorrection: (i: number) => void;
}) {
  const [adaptOpen, setAdaptOpen] = useState(false)
  const [adaptInstructions, setAdaptInstructions] = useState('')
  const canAdapt = ex.status === 'kb_original' || ex.status === 'adapted_from_kb'

  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-bold text-slate-900">Exercice {index + 1}</span>
          {ex.exercise_type && <TypeBadge type={ex.exercise_type} />}
          {ex.level && <span className={cn('rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset', LEVEL_BADGE[ex.level] ?? 'text-slate-600 ring-slate-200')}>{ex.level}{ex.level_source !== 'explicit' ? ' (estimé)' : ''}</span>}
          {ex.status && <span className="text-xs italic text-slate-400">{STATUS_LABEL[ex.status]}</span>}
        </div>
        {ex.skill && <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs text-amber-800">{ex.skill}</span>}
      </div>

      {editing ? (
        <div className="mt-3 space-y-3 text-sm">
          <label className="block"><span className="mb-1 block text-xs text-slate-500">Titre</span><input value={ex.title} onChange={e => onUpdate(index, 'title', e.target.value)} className="w-full rounded-lg border px-3 py-2" /></label>
          <label className="block"><span className="mb-1 block text-xs text-slate-500">Consigne</span><textarea rows={3} value={ex.prompt} onChange={e => onUpdate(index, 'prompt', e.target.value)} className="w-full rounded-lg border p-3" /></label>
          <label className="block"><span className="mb-1 block text-xs text-slate-500">Correction attendue</span><textarea rows={2} value={ex.answer_expectation ?? ''} onChange={e => onUpdate(index, 'answer_expectation', e.target.value)} className="w-full rounded-lg border p-3" /></label>
        </div>
      ) : (
        <div className="mt-3 space-y-3 text-sm">
          <p className="font-medium text-slate-500">Consigne</p>
          <div className="whitespace-pre-wrap leading-relaxed text-slate-800">{ex.prompt}</div>
          {ex.context ? <p className="whitespace-pre-wrap text-slate-500">{ex.context}</p> : null}
          <TypedExerciseContent ex={ex} />
        </div>
      )}

      {ex.answer_expectation && !editing && (
        <div className="mt-3">
          <button type="button" onClick={() => onToggleCorrection(index)} className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700 hover:text-emerald-800">{correctionVisible ? 'Masquer la correction' : '✓ Afficher la correction'}</button>
          {correctionVisible && <div className="mt-2 rounded-xl border border-emerald-100 bg-emerald-50/60 p-3 text-sm"><span className="font-semibold text-emerald-700">Correction : </span>{ex.answer_expectation}</div>}
        </div>
      )}

      <p className="mt-3 border-t border-slate-100 pt-3 text-xs text-slate-400">
        {ex.document_title ? <><span className="font-semibold text-slate-500">📚 Source :</span> {ex.document_title}{ex.page_start ? ` · page ${ex.page_start}${ex.page_end && ex.page_end !== ex.page_start ? `-${ex.page_end}` : ''}` : ''}{ex.original_level ? ` · adapté du niveau ${ex.original_level}` : ''}</> : <span className="italic text-slate-400">Créé avec l’IA (non lié à un document)</span>}
      </p>

      {!editing && canAdapt && (
        <div className="mt-3 flex flex-wrap items-center justify-end gap-2">
          <button type="button" onClick={() => setAdaptOpen(o => !o)} disabled={adapting} className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-emerald-200 bg-emerald-50 px-2.5 text-xs font-semibold text-emerald-700 hover:bg-emerald-100 disabled:opacity-50"><Wand2 className="size-3.5" />{adapting ? 'Adaptation…' : '✨ Modifier avec l’IA'}</button>
          {adaptOpen && !adapting && (<><input value={adaptInstructions} onChange={e => setAdaptInstructions(e.target.value)} placeholder="Ex. Rendre plus facile" aria-label="Instructions d’adaptation" className="h-8 w-44 rounded-lg border px-2 text-xs" /><button type="button" onClick={() => { onAdapt(index, adaptInstructions); setAdaptOpen(false); setAdaptInstructions('') }} className="inline-flex h-8 items-center rounded-lg bg-[#065F46] px-2.5 text-xs font-semibold text-white">Confirmer</button></>)}
        </div>
      )}
    </article>
  )
}

function TypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    'QCM': 'bg-sky-100 text-sky-800', 'Vrai ou faux': 'bg-amber-100 text-amber-800',
    'Compléter les phrases': 'bg-teal-100 text-teal-800', 'Relier les mots': 'bg-violet-100 text-violet-800',
    'Remettre en ordre': 'bg-indigo-100 text-indigo-800', 'Transformation': 'bg-fuchsia-100 text-fuchsia-800',
    'Question ouverte': 'bg-emerald-100 text-emerald-800', 'Production écrite': 'bg-rose-100 text-rose-800',
  }
  return <span className={cn('rounded-full px-2.5 py-0.5 text-xs font-semibold', colors[type] ?? 'bg-slate-100 text-slate-600')}>{type}</span>
}

function RecentCard({ item }: { item: TeacherLibraryItem }) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <h3 className="font-semibold text-slate-900">{item.title}</h3>
        {item.cefr_level && <span className={cn('shrink-0 rounded-full px-2 py-0.5 text-xs font-semibold ring-1 ring-inset', LEVEL_BADGE[item.cefr_level] ?? 'text-slate-600 ring-slate-200')}>{item.cefr_level}</span>}
      </div>
      {item.kind === 'creation' && item.theme ? <p className="mt-1 text-xs text-slate-500">{item.theme}</p> : null}
      <p className="mt-1 text-xs text-slate-400">Il y a <RelativeTime iso={item.created_at} /></p>
      <a href={`/teacher/tools/exercises?resourceId=${item.id}`} onClick={e => { e.preventDefault(); window.location.assign(`/teacher/tools/exercises?resourceId=${item.id}`) }} className="mt-3 inline-flex items-center gap-1 text-sm font-semibold text-emerald-700">Ouvrir</a>
    </article>
  )
}

function RelativeTime({ iso }: { iso: string }) {
  const [label, setLabel] = useState('récemment')
  useEffect(() => {
    const delta = Date.now() - new Date(iso).getTime()
    const minutes = Math.floor(delta / 60000)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)
    setLabel(days > 0 ? `${days} jour${days > 1 ? 's' : ''}` : hours > 0 ? `${hours} heure${hours > 1 ? 's' : ''}` : minutes > 0 ? `${minutes} minute${minutes > 1 ? 's' : ''}` : 'à l’instant')
  }, [iso])
  return <span>{label}</span>
}