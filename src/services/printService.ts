import type { Activity } from './activityService'
import type { LessonPlan } from './lessonPlanService'
import type { Course } from './courseService'
import type { ExerciseSet, ExerciseItem } from './exerciseService'

/* =====================================================================
   Mo3allimAI — Reusable professional print templates
   Renders A4, portrait, print-ready documents (SEC pedagogical sheet
   and activity). Uses the browser's native print pipeline (window.print)
   which provides correct Unicode Arabic shaping and RTL handling.
   ===================================================================== */

const BRAND = 'Mo3allimAI'
const TAGLINE = "Plateforme intelligente d'accompagnement pédagogique"

const esc = (value: string) => value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

const hasArabic = (text: string) => /[\u0600-\u06FF]/.test(text)

/* Apply RTL direction only to blocks that actually contain Arabic,
   leaving French/English/Spanish content LTR (requirement n°9). */
const rtlAttr = (text: string) => (hasArabic(text) ? ' dir="rtl" style="text-align:right;unicode-bidi:embed"' : '')

const sectionTitle = (label: string) =>
  `<div class="section-title avoid-break"><span>${esc(label)}</span></div>`

const section = (label: string, inner: string) => `${sectionTitle(label)}${inner}`

const list = (items: string[]) =>
  items.length
    ? `<ul>${items.map((item) => `<li${rtlAttr(item)}>${esc(item)}</li>`).join('')}</ul>`
    : '<p class="muted">—</p>'

/* Professional checkbox list (used for assessment criteria). */
const checkList = (items: string[]) =>
  items.length
    ? `<ul class="checks">${items.map((item) => `<li${rtlAttr(item)}>${esc(item)}</li>`).join('')}</ul>`
    : '<p class="muted">—</p>'

/* Two-column presentation for teacher/learner roles. */
const roles = (teacher: string, learners: string) => `
  <div class="roles avoid-break">
    <div class="roles-col">
      <div class="roles-title">Enseignant</div>
      <div${rtlAttr(teacher)}>${esc(teacher)}</div>
    </div>
    <div class="roles-col">
      <div class="roles-title">Apprenants</div>
      <div${rtlAttr(learners)}>${esc(learners)}</div>
    </div>
  </div>`

const metaGrid = (cells: { label: string; value: string }[]) => `
  <div class="meta-grid avoid-break">
    ${cells.map((cell) => `<div class="meta-item"><div class="meta-label">${esc(cell.label)}</div><div class="meta-value"${rtlAttr(cell.value)}>${esc(cell.value)}</div></div>`).join('')}
  </div>`

/* Reusable procedure table with a repeating header across page breaks and
   rows that never split mid-line (requirement n°8, n°11). */
interface TableColumn { head: string; render: (row: unknown, rowIndex: number) => string; width?: string }
const table = (columns: TableColumn[], rows: unknown[]) => `
  <table class="data-table">
    <thead><tr>${columns.map((col) => `<th style="width:${col.width ?? 'auto'}">${esc(col.head)}</th>`).join('')}</tr></thead>
    <tbody>
      ${rows.map((row, index) => `<tr>${columns.map((col) => `<td>${col.render(row, index)}</td>`).join('')}</tr>`).join('')}
    </tbody>
  </table>`

const labelValue = (label: string, value: string) =>
  value ? `<p><strong>${esc(label)} :</strong> <span${rtlAttr(value)}>${esc(value)}</span></p>` : ''

const paragraph = (value: string) =>
  value ? `<p${rtlAttr(value)}>${esc(value)}</p>` : '<p class="muted">—</p>'

function buildDocument(opts: { type: string; title: string; body: string; footerNote?: string }): string {
  return `<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>${esc(opts.title)}</title><style>${CSS}</style></head><body>
<nav class="doc-header">
  <div class="brand">
    <span class="brand-name">${BRAND}</span>
    <span class="brand-tag">${TAGLINE}</span>
  </div>
  <div class="doc-type">${esc(opts.type)}</div>
</nav>
<main>
  <h1 class="doc-title">${esc(opts.title)}</h1>
  ${opts.body}
</main>
</body></html>`
}

/**
 * Open a new window, inject the full HTML document, wait for the DOM/layout to
 * actually be ready, then trigger printing. Returns false (never a silent blank
 * window) when the popup cannot be opened.
 *
 * The print call is deliberately NOT fired from inside the written HTML (an
 * inline `window.onload=()=>window.print()` races the CSS/layout and can print
 * an empty page). Instead we drive it from this parent frame once the document
 * reports it is complete, plus a short settle delay so print captures real
 * rendered content.
 */
export function openPrintWindow(filename: string, template: { type: string; title: string; html: string; footerNote?: string }): boolean {
  console.log('[PDF] opening print window', filename)
  const printWindow = window.open('', '_blank')
  if (!printWindow) {
    console.warn('[PDF] popup blocked: window.open returned null')
    return false
  }
  try {
    const html = buildDocument({ type: template.type, title: template.title, body: template.html, footerNote: template.footerNote })
    console.log('[PDF] HTML length:', html.length)
    printWindow.document.open()
    printWindow.document.write(html)
    printWindow.document.close()
    console.log('[PDF] document readyState after close:', printWindow.document.readyState)

    // Let the browser finish parsing/painting before printing.
    const printWhenReady = () => {
      setTimeout(() => {
        const body = printWindow.document.body
        console.log('[PDF] document.body.innerHTML length:', body ? body.innerHTML.length : -1)
        if (body && body.innerHTML.length > 0) {
          printWindow.focus()
          printWindow.print()
        } else {
          console.warn('[PDF] body is empty before print; aborting to avoid a blank page.')
        }
      }, 300)
    }

    if (printWindow.document.readyState === 'complete') {
      printWhenReady()
    } else {
      printWindow.addEventListener('load', printWhenReady, { once: true })
    }

    // Close the popup only after the user has finished printing.
    printWindow.onafterprint = () => { printWindow.close() }
    return true
  } catch (error) {
    console.error('[PDF] print flow error', error)
    return false
  }
}

/* =====================================================================
   Shared helpers exported for both templates
   ===================================================================== */
export const print = { esc, rtlAttr, section, sectionTitle, list, checkList, roles, metaGrid, table, labelValue, paragraph }

/* =====================================================================
   ACTIVITY TEMPLATE
   ===================================================================== */
export function activityTemplate(activity: Activity) {
  const body = [
    metaGrid([
      { label: 'Niveau', value: activity.level },
      { label: 'Thème', value: activity.theme },
      { label: 'Type', value: activity.activity_type },
      { label: 'Durée', value: `${activity.duration} min` },
    ]),
    section('Objectif', paragraph(activity.objective)),
    section('Compétences', list(activity.skills)),
    section('Consigne', paragraph(activity.instructions)),
    section('Matériel', list(activity.materials)),
    section('Déroulement', table(
      [
        { head: 'Étape', width: '10%', render: (row) => String((row as { step: number }).step) },
        { head: 'Durée', width: '14%', render: (row) => `${(row as { duration: number }).duration} min` },
        { head: 'Titre', width: '22%', render: (row) => `<span${rtlAttr((row as { title: string }).title)}>${esc((row as { title: string }).title)}</span>` },
        { head: 'Déroulement', render: (row) => `<span${rtlAttr((row as { description: string }).description)}>${esc((row as { description: string }).description)}</span>` },
      ],
      activity.procedure,
    )),
    section('Rôles', roles(activity.teacher_role, activity.learner_role)),
    section('Résultat attendu', paragraph(activity.expected_outcome)),
    section('Critères d’évaluation', checkList(activity.assessment.criteria)),
    section('Différenciation', [
      labelValue('En difficulté', activity.differentiation.support),
      labelValue('Standard', activity.differentiation.standard),
      labelValue('Plus avancés', activity.differentiation.advanced),
    ].join('')),
  ].join('')
  return { type: 'ACTIVITÉ PÉDAGOGIQUE', title: activity.title, html: body }
}

/* =====================================================================
   EXERCISES TEMPLATE
   ===================================================================== */
export function exerciseTemplate(set: ExerciseSet) {
  const body = [
    metaGrid([
      { label: 'Niveau', value: set.level },
      { label: 'Thème', value: set.theme },
      { label: 'Type', value: set.exercise_type },
      { label: 'Exercices', value: `${set.exercises.length}` },
    ]),
    section('Consigne générale', list(set.skills)),
    section('Exercices', set.exercises.map((ex, index) => `${exerciseBlock(ex, index + 1)}`).join('')),
  ].join('')
  return { type: 'FICHE D\'EXERCICES', title: set.title, html: body }
}

function exerciseBlock(ex: ExerciseItem, index: number) {
  const source = ex.document_title
    ? `<p class="muted">Source : ${esc(ex.document_title)}${ex.page_start ? ` · p. ${esc(String(ex.page_start))}` : ''}</p>`
    : '<p class="muted">Exercice créé avec l’IA (non lié à un document).</p>'
  return `
  <div class="exercise-item avoid-break">
    <div class="exercise-head">Exercice ${index}${ex.title ? ` — ${esc(ex.title)}` : ''}</div>
    ${ex.context ? paragraph(ex.context) : ''}
    ${paragraph(ex.prompt)}
    ${ex.answer_expectation ? sectionTitle('Correction attendue') + paragraph(ex.answer_expectation) : ''}
    ${source}
  </div>`
}
export function lessonPlanTemplate(plan: LessonPlan) {
  const body = [
    metaGrid([
      { label: 'Niveau', value: plan.level },
      { label: 'Thème', value: plan.theme },
      { label: 'Type', value: plan.session_type },
      { label: 'Durée', value: `${plan.duration} min` },
    ]),
    paragraph(`<strong>Public :</strong> <span${rtlAttr(plan.audience)}>${esc(plan.audience)}</span>${plan.age_approximation ? ` · ${esc(plan.age_approximation)}` : ''}`),
    section('Objectifs pédagogiques', [
      labelValue('Objectif général', plan.general_objective),
      labelValue('Objectifs communicatifs', plan.communicative_objectives.join(' · ')),
      labelValue('Objectifs linguistiques', plan.linguistic_objectives.join(' · ')),
      labelValue('Objectifs spécifiques', plan.specific_objectives.join(' · ')),
    ].join('')),
    section('Compétences', list(plan.skills)),
    section('Contenu linguistique', Object.entries(plan.linguistic_content).map(([key, items]) =>
      `<p><strong>${esc(key)} :</strong> ${items.map((item) => `<span${rtlAttr(item)}>${esc(item)}</span>`).join(', ')}</p>`,
    ).join('')),
    section('Prérequis', list(plan.prerequisites)),
    section('Matériel et ressources', list(plan.materials)),
    section('Déroulement', table(
      [
        { head: 'Phase', width: '18%', render: (row) => `<span${rtlAttr((row as { phase: string }).phase)}>${esc((row as { phase: string }).phase)}</span>` },
        { head: 'Durée', width: '12%', render: (row) => `${(row as { duration: number }).duration} min` },
        { head: 'Déroulement', render: (row) => {
          const r = row as { objective: string; work_mode: string; teacher_role: string; learner_activity: string; instructions: string; example: string; expected_result: string; materials: string[] }
          const parts = [
            r.objective && `<b>Objectif :</b> ${esc(r.objective)}`,
            r.work_mode && `<b>Modalité :</b> ${esc(r.work_mode)}`,
            r.materials && r.materials.length && `<b>Matériel :</b> ${esc(r.materials.join(', '))}`,
            r.teacher_role && `<b>Enseignant :</b> ${esc(r.teacher_role)}`,
            r.learner_activity && `<b>Apprenants :</b> ${esc(r.learner_activity)}`,
            r.instructions && `<b>Consigne :</b> <span${rtlAttr(r.instructions)}>${esc(r.instructions)}</span>`,
            r.example && `<b>Exemple :</b> ${esc(r.example)}`,
            r.expected_result && `<b>Résultat attendu :</b> ${esc(r.expected_result)}`,
          ].filter(Boolean)
          return parts.map((p) => `<div>${p}</div>`).join('')
        } },
      ],
      plan.lesson_flow,
    )),
    section('Évaluation', [
      paragraph([
        plan.assessment.assessment_type && `<b>Type :</b> ${esc(plan.assessment.assessment_type)}`,
        plan.assessment.moment && `<b>Moment :</b> ${esc(plan.assessment.moment)}`,
        plan.assessment.method && `<b>Méthode :</b> ${esc(plan.assessment.method)}`,
      ].filter(Boolean).join(' · ')),
      labelValue('Activité d’évaluation', plan.assessment.activity),
      labelValue('Consigne', plan.assessment.instructions),
      paragraph('<strong>Critères :</strong>'),
      checkList(plan.assessment.criteria),
      plan.assessment.success_indicators?.length ? `${paragraph('<strong>Indicateurs de réussite :</strong>')}${checkList(plan.assessment.success_indicators)}` : '',
      plan.assessment.rubric?.length ? section('Grille d’évaluation', table(
        [
          { head: 'Critère', render: (row) => esc((row as { criterion: string }).criterion) },
          { head: 'Acquis', render: (row) => esc((row as { achieved: string }).achieved) },
          { head: 'À renforcer', render: (row) => esc((row as { to_reinforce: string }).to_reinforce) },
        ],
        plan.assessment.rubric,
      )) : '',
    ].join('')),
    section('Différenciation', [
      paragraph('<strong>En difficulté :</strong>'),
      list(plan.differentiation.support),
      paragraph('<strong>Plus avancés :</strong>'),
      list(plan.differentiation.extension),
    ].join('')),
    section('Prolongement', paragraph([plan.extension?.homework, plan.extension?.follow_up].filter(Boolean).join(' · ') || '—')),
  ].join('')
  return { type: 'FICHE PÉDAGOGIQUE', title: plan.title, html: body }
}

/* =====================================================================
   COURSE TEMPLATE (contenu pédagogique)
   ===================================================================== */
const contentBlock = (title: string, body: string) => `${sectionTitle(title)}${paragraph(body)}`
export function courseTemplate(course: Course) {
  const grammar = course.grammar ?? []
  const content = course.content ?? []
  const summary = Array.isArray(course.summary) ? course.summary : (course.summary ? [String(course.summary)] : [])
  const dialogue = course.dialogue && typeof course.dialogue === 'object' && Array.isArray(course.dialogue.lines) ? course.dialogue : null

  const sections = (items: Array<{ title: string; instructions?: string; example?: { title?: string; body?: string } | null }>) =>
    items.map((ex) => `${sectionTitle(ex.title)}${paragraph(ex.instructions || '')}${ex.example ? paragraph(`<em>${esc(ex.example.title || 'Exemple')} · ${esc(ex.example.body || '')}</em>`) : ''}`).join('')

  const body = [
    metaGrid([
      { label: 'Niveau', value: course.level },
      { label: 'Thème', value: course.theme },
      { label: 'Durée', value: `${course.duration} min` },
      { label: 'Compétences', value: (course.skills || []).join(', ') },
    ]),
    section('Objectifs', list(course.objectives || [])),
    section('Introduction', paragraph(course.introduction || '')),
    section('Vocabulaire', list(course.vocabulary || [])),
    (course.expressions && course.expressions.length) ? section('Expressions', list(course.expressions)) : '',
    grammar.length ? section('Grammaire', grammar.map((g) => [
      paragraph(`<strong>${esc(g.title)}</strong>`),
      paragraph(g.body),
      (g.examples || []).map((ex) => paragraph(`<em>${esc(ex.title)} · ${esc(ex.body)}</em>`)).join(''),
    ].join('')).join('')) : '',
    content.length ? section('Contenu du cours', content.map((b) => [
      paragraph(`<strong>${esc(b.title)}</strong>`),
      paragraph(b.body),
      (b.examples || []).map((ex) => paragraph(`<em>${esc(ex.title)} · ${esc(ex.body)}</em>`)).join(''),
    ].join('')).join('')) : '',
    dialogue ? section('Dialogue', [
      dialogue.context ? paragraph(dialogue.context) : '',
      dialogue.lines.map(paragraph).join(''),
    ].join('')) : '',
    (course.comprehension || []).length ? section('Compréhension', sections(course.comprehension)) : '',
    (course.guided_practice || []).length ? section('Pratique guidée', sections(course.guided_practice)) : '',
    (course.communicative_practice || []).length ? section('Pratique communicative', sections(course.communicative_practice)) : '',
    (course.production || []).length ? section('Production', sections(course.production)) : '',
    section('Synthèse', list(summary)),
    course.homework ? section('Devoir', paragraph(course.homework)) : '',
  ].join('')
  return { type: 'COURS PÉDAGOGIQUE', title: course.title, html: body }
}

/* =====================================================================
   Document CSS — A4 professional print stylesheet
   ===================================================================== */
const CSS = `
@page {
  size: A4 portrait;
  margin: 16mm 14mm 18mm 14mm;
  @bottom-left { content: "Mo3allimAI · Document pédagogique généré avec assistance IA"; font: 7.5pt 'Segoe UI', Arial, sans-serif; color: #94a3b8; }
  @bottom-right { content: "Page " counter(page) " / " counter(pages); font: 7.5pt 'Segoe UI', Arial, sans-serif; color: #94a3b8; }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', 'Noto Sans Arabic', 'Tahoma', Arial, sans-serif;
  color: #1e293b; font-size: 10.5pt; line-height: 1.5;
  orphans: 3; widows: 3;
}
.doc-header {
  display: flex; justify-content: space-between; align-items: flex-end;
  border-bottom: 1.5pt solid #065f46; padding-bottom: 3pt; margin-bottom: 7mm;
}
.brand { display: flex; flex-direction: column; }
.brand-name { font-weight: 700; font-size: 13pt; color: #065f46; letter-spacing: .5px; line-height: 1; }
.brand-tag { font-size: 7.5pt; color: #64748b; margin-top: 1mm; }
.doc-type {
  font-size: 9.5pt; font-weight: 700; color: #065f46; letter-spacing: 1.5px;
  border: 1pt solid #065f46; border-radius: 3mm; padding: 2mm 4mm; text-transform: uppercase;
  white-space: nowrap;
}
main { margin-top: 0; }
.doc-title {
  font-size: 16pt; font-weight: 700; color: #0f172a; margin: 0 0 5mm;
  text-align: center; line-height: 1.3;
  page-break-after: avoid; page-break-inside: avoid;
}
.section-title {
  display: flex; align-items: center; gap: 2.5mm;
  font-size: 11.5pt; font-weight: 700; color: #065f46; text-transform: uppercase;
  letter-spacing: .5px; margin: 6mm 0 2.5mm;
  border-bottom: 1pt solid #cbd5e1; padding-bottom: 1.5mm;
  page-break-after: avoid; page-break-inside: avoid;
}
.section-title::before {
  content: ""; display: inline-block; width: 4mm; height: 4mm;
  background: #065f46; border-radius: 1mm; flex: 0 0 auto; margin-right: 2mm;
}
.avoid-break { page-break-inside: avoid; }
p { margin: 0 0 2.5mm; page-break-inside: avoid; }
p.muted { color: #94a3b8; font-style: italic; }
strong { color: #0f172a; }
ul { margin: 0 0 3mm; padding-inline-start: 6mm; }
li { margin-bottom: 1.5mm; page-break-inside: avoid; }
ul.checks { list-style: none; padding-inline-start: 1mm; }
ul.checks li { position: relative; padding-inline-start: 7mm; margin-bottom: 1.8mm; }
ul.checks li::before {
  content: "☐"; position: absolute; inset-inline-start: 0;
  font-size: 10.5pt; color: #065f46; line-height: 1.45;
}
.meta-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); grid-auto-rows: 1fr;
  gap: 2.5mm; margin-bottom: 3mm; page-break-inside: avoid;
}
.meta-item {
  border: 1pt solid #e2e8f0; border-top: 2.5pt solid #065f46;
  border-radius: 2mm; padding: 2.5mm 3mm 3mm; background: #f8fafc;
  min-width: 0;
}
.meta-label { font-size: 7pt; text-transform: uppercase; letter-spacing: .6px; color: #64748b; margin-bottom: 1mm; }
.meta-value { font-size: 10.5pt; font-weight: 700; color: #0f172a; word-wrap: break-word; }
.roles { display: grid; grid-template-columns: 1fr 1fr; gap: 3mm; }
.roles-col {
  border: 1pt solid #e2e8f0; border-radius: 2mm; padding: 3mm;
  background: #f8fafc; page-break-inside: avoid;
}
.roles-title { font-size: 8pt; font-weight: 700; text-transform: uppercase; letter-spacing: .6px; color: #065f46; margin-bottom: 1.5mm; }
.exercise-item { border: 1pt solid #e2e8f0; border-inline-start: 3pt solid #065f46; border-radius: 2mm; padding: 3mm 3.5mm; margin-bottom: 3mm; background: #f8fafc; }
.exercise-head { font-weight: 700; color: #065f46; font-size: 10.5pt; margin-bottom: 1.5mm; }
.data-table { width: 100%; border-collapse: collapse; margin-bottom: 3mm; table-layout: fixed; }
.data-table thead { display: table-header-group; }
.data-table th {
  background: #065f46; color: #ffffff; text-align: left; font-weight: 600;
  padding: 2.2mm 3mm; font-size: 9pt; border: 1pt solid #065f46;
}
.data-table td {
  border: 1pt solid #e2e8f0; padding: 2.2mm 3mm; vertical-align: top; font-size: 9.5pt;
  word-wrap: break-word; overflow-wrap: break-word; min-width: 0;
}
.data-table tr { page-break-inside: avoid; }
.data-table tbody tr:nth-child(even) { background: #f8fafc; }
.data-table td div { margin-bottom: 1mm; }
.data-table td div:last-child { margin-bottom: 0; }
.data-table td > span[dir="rtl"] { display: block; }
[dir="rtl"], .rtl { direction: rtl; unicode-bidi: embed; }
`

