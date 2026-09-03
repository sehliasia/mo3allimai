import assert from 'node:assert/strict'
import { courseTemplate } from '../src/services/printService.ts'

const course = {
  title: 'Cours : La famille en arabe — Découverte et pratique',
  level: 'A1',
  theme: 'La famille',
  duration: 45,
  objectives: ['Reconnaître et nommer les membres de la famille', 'Présenter sa famille en phrases courtes'],
  skills: ['Expression orale'],
  vocabulary: ['أب', 'أم', 'أخ', 'أخت', 'جد', 'جدة'],
  expressions: ['هذا أبي', 'هذه أمي', 'مَنْ هذا؟'],
  introduction: 'Aujourd’hui nous découvrons la famille et nous apprenons à présenter chaque membre en arabe avec des phrases courtes et claires.',
  grammar: [
    { title: 'Les démonstratifs هذا / هذه', body: 'On utilise هذا au masculin et هذه au féminin pour montrer une personne.', examples: [{ title: 'Exemple', body: 'هذا أبي، هذه أمي.' }] },
  ],
  content: [
    { title: 'Le lexique de la famille', body: 'Présentation illustrée des membres de la famille : الأب (le père), الأم (la mère), الأخ (le frère), الأخت (la sœur).', examples: [] },
    { title: 'Présentation guidée', body: 'L’apprenant présente les membres de sa famille en utilisant les démonstratifs.', examples: [] },
  ],
  dialogue: { context: 'À la maison, la famille présente ses membres.', lines: ['الأم: من هذا؟', 'الطفل: هذا أبي.', 'الأم: ومن هذه؟', 'الطفل: هذه أمي.'] },
  comprehension: [{ title: 'Vrai ou faux ?', instructions: 'Réponds d’après le dialogue.', example: null }],
  guided_practice: [
    { title: 'Complète la phrase', instructions: 'Complète avec هذا ou هذه.', example: { title: 'Exemple', body: 'هذا أخي.' } },
  ],
  communicative_practice: [{ title: 'En binôme', instructions: 'Présente un membre de ta famille, ton camarade devine qui c’est.', example: null }],
  production: [{ title: 'Présente ta famille', instructions: 'Présente trois membres de ta famille avec هذا / هذه.', example: null }],
  summary: ['L’apprenant sait nommer les membres de la famille', 'Il sait les présenter avec هذا / هذه'],
  homework: 'Apprendre le vocabulaire de la famille et préparer une courte présentation orale.',
  rag_sources_used: 0,
  provider_model: 'test',
}

const out = courseTemplate(course)

const checks = {
  'type is COURS PÉDAGOGIQUE': out.type === 'COURS PÉDAGOGIQUE',
  'title present': out.title === course.title,
  'non-empty html': out.html.trim().length > 100,
  'title is the document header field': out.title === 'Cours : La famille en arabe — Découverte et pratique',
  'meta grid level/theme/duration': out.html.includes(course.level) && out.html.includes(course.theme) && out.html.includes(`${course.duration} min`),
  'objectives section': out.html.includes('Objectifs'),
  'introduction section + arabic': out.html.includes('Introduction') && out.html.includes('Aujourd’hui nous découvrons'),
  'vocabulary section + arabic': out.html.includes('Vocabulaire') && out.html.includes('أب') && out.html.includes('أم'),
  'expressions section + arabic': out.html.includes('Expressions') && out.html.includes('هذا أبي'),
  'grammar section with explanation': out.html.includes('Grammaire') && out.html.includes('Les démonstratifs') && out.html.includes('On utilise') && out.html.includes('هذا أبي، هذه أمي.'),
  'content section with blocks': out.html.includes('Contenu du cours') && out.html.includes('Le lexique de la famille') && out.html.includes('Présentation guidée'),
  'dialogue section + arabic': out.html.includes('Dialogue') && out.html.includes('من هذا؟'),
  'comprehension section': out.html.includes('Compréhension') && out.html.includes('Vrai ou faux ?'),
  'guided practice section with example': out.html.includes('Pratique guidée') && out.html.includes('Complète la phrase') && out.html.includes('هذا أخي.'),
  'communicative practice section': out.html.includes('Pratique communicative') && out.html.includes('En binôme'),
  'production section': out.html.includes('Production') && out.html.includes('Présente ta famille'),
  'summary section + points': out.html.includes('Synthèse') && out.html.includes('Il sait les présenter avec هذا / هذه'),
  'homework section': out.html.includes('Devoir') && out.html.includes('Apprendre le vocabulaire'),
}

let pass = 0
let fail = 0
for (const [label, ok] of Object.entries(checks)) {
  if (ok) { pass += 1; console.log('  ✅', label) } else { fail += 1; console.log('  ❌', label) }
}

assert.ok(out.html.includes('section-title'), 'CSS class section-title present')

console.log(`\nPDF course harness: ${pass} passed, ${fail} failed`)
if (fail > 0) process.exit(1)
console.log('HARNESS COMPLETE — courseTemplate produces complete, structured A4 HTML.')
