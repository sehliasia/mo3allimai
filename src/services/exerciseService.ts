import { teacherRequest } from './teacherLibraryService'

export type ExerciseInput = {
  level: string
  theme: string
  objective?: string | null
  skills: string[]
  exercise_type: string
  count: number
  language: 'ar' | 'fr' | 'en' | 'es'
  adapt_with_ai: boolean
  special_instructions?: string | null
}

export type ExerciseStatus = 'kb_original' | 'adapted_from_kb' | 'ai_generated'

export type ExerciseItem = {
  title: string
  skill: string
  exercise_type: string
  prompt: string
  context: string
  answer_expectation: string | null
  level: string
  level_source?: 'explicit' | 'inferred' | 'generated'
  status?: ExerciseStatus
  difficulty?: 'easy' | 'medium' | 'hard' | null
  // Structured representation for typed display (QCM options, vrai/faux,
  // paires d'association). Absent/empty on older exercises (produced before
  // these fields existed) — the UI must fall back to `prompt` when missing.
  options?: string[]
  is_true?: boolean | null
  pairs?: Array<{ left: string; right: string }>
  document_title: string | null
  document_id: number | null
  page_start: number | null
  page_end: number | null
  chunk_ids: number[]
  heading_context?: string[]
  original_level?: string | null
  original_document_title?: string | null
  original_document_id?: number | null
  original_page_start?: number | null
  original_page_end?: number | null
  original_chunk_ids?: number[]
  summary_score?: number | null
}

export type ExerciseSearchIn = {
  query: string
  limit?: number
  level?: string | null
  skills?: string[]
  exercise_type?: string | null
  source_document_ids?: number[]
}

export type ExerciseFilterFacet = { value: string; count: number }

export type ExerciseSearchOut = {
  query: string
  items: ExerciseSearchItem[]
  total: number
  facets: Record<string, ExerciseFilterFacet[]>
  meta: {
    llm_calls: number
    retrieval_llm_calls: number
    extraction_llm_calls: number
    retrieval_mode: string | null
    dense_candidate_count: number
    sparse_candidate_count: number
    union_candidate_count: number
    candidate_blocks: number
    expanded_blocks: number
    extracted_blocks: number
    deduplicated_count: number
    detected_blocks: number
    parsed: Record<string, unknown>
  }
}

export type ExerciseSearchItem = ExerciseItem & {
  theme_label?: string | null
  skill_label?: string | null
}

export type ExerciseAdaptIn = {
  source: ExerciseItem
  target_level: string
  language?: 'ar' | 'fr' | 'en' | 'es'
  instructions?: string | null
}

export type ExercisePlan = {
  level: string
  theme: string
  skills: string[]
  objective: string
  learning_objectives: string[]
  target_vocabulary: string[]
  target_grammar: string[]
  exercise_distribution: string[]
  rationale: string
}

export type ExerciseSet = {
  title: string
  level: string
  theme: string
  exercise_type: string
  language: string
  skills: string[]
  exercises: ExerciseItem[]
  kb_sourced_count: number
  rag_sources_used: number
  provider_model?: string
  adapt_with_ai: boolean
  plan?: ExercisePlan | null
}

export function generateExercises(payload: ExerciseInput): Promise<ExerciseSet> {
  return teacherRequest<ExerciseSet>(
    '/teacher/ai/exercises/generate',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
    'La génération des exercices a échoué.',
  )
}

export function searchExercises(payload: ExerciseSearchIn): Promise<ExerciseSearchOut> {
  return teacherRequest<ExerciseSearchOut>(
    '/teacher/ai/exercises/search',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
    'La recherche d’exercices a échoué.',
  )
}

export function adaptExercise(payload: ExerciseAdaptIn): Promise<ExerciseItem> {
  return teacherRequest<ExerciseItem>(
    '/teacher/ai/exercises/adapt',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
    'L’adaptation de l’exercice a échoué.',
  )
}
