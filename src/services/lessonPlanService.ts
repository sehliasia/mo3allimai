import { teacherRequest } from './teacherLibraryService'

export type LessonPlanInput = { level: string; theme: string; general_objective: string; duration_minutes: number; skills: string[]; audience: string; learner_count?: number; session_type: string; prerequisites: string[]; linguistic_points: string[]; special_instructions?: string; language: 'ar' | 'fr' | 'en' | 'es' }
export type LessonPlan = { title: string; level: string; theme: string; duration: number; audience: string; session_type: string; skills: string[]; age_approximation?: string; communicative_objectives: string[]; linguistic_objectives: string[]; general_objective: string; specific_objectives: string[]; prerequisites: string[]; linguistic_content: Record<string, string[]>; materials: string[]; lesson_flow: { phase: string; duration: number; objective: string; teacher_role: string; learner_activity: string; instructions: string; materials: string[]; work_mode: string; example: string; expected_result: string }[]; assessment: { assessment_type: string; moment: string; criteria: string[]; method: string; activity: string; instructions: string; success_indicators: string[]; rubric: { criterion: string; achieved: string; to_reinforce: string }[] }; differentiation: { support: string[]; extension: string[] }; extension: { homework?: string; follow_up?: string }; rag_sources_used: number; provider_model?: string }
export function generateLessonPlan(payload: LessonPlanInput): Promise<LessonPlan> {
  return teacherRequest<LessonPlan>(
    '/teacher/ai/lesson-plans/generate',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
    'La génération a échoué.',
  )
}
