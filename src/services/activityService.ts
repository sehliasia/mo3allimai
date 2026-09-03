import { teacherRequest } from './teacherLibraryService'

export type ActivityInput = {
  level: string; theme: string; objective: string; skills: string[]; activity_type: string
  duration_minutes: number; audience?: string; learner_count?: number; materials: string[]
  special_instructions?: string; language: 'ar' | 'fr' | 'en' | 'es'
}
export type Activity = {
  title: string; level: string; theme: string; activity_type: string; duration: number; objective: string
  skills: string[]; materials: string[]; instructions: string
  procedure: { step: number; title: string; duration: number; description: string }[]
  teacher_role: string; learner_role: string; expected_outcome: string
  assessment: { criteria: string[] }
  differentiation: { support: string; standard: string; advanced: string }
  rag_sources_used: number; provider_model?: string
}
export function generateActivity(payload: ActivityInput): Promise<Activity> {
  return teacherRequest<Activity>(
    '/teacher/ai/activities/generate',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
    'La génération de l’activité a échoué.',
  )
}
