import { teacherRequest } from './teacherLibraryService'

export type CourseInput = {
  level: string; theme: string; objective?: string; skills: string[]
  duration_minutes: number; audience?: string; learner_count?: number
  language: 'ar' | 'fr' | 'en' | 'es'; special_instructions?: string
}
export type CourseExample = { title: string; body: string }
export type CourseSection = { title: string; body: string; examples: CourseExample[] }
export type CourseExercise = { title: string; instructions: string; example?: CourseExample | null }
export type CourseDialogue = { context: string; lines: string[] }
export type Course = {
  title: string; level: string; theme: string; duration: number
  objectives: string[]; skills: string[]; vocabulary: string[]; expressions: string[]
  introduction: string
  grammar: CourseSection[]; content: CourseSection[]
  dialogue: CourseDialogue | null
  comprehension: CourseExercise[]; guided_practice: CourseExercise[]
  communicative_practice: CourseExercise[]; production: CourseExercise[]
  summary: string[]; homework: string | null
  rag_sources_used: number; provider_model?: string
}
export function generateCourse(payload: CourseInput): Promise<Course> {
  return teacherRequest<Course>(
    '/teacher/ai/courses/generate',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
    'La génération du cours a échoué.',
  )
}