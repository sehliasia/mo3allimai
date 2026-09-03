import { BookOpen, Bot, ClipboardList, FileCheck, ListChecks, Shapes } from 'lucide-react'

export const teacherTools = [
  { id: 'lesson-plan', title: 'lessonPlan', description: 'lessonPlanDescription', icon: ClipboardList, route: '/teacher/tools/lesson-plan', category: 'preparation' },
  { id: 'lesson', title: 'lesson', description: 'lessonDescription', icon: BookOpen, route: '/teacher/tools/lesson', category: 'preparation' },
  { id: 'exercises', title: 'exercises', description: 'exercisesDescription', icon: ListChecks, route: '/teacher/tools/exercises', category: 'activities' },
  { id: 'exam', title: 'exam', description: 'examDescription', icon: FileCheck, route: '/teacher/tools/exam', category: 'evaluation' },
  { id: 'activity', title: 'activity', description: 'activityDescription', icon: Shapes, route: '/teacher/tools/activity', category: 'activities' },
  { id: 'assistant', title: 'assistant', description: 'assistantDescription', icon: Bot, route: '/teacher/assistant', category: 'assistant' },
] as const
