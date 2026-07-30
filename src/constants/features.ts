import { Bot, BookOpen, BrainCircuit, CalendarDays, ClipboardCheck, FilePenLine, FolderKanban, LibraryBig } from 'lucide-react'
import type { Feature } from '../types/landing'

export const features: Feature[] = [
  { title: 'مساعد Mo3allimAI', description: 'مساعد ذكي يدعمك في جميع مراحل التحضير.', icon: Bot, accent: 'bg-emerald-50 text-emerald-700' },
  { title: 'تحضير الدروس', description: 'أنشئ دروساً احترافية خلال ثوانٍ.', icon: BookOpen, accent: 'bg-sky-50 text-sky-700' },
  { title: 'إنشاء الأنشطة', description: 'أنشئ أنشطة تعليمية متنوعة بسهولة.', icon: FilePenLine, accent: 'bg-violet-50 text-violet-700' },
  { title: 'إعداد الاختبارات', description: 'أنشئ اختبارات وتقويمات جاهزة.', icon: ClipboardCheck, accent: 'bg-amber-50 text-amber-700' },
  { title: 'التخطيط السنوي', description: 'خطط السنة الدراسية بوضوح.', icon: CalendarDays, accent: 'bg-rose-50 text-rose-700' },
  { title: 'بنك الموارد', description: 'مكتبة تعليمية منظمة وجاهزة.', icon: LibraryBig, accent: 'bg-cyan-50 text-cyan-700' },
  { title: 'الخرائط الذهنية', description: 'حوّل الدروس إلى خرائط ذهنية.', icon: BrainCircuit, accent: 'bg-indigo-50 text-indigo-700' },
  { title: 'إدارة المحتوى', description: 'رتب جميع مواردك في مكان واحد.', icon: FolderKanban, accent: 'bg-teal-50 text-teal-700' },
]
