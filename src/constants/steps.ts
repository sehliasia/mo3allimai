import { Grid2X2, PencilLine, Sparkles, UserRound } from 'lucide-react'
import type { Step } from '../types/landing'

export const howItWorksContent = {
  title: 'كيف تعمل منصة Mo3allimAI؟',
  description: 'ابدأ في دقائق قليلة، واترك الذكاء الاصطناعي يساعدك في إعداد مواردك التعليمية.',
}

export const steps: Step[] = [
  { number: '١', title: 'سجل الدخول', description: 'أنشئ حسابك أو سجّل الدخول إلى المنصة.', icon: UserRound },
  { number: '٢', title: 'اختر الأداة', description: 'اختر الدرس أو النشاط أو الاختبار أو التخطيط الذي تريد إنشاءه.', icon: Grid2X2 },
  { number: '٣', title: 'أدخل تعليماتك', description: 'اكتب موضوع الدرس وحدد المستوى والأهداف المطلوبة.', icon: PencilLine },
  { number: '٤', title: 'احصل على النتيجة', description: 'ينشئ Mo3allimAI المحتوى خلال ثوانٍ ويمكنك مراجعته وتعديله وتنزيله.', icon: Sparkles },
]
