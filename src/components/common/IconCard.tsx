import type { LucideIcon } from 'lucide-react'; import type { ReactNode } from 'react'; import { Card } from '../ui/Card'
interface IconCardProps { icon: LucideIcon; children: ReactNode; className?: string }
export function IconCard({ icon: Icon, children, className }: IconCardProps) { return <Card className={className}><span className="grid size-11 place-items-center rounded-xl bg-emerald-100 text-emerald-700"><Icon className="size-5"/></span>{children}</Card> }
