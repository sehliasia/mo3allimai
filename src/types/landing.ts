import type { LucideIcon } from 'lucide-react'

export interface NavigationItem { label: string; href: string }
export interface Feature { title: string; description: string; icon: LucideIcon; accent: string }
export interface Step { number: string; title: string; description: string; icon: LucideIcon }
