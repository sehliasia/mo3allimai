import type { ReactNode } from 'react'
import { Footer } from '../components/layout/Footer'
import { Navbar } from '../components/layout/Navbar'
export function PublicLayout({ children }: { children: ReactNode }) { return <div className="min-h-screen overflow-x-clip bg-white text-slate-900"><Navbar /><main>{children}</main><Footer /></div> }
