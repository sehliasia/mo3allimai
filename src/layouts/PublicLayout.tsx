import type { ReactNode } from 'react'; import { Navbar } from '../components/layout/Navbar'; import { Footer } from '../components/layout/Footer'
export function PublicLayout({ children }: { children: ReactNode }) { return <div dir="rtl" lang="ar" className="min-h-screen overflow-x-clip bg-white text-slate-900"><Navbar/><main>{children}</main><Footer/></div> }
