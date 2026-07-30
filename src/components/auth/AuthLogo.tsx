import { Link } from 'react-router-dom'
import logo1 from '../../assets/logo1.png'

export function AuthLogo() {
  return <Link to="/" dir="ltr" aria-label="Mo3allimAI" className="inline-flex flex-col items-center gap-3 rounded-2xl focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:ring-offset-4">
    <span className="grid size-[72px] place-items-center rounded-full border border-emerald-200 bg-white p-2 shadow-[0_8px_20px_rgba(6,95,70,0.12)] sm:size-20"><img src={logo1} alt="Mo3allimAI" className="size-full object-contain" /></span>
    <span className="whitespace-nowrap text-2xl font-extrabold leading-none tracking-[-0.03em]"><span className="text-[#065F46]">Mo3allim</span><span className="text-[#C89B3C]">AI</span></span>
  </Link>
}
