export function useScrollToSection() { return (sectionId: string) => document.querySelector(sectionId)?.scrollIntoView({ behavior: 'smooth' }) }
