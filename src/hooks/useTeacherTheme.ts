import { useEffect, useState } from 'react'

export type TeacherTheme = 'light' | 'dark' | 'system'

const storageKey = 'mo3allimai_theme'

function storedTheme(): TeacherTheme {
  const value = localStorage.getItem(storageKey)
  return value === 'light' || value === 'dark' || value === 'system' ? value : 'system'
}

function resolved(theme: TeacherTheme) {
  return theme === 'system' ? window.matchMedia('(prefers-color-scheme: dark)').matches : theme === 'dark'
}

export function useTeacherTheme() {
  const [theme, setTheme] = useState<TeacherTheme>(storedTheme)
  const [isDark, setIsDark] = useState(() => resolved(storedTheme()))

  useEffect(() => {
    setIsDark(resolved(theme))
    if (theme === 'system') localStorage.removeItem(storageKey)
    else localStorage.setItem(storageKey, theme)
  }, [theme])

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => { if (theme === 'system') setIsDark(media.matches) }
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [theme])

  return { theme, isDark, setTheme, toggleDark: () => setTheme(isDark ? 'light' : 'dark') }
}
