import { createServer } from 'vite'
import React from 'react'
import { renderToString } from 'react-dom/server'
import { MemoryRouter, Routes, Route } from 'react-router-dom'

const server = await createServer({
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
  optimizeDeps: { noDiscovery: true },
})

globalThis.window = globalThis
globalThis.localStorage = {
  getItem: () => null, setItem: () => {}, removeItem: () => {},
}

const layoutMod = await server.ssrLoadModule('/src/layouts/TeacherLayout.tsx')
const Layout = layoutMod.TeacherLayout ?? layoutMod.default

const pages = {
  exercises: ['/src/pages/teacher/TeacherExercisesPage.tsx', '/teacher/tools/exercises'],
  activity: ['/src/pages/teacher/TeacherActivityPage.tsx', '/teacher/tools/activity'],
  course: ['/src/pages/teacher/TeacherCoursePage.tsx', '/teacher/tools/lesson'],
  dashboard: ['/src/pages/teacher/TeacherDashboardPage.tsx', '/teacher/dashboard'],
  resource: ['/src/pages/teacher/TeacherResourcePage.tsx', '/teacher/library'],
  tools: ['/src/pages/teacher/TeacherToolsPage.tsx', '/teacher/tools'],
}

try {
  for (const [name, [path, url]] of Object.entries(pages)) {
    const mod = await server.ssrLoadModule(path)
    const Comp = mod.default ?? Object.values(mod).find(v => typeof v === 'function')
    try {
      const el = React.createElement(Comp)
      const html = renderToString(el)
      console.log(`RENDER_OK    ${name}  (${html.length} chars)`)
    } catch (err) {
      console.log(`RENDER_FAIL  ${name}`)
      console.log(String(err && err.stack ? err.stack : err))
    }
  }
} finally {
  await server.close()
}