import { createServer } from 'vite'
import React from 'react'
import { renderToString } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'

globalThis.window = globalThis
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }

const server = await createServer({
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
  optimizeDeps: { noDiscovery: true },
})

try {
  const { AuthProvider } = await server.ssrLoadModule('/src/contexts/AuthContext.tsx')
  const { TeacherExercisesPage } = await server.ssrLoadModule('/src/pages/teacher/TeacherExercisesPage.tsx')
  const { TeacherLayout } = await server.ssrLoadModule('/src/layouts/TeacherLayout.tsx')
  for (const comp of [
    ['TeacherExercisesPage', TeacherExercisesPage],
    ['TeacherExercisesPage inside TeacherLayout', () => React.createElement(TeacherLayout, null, React.createElement(TeacherExercisesPage))],
  ]) {
    try {
      const html = renderToString(
        React.createElement(MemoryRouter, { initialEntries: ['/teacher/tools/exercises'] },
          React.createElement(AuthProvider, null, React.createElement(comp[1]))),
      )
      console.log(`RENDER_OK    ${comp[0]}  (${html.length} chars)`)
    } catch (err) {
      console.log(`RENDER_FAIL  ${comp[0]}`)
      console.log(String(err && err.stack ? err.stack : err))
    }
  }
} finally {
  await server.close()
}