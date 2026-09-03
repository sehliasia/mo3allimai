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
  const { AppRoutes } = await server.ssrLoadModule('/src/routes/AppRoutes.tsx')

  const urls = [
    '/', '/login',
    '/teacher/dashboard', '/teacher/tools', '/teacher/tools/exercises',
    '/teacher/tools/activity', '/teacher/tools/lesson', '/teacher/assistant',
    '/teacher/library', '/teacher/history', '/teacher/profile',
    '/admin/dashboard',
  ]
  for (const url of urls) {
    try {
      const html = renderToString(
        React.createElement(MemoryRouter, { initialEntries: [url] },
          React.createElement(AuthProvider, null,
            React.createElement(AppRoutes))),
      )
      console.log(`BOOT_OK      ${url}  (${html.length} chars)`)
    } catch (err) {
      console.log(`BOOT_FAIL    ${url}`)
      console.log(String(err && err.stack ? err.stack : err))
    }
  }
} finally {
  await server.close()
}