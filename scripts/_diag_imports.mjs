import { createServer } from 'vite'
import React from 'react'
import { renderToString } from 'react-dom/server'

const server = await createServer({
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
  optimizeDeps: { noDiscovery: true },
})

const routes = {
  'teacher/exercises': '/src/pages/teacher/TeacherExercisesPage.tsx',
  'teacher/activity': '/src/pages/teacher/TeacherActivityPage.tsx',
  'teacher/course': '/src/pages/teacher/TeacherCoursePage.tsx',
  'teacher/dashboard': '/src/pages/teacher/TeacherDashboardPage.tsx',
  'teacher/generator': '/src/pages/teacher/TeacherGeneratorPage.tsx',
  'teacher/resource': '/src/pages/teacher/TeacherResourcePage.tsx',
  'teacher/tools': '/src/pages/teacher/TeacherToolsPage.tsx',
  'teacher/assistant': '/src/pages/teacher/TeacherAssistantPage.tsx',
  'AppRoutes': '/src/routes/AppRoutes.tsx',
  'App': '/src/App.tsx',
}

try {
  for (const [name, path] of Object.entries(routes)) {
    try {
      const mod = await server.ssrLoadModule(path)
      console.log(`MODULE_EVAL_OK  ${name}`)
    } catch (err) {
      console.log(`MODULE_EVAL_FAIL ${name}`)
      console.log(String(err && err.stack ? err.stack : err))
      continue
    }
  }
} finally {
  await server.close()
}