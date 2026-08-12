import { Route, Routes } from 'react-router-dom'
import { LandingPage } from '../pages/LandingPage'
import { LoginPage } from '../pages/LoginPage'
import { RegisterPage } from '../pages/RegisterPage'
import { DashboardPage } from '../pages/DashboardPage'
import { ProtectedRoute } from '../components/auth/ProtectedRoute'
import { AdminLayout } from '../layouts/AdminLayout'
import { AdminDashboardPage } from '../pages/admin/AdminDashboardPage'
import { AdminTeachersPage } from '../pages/admin/AdminTeachersPage'
import { AdminStatisticsPage } from '../pages/admin/AdminStatisticsPage'
import { AdminSettingsPage } from '../pages/admin/AdminSettingsPage'
import { KnowledgeBasePage } from '../pages/admin/KnowledgeBasePage'
import { Navigate } from 'react-router-dom'
import { TeacherLayout } from '../layouts/TeacherLayout'
import { TeacherDashboardPage } from '../pages/teacher/TeacherDashboardPage'
import { TeacherPlaceholderPage } from '../pages/teacher/TeacherPlaceholderPage'

export function AppRoutes() {
  return <Routes>
    <Route path="/" element={<LandingPage />} />
    <Route path="/login" element={<LoginPage />} />
    <Route path="/register" element={<RegisterPage />} />
    <Route path="/teacher" element={<ProtectedRoute allowedRoles={['teacher']}><TeacherLayout /></ProtectedRoute>}><Route path="dashboard" element={<TeacherDashboardPage />} /><Route path="assistant" element={<TeacherPlaceholderPage page="assistant" />} /><Route path="tools" element={<TeacherPlaceholderPage page="tools" />} /><Route path="library" element={<TeacherPlaceholderPage page="library" />} /><Route path="history" element={<TeacherPlaceholderPage page="history" />} /><Route path="settings" element={<TeacherPlaceholderPage page="settings" />} /></Route>
    <Route path="/admin" element={<ProtectedRoute allowedRoles={['admin']}><AdminLayout /></ProtectedRoute>}><Route index element={<Navigate to="dashboard" replace />} /><Route path="dashboard" element={<AdminDashboardPage />} /><Route path="teachers" element={<AdminTeachersPage />} /><Route path="knowledge-base" element={<KnowledgeBasePage />} /><Route path="statistics" element={<AdminStatisticsPage />} /><Route path="settings" element={<AdminSettingsPage />} /><Route path="*" element={<Navigate to="dashboard" replace />} /></Route>
    <Route path="*" element={<LandingPage />} />
  </Routes>
}
