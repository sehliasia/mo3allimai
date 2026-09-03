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
import { KnowledgeDocumentDetailsPage } from '../pages/admin/KnowledgeDocumentDetailsPage'
import { Navigate } from 'react-router-dom'
import { TeacherLayout } from '../layouts/TeacherLayout'
import { TeacherDashboardPage } from '../pages/teacher/TeacherDashboardPage'
import { TeacherAssistantPage } from '../pages/teacher/TeacherAssistantPage'
import { TeacherGeneratorPage } from '../pages/teacher/TeacherGeneratorPage'
import { TeacherAiToolPage } from '../pages/teacher/TeacherAiToolPage'
import { TeacherResourcePage } from '../pages/teacher/TeacherResourcePage'
import { TeacherProfilePage } from '../pages/teacher/TeacherProfilePage'
import { TeacherToolsPage } from '../pages/teacher/TeacherToolsPage'

export function AppRoutes() {
  return <Routes>
    <Route path="/" element={<LandingPage />} />
    <Route path="/login" element={<LoginPage />} />
    <Route path="/register" element={<RegisterPage />} />
    <Route path="/teacher" element={<ProtectedRoute allowedRoles={['teacher']}><TeacherLayout /></ProtectedRoute>}><Route index element={<Navigate to="dashboard" replace />} /><Route path="dashboard" element={<TeacherDashboardPage />} /><Route path="assistant" element={<TeacherAssistantPage />} /><Route path="tools" element={<TeacherToolsPage />} /><Route path="tools/lesson-plan" element={<TeacherGeneratorPage />} /><Route path="tools/:toolId" element={<TeacherAiToolPage />} /><Route path="library" element={<TeacherResourcePage page="library" />} /><Route path="history" element={<TeacherResourcePage page="history" />} /><Route path="profile" element={<TeacherProfilePage />} /><Route path="settings" element={<TeacherResourcePage page="settings" />} /></Route>
    <Route path="/admin" element={<ProtectedRoute allowedRoles={['admin']}><AdminLayout /></ProtectedRoute>}><Route index element={<Navigate to="dashboard" replace />} /><Route path="dashboard" element={<AdminDashboardPage />} /><Route path="teachers" element={<AdminTeachersPage />} /><Route path="knowledge-base" element={<KnowledgeBasePage />} /><Route path="knowledge-base/:documentId" element={<KnowledgeDocumentDetailsPage />} /><Route path="statistics" element={<AdminStatisticsPage />} /><Route path="settings" element={<AdminSettingsPage />} /><Route path="*" element={<Navigate to="dashboard" replace />} /></Route>
    <Route path="*" element={<LandingPage />} />
  </Routes>
}

