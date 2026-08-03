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
import { Navigate } from 'react-router-dom'

export function AppRoutes() {
  return <Routes>
    <Route path="/" element={<LandingPage />} />
    <Route path="/login" element={<LoginPage />} />
    <Route path="/register" element={<RegisterPage />} />
    <Route path="/teacher/dashboard" element={<ProtectedRoute allowedRoles={['teacher']}><DashboardPage role="teacher" /></ProtectedRoute>} />
    <Route path="/admin" element={<ProtectedRoute allowedRoles={['admin']}><AdminLayout /></ProtectedRoute>}><Route index element={<Navigate to="dashboard" replace />} /><Route path="dashboard" element={<AdminDashboardPage />} /><Route path="teachers" element={<AdminTeachersPage />} /><Route path="statistics" element={<AdminStatisticsPage />} /><Route path="settings" element={<AdminSettingsPage />} /><Route path="*" element={<Navigate to="dashboard" replace />} /></Route>
    <Route path="*" element={<LandingPage />} />
  </Routes>
}
