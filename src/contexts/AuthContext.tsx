import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { AuthApiError, clearSession, getCurrentUser, getStoredToken, login as loginRequest, saveToken, type AuthUser } from '../services/authService'

type AuthContextValue = { user: AuthUser | null; accessToken: string | null; isAuthenticated: boolean; isLoading: boolean; serviceError: string | null; login: (email: string, password: string) => Promise<AuthUser>; logout: () => void; loadCurrentUser: () => Promise<void> }
const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null); const [accessToken, setAccessToken] = useState<string | null>(getStoredToken); const [isLoading, setIsLoading] = useState(true); const [serviceError, setServiceError] = useState<string | null>(null)
  const logout = useCallback(() => { clearSession(); setAccessToken(null); setUser(null); setServiceError(null) }, [])
  const loadCurrentUser = useCallback(async () => { const token = getStoredToken(); if (!token) { setUser(null); setAccessToken(null); setServiceError(null); return }; try { const currentUser = await getCurrentUser(token); setAccessToken(token); setUser(currentUser); setServiceError(null) } catch (error) { if (error instanceof AuthApiError && (error.status === 401 || error.status === 403)) { logout(); return }; setServiceError('الخدمة غير متاحة مؤقتاً. يرجى المحاولة مرة أخرى.') } }, [logout])
  useEffect(() => { loadCurrentUser().finally(() => setIsLoading(false)) }, [loadCurrentUser])
  const login = useCallback(async (email: string, password: string) => { const result = await loginRequest({ email, password }); saveToken(result.access_token); const currentUser = await getCurrentUser(result.access_token); setAccessToken(result.access_token); setUser(currentUser); setServiceError(null); return currentUser }, [])
  const value = useMemo(() => ({ user, accessToken, isAuthenticated: Boolean(user && accessToken), isLoading, serviceError, login, logout, loadCurrentUser }), [user, accessToken, isLoading, serviceError, login, logout, loadCurrentUser])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
export function useAuth() { const context = useContext(AuthContext); if (!context) throw new Error('useAuth must be used within AuthProvider'); return context }
