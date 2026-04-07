import { create } from 'zustand'

interface AuthState {
  token: string | null
  isAuthenticated: boolean
  login: (token: string) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  token: localStorage.getItem('veda_token'),
  isAuthenticated: !!localStorage.getItem('veda_token'),
  login: (token) => {
    localStorage.setItem('veda_token', token)
    set({ token, isAuthenticated: true })
  },
  logout: () => {
    localStorage.removeItem('veda_token')
    set({ token: null, isAuthenticated: false })
  },
}))
