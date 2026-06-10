import axios from 'axios'
import api from '../api/client'

const TOKEN_KEY = 'veda_token'

/** POST credentials to the gateway; on success persist the JWT. */
export async function login(username: string, password: string): Promise<void> {
  try {
    const { data } = await api.post<{ token?: string }>('/auth/login', { username, password })
    if (!data?.token) throw new Error('No token returned')
    localStorage.setItem(TOKEN_KEY, data.token)
  } catch (e) {
    if (axios.isAxiosError(e)) {
      const body = e.response?.data as { error?: string } | undefined
      throw new Error(body?.error ?? e.message)
    }
    throw e
  }
}

export function logout(): void {
  try {
    localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* storage unavailable — non-fatal */
  }
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function isAuthed(): boolean {
  return !!getToken()
}
