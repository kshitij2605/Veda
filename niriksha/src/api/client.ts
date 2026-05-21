import axios from 'axios'

// GAP-2 fix (2026-05-21): API base URL is now env-driven.
//
// - During `npm run dev`, baseURL stays `/v1` (the relative path), so
//   the Vite proxy in `vite.config.ts` forwards it to the backend
//   configured by VITE_API_TARGET. This lets one Niriksha dev server
//   talk to any test Veda instance without code changes.
// - For production builds (`npm run build`), set VITE_API_BASE to the
//   absolute backend URL (e.g. `https://veda.prod.example.com/v1`)
//   before building. Defaults to `/v1` so the existing same-origin
//   deployment shape still works.
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/v1'

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('veda_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

export default api
