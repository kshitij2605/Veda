import axios from 'axios'

const api = axios.create({
  baseURL: '/v1',
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('veda_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

export default api
