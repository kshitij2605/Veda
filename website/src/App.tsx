import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Landing } from './pages/Landing'
import { DocPage } from './pages/DocPage'

// Router-agnostic route table, shared by the client (BrowserRouter) and the
// SSR prerender entry (MemoryRouter in entry-server.tsx).
export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/docs" element={<Navigate to="/docs/introduction" replace />} />
      <Route path="/docs/:slug" element={<DocPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    // Must match Vite's `base` or deep links 404 under a sub-path deploy.
    // Reading BASE_URL keeps them from drifting apart.
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <AppRoutes />
    </BrowserRouter>
  )
}
