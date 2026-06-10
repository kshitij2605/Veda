import { useEffect } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { applyTheme, getStoredTheme } from './lib/theme'
import { Landing } from './pages/Landing'

export default function App() {
  useEffect(() => {
    applyTheme(getStoredTheme())
  }, [])
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  )
}
