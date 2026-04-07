import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Sidebar } from './components/Sidebar'
import { DashboardHome } from './pages/DashboardHome'
import { CollectionBrowser } from './pages/CollectionBrowser'
import { SearchPlayground } from './pages/SearchPlayground'

export default function App() {
  return (
    <BrowserRouter basename="/niriksha">
      <div className="flex h-screen bg-bg-primary text-text-primary">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<DashboardHome />} />
            <Route path="/collections" element={<CollectionBrowser />} />
            <Route path="/collections/:name" element={<CollectionBrowser />} />
            <Route path="/search" element={<SearchPlayground />} />
            <Route path="/search/:collection" element={<SearchPlayground />} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
