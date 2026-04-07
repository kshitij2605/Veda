import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Sidebar } from './components/Sidebar'
import { DashboardHome } from './pages/DashboardHome'
import { CollectionBrowser } from './pages/CollectionBrowser'
import { SearchPlayground } from './pages/SearchPlayground'
import { VectorAtlas } from './pages/VectorAtlas'
import { SystemHealth } from './pages/SystemHealth'
import { IndexManager } from './pages/IndexManager'
import { EvaluationView } from './pages/EvaluationView'

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
            <Route path="/atlas" element={<VectorAtlas />} />
            <Route path="/health" element={<SystemHealth />} />
            <Route path="/indexes" element={<IndexManager />} />
            <Route path="/evaluation" element={<EvaluationView />} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
