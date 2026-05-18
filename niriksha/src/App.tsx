import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AppShell } from './components/shell/AppShell'
import { DashboardHome } from './pages/DashboardHome'
import { CollectionBrowser } from './pages/CollectionBrowser'
import { RecordPage } from './pages/collections/RecordPage'
import { SearchPlayground } from './pages/SearchPlayground'
import { AtlasPage } from './pages/atlas/AtlasPage'
import { SystemHealth } from './pages/SystemHealth'
import { IndexManager } from './pages/IndexManager'
import { EvaluationView } from './pages/EvaluationView'
import { Pipelines } from './pages/Pipelines'
import { ChunkerAB } from './pages/ChunkerAB'
import { DashboardsList } from './pages/dashboards/DashboardsList'
import { DashboardPage } from './pages/dashboards/DashboardPage'
import { QualityLab } from './pages/quality/QualityLab'
import { Console } from './pages/Console'
import { Observability } from './pages/Observability'
import { Settings } from './pages/Settings'

export default function App() {
  return (
    <BrowserRouter basename="/niriksha">
      <AppShell>
        <Routes>
          <Route path="/" element={<Navigate to="/collections" replace />} />
          <Route path="/home" element={<DashboardHome />} />
          <Route path="/collections" element={<CollectionBrowser />} />
          <Route path="/collections/:name" element={<CollectionBrowser />} />
          <Route
            path="/collections/:name/records/:id"
            element={<RecordPage />}
          />
          <Route path="/search" element={<SearchPlayground />} />
          <Route path="/search/:collection" element={<SearchPlayground />} />
          <Route path="/atlas" element={<AtlasPage />} />
          <Route path="/health" element={<SystemHealth />} />
          <Route path="/indexes" element={<IndexManager />} />
          <Route path="/evaluation" element={<EvaluationView />} />
          {/* Surfaces queued for later phases — render placeholder until they're built */}
          <Route path="/quality" element={<QualityLab />} />
          <Route path="/pipelines" element={<Pipelines />} />
          <Route path="/chunker" element={<ChunkerAB />} />
          <Route path="/dashboards" element={<DashboardsList />} />
          <Route path="/dashboards/:id" element={<DashboardPage />} />
          <Route path="/observability" element={<Observability />} />
          <Route path="/console" element={<Console />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/collections" replace />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  )
}

