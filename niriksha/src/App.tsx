import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AppShell } from './components/shell/AppShell'
import { DashboardHome } from './pages/DashboardHome'
import { CollectionBrowser } from './pages/CollectionBrowser'
import { RecordPage } from './pages/collections/RecordPage'
import { SearchPlayground } from './pages/SearchPlayground'
import { VectorAtlas } from './pages/VectorAtlas'
import { SystemHealth } from './pages/SystemHealth'
import { IndexManager } from './pages/IndexManager'
import { EvaluationView } from './pages/EvaluationView'
import { Pipelines } from './pages/Pipelines'

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
          <Route path="/atlas" element={<VectorAtlas />} />
          <Route path="/health" element={<SystemHealth />} />
          <Route path="/indexes" element={<IndexManager />} />
          <Route path="/evaluation" element={<EvaluationView />} />
          {/* Surfaces queued for later phases — render placeholder until they're built */}
          <Route path="/quality" element={<Placeholder name="Quality Lab" phase="Phase 5" />} />
          <Route path="/pipelines" element={<Pipelines />} />
          <Route path="/observability" element={<Placeholder name="Observability" phase="Phase 6" />} />
          <Route path="/console" element={<Placeholder name="Dev Console" phase="Phase 6" />} />
          <Route path="/settings" element={<Placeholder name="Settings" phase="Phase 6" />} />
          <Route path="*" element={<Navigate to="/collections" replace />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  )
}

function Placeholder({ name, phase }: { name: string; phase: string }) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="dot-grid absolute inset-0 opacity-30" aria-hidden />
      <div className="relative text-center max-w-md">
        <div className="text-[11px] uppercase tracking-[0.1em] text-[var(--color-text-subtle)] mb-2">
          {phase}
        </div>
        <h2 className="text-[20px] font-medium text-[var(--color-text)] mb-2">
          {name}
        </h2>
        <p className="text-[12.5px] text-[var(--color-text-muted)]">
          This surface is part of the multi-phase rebuild. The route is
          reserved and the navigation reaches it — the implementation lands
          in {phase}.
        </p>
      </div>
    </div>
  )
}
