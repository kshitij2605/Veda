import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AppShell } from './components/shell/AppShell'

// Lazy-load every route. The shell + the palette are part of the
// initial bundle; the page chunks are fetched on demand. This drops
// the initial JS from ~807 kB to ~150 kB and parallel-loads the
// heavy modules (Monaco for Console, canvas + cluster math for
// Atlas) only when their route is actually visited.
const DashboardHome = lazy(() =>
  import('./pages/DashboardHome').then((m) => ({ default: m.DashboardHome })),
)
const CollectionBrowser = lazy(() =>
  import('./pages/CollectionBrowser').then((m) => ({
    default: m.CollectionBrowser,
  })),
)
const RecordPage = lazy(() =>
  import('./pages/collections/RecordPage').then((m) => ({
    default: m.RecordPage,
  })),
)
const SearchPlayground = lazy(() =>
  import('./pages/SearchPlayground').then((m) => ({
    default: m.SearchPlayground,
  })),
)
const AtlasPage = lazy(() =>
  import('./pages/atlas/AtlasPage').then((m) => ({ default: m.AtlasPage })),
)
const SystemHealth = lazy(() =>
  import('./pages/SystemHealth').then((m) => ({ default: m.SystemHealth })),
)
const IndexManager = lazy(() =>
  import('./pages/IndexManager').then((m) => ({ default: m.IndexManager })),
)
const EvaluationView = lazy(() =>
  import('./pages/EvaluationView').then((m) => ({ default: m.EvaluationView })),
)
const Pipelines = lazy(() =>
  import('./pages/Pipelines').then((m) => ({ default: m.Pipelines })),
)
const ChunkerAB = lazy(() =>
  import('./pages/ChunkerAB').then((m) => ({ default: m.ChunkerAB })),
)
const DashboardsList = lazy(() =>
  import('./pages/dashboards/DashboardsList').then((m) => ({
    default: m.DashboardsList,
  })),
)
const DashboardPage = lazy(() =>
  import('./pages/dashboards/DashboardPage').then((m) => ({
    default: m.DashboardPage,
  })),
)
const QualityLab = lazy(() =>
  import('./pages/quality/QualityLab').then((m) => ({ default: m.QualityLab })),
)
const Console = lazy(() =>
  import('./pages/Console').then((m) => ({ default: m.Console })),
)
const Observability = lazy(() =>
  import('./pages/Observability').then((m) => ({ default: m.Observability })),
)
const Settings = lazy(() =>
  import('./pages/Settings').then((m) => ({ default: m.Settings })),
)

function RouteFallback() {
  return (
    <div className="flex items-center justify-center h-full text-[12px] text-[var(--color-text-subtle)]">
      Loading…
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter basename="/niriksha">
      <AppShell>
        <Suspense fallback={<RouteFallback />}>
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
        </Suspense>
      </AppShell>
    </BrowserRouter>
  )
}
