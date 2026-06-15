import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import type { ResourcesView } from '../../api/resources'

// Metrics fetch path (api/client) — reject so the panel under test is
// unaffected by the Prometheus table.
vi.mock('../../api/client', () => ({
  default: { get: vi.fn(() => Promise.reject(new Error('no server'))) },
}))

vi.mock('../../api/collections', () => ({
  listNodes: vi.fn(() => Promise.reject(new Error('no nodes'))),
}))

const RESOURCES: ResourcesView = {
  disk: { total_bytes: 100 * 1024 ** 3, free_bytes: 8 * 1024 ** 3, used_pct: 92.0 },
  cpu_pct: 37.5,
  load: { load1: 1.25, load5: 0.8, load15: 0.4 },
  num_cpus: 8,
  fds: { open: 42, limit: 1024 },
  memory: { total_bytes: 32 * 1024 ** 3, available_bytes: 12 * 1024 ** 3 },
  rss_bytes: 2 * 1024 ** 3,
}

vi.mock('../../api/resources', () => ({
  getResources: vi.fn(() => Promise.resolve({ data: RESOURCES })),
}))

const STORE = {
  collections: [],
  health: { status: 'healthy', version: '0.2.1', checks: {} },
  loading: false,
  fetchCollections: vi.fn(() => Promise.resolve()),
  fetchHealth: vi.fn(() => Promise.resolve()),
}

vi.mock('../../stores/collectionStore', () => ({
  useCollectionStore: () => STORE,
}))

// Import after mocks are registered.
import { SystemHealth } from '../SystemHealth'

test('renders the Resources panel with formatted, null-safe values', async () => {
  render(
    <MemoryRouter>
      <SystemHealth />
    </MemoryRouter>,
  )
  expect(screen.getByText('Resources')).toBeInTheDocument()
  // CPU% formatted to one decimal.
  await waitFor(() => expect(screen.getByText('37.5%')).toBeInTheDocument())
  // Disk used% (danger threshold) and free GiB.
  expect(screen.getByText('92.0%')).toBeInTheDocument()
  expect(screen.getByText('8.0 GiB free')).toBeInTheDocument()
  // Load triple and open fds / limit.
  expect(screen.getByText('1.25 / 0.80 / 0.40')).toBeInTheDocument()
  expect(screen.getByText('42 / 1024')).toBeInTheDocument()
})

test('renders dashes for null resource signals (non-Linux host)', async () => {
  const mod = await import('../../api/resources')
  ;(mod.getResources as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
    data: {
      disk: { total_bytes: null, free_bytes: null, used_pct: null },
      cpu_pct: null,
      load: { load1: null, load5: null, load15: null },
      num_cpus: 4,
      fds: { open: null, limit: null },
      memory: { total_bytes: null, available_bytes: null },
      rss_bytes: null,
    } satisfies ResourcesView,
  })
  render(
    <MemoryRouter>
      <SystemHealth />
    </MemoryRouter>,
  )
  // The null load triple renders as em-dashes.
  await waitFor(() =>
    expect(screen.getByText('— / — / —')).toBeInTheDocument(),
  )
})
