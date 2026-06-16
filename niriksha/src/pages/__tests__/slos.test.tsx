import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import type { SloDef, SloStatus } from '../../api/slos'

const SLO: SloDef = {
  id: 'slo-1',
  name: 'API availability',
  kind: { kind: 'availability' },
  target_pct: 99.0,
  window_secs: 2_592_000,
  fast_burn: { window_secs: 300, threshold: 14.4 },
  slow_burn: { window_secs: 3600, threshold: 6.0 },
}

const STATUS: SloStatus = {
  slo_id: 'slo-1',
  sli_pct: 99.5,
  target_pct: 99.0,
  good: 995,
  total: 1000,
  error_budget_remaining_pct: 50.0,
  burn_rate_fast: 0.5,
  burn_rate_slow: 0.5,
  compliant: true,
  fast_burn_threshold: 14.4,
  slow_burn_threshold: 6.0,
  breaching_fast: false,
  breaching_slow: false,
  budget_exhausted: false,
}

vi.mock('../../api/slos', async () => {
  const actual = await vi.importActual<typeof import('../../api/slos')>('../../api/slos')
  return {
    ...actual,
    listSlos: vi.fn(() => Promise.resolve({ data: { slos: [SLO] } })),
    getSloStatus: vi.fn(() => Promise.resolve({ data: STATUS })),
  }
})

// Import after mocks are registered.
import { Slos } from '../Slos'

test('renders an SLO with live SLI, error-budget gauge, and burn rates', async () => {
  render(
    <MemoryRouter>
      <Slos />
    </MemoryRouter>,
  )
  expect(screen.getByText('SLOs')).toBeInTheDocument()
  await waitFor(() => expect(screen.getByText('API availability')).toBeInTheDocument())
  // Live SLI %.
  expect(screen.getByText('99.5%')).toBeInTheDocument()
  // Error-budget gauge value.
  expect(screen.getByText('50.0%')).toBeInTheDocument()
  // Compliant badge + burn rate formatting.
  expect(screen.getByText('compliant')).toBeInTheDocument()
  expect(screen.getAllByText('0.50×').length).toBeGreaterThan(0)
})

test('shows "no data" and dashes when status is unavailable', async () => {
  const mod = await import('../../api/slos')
  ;(mod.getSloStatus as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
    new Error('no status'),
  )
  render(
    <MemoryRouter>
      <Slos />
    </MemoryRouter>,
  )
  await waitFor(() => expect(screen.getByText('no data')).toBeInTheDocument())
  // The null SLI renders as an em-dash.
  expect(screen.getAllByText('—').length).toBeGreaterThan(0)
})
