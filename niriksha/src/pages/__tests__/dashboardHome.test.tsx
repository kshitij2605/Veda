import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'

vi.mock('../../api/client', () => ({
  default: { get: vi.fn(() => Promise.reject(new Error('no server'))) },
}))

const STORE = {
  collections: [
    { name: 'articles', field_count: 3, shard_count: 1 },
    { name: 'product_images', field_count: 3, shard_count: 1 },
    { name: 'product_catalog', field_count: 5, shard_count: 1 },
  ],
  details: [
    {
      name: 'articles',
      fields: [{ name: 'text_embedding', type: 'vector', filterable: false }],
      shard_count: 1,
      replication_factor: 1,
    },
    {
      name: 'product_images',
      fields: [{ name: 'image_embedding', type: 'vector', filterable: false }],
      shard_count: 1,
      replication_factor: 1,
    },
    {
      name: 'product_catalog',
      fields: [
        { name: 'text_embedding', type: 'vector', filterable: false },
        { name: 'image_embedding', type: 'vector', filterable: false },
      ],
      shard_count: 1,
      replication_factor: 1,
    },
  ],
  health: { status: 'healthy', version: '0.2.1' },
  loading: false,
  error: null,
  fetchCollections: vi.fn(() => Promise.resolve()),
  fetchHealth: vi.fn(() => Promise.resolve()),
  fetchAllDetails: vi.fn(() => Promise.resolve()),
}

vi.mock('../../stores/collectionStore', () => ({ useCollectionStore: () => STORE }))

// Import after mocks are registered.
import { DashboardHome } from '../DashboardHome'

test('renders KPIs, collections, and modality chips', () => {
  render(
    <MemoryRouter>
      <DashboardHome />
    </MemoryRouter>,
  )
  expect(screen.getByText('articles')).toBeInTheDocument()
  expect(screen.getByText('product_catalog')).toBeInTheDocument()
  // product_catalog has both text + image vectors → "Multimodal" (table chip + legend)
  expect(screen.getAllByText('Multimodal').length).toBeGreaterThan(0)
  // KPI label (the table header is "All collections", so this is unambiguous)
  expect(screen.getByText('Collections')).toBeInTheDocument()
})
