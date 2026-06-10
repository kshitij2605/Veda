import { vi } from 'vitest'
import * as api from '../../api/collections'
import { useCollectionStore } from '../collectionStore'

vi.mock('../../api/collections', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/collections')>()
  return { ...actual, fetchCollection: vi.fn() }
})

test('fetchAllDetails populates details and skips failures', async () => {
  useCollectionStore.setState({
    collections: [
      { name: 'a', field_count: 1, shard_count: 1 },
      { name: 'b', field_count: 1, shard_count: 1 },
    ],
    details: [],
  })

  vi.mocked(api.fetchCollection).mockImplementation((name: string) =>
    name === 'a'
      ? Promise.resolve({
          data: {
            name: 'a',
            fields: [{ name: 'text_embedding', type: 'vector', filterable: false }],
            shard_count: 1,
            replication_factor: 1,
          },
        } as Awaited<ReturnType<typeof api.fetchCollection>>)
      : Promise.reject(new Error('boom')),
  )

  await useCollectionStore.getState().fetchAllDetails()

  const details = useCollectionStore.getState().details
  expect(details).toHaveLength(1)
  expect(details[0].name).toBe('a')
})
