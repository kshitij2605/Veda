import { vi } from 'vitest'
import api from '../../api/client'
import { login, getToken, logout } from '../auth'

vi.mock('../../api/client', () => ({ default: { post: vi.fn() } }))

test('login stores the returned token; logout clears it', async () => {
  vi.mocked(api.post).mockResolvedValue({ data: { token: 'abc123' } })
  await login('user', 'pass')
  expect(getToken()).toBe('abc123')
  logout()
  expect(getToken()).toBeNull()
})
