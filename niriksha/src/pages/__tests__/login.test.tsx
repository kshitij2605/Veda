import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'

// Aurora is WebGL (ogl) — stub it so jsdom doesn't try to create a GL context.
vi.mock('../../components/reactbits/Aurora', () => ({ default: () => null }))
vi.mock('../../lib/auth', () => ({ login: vi.fn(() => Promise.resolve()) }))

import { Login } from '../Login'
import { login } from '../../lib/auth'

test('renders the form and submits credentials', () => {
  render(
    <MemoryRouter>
      <Login />
    </MemoryRouter>,
  )
  expect(screen.getByPlaceholderText('Username')).toBeInTheDocument()
  expect(screen.getByPlaceholderText('Password')).toBeInTheDocument()

  fireEvent.change(screen.getByPlaceholderText('Username'), { target: { value: 'u' } })
  fireEvent.change(screen.getByPlaceholderText('Password'), { target: { value: 'p' } })
  fireEvent.click(screen.getByRole('button', { name: /sign in/i }))

  expect(login).toHaveBeenCalledWith('u', 'p')
})
