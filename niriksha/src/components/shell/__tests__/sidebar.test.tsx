import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Sidebar } from '../Sidebar'

test('shows the Home link and the regrouped sections', () => {
  render(
    <MemoryRouter>
      <Sidebar onOpenPalette={() => {}} />
    </MemoryRouter>,
  )
  expect(screen.getByText('Home')).toBeInTheDocument()
  expect(screen.getByText('Overview')).toBeInTheDocument()
  expect(screen.getByText('Data')).toBeInTheDocument()
  expect(screen.getByText('Observe')).toBeInTheDocument()
  expect(screen.getByText('Developer')).toBeInTheDocument()
})

test('collapsed sidebar hides labels (icon rail)', () => {
  render(
    <MemoryRouter>
      <Sidebar onOpenPalette={() => {}} collapsed onToggle={() => {}} />
    </MemoryRouter>,
  )
  expect(screen.queryByText('Home')).not.toBeInTheDocument()
  expect(screen.queryByText('Overview')).not.toBeInTheDocument()
})
