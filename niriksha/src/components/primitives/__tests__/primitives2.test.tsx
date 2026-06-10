import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import { Tabs, SegmentedControl, Switch, PageHeader, ThemeToggle } from '../index'

test('Tabs shows triggers and the active panel', () => {
  render(
    <Tabs
      items={[
        { value: 'a', label: 'Alpha', content: <div>Apanel</div> },
        { value: 'b', label: 'Beta', content: <div>Bpanel</div> },
      ]}
    />,
  )
  expect(screen.getByText('Alpha')).toBeInTheDocument()
  expect(screen.getByText('Beta')).toBeInTheDocument()
  expect(screen.getByText('Apanel')).toBeInTheDocument()
})

test('SegmentedControl fires onChange', () => {
  const onChange = vi.fn()
  render(
    <SegmentedControl
      options={[
        { value: 'x', label: 'X' },
        { value: 'y', label: 'Y' },
      ]}
      value="x"
      onChange={onChange}
    />,
  )
  fireEvent.click(screen.getByText('Y'))
  expect(onChange).toHaveBeenCalledWith('y')
})

test('Switch toggles', () => {
  const onChange = vi.fn()
  render(<Switch checked={false} onChange={onChange} label="wifi" />)
  fireEvent.click(screen.getByRole('switch'))
  expect(onChange).toHaveBeenCalledWith(true)
})

test('PageHeader renders title and actions', () => {
  render(<PageHeader title="My Page" description="desc" actions={<button>Act</button>} />)
  expect(screen.getByText('My Page')).toBeInTheDocument()
  expect(screen.getByText('Act')).toBeInTheDocument()
})

test('ThemeToggle renders a button', () => {
  render(<ThemeToggle />)
  expect(screen.getByRole('button')).toBeInTheDocument()
})
