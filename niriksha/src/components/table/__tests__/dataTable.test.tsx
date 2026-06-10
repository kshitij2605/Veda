import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import { type ColumnDef } from '@tanstack/react-table'
import { DataTable } from '../DataTable'

interface Row {
  name: string
  n: number
}

const columns: ColumnDef<Row>[] = [
  { accessorKey: 'name', header: 'Name' },
  { accessorKey: 'n', header: 'Count' },
]

const data: Row[] = [
  { name: 'alpha', n: 1 },
  { name: 'beta', n: 2 },
  { name: 'gamma', n: 3 },
]

test('renders headers and cells', () => {
  render(<DataTable columns={columns} data={data} />)
  expect(screen.getByText('Name')).toBeInTheDocument()
  expect(screen.getByText('alpha')).toBeInTheDocument()
  expect(screen.getByText('gamma')).toBeInTheDocument()
})

test('onRowClick fires with the original row', () => {
  const onRowClick = vi.fn()
  render(<DataTable columns={columns} data={data} onRowClick={onRowClick} />)
  fireEvent.click(screen.getByText('beta'))
  expect(onRowClick).toHaveBeenCalledWith({ name: 'beta', n: 2 })
})
