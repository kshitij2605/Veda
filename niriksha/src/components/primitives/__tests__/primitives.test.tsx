import { render, screen } from '@testing-library/react'
import { Card, StatTile, ModalityChip, ConnectionPill, EmptyState } from '../index'

test('Card renders children', () => {
  render(<Card>hi</Card>)
  expect(screen.getByText('hi')).toBeInTheDocument()
})

test('StatTile shows label + value', () => {
  render(<StatTile label="Collections" value="12" />)
  expect(screen.getByText('Collections')).toBeInTheDocument()
  expect(screen.getByText('12')).toBeInTheDocument()
})

test('ModalityChip labels modality', () => {
  render(<ModalityChip modality="image" />)
  expect(screen.getByText('Image')).toBeInTheDocument()
})

test('ConnectionPill shows status', () => {
  render(<ConnectionPill ok label="connected" />)
  expect(screen.getByText('connected')).toBeInTheDocument()
})

test('EmptyState shows title', () => {
  render(<EmptyState title="Nothing here" />)
  expect(screen.getByText('Nothing here')).toBeInTheDocument()
})
