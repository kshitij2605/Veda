import { render, screen } from '@testing-library/react'
import { CountUp } from '../CountUp'
import { SpotlightCard } from '../SpotlightCard'

test('CountUp renders the target value (reduced motion path)', () => {
  render(<CountUp to={42} />)
  expect(screen.getByText('42')).toBeInTheDocument()
})

test('SpotlightCard renders its children', () => {
  render(<SpotlightCard>content</SpotlightCard>)
  expect(screen.getByText('content')).toBeInTheDocument()
})
