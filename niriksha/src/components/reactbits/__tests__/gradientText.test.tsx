import { render, screen } from '@testing-library/react'
import { GradientText } from '../GradientText'

test('renders its children', () => {
  render(<GradientText>Veda</GradientText>)
  expect(screen.getByText('Veda')).toBeInTheDocument()
})
