import { render } from '@testing-library/react'
import { DonutChart } from '../DonutChart'

test('renders without crashing', () => {
  const { container } = render(
    <DonutChart
      data={[
        { label: 'A', value: 3, color: 'var(--color-chart-1)' },
        { label: 'B', value: 1, color: 'var(--color-chart-2)' },
      ]}
    />,
  )
  expect(container.firstChild).toBeTruthy()
})
