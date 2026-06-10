import { compactNumber, formatInt } from '../format'

test('compact', () => {
  expect(compactNumber(3200000)).toBe('3.2M')
  expect(compactNumber(950)).toBe('950')
  expect(compactNumber(1500)).toBe('1.5K')
})

test('int', () => {
  expect(formatInt(12345)).toBe('12,345')
})
