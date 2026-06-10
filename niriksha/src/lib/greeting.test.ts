import { describe, it, expect } from 'vitest'
import { greetingFor } from './greeting'

describe('greetingFor', () => {
  it('returns Good morning before noon', () => {
    expect(greetingFor(new Date('2026-06-10T08:00:00'))).toBe('Good morning')
  })
  it('returns Good afternoon midday', () => {
    expect(greetingFor(new Date('2026-06-10T14:00:00'))).toBe('Good afternoon')
  })
  it('returns Good evening at night', () => {
    expect(greetingFor(new Date('2026-06-10T21:00:00'))).toBe('Good evening')
  })
})
