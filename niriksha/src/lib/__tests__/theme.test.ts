import { applyTheme, getStoredTheme } from '../theme'

test('light sets data-theme and persists', () => {
  applyTheme('light')
  expect(document.documentElement.dataset.theme).toBe('light')
  expect(getStoredTheme()).toBe('light')
})

test('dark removes data-theme and persists', () => {
  applyTheme('light')
  applyTheme('dark')
  expect(document.documentElement.dataset.theme).toBeUndefined()
  expect(getStoredTheme()).toBe('dark')
})
