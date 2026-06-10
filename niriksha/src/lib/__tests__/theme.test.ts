import { applyTheme, getStoredTheme } from '../theme'

afterEach(() => {
  localStorage.clear()
  delete document.documentElement.dataset.theme
})

test('light sets data-theme and persists', () => {
  applyTheme('light')
  expect(document.documentElement.dataset.theme).toBe('light')
  expect(getStoredTheme()).toBe('light')
})

test('dark sets data-theme and persists', () => {
  applyTheme('dark')
  expect(document.documentElement.dataset.theme).toBe('dark')
  expect(getStoredTheme()).toBe('dark')
})

test('defaults to light when nothing is stored', () => {
  expect(getStoredTheme()).toBe('light')
})
