/* eslint-disable @typescript-eslint/no-explicit-any */
import '@testing-library/jest-dom'

// jsdom is missing browser APIs that framer-motion / recharts rely on.
// Report reduced-motion so animations resolve to their final value in tests.
if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: query.includes('prefers-reduced-motion'),
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as any
}

class StubObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return []
  }
}
;(globalThis as any).IntersectionObserver ??= StubObserver as any
;(globalThis as any).ResizeObserver ??= StubObserver as any
