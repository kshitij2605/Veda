/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'

// Note: we deliberately do NOT use @vitejs/plugin-react here. Vitest's
// bundled Vite + plugin-react v6 can emit the classic JSX runtime
// ("React is not defined"). esbuild's automatic runtime is all the
// component tests need (no Fast Refresh in a test run).
export default defineConfig({
  esbuild: { jsx: 'automatic', jsxImportSource: 'react' },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
})
