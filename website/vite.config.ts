import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  // A GitHub Pages *project* site is served from /<repo>/, so every asset and
  // route link needs that prefix. Driven by an env var rather than hardcoded
  // so a move to a custom domain or user site is `SITE_BASE=/` and nothing
  // else. Vite re-exports this as `import.meta.env.BASE_URL`, which is what
  // the router's basename reads — one value, no second place to forget.
  base: process.env.SITE_BASE || '/',
  plugins: [react(), tailwindcss()],
  server: {
    // Vite 8 blocks non-local Host headers by default; allow the gpu-host-a
    // hostname so the dev server is reachable from a workstation.
    allowedHosts: ['.internal.example', 'localhost'],
  },
})
