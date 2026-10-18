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
    // Vite 8 blocks non-local Host headers. The real hostname used to be
    // hardcoded here; it was scrubbed out of history on 2026-10-18 before the
    // repo went public, which would otherwise leave `npm run dev` unreachable
    // from anything but localhost. Supply it at run time instead:
    //   VITE_ALLOWED_HOSTS=.example.internal npm run dev -- --host 0.0.0.0
    allowedHosts: [
      ...(process.env.VITE_ALLOWED_HOSTS?.split(',').filter(Boolean) ?? []),
      'localhost',
    ],
  },
})
