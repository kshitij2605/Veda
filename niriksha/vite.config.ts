import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// GAP-2 fix (2026-05-21): the dev proxy target is now env-driven so
// two Niriksha instances can coexist against two different Vedas
// without editing this file:
//
//   VITE_API_TARGET=http://gpu-host-a:6341 npm run dev -- --port 5175
//   VITE_API_TARGET=http://gpu-host-a:6340 npm run dev -- --port 5174
//
// Defaults to localhost:6340 (the historical hardcoded value) so
// existing workflows keep working with no env change.
const apiTarget = process.env.VITE_API_TARGET ?? 'http://localhost:6340'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Vite 8 blocks non-local Host headers by default. Allow the
    // gpu-host-a hostname (and any internal LAN host) so the dev server
    // is reachable from a workstation.
    allowedHosts: ['.internal.example', 'localhost'],
    proxy: {
      '/v1': apiTarget,
    },
  },
})
