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
    // Vite 8 blocks non-local Host headers. The real hostname used to be
    // hardcoded here; it was scrubbed out of history on 2026-10-18 before the
    // repo went public, which would otherwise leave `npm run dev` unreachable
    // from anything but localhost. Supply it at run time instead:
    //   VITE_ALLOWED_HOSTS=.example.internal npm run dev -- --host 0.0.0.0
    allowedHosts: [
      ...(process.env.VITE_ALLOWED_HOSTS?.split(',').filter(Boolean) ?? []),
      'localhost',
    ],
    proxy: {
      '/v1': apiTarget,
    },
  },
})
