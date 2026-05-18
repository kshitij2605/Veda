import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Vite 8 blocks non-local Host headers by default. Allow the
    // gpu-host-a hostname (and any internal LAN host) so the dev server
    // is reachable from a workstation.
    allowedHosts: ['.internal.example', 'localhost'],
    proxy: {
      '/v1': 'http://localhost:6340',
    },
  },
})
