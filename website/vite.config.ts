import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Vite 8 blocks non-local Host headers by default; allow the gpu-host-a
    // hostname so the dev server is reachable from a workstation.
    allowedHosts: ['.internal.example', 'localhost'],
  },
})
