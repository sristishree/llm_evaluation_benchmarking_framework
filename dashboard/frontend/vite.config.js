import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const backendUrl = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api':          { target: backendUrl, changeOrigin: true },
      '/docs':         { target: backendUrl, changeOrigin: true },
      '/redoc':        { target: backendUrl, changeOrigin: true },
      '/openapi.json': { target: backendUrl, changeOrigin: true },
    },
  },
})
