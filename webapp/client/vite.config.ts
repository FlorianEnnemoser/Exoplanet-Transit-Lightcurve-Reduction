import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev: Vite serves the SPA, FastAPI answers /api (uvicorn on :8000)
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
