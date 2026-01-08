import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 58735,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:58736',
        changeOrigin: true,
      }
    }
  }
})

