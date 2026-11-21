import { defineConfig } from 'vite';

// BASE_PATH is set by the GitHub Pages workflow (e.g. "/GPUgraphProcessing/").
export default defineConfig({
  base: process.env.BASE_PATH || '/',
  server: { port: 5173, open: false },
  build: { target: 'esnext' },
  test: { environment: 'node', include: ['tests/**/*.test.js'] },
});
