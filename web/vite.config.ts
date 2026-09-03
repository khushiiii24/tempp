import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true },
  build: {
    // three is the only genuinely large dependency and it is used by exactly one
    // component. Splitting it keeps it out of the entry chunk so the copy above the fold
    // paints before the 3D scene has finished parsing.
    rollupOptions: {
      output: {
        manualChunks: { three: ["three"] },
      },
    },
  },
});
