import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import type { Declaration } from "postcss";

export default defineConfig({
  plugins: [react()],
  css: {
    postcss: {
      plugins: [{
        postcssPlugin: "woff2-font-delivery",
        Declaration(declaration: Declaration) {
          // All browsers supporting this app's container queries also support WOFF2.
          // Preserve the original WOFF2 font; omit unused legacy WOFF/TTF fallbacks.
          if (declaration.prop !== "src" || declaration.parent?.type !== "atrule" || declaration.parent.name !== "font-face") return;
          const woff2 = declaration.value.split(",").find(source => /format\(["']?woff2["']?\)/.test(source));
          if (woff2) declaration.value = woff2.trim();
        },
      }],
    },
  },
  build: {
    cssMinify: "lightningcss",
    manifest: true,
    rollupOptions: { output: { manualChunks: { "math-typesetting": ["katex"] } } },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  preview: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
