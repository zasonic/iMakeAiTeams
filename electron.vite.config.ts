import { defineConfig, externalizeDepsPlugin } from "electron-vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        input: { index: resolve(__dirname, "desktop-shell/main.ts") },
      },
      outDir: "out/main",
    },
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        input: { index: resolve(__dirname, "desktop-shell/preload.ts") },
      },
      outDir: "out/preload",
    },
  },
  renderer: {
    // Scope the renderer's project root to desktop-ui/ so vite only watches
    // the renderer source tree. With root="." the dev server was scanning
    // backend/, branding/, archive/, and node_modules — all irrelevant to
    // the React bundle and a small but real performance hit on first start.
    root: resolve(__dirname, "desktop-ui"),
    plugins: [react()],
    resolve: {
      alias: {
        "@": resolve(__dirname, "desktop-ui"),
      },
    },
    build: {
      rollupOptions: {
        input: { index: resolve(__dirname, "desktop-ui/index.html") },
      },
      outDir: resolve(__dirname, "out/renderer"),
    },
    server: {
      port: 5173,
      strictPort: true,
    },
  },
});
