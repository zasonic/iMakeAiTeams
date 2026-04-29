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
    root: ".",
    plugins: [react()],
    resolve: {
      alias: {
        "@": resolve(__dirname, "desktop-ui"),
      },
    },
    build: {
      rollupOptions: {
        input: { index: resolve(__dirname, "index.html") },
      },
      outDir: "out/renderer",
    },
    server: {
      port: 5173,
      strictPort: true,
    },
  },
});
