import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./desktop-ui", import.meta.url)),
    },
  },
  test: {
    include: ["desktop-ui/**/*.test.{ts,tsx}"],
    environment: "jsdom",
    globals: false,
  },
});
