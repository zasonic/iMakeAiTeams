/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Helvetica Neue",
          "sans-serif",
        ],
        mono: [
          "SF Mono",
          "Cascadia Code",
          "JetBrains Mono",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      colors: {
        bg: {
          DEFAULT: "#0a0a0c",
          1: "#111114",
          2: "#19191e",
          3: "#222228",
          4: "#2a2a33",
        },
        line: {
          DEFAULT: "#2a2a33",
          soft: "#1e1e26",
        },
        ink: {
          DEFAULT: "#ececf0",
          dim: "#9d9db0",
          faint: "#5c5c72",
        },
        accent: {
          DEFAULT: "#8b7cf6",
          dark: "#7366e0",
        },
        ok: "#3dd68c",
        warn: "#f0b440",
        err: "#f0564a",
        claude: "#c49bff",
        local: "#5eead4",
      },
      boxShadow: {
        glass: "0 4px 24px rgba(0,0,0,.4),0 8px 40px rgba(0,0,0,.25)",
      },
    },
  },
  plugins: [],
};
