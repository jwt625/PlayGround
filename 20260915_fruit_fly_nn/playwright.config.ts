import { defineConfig } from "@playwright/test";

const existingServer = process.env.CBC_INSPECTOR_URL;

export default defineConfig({
  testDir: "./tests/browser",
  timeout: 60_000,
  use: {
    baseURL: existingServer ?? "http://localhost:4173",
    headless: true,
  },
  webServer: existingServer ? undefined : {
    command: "npm run build && npm run preview -- --port 4173 --strictPort",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
