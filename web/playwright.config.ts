import { defineConfig, devices } from "@playwright/test";

// Backend is mocked per-test via page.route("**/api/query"), so only the Next
// dev server runs — no FastAPI, no chroma_db, no paid OpenRouter calls.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  use: { baseURL: "http://localhost:3000" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
