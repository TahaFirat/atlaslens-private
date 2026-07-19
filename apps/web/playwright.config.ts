import { defineConfig, devices } from "@playwright/test";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const externalStack = process.env.ATLASLENS_E2E_EXTERNAL === "1";
const realModel = process.env.ATLASLENS_REAL_MODEL_E2E === "1";
const missingModel = process.env.ATLASLENS_MISSING_MODEL_E2E === "1";
const developmentMock = process.env.ATLASLENS_MOCK_E2E === "1";
const modelConfigured = realModel || missingModel;
const webBaseUrl = process.env.ATLASLENS_E2E_BASE_URL ?? "http://127.0.0.1:5174";
const repoRoot = fileURLToPath(new URL("../../", import.meta.url));
const apiPython = process.env.ATLASLENS_E2E_PYTHON ?? resolve(
  repoRoot,
  "services",
  "api",
  ".venv",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results/playwright",
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  use: {
    baseURL: webBaseUrl,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: externalStack ? undefined : [
    {
      command: `"${apiPython}" -m uvicorn atlaslens_api.main:app --host 127.0.0.1 --port 8100`,
      cwd: resolve(repoRoot, "services", "api"),
      url: "http://127.0.0.1:8100/api/v1/health",
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
      env: {
        APP_ENV: developmentMock ? "development" : "production",
        ENABLE_MOCK_INFERENCE: developmentMock ? "true" : "false",
        MOCK_SCENARIO: "turkiye_kayseri_demo",
        DATABASE_URL: "sqlite:///:memory:",
        TEMP_STORAGE_DIR: "./test-results/api-tmp",
        KEEP_UPLOADS: "false",
        OCR_ENABLED: "false",
        GLOBAL_MODEL_ENABLED: modelConfigured && !developmentMock ? "true" : "false",
        GLOBAL_MODEL_DEVICE: process.env.GLOBAL_MODEL_DEVICE ?? "auto",
        GLOBAL_MODEL_TIMEOUT_SECONDS: process.env.GLOBAL_MODEL_TIMEOUT_SECONDS ?? "180",
        MODEL_CACHE_DIR: process.env.MODEL_CACHE_DIR ?? resolve(repoRoot, ".local/models"),
        GAZETTEER_CACHE_DIR: process.env.GAZETTEER_CACHE_DIR ?? resolve(repoRoot, ".local/gazetteer"),
        ALLOWED_ORIGINS: "http://127.0.0.1:5174,http://localhost:5174",
        OPENAI_API_KEY: "",
        LOG_LEVEL: "WARNING",
      },
    },
    {
      command: `${npm} run dev -- --host 127.0.0.1 --port 5174`,
      url: "http://127.0.0.1:5174",
      timeout: 60_000,
      reuseExistingServer: !process.env.CI,
      env: {
        VITE_DEV_API_TARGET: "http://127.0.0.1:8100",
        VITE_API_BASE_URL: "",
        VITE_MAP_PROVIDER: "maplibre",
        VITE_MAP_STYLE_URL: "atlaslens://offline",
        VITE_ENABLE_OPERATOR_UI: "true",
      },
    },
  ],
});
