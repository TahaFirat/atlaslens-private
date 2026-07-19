import { execFileSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";

const enabled = process.env.ATLASLENS_MOCK_E2E === "1";
const apiProject = fileURLToPath(new URL("../../../services/api", import.meta.url));
const apiPython = process.env.ATLASLENS_E2E_PYTHON ?? resolve(
  apiProject,
  ".venv",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);

function captureBrowserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}

function noExifFixture(): string {
  const output = fileURLToPath(new URL("../test-results/phase5c-mock.jpg", import.meta.url));
  mkdirSync(fileURLToPath(new URL("../test-results", import.meta.url)), { recursive: true });
  execFileSync(
    apiPython,
    [
      "-c",
      "import sys; from PIL import Image; Image.new('RGB',(96,72),(64,92,118)).save(sys.argv[1],'JPEG',quality=90)",
      output,
    ],
    { cwd: apiProject, stdio: "pipe" },
  );
  return output;
}

test("explicit development mock drives result, map, bilingual warning, history, and deletion", async ({ page }) => {
  test.skip(!enabled, "Set ATLASLENS_MOCK_E2E=1 for the quarantined simulation flow.");
  const browserErrors = captureBrowserErrors(page);
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles(noExifFixture());
  await page.getByRole("checkbox", { name: /I own this image/ }).check();
  await page.getByRole("button", { name: "Analyze photo" }).click();

  await expect(page.getByRole("alert", { name: "SIMULATED DEVELOPMENT RESULT" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("heading", { name: "Candidate regions" })).toBeVisible();
  await expect(page.getByTestId("map-region")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Textual map alternative" })).toBeVisible();

  await page.getByRole("button", { name: "TR", exact: true }).click();
  await expect(page.getByRole("alert", { name: "SİMÜLE EDİLMİŞ GELİŞTİRME SONUCU" })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);

  await page.getByRole("button", { name: "Geçmiş" }).click();
  const historyItem = page.locator(".history-item").filter({ hasText: "Simüle" });
  await expect(historyItem.getByText("Simüle", { exact: true })).toBeVisible();
  await historyItem.getByRole("button", { name: "Analizi aç" }).click();
  await page.getByRole("button", { name: "Analizi sil" }).click();
  await expect(page.getByText("Analiz sunucudan silindi.")).toBeVisible();
  await page.getByRole("button", { name: "Geçmiş" }).click();
  await expect(page.getByRole("heading", { name: "Bu filtrelerle eşleşen analiz yok" })).toBeVisible();
  expect(browserErrors).toEqual([]);
});
