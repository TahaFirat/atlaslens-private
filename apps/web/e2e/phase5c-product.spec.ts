import { expect, test, type Page } from "@playwright/test";

function captureBrowserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}

test("dataset-QA fixture renders a read-only issue report without private paths", async ({ page }) => {
  const browserErrors = captureBrowserErrors(page);
  const summary = {
    report_id: "qa-e2e",
    schema_version: 1,
    dataset_fingerprint: "d".repeat(64),
    dataset_type: "geolocation",
    created_at: "2026-07-12T09:00:00Z",
    scanned_images: 3,
    scanned_masks: 0,
    error_count: 1,
    warning_count: 1,
  };
  await page.route("**/api/v1/datasets/qa/qa-e2e", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      summary,
      checks: { images: "passed", gps: "failed", duplicates: "warning" },
      distributions: { country: { TR: 3 } },
      issues: [{
        code: "gps.out_of_range",
        severity: "error",
        asset_key: "asset-opaque-17",
        field: "coordinates",
        message_key: "dataset_qa.gps.out_of_range",
        safe_metrics: {},
      }],
      outputs: ["report.json", "issues.csv", "summary.md", "report.html"],
      limitations: ["country_boundary_checks_unavailable"],
    }),
  }));
  await page.route("**/api/v1/datasets/qa", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ reports: [summary] }),
  }));

  await page.goto("/");
  await page.getByRole("button", { name: "Dataset QA" }).click();
  await expect(page.getByRole("heading", { name: "Dataset quality" })).toBeVisible();
  await expect(page.getByText("d".repeat(64))).toBeVisible();
  await expect(page.getByText("asset-opaque-17")).toBeVisible();
  await expect(page.getByText("Dataset qa gps out of range")).toBeVisible();
  await expect(page.getByText(/C:\\|\/Users\/|\/home\//)).toHaveCount(0);
  expect(browserErrors).toEqual([]);
});
