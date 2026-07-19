import { expect, test, type Page } from "@playwright/test";

const enabled = process.env.ATLASLENS_REAL_MODEL_E2E === "1";
const missingModelEnabled = process.env.ATLASLENS_MISSING_MODEL_E2E === "1";
const imagePath = process.env.ATLASLENS_REAL_IMAGE;
const apiBaseUrl = process.env.ATLASLENS_E2E_API_URL ?? "http://127.0.0.1:8100";

test.setTimeout(300_000);

function captureBrowserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}

function captureExternalRequests(page: Page): string[] {
  const requests: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (
      !["blob:", "data:"].includes(url.protocol)
      && !["127.0.0.1", "localhost"].includes(url.hostname)
    ) requests.push(url.origin);
  });
  return requests;
}

test("real no-EXIF image returns clustered uncalibrated model hypotheses and can be deleted", async ({ page, request }) => {
  test.skip(!enabled, "Set ATLASLENS_REAL_MODEL_E2E=1 to run the installed-model E2E test.");
  test.skip(!imagePath, "Set ATLASLENS_REAL_IMAGE to an authorized no-EXIF image.");
  const browserErrors = captureBrowserErrors(page);
  const externalRequests = captureExternalRequests(page);

  await page.goto("/");
  await expect(page.getByText("Global visual prediction")).toBeVisible();
  await expect(page.getByText(/Ready.*Server-local/)).toBeVisible({ timeout: 120_000 });
  await page.getByTestId("file-input").setInputFiles(imagePath as string);
  await page.getByRole("checkbox", { name: /I own this image/ }).check();

  const createResponse = page.waitForResponse(
    (response) => response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1/analyses",
  );
  const eventRequest = page.waitForRequest(
    (request) => request.method() === "GET" && /\/api\/v1\/analyses\/[^/]+\/events$/.test(new URL(request.url()).pathname),
  );
  await page.getByRole("button", { name: "Analyze photo" }).click();
  const response = await createResponse;
  expect(response.status()).toBe(202);
  const accepted = (await response.json()) as { id: string };
  await eventRequest;

  await expect(page.getByRole("heading", { name: "Candidate regions" })).toBeVisible({ timeout: 180_000 });
  const modelCards = page.locator(".candidate-summary");
  await expect(modelCards.first()).toBeVisible();
  const displayedModelCount = await modelCards.count();
  expect(displayedModelCount).toBeGreaterThanOrEqual(1);
  expect(displayedModelCount).toBeLessThanOrEqual(5);
  await expect(page.getByText("Uncalibrated relative support").first()).toBeVisible();
  await page.getByText("Advanced diagnostics and raw relative scores").click();
  await expect(page.getByText(/Raw relative score:/).first()).toBeVisible();
  await expect(page.getByText(/not a probability/).first()).toBeVisible();
  await expect(page.getByText(/ ms$/).first()).toBeVisible();
  await expect(page.getByTestId("map-region")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Textual map alternative" })).toBeVisible();
  await expect(page.locator(".text-map").getByText(/Uncertainty radius:/)).toHaveCount(displayedModelCount);

  const analysis = await request.get(`${apiBaseUrl}/api/v1/analyses/${accepted.id}`);
  expect(analysis.status()).toBe(200);
  const payload = (await analysis.json()) as {
    candidates: Array<{
      model_prediction?: { device?: string };
      phase5b_assessment?: { classification?: string };
    }>;
  };
  expect(payload.candidates).toHaveLength(displayedModelCount);
  expect(payload.candidates.every((candidate) => Boolean(candidate.model_prediction?.device))).toBe(true);
  expect(
    payload.candidates.every(
      (candidate) => candidate.phase5b_assessment?.classification === "model_only",
    ),
  ).toBe(true);
  const capabilityResponse = await request.get(`${apiBaseUrl}/api/v1/capabilities`);
  expect(capabilityResponse.status()).toBe(200);
  const capabilities = (await capabilityResponse.json()) as {
    providers: {
      global_geolocation?: {
        installed?: boolean;
        verified?: boolean;
        operational_status?: string;
        device?: string;
      };
    };
  };
  expect(capabilities.providers.global_geolocation).toMatchObject({
    installed: true,
    verified: true,
    operational_status: "ready",
    device: "cuda",
  });

  await page.getByRole("button", { name: "Delete analysis" }).click();
  await expect(page.getByText("Analysis deleted from the server.")).toBeVisible();
  const deleted = await request.get(`${apiBaseUrl}/api/v1/analyses/${accepted.id}`);
  expect(deleted.status()).toBe(404);
  expect(browserErrors).toEqual([]);
  expect(externalRequests).toEqual([]);
});

test("missing model is installation-specific and safely abstains", async ({ page, request }) => {
  test.skip(!missingModelEnabled, "Set ATLASLENS_MISSING_MODEL_E2E=1 to run the missing-model E2E test.");
  test.skip(!imagePath, "Set ATLASLENS_REAL_IMAGE to an authorized no-EXIF image.");
  const browserErrors = captureBrowserErrors(page);
  const externalRequests = captureExternalRequests(page);

  await page.goto("/");
  await expect(page.getByText("Global visual prediction")).toBeVisible();
  await expect(page.getByText(/Model not installed.*Server-local/)).toBeVisible();
  await page.getByTestId("file-input").setInputFiles(imagePath as string);
  await page.getByRole("checkbox", { name: /I own this image/ }).check();
  const createResponse = page.waitForResponse(
    (response) => response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1/analyses",
  );
  await page.getByRole("button", { name: "Analyze photo" }).click();
  const response = await createResponse;
  expect(response.status()).toBe(202);
  const accepted = (await response.json()) as { id: string };

  await expect(page.getByRole("heading", { name: "Not enough evidence to estimate a location" })).toBeVisible();
  await expect(page.getByText("Global model prediction was skipped because the model is not installed.")).toBeVisible();
  await page.getByRole("button", { name: "Delete analysis" }).click();
  await expect(page.getByText("Analysis deleted from the server.")).toBeVisible();
  const deleted = await request.get(`${apiBaseUrl}/api/v1/analyses/${accepted.id}`);
  expect(deleted.status()).toBe(404);
  expect(browserErrors).toEqual([]);
  expect(externalRequests).toEqual([]);
});
