import { randomUUID } from "node:crypto";
import { Buffer } from "node:buffer";
import { mkdirSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { deflateSync } from "node:zlib";
import { expect, test, type Page } from "@playwright/test";

const enabled = process.env.ATLASLENS_INVESTOR_DEMO_E2E === "1";
const canonicalUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
let blockedOrigins: string[] = [];
let mockTileRequests = 0;
let failMockTiles = false;

function crc32(input: Buffer): number {
  let crc = 0xffffffff;
  for (const value of input) {
    crc ^= value;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type: string, payload: Buffer): Buffer {
  const label = Buffer.from(type, "ascii");
  const output = Buffer.alloc(12 + payload.length);
  output.writeUInt32BE(payload.length, 0);
  label.copy(output, 4);
  payload.copy(output, 8);
  output.writeUInt32BE(crc32(Buffer.concat([label, payload])), 8 + payload.length);
  return output;
}

function controlledMapTile(): Buffer {
  const size = 256;
  const scanlines = Buffer.alloc(size * (1 + size * 3));
  for (let y = 0; y < size; y += 1) {
    const row = y * (1 + size * 3);
    scanlines[row] = 0;
    for (let x = 0; x < size; x += 1) {
      const roadA = Math.abs(y - (0.44 * x + 54));
      const roadB = Math.abs(y - (-0.7 * x + 230));
      const river = Math.abs(y - (136 + Math.sin(x / 30) * 17));
      const block = (Math.floor(x / 48) + Math.floor(y / 48)) % 3 === 0;
      let color = block ? [220, 230, 218] : [232, 236, 231];
      if (river < 4) color = [170, 202, 210];
      if (roadA < 9 || roadB < 9) color = [197, 205, 199];
      if (roadA < 5 || roadB < 5) color = [255, 255, 253];
      const offset = row + 1 + x * 3;
      scanlines[offset] = color[0]!;
      scanlines[offset + 1] = color[1]!;
      scanlines[offset + 2] = color[2]!;
    }
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0);
  header.writeUInt32BE(size, 4);
  header.set([8, 2, 0, 0, 0], 8);
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", header),
    pngChunk("IDAT", deflateSync(scanlines)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

const mockTile = controlledMapTile();

function loopbackUrl(raw: string | undefined, label: string): URL {
  if (!raw) throw new Error(`${label} is required for the external investor-demo gate.`);
  const parsed = new URL(raw);
  if (
    parsed.protocol !== "http:" ||
    parsed.hostname !== "127.0.0.1" ||
    parsed.username ||
    parsed.password
  ) {
    throw new Error(`${label} must be an unauthenticated http://127.0.0.1 URL.`);
  }
  return parsed;
}

function externalContract(): { baseUrl: URL; caseId: string; launcherUrl: URL } {
  const baseUrl = loopbackUrl(process.env.ATLASLENS_E2E_BASE_URL, "ATLASLENS_E2E_BASE_URL");
  const launcherUrl = loopbackUrl(
    process.env.ATLASLENS_INVESTOR_DEMO_URL,
    "ATLASLENS_INVESTOR_DEMO_URL",
  );
  const caseIds = launcherUrl.searchParams.getAll("caseId");
  const caseId = caseIds[0] ?? "";
  if (
    launcherUrl.origin !== baseUrl.origin ||
    launcherUrl.pathname !== "/" ||
    launcherUrl.searchParams.get("demo") !== "investor" ||
    launcherUrl.searchParams.get("lang") !== "tr" ||
    caseIds.length !== 1 ||
    !canonicalUuid.test(caseId)
  ) {
    throw new Error("The launcher URL does not satisfy the official investor-demo route contract.");
  }
  return { baseUrl, caseId: caseId.toLowerCase(), launcherUrl };
}

function originOnly(raw: string): string {
  try {
    const url = new URL(raw);
    return `${url.protocol}//${url.host}`;
  } catch {
    return "invalid-url";
  }
}

function isOsmViewportTile(url: URL): boolean {
  return (
    url.protocol === "https:" &&
    url.hostname === "tile.openstreetmap.org" &&
    !url.port &&
    !url.username &&
    !url.password &&
    /^\/\d+\/\d+\/\d+\.png$/.test(url.pathname) &&
    !url.search &&
    !url.hash
  );
}

async function readyWorkspace(page: Page): Promise<string> {
  const workspace = page.locator("main.investor-workspace");
  await expect(workspace).toBeVisible();
  const heading = workspace.locator("#investor-result-title");
  await expect(heading).toContainText(/\S/);
  return (await heading.innerText()).trim();
}

function currentCaseId(page: Page): string | null {
  return new URL(page.url()).searchParams.get("caseId");
}

test.describe("official investor demo stack", () => {
  test.skip(!enabled, "Set ATLASLENS_INVESTOR_DEMO_E2E=1 through the official runner.");
  test.describe.configure({ mode: "serial", timeout: 60_000 });

  test.beforeEach(async ({ context }) => {
    blockedOrigins = [];
    mockTileRequests = 0;
    failMockTiles = false;
    await context.route("**/*", async (route) => {
      const requestUrl = new URL(route.request().url());
      if (isOsmViewportTile(requestUrl)) {
        mockTileRequests += 1;
        if (failMockTiles) {
          await route.abort("failed");
          return;
        }
        await route.fulfill({
          status: 200,
          contentType: "image/png",
          body: mockTile,
          headers: { "Cache-Control": "no-store" },
        });
        return;
      }
      if (
        ["http:", "https:", "ws:", "wss:"].includes(requestUrl.protocol) &&
        requestUrl.hostname !== "127.0.0.1"
      ) {
        blockedOrigins.push(originOnly(requestUrl.href));
        await route.abort("blockedbyclient");
        return;
      }
      await route.continue();
    });
    await context.routeWebSocket("**/*", async (socket) => {
      const socketUrl = new URL(socket.url());
      if (socketUrl.hostname !== "127.0.0.1") {
        blockedOrigins.push(originOnly(socket.url()));
        await socket.close({ code: 1008, reason: "non-loopback blocked" });
        return;
      }
      socket.connectToServer();
    });
  });

  test.afterEach(() => {
    expect(blockedOrigins, "the official demo attempted a non-loopback request").toEqual([]);
  });

  test("direct caseId load survives refresh and keeps honest provider limits", async ({ page }) => {
    const { caseId, launcherUrl } = externalContract();
    await page.goto(launcherUrl.href);

    const title = await readyWorkspace(page);
    await expect.poll(() => mockTileRequests).toBeGreaterThan(0);
    expect(currentCaseId(page)).toBe(caseId);
    await expect(page.getByText("Ankara referans koleksiyonu içinde görsel benzerlik araması.")).toBeVisible();
    await expect(page.getByText("Türkiye-geneli konum tespiti veya kalibre edilmiş doğruluk değildir.")).toBeVisible();
    await expect(page.getByText(/confidence veya olasılık değildir/)).toBeVisible();
    await expect(page.getByTestId("investor-query-preview")).toContainText("Analiz sonrası görsel kaldırıldı");
    await expect(page.getByTestId("investor-map")).toBeVisible();
    await expect(page.getByTestId("investor-map-offline")).toHaveCount(0);
    await expect(page.getByRole("link", { name: "© OpenStreetMap contributors" })).toBeVisible();
    const cards = page.locator(".investor-alternatives button");
    const markers = page.locator(".investor-numbered-marker");
    await expect(cards).toHaveCount(5);
    await expect(markers).toHaveCount(5);
    await expect(cards.first()).toBeVisible();
    await expect(page.locator('.investor-numbered-marker[data-candidate-rank="1"]')).toBeVisible();
    if (await cards.count() > 1) {
      await cards.nth(1).click();
      await expect(cards.nth(1)).toHaveAttribute("aria-pressed", "true");
      await page.locator('.investor-numbered-marker[data-candidate-rank="2"]').click();
      await expect(cards.nth(1)).toHaveAttribute("aria-pressed", "true");
    }
    await page.getByRole("button", { name: "Sıfırla / adaylara sığdır", exact: true }).click();

    await page.reload();
    expect(await readyWorkspace(page)).toBe(title);
    expect(currentCaseId(page)).toBe(caseId);
  });

  test("missing caseId requires one explicit authorized-case selection and updates the URL", async ({ page }) => {
    const { caseId, launcherUrl } = externalContract();
    const landingUrl = new URL(launcherUrl);
    landingUrl.searchParams.delete("caseId");
    await page.goto(landingUrl.href);

    await expect(page.getByTestId("investor-route-notice")).toBeVisible();
    const authorizedCases = page.getByTestId("investor-authorized-cases");
    await expect(authorizedCases).toBeVisible();
    await authorizedCases.locator("summary").click();
    const openButtons = authorizedCases.getByTestId("investor-authorized-case-open");
    await expect(openButtons).toHaveCount(1);
    await openButtons.first().click();

    await readyWorkspace(page);
    await expect.poll(() => currentCaseId(page)).toBe(caseId);
  });

  test("malformed caseId is never loaded and recovers to the authorized landing", async ({ page }) => {
    const { launcherUrl } = externalContract();
    const malformed = "not-a-uuid";
    const malformedUrl = new URL(launcherUrl);
    malformedUrl.searchParams.set("caseId", malformed);
    let malformedDetailRequested = false;
    page.on("request", (request) => {
      const url = new URL(request.url());
      if (url.pathname.includes(`/api/v1/cases/${malformed}`)) malformedDetailRequested = true;
    });

    await page.goto(malformedUrl.href);

    await expect(page.getByTestId("investor-route-notice")).toContainText("caseId");
    await expect(page.getByTestId("investor-authorized-cases")).toBeVisible();
    await expect.poll(() => currentCaseId(page)).toBeNull();
    expect(malformedDetailRequested).toBe(false);
  });

  test("valid nonexistent caseId stays visible and supports an explicit retry", async ({ page }) => {
    const { launcherUrl } = externalContract();
    const nonexistentId = randomUUID();
    const nonexistentUrl = new URL(launcherUrl);
    nonexistentUrl.searchParams.set("caseId", nonexistentId);
    let exactCaseRequests = 0;
    page.on("request", (request) => {
      const url = new URL(request.url());
      if (request.method() === "GET" && url.pathname === `/api/v1/cases/${nonexistentId}`) {
        exactCaseRequests += 1;
      }
    });

    await page.goto(nonexistentUrl.href);

    const error = page.locator(".workspace-state[role='alert']");
    await expect(error.getByRole("heading", { name: "İstenen vaka bulunamadı." })).toBeVisible();
    expect(currentCaseId(page)).toBe(nonexistentId);
    expect(exactCaseRequests).toBe(1);
    const beforeRetry = exactCaseRequests;
    await error.getByRole("button", { name: "Yeniden dene" }).click();
    await expect.poll(() => exactCaseRequests).toBeGreaterThan(beforeRetry);
    await expect(error).toBeVisible();
    expect(currentCaseId(page)).toBe(nonexistentId);
  });

  test("tile failure keeps candidate geometry and exposes an honest retry", async ({ page }) => {
    const { launcherUrl } = externalContract();
    failMockTiles = true;
    await page.goto(launcherUrl.href);
    await readyWorkspace(page);

    const offline = page.getByTestId("investor-map-offline");
    await expect(offline).toContainText("Alt harita çevrimdışı");
    await expect(page.locator('.investor-numbered-marker[data-candidate-rank="1"]')).toBeVisible();
    await expect(page.getByRole("link", { name: "© OpenStreetMap contributors" })).toBeVisible();
    const beforeRetry = mockTileRequests;
    await offline.getByRole("button", { name: "Alt haritayı yeniden dene" }).click();
    await expect.poll(() => mockTileRequests).toBeGreaterThan(beforeRetry);
    await expect(offline).toBeVisible();
  });

  test("five required viewports keep the map, result, controls, and attribution usable", async ({ page }) => {
    const { launcherUrl } = externalContract();
    const outputDirectory = process.env.ATLASLENS_PHASE3E_SCREENSHOT_DIR;
    if (!outputDirectory || !isAbsolute(outputDirectory)) {
      throw new Error("ATLASLENS_PHASE3E_SCREENSHOT_DIR must be an absolute external path.");
    }
    mkdirSync(outputDirectory, { recursive: true });
    const viewports = [
      { width: 1920, height: 1080 },
      { width: 1440, height: 900 },
      { width: 1280, height: 720 },
      { width: 768, height: 1024 },
      { width: 390, height: 844 },
    ];

    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.goto(launcherUrl.href);
      await readyWorkspace(page);
      await expect.poll(() => mockTileRequests).toBeGreaterThan(0);
      const layout = await page.evaluate(() => {
        const rectangle = (selector: string) => {
          const rect = document.querySelector<HTMLElement>(selector)?.getBoundingClientRect();
          return rect ? { top: rect.top, bottom: rect.bottom, height: rect.height } : null;
        };
        const map = document.querySelector<HTMLElement>(".investor-map-surface");
        return {
          scrollWidth: document.documentElement.scrollWidth,
          viewportWidth: window.innerWidth,
          viewportHeight: window.innerHeight,
          map: map ? rectangle(".investor-map-surface") : null,
          result: rectangle(".investor-result-panel"),
          title: rectangle("#investor-result-title"),
          action: rectangle(".investor-primary-action"),
          attribution: rectangle(".investor-map-attribution"),
        };
      });
      expect(layout.scrollWidth).toBeLessThanOrEqual(layout.viewportWidth + 1);
      expect(layout.map?.height ?? 0).toBeGreaterThan(240);
      expect(layout.title?.top ?? layout.viewportHeight).toBeLessThan(layout.viewportHeight);
      if (viewport.width >= 1280) expect(layout.action?.bottom ?? Infinity).toBeLessThanOrEqual(layout.viewportHeight);
      if (viewport.width === 390) expect(layout.map?.top ?? Infinity).toBeLessThan(layout.result?.top ?? -Infinity);
      if (viewport.width <= 900) expect(layout.attribution?.bottom ?? Infinity).toBeLessThanOrEqual(layout.result?.top ?? -Infinity);
      await expect(page.getByRole("link", { name: "© OpenStreetMap contributors" })).toBeVisible();
      await page.screenshot({
        path: join(outputDirectory, `${viewport.width}x${viewport.height}.png`),
        fullPage: false,
      });
    }
  });
});
