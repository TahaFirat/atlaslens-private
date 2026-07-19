import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("responsive accessibility styles", () => {
  const styles = readFileSync("src/styles.css", "utf8");
  const mapSource = readFileSync("src/components/ResultMap.tsx", "utf8");
  const hypothesisMapSource = readFileSync("src/cases/CaseHypothesisMap.tsx", "utf8");
  const investorMapSource = readFileSync("src/components/InvestorWorkspaceMap.tsx", "utf8");

  it("has a mobile layout that covers an approximately 390px viewport", () => {
    expect(styles).toContain("@media (max-width: 560px)");
    expect(styles).toContain("height: 320px");
    expect(styles).toContain("min-height: 44px");
    expect(styles).toContain("grid-template-columns: 1fr");
  });

  it("honors reduced-motion and forced-color preferences", () => {
    expect(styles).toContain("prefers-reduced-motion: reduce");
    expect(styles).toContain("forced-colors: active");
    expect(styles).toContain(":focus-visible");
    expect(styles).toContain("--faint: #8e9995");
  });

  it("keeps visible OpenStreetMap attribution on the default basemap", () => {
    expect(mapSource).toContain("customAttribution");
    expect(mapSource).toContain("OpenStreetMap contributors");
    expect(mapSource).toContain("https://www.openstreetmap.org/copyright");
  });

  it("keeps the case map local when the launcher selects the offline style", () => {
    expect(hypothesisMapSource).toContain('const OFFLINE_STYLE_URL = "atlaslens://offline"');
    expect(hypothesisMapSource).toContain("configured === OFFLINE_STYLE_URL ? offlineStyle");
  });

  it("keeps the investor map full-height, responsive, attributed, and bounded to viewport tiles", () => {
    expect(styles).toContain("height: 100dvh");
    expect(styles).toContain("grid-template-areas:");
    expect(styles).toContain("investor-query-panel:not(.investor-query-panel--open)");
    expect(styles).toContain("@media (min-width: 901px) and (max-height: 760px)");
    expect(investorMapSource).toContain("VITE_MAP_TILE_URL");
    expect(investorMapSource).toContain("prefetchZoomDelta: 0");
    expect(investorMapSource).toContain("© OpenStreetMap contributors");
    expect(investorMapSource).toContain('INVESTOR_OFFLINE_MAP = "atlaslens://offline"');
    expect(investorMapSource).toContain('marker.style.zIndex = item.id === selectedRef.current ? "4"');
  });
});
