import { describe, expect, it } from "vitest";
import type { Candidate } from "../src/api/schemas";
import { copyableCoordinates, openStreetMapUrl } from "../src/display";
import { completedAnalysis } from "./fixtures";

describe("privacy-bounded coordinate actions", () => {
  it("does not copy or link more precision than the visible candidate center", () => {
    const base = completedAnalysis.candidates[0];
    expect(base).toBeDefined();
    const candidate: Candidate = {
      ...base!,
      center: { latitude: 41.008234, longitude: 28.978456 },
      radius_km: 0.05,
      granularity: "exact_metadata" as const,
    };

    expect(copyableCoordinates(candidate)).toBe("41.008, 28.978");
    expect(openStreetMapUrl(candidate)).toContain("mlat=41.008&mlon=28.978");
    expect(openStreetMapUrl(candidate)).not.toContain("41.008234");
  });
});
