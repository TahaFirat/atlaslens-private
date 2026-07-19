import { describe, expect, it } from "vitest";
import { candidateCenterBounds, longitudeNear, uncertaintyCircles } from "../src/map/geo";
import { completedAnalysis, modelAnalysis } from "./fixtures";

describe("uncertaintyCircles", () => {
  it("creates a closed GeoJSON polygon from each positive candidate radius", () => {
    const collection = uncertaintyCircles(completedAnalysis.candidates, 24);
    expect(collection.type).toBe("FeatureCollection");
    expect(collection.features).toHaveLength(1);
    const ring = collection.features[0]?.geometry.coordinates[0];
    expect(ring).toHaveLength(25);
    expect(ring?.[0]).toEqual(ring?.at(-1));
    expect(collection.features[0]?.properties.radius_km).toBe(0.25);
  });

  it("keeps antimeridian circles and fit longitudes in one world copy", () => {
    const candidates = modelAnalysis.candidates.filter((candidate) => Math.abs(candidate.center.longitude) > 170);
    const collection = uncertaintyCircles(candidates, 24);
    for (const feature of collection.features) {
      const ring = feature.geometry.coordinates[0] ?? [];
      for (let index = 1; index < ring.length; index += 1) {
        expect(Math.abs((ring[index]?.[0] ?? 0) - (ring[index - 1]?.[0] ?? 0))).toBeLessThan(180);
      }
    }
    expect(longitudeNear(-179.4, 175)).toBeCloseTo(180.6);
  });

  it("fits candidate centers across the antimeridian without expanding for uncertainty radii", () => {
    const bounds = candidateCenterBounds(modelAnalysis.candidates);
    const hugeRadiusBounds = candidateCenterBounds(
      modelAnalysis.candidates.map((candidate) => ({ ...candidate, radius_km: 20_000 })),
    );

    expect(bounds).not.toBeNull();
    expect((bounds?.east ?? 360) - (bounds?.west ?? 0)).toBeLessThan(10);
    expect(hugeRadiusBounds).toEqual(bounds);
  });
});
