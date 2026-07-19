import type { Candidate } from "../api/schemas";

const EARTH_RADIUS_KM = 6371.0088;

export interface UncertaintyFeatureCollection {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    properties: { id: string; rank: number; radius_km: number };
    geometry: { type: "Polygon"; coordinates: number[][][] };
  }>;
}

export interface UncertaintyPoint {
  id: string;
  rank: number;
  center: { latitude: number; longitude: number };
  radius_km: number;
}

export function longitudeNear(longitude: number, anchor: number): number {
  let adjusted = longitude;
  while (adjusted - anchor > 180) adjusted -= 360;
  while (adjusted - anchor < -180) adjusted += 360;
  return adjusted;
}

export interface CandidateCenterBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

export function candidateCenterBounds(candidates: Candidate[]): CandidateCenterBounds | null {
  const first = candidates[0];
  if (!first) return null;
  const anchor = first.center.longitude;
  const longitudes = candidates.map((candidate) => longitudeNear(candidate.center.longitude, anchor));
  const latitudes = candidates.map((candidate) => candidate.center.latitude);
  return {
    west: Math.min(...longitudes),
    south: Math.min(...latitudes),
    east: Math.max(...longitudes),
    north: Math.max(...latitudes),
  };
}

export function uncertaintyRegions(candidates: UncertaintyPoint[], steps = 72): UncertaintyFeatureCollection {
  return {
    type: "FeatureCollection",
    features: candidates.map((candidate) => {
      const latitudeRadians = (candidate.center.latitude * Math.PI) / 180;
      const longitudeRadians = (candidate.center.longitude * Math.PI) / 180;
      const angularDistance = candidate.radius_km / EARTH_RADIUS_KM;
      const ring: number[][] = [];
      let previousLongitude = candidate.center.longitude;
      for (let index = 0; index <= steps; index += 1) {
        const bearing = (index / steps) * Math.PI * 2;
        const latitude = Math.asin(
          Math.sin(latitudeRadians) * Math.cos(angularDistance) +
            Math.cos(latitudeRadians) * Math.sin(angularDistance) * Math.cos(bearing),
        );
        const longitude =
          longitudeRadians +
          Math.atan2(
            Math.sin(bearing) * Math.sin(angularDistance) * Math.cos(latitudeRadians),
            Math.cos(angularDistance) - Math.sin(latitudeRadians) * Math.sin(latitude),
          );
        const continuousLongitude = longitudeNear((longitude * 180) / Math.PI, previousLongitude);
        previousLongitude = continuousLongitude;
        ring.push([continuousLongitude, (latitude * 180) / Math.PI]);
      }
      return {
        type: "Feature" as const,
        properties: { id: candidate.id, rank: candidate.rank, radius_km: candidate.radius_km },
        geometry: { type: "Polygon" as const, coordinates: [ring] },
      };
    }),
  };
}

export function uncertaintyCircles(candidates: Candidate[], steps = 72): UncertaintyFeatureCollection {
  return uncertaintyRegions(candidates, steps);
}
