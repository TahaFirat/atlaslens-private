import type { Candidate } from "./api/schemas";

export function humanizeToken(value: string): string {
  const safe = value.replaceAll(/[^a-zA-Z0-9._ -]/g, "").slice(0, 160);
  const words = safe.replaceAll(/[._-]+/g, " ").replaceAll(/\s+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Additional diagnostic";
}

export function candidateLabel(candidate: Candidate, fallback: string): string {
  if (candidate.label?.trim()) return candidate.label.trim();
  const place = candidate.model_prediction?.place_label;
  if (place && place.source !== "coordinate_fallback") {
    const parts = [place.city, place.region, place.country].filter((part): part is string => Boolean(part?.trim()));
    if (parts.length) return [...new Set(parts)].join(", ");
  }
  return fallback;
}

export function coordinateDecimals(candidate: Candidate): number {
  if (candidate.granularity === "exact_metadata") return 3;
  if (candidate.radius_km < 1) return 3;
  if (candidate.radius_km < 25) return 3;
  if (candidate.radius_km < 250) return 2;
  return 1;
}

export function displayedCoordinates(candidate: Candidate): string {
  const decimals = coordinateDecimals(candidate);
  return `${candidate.center.latitude.toFixed(decimals)}, ${candidate.center.longitude.toFixed(decimals)}`;
}

export function displayedRadius(radiusKm: number): string {
  if (radiusKm < 10) return radiusKm.toFixed(1);
  if (radiusKm < 100) return String(Math.round(radiusKm));
  return String(Math.round(radiusKm / 10) * 10);
}

export function copyableCoordinates(candidate: Candidate): string {
  return displayedCoordinates(candidate);
}

export function openStreetMapUrl(candidate: Candidate): string {
  const decimals = coordinateDecimals(candidate);
  const latitude = candidate.center.latitude.toFixed(decimals);
  const longitude = candidate.center.longitude.toFixed(decimals);
  return `https://www.openstreetmap.org/?mlat=${latitude}&mlon=${longitude}#map=8/${latitude}/${longitude}`;
}
