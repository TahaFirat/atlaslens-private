import type { Phase6CAnalysisSummary, Phase6BModelPredictionSummary } from "../api/schemas";

export type Phase6CMapLayer = "geoclip" | "hierarchy" | "osv" | "plonk" | "megaloc" | "fused";
export const PHASE6C_MAP_LAYER_LIMIT = 12;

interface Phase6CMapFeature {
  type: "Feature";
  properties: { layer: Phase6CMapLayer; rank: number; support: number; radius_km: number };
  geometry: { type: "Point"; coordinates: [number, number] };
}

function predictionFeatures(
  predictions: Record<string, Phase6BModelPredictionSummary | null> | null | undefined,
  layer: "geoclip" | "osv" | "plonk",
): Phase6CMapFeature[] {
  const matches = Object.values(predictions ?? {}).filter((prediction): prediction is Phase6BModelPredictionSummary => {
    if (!prediction) return false;
    const provider = prediction.provider.toLowerCase();
    return layer === "geoclip" ? provider.includes("geoclip") : provider.includes(layer);
  });
  return matches.flatMap((prediction) => prediction.candidates).slice(0, PHASE6C_MAP_LAYER_LIMIT).map((candidate) => ({
    type: "Feature" as const,
    properties: {
      layer,
      rank: candidate.provider_rank,
      support: candidate.sample_support,
      radius_km: 0,
    },
    geometry: { type: "Point" as const, coordinates: [candidate.longitude, candidate.latitude] },
  }));
}

export function phase6CMapPoints(
  phase6c: Phase6CAnalysisSummary | null | undefined,
  predictions: Record<string, Phase6BModelPredictionSummary | null> | null | undefined,
) {
  const features: Phase6CMapFeature[] = [
    ...predictionFeatures(predictions, "geoclip"),
    ...predictionFeatures(predictions, "osv"),
    ...predictionFeatures(predictions, "plonk"),
  ];
  if (phase6c) {
    features.push(
      ...phase6c.hierarchical_candidates.slice(0, PHASE6C_MAP_LAYER_LIMIT).map((candidate) => ({
        type: "Feature" as const,
        properties: { layer: "hierarchy" as const, rank: candidate.provider_rank, support: 1, radius_km: candidate.grid_resolution_km },
        geometry: { type: "Point" as const, coordinates: [candidate.longitude, candidate.latitude] as [number, number] },
      })),
      ...phase6c.megaloc_matches.slice(0, PHASE6C_MAP_LAYER_LIMIT).map((candidate) => ({
        type: "Feature" as const,
        properties: { layer: "megaloc" as const, rank: candidate.rank, support: 1, radius_km: candidate.uncertainty_radius_m / 1_000 },
        geometry: { type: "Point" as const, coordinates: [candidate.longitude, candidate.latitude] as [number, number] },
      })),
      ...phase6c.fusion_candidates.slice(0, PHASE6C_MAP_LAYER_LIMIT).map((candidate) => ({
        type: "Feature" as const,
        properties: { layer: "fused" as const, rank: candidate.final_rank, support: candidate.independent_source_family_count, radius_km: candidate.uncertainty_radius_km },
        geometry: { type: "Point" as const, coordinates: [candidate.longitude, candidate.latitude] as [number, number] },
      })),
    );
  }
  return { type: "FeatureCollection" as const, features };
}
