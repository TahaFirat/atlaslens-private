import { useEffect, useMemo, useRef } from "react";
import maplibregl, { LngLatBounds, type Map as MapLibreMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { MapillaryDemoCandidate } from "../mapillary-demo/models";
import { DEFAULT_MAP_STYLE_URL } from "./ResultMap";

interface MapillaryDemoMapProps {
  candidates: MapillaryDemoCandidate[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  mapLabel: string;
  alternativeLabel: string;
}

export function MapillaryDemoMap({ candidates, selectedId, onSelect, mapLabel, alternativeLabel }: MapillaryDemoMapProps) {
  const container = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;
  const points = useMemo(() => ({
    type: "FeatureCollection" as const,
    features: candidates.map((candidate) => ({
      type: "Feature" as const,
      id: candidate.mapillary_image_id,
      properties: { id: candidate.mapillary_image_id, rank: candidate.rank },
      geometry: {
        type: "Point" as const,
        coordinates: [candidate.longitude, candidate.latitude] as [number, number],
      },
    })),
  }), [candidates]);

  useEffect(() => {
    if (!container.current || candidates.length === 0 || typeof WebGLRenderingContext === "undefined") return;
    let map: MapLibreMap | null = null;
    try {
      map = new maplibregl.Map({
        container: container.current,
        style: import.meta.env.VITE_MAP_STYLE_URL || DEFAULT_MAP_STYLE_URL,
        center: [35.2, 39.0],
        zoom: 5,
        attributionControl: { compact: true },
      });
      mapRef.current = map;
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      map.on("load", () => {
        if (!map) return;
        map.addSource("mapillary-demo-references", { type: "geojson", data: points });
        map.addLayer({
          id: "mapillary-demo-reference-points",
          type: "circle",
          source: "mapillary-demo-references",
          paint: {
            "circle-color": ["case", ["==", ["get", "id"], selectedId ?? ""], "#f1c96a", "#5cc8b0"],
            "circle-radius": ["case", ["==", ["get", "id"], selectedId ?? ""], 11, 8],
            "circle-stroke-color": "#0f1413",
            "circle-stroke-width": 2,
          },
        });
        const bounds = new LngLatBounds();
        for (const candidate of candidates) bounds.extend([candidate.longitude, candidate.latitude]);
        map.fitBounds(bounds, { padding: 48, maxZoom: 13, duration: 0 });
      });
      map.on("click", "mapillary-demo-reference-points", (event) => {
        const id: unknown = event.features?.[0]?.properties?.id;
        if (typeof id === "string") selectRef.current(id);
      });
    } catch {
      mapRef.current = null;
    }
    return () => {
      map?.remove();
      mapRef.current = null;
    };
  }, [candidates, points, selectedId]);

  return (
    <section className="mapillary-demo-map" aria-label={mapLabel}>
      <div ref={container} className="map-frame mapillary-demo-map__frame" data-testid="mapillary-demo-map" />
      <div className="text-map">
        <h3>{alternativeLabel}</h3>
        <ol>
          {candidates.map((candidate) => (
            <li key={candidate.mapillary_image_id}>
              <button
                type="button"
                className="text-map-focus"
                aria-pressed={candidate.mapillary_image_id === selectedId}
                onClick={() => onSelect(candidate.mapillary_image_id)}
              >
                <strong>#{candidate.rank} · {candidate.mapillary_image_id}</strong>
                <span>{candidate.latitude.toFixed(5)}, {candidate.longitude.toFixed(5)}</span>
              </button>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
