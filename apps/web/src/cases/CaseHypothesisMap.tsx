import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl, {
  LngLatBounds,
  type Map as MapLibreMap,
  type StyleSpecification,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { LocationHypothesis } from "../api/schemas";
import { longitudeNear, uncertaintyRegions } from "../map/geo";
import { useI18n } from "../i18n";
import { caseCopy, decisionLabel, originLabel } from "./copy";

const DEFAULT_STYLE = "https://tiles.openfreemap.org/styles/positron";
const OFFLINE_STYLE_URL = "atlaslens://offline";
const offlineStyle: StyleSpecification = {
  version: 8,
  name: "AtlasLens hypothesis canvas",
  sources: {},
  layers: [{ id: "background", type: "background", paint: { "background-color": "#14191b" } }],
};

interface CaseHypothesisMapProps {
  hypotheses: LocationHypothesis[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  sectionId?: string;
}

function hypothesisPoints(hypotheses: LocationHypothesis[]) {
  return {
    type: "FeatureCollection" as const,
    features: hypotheses.map((item, index) => ({
      type: "Feature" as const,
      id: item.id,
      properties: {
        id: item.id,
        rank: item.rank ?? index + 1,
        origin: item.origin,
        decision: item.latest_adjudication?.decision ?? "unverified",
      },
      geometry: {
        type: "Point" as const,
        coordinates: [item.longitude, item.latitude] as [number, number],
      },
    })),
  };
}

export function CaseHypothesisMap({ hypotheses, selectedId, onSelect, sectionId }: CaseHypothesisMapProps) {
  const { locale } = useI18n();
  const text = caseCopy[locale];
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const onSelectRef = useRef(onSelect);
  const selectedIdRef = useRef(selectedId);
  const [issue, setIssue] = useState(false);
  const points = useMemo(() => hypothesisPoints(hypotheses), [hypotheses]);
  const uncertainty = useMemo(
    () => uncertaintyRegions(hypotheses.map((item, index) => ({
      id: item.id,
      rank: item.rank ?? index + 1,
      center: { latitude: item.latitude, longitude: item.longitude },
      radius_km: item.uncertainty_radius_m / 1000,
    })), 64),
    [hypotheses],
  );
  onSelectRef.current = onSelect;
  selectedIdRef.current = selectedId;

  useEffect(() => {
    if (!containerRef.current || hypotheses.length === 0 || typeof WebGLRenderingContext === "undefined") {
      setIssue(hypotheses.length > 0);
      return undefined;
    }
    let disposed = false;
    let fallbackActivated = false;
    const configured = import.meta.env.VITE_MAP_STYLE_URL?.trim();
    const buildMap = (style: string | StyleSpecification) => {
      const first = hypotheses[0];
      if (!first || !containerRef.current) return;
      const map = new maplibregl.Map({
        container: containerRef.current,
        style,
        center: [first.longitude, first.latitude],
        zoom: 3,
        attributionControl: {
          customAttribution: configured
            ? undefined
            : [
                '<a href="https://openfreemap.org/">OpenFreeMap</a>',
                '<a href="https://www.openstreetmap.org/copyright">&copy; OpenStreetMap contributors</a>',
              ],
        },
      });
      mapRef.current = map;
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      map.on("error", () => {
        if (disposed || fallbackActivated || style === offlineStyle) return;
        fallbackActivated = true;
        map.remove();
        buildMap(offlineStyle);
      });
      map.on("load", () => {
        if (disposed) return;
        setIssue(false);
        map.addSource("case-hypothesis-uncertainty", { type: "geojson", data: uncertainty });
        map.addLayer({
          id: "case-hypothesis-uncertainty-fill",
          type: "fill",
          source: "case-hypothesis-uncertainty",
          paint: { "fill-color": "#69d2b9", "fill-opacity": 0.09 },
        });
        map.addLayer({
          id: "case-hypothesis-uncertainty-line",
          type: "line",
          source: "case-hypothesis-uncertainty",
          paint: { "line-color": "#69d2b9", "line-width": 1.5, "line-dasharray": [2, 2] },
        });
        map.addSource("case-hypothesis-points", { type: "geojson", data: points });
        map.addLayer({
          id: "case-hypothesis-points",
          type: "circle",
          source: "case-hypothesis-points",
          paint: {
            "circle-color": [
              "match", ["get", "origin"],
              "operator_correction", "#f1c96a",
              "imported", "#9b8cff",
              "#69d2b9",
            ],
            "circle-radius": 8,
            "circle-stroke-color": "#101214",
            "circle-stroke-width": 2,
          },
        });
        map.addLayer({
          id: "case-hypothesis-selected",
          type: "circle",
          source: "case-hypothesis-points",
          filter: ["==", ["get", "id"], selectedIdRef.current ?? "__none__"],
          paint: {
            "circle-color": "transparent",
            "circle-radius": 13,
            "circle-stroke-color": "#ffffff",
            "circle-stroke-width": 2,
          },
        });
        map.on("click", "case-hypothesis-points", (event) => {
          const id = event.features?.[0]?.id;
          if (typeof id === "string") onSelectRef.current(id);
        });
        map.on("mouseenter", "case-hypothesis-points", () => { map.getCanvas().style.cursor = "pointer"; });
        map.on("mouseleave", "case-hypothesis-points", () => { map.getCanvas().style.cursor = ""; });

        const anchor = first.longitude;
        const bounds = new LngLatBounds();
        hypotheses.forEach((item) => bounds.extend([longitudeNear(item.longitude, anchor), item.latitude]));
        if (hypotheses.length > 1) map.fitBounds(bounds, { padding: 70, maxZoom: 7, duration: 0 });
      });
    };

    try {
      buildMap(configured === OFFLINE_STYLE_URL ? offlineStyle : configured || DEFAULT_STYLE);
    } catch {
      setIssue(true);
    }
    return () => {
      disposed = true;
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, [hypotheses, points, uncertainty]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getLayer("case-hypothesis-selected")) return;
    map.setFilter("case-hypothesis-selected", ["==", ["get", "id"], selectedId ?? "__none__"]);
  }, [selectedId]);

  return (
    <section id={sectionId} className="case-map-card" aria-labelledby="case-map-title">
      <header>
        <div>
          <p className="eyebrow">MapLibre</p>
          <h2 id="case-map-title">{text.mapTitle}</h2>
        </div>
        <p>{text.mapHelp}</p>
      </header>
      <div
        ref={containerRef}
        className="case-map"
        data-testid="case-hypothesis-map"
        aria-label={text.mapTitle}
      >
        {hypotheses.length === 0 || issue ? <p className="case-map__fallback">{text.mapUnavailable}</p> : null}
      </div>
      <ol className="case-map-alternative">
        {hypotheses.map((item, index) => (
          <li key={item.id}>
            <button
              type="button"
              aria-pressed={selectedId === item.id}
              onClick={() => onSelect(item.id)}
            >
              <strong>#{item.rank ?? index + 1} · {originLabel(item.origin, locale)}</strong>
              <span>{decisionLabel(item.latest_adjudication?.decision ?? null, locale)}</span>
              <small>{item.latitude.toFixed(5)}, {item.longitude.toFixed(5)} · {Math.round(item.uncertainty_radius_m).toLocaleString(locale)} m</small>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}
