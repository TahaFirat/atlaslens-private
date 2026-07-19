import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl, {
  LngLatBounds,
  type FilterSpecification,
  type GeoJSONSource,
  type Map as MapLibreMap,
  type StyleSpecification,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type {
  Candidate,
  Phase6CAnalysisSummary,
  Phase6BModelPredictionSummary,
} from "../api/schemas";
import { displayedCoordinates, displayedRadius } from "../display";
import { useI18n, type TranslationKey } from "../i18n";
import { candidateCenterBounds, longitudeNear, uncertaintyCircles, uncertaintyRegions } from "../map/geo";
import { PHASE6C_MAP_LAYER_LIMIT, phase6CMapPoints, type Phase6CMapLayer } from "../map/phase6c";

export const DEFAULT_MAP_STYLE_URL = "https://tiles.openfreemap.org/styles/positron";
export const OFFLINE_MAP_STYLE = "atlaslens://offline";
export const MAP_LABEL_FONT = "Noto Sans Regular";

const offlineStyle: StyleSpecification = {
  version: 8,
  name: "AtlasLens unavailable basemap",
  sources: {},
  layers: [{ id: "background", type: "background", paint: { "background-color": "#171a1d" } }],
};

type MapView = "candidates" | "uncertainty" | "evidence";
type MapIssue = "webgl" | "style" | "resource" | null;
const PHASE6C_LAYERS: Phase6CMapLayer[] = ["geoclip", "hierarchy", "osv", "plonk", "megaloc", "fused"];
const PHASE6C_LAYER_COLORS: Record<Phase6CMapLayer, string> = {
  geoclip: "#77bdfb",
  hierarchy: "#9b8cff",
  osv: "#f09f64",
  plonk: "#d278cf",
  megaloc: "#f1c96a",
  fused: "#5cc8b0",
};
const PHASE6C_LAYER_LABELS: Record<Phase6CMapLayer, TranslationKey> = {
  geoclip: "phase6c.layer.geoclip",
  hierarchy: "phase6c.layer.hierarchy",
  osv: "phase6c.layer.osv",
  plonk: "phase6c.layer.plonk",
  megaloc: "phase6c.layer.megaloc",
  fused: "phase6c.layer.fused",
};

interface ResultMapProps {
  candidates: Candidate[];
  phase6c?: Phase6CAnalysisSummary | null;
  modelPredictions?: Record<string, Phase6BModelPredictionSummary | null> | null;
  selectedCandidateId?: string | null;
  onSelectCandidate?: (candidateId: string) => void;
}

function applyPhase6CLayers(map: MapLibreMap, activeLayers: ReadonlySet<Phase6CMapLayer>, view: MapView) {
  for (const layer of PHASE6C_LAYERS) {
    const layerId = `phase6c-${layer}-points`;
    if (map.getLayer(layerId)) map.setLayoutProperty(layerId, "visibility", activeLayers.has(layer) ? "visible" : "none");
    const uncertaintyFill = `phase6c-${layer}-uncertainty-fill`;
    if (map.getLayer(uncertaintyFill)) {
      const visible = activeLayers.has(layer) && view === "uncertainty" ? "visible" : "none";
      map.setLayoutProperty(uncertaintyFill, "visibility", visible);
      map.setLayoutProperty(`phase6c-${layer}-uncertainty-line`, "visibility", visible);
    }
  }
}

function candidatePoints(candidates: Candidate[]) {
  return {
    type: "FeatureCollection" as const,
    features: candidates.map((candidate) => ({
      type: "Feature" as const,
      id: candidate.id,
      properties: {
        candidate_id: candidate.id,
        rank: candidate.rank,
        relative_weight: candidate.phase5b_assessment?.relative_rank_score ?? 0,
      },
      geometry: {
        type: "Point" as const,
        coordinates: [candidate.center.longitude, candidate.center.latitude] as [number, number],
      },
    })),
  };
}

function applyView(map: MapLibreMap, view: MapView, selectedCandidateId: string | null) {
  if (!map.getLayer("uncertainty-fill")) return;
  const showAllUncertainty = view === "uncertainty";
  const showSelectedUncertainty = view === "candidates" && selectedCandidateId !== null;
  const selectedFilter: FilterSpecification = [
    "==",
    ["get", "id"],
    selectedCandidateId ?? "__atlaslens-no-selection__",
  ];
  map.setFilter("uncertainty-fill", showAllUncertainty ? null : selectedFilter);
  map.setFilter("uncertainty-line", showAllUncertainty ? null : selectedFilter);
  map.setLayoutProperty(
    "uncertainty-fill",
    "visibility",
    showAllUncertainty || showSelectedUncertainty ? "visible" : "none",
  );
  map.setLayoutProperty(
    "uncertainty-line",
    "visibility",
    showAllUncertainty || showSelectedUncertainty ? "visible" : "none",
  );
  map.setLayoutProperty("candidate-heatmap", "visibility", view === "evidence" ? "visible" : "none");
}

export default function ResultMap({
  candidates,
  phase6c,
  modelPredictions,
  selectedCandidateId = null,
  onSelectCandidate,
}: ResultMapProps) {
  const { t } = useI18n();
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const layersReadyRef = useRef(false);
  const selectedRef = useRef<string | null>(selectedCandidateId);
  const previousSelectedRef = useRef<string | null>(null);
  const viewRef = useRef<MapView>("candidates");
  const phase6CLayersRef = useRef<ReadonlySet<Phase6CMapLayer>>(new Set(["fused"]));
  const onSelectRef = useRef(onSelectCandidate);
  const [mapIssue, setMapIssue] = useState<MapIssue>(null);
  const [offlineFallbackIssue, setOfflineFallbackIssue] = useState<Exclude<MapIssue, "webgl">>(null);
  const [mapReady, setMapReady] = useState(false);
  const [view, setView] = useState<MapView>("candidates");
  const [phase6CLayers, setPhase6CLayers] = useState<Set<Phase6CMapLayer>>(() => new Set(["fused"]));
  const [retryNonce, setRetryNonce] = useState(0);
  const circles = useMemo(() => uncertaintyCircles(candidates), [candidates]);
  const points = useMemo(() => candidatePoints(candidates), [candidates]);
  const phase6cPoints = useMemo(() => phase6CMapPoints(phase6c, modelPredictions), [modelPredictions, phase6c]);
  const phase6cLayerCounts = useMemo(
    () => Object.fromEntries(PHASE6C_LAYERS.map((layer) => [layer, phase6cPoints.features.filter((feature) => feature.properties.layer === layer).length])) as Record<Phase6CMapLayer, number>,
    [phase6cPoints],
  );
  const phase6cUncertainty = useMemo(() => {
    const bounded = phase6cPoints.features.filter((feature) => ["megaloc", "fused"].includes(feature.properties.layer) && feature.properties.radius_km > 0);
    const regions = uncertaintyRegions(bounded.map((feature, index) => ({
      id: `phase6c-${feature.properties.layer}-${index}`,
      rank: feature.properties.rank,
      center: { latitude: feature.geometry.coordinates[1], longitude: feature.geometry.coordinates[0] },
      radius_km: feature.properties.radius_km,
    })), 48);
    return {
      type: "FeatureCollection" as const,
      features: regions.features.map((feature, index) => ({
        ...feature,
        properties: { ...feature.properties, layer: bounded[index]?.properties.layer ?? "fused" },
      })),
    };
  }, [phase6cPoints]);
  const hasPhase6CLayers = phase6cPoints.features.length > 0;
  const mapProvider = import.meta.env.VITE_MAP_PROVIDER ?? "maplibre";
  const configuredStyle = import.meta.env.VITE_MAP_STYLE_URL;
  const styleUrl = configuredStyle === OFFLINE_MAP_STYLE
    ? OFFLINE_MAP_STYLE
    : configuredStyle?.trim() || DEFAULT_MAP_STYLE_URL;
  const activeStyleUrl = offlineFallbackIssue ? OFFLINE_MAP_STYLE : styleUrl;
  const hasHeatmap = candidates.some(
    (candidate) => (candidate.phase5b_assessment?.provider_diversity ?? 0) >= 2,
  );
  selectedRef.current = selectedCandidateId;
  viewRef.current = view;
  phase6CLayersRef.current = phase6CLayers;
  onSelectRef.current = onSelectCandidate;

  useEffect(() => {
    if (!hasPhase6CLayers || [...phase6CLayers].some((layer) => phase6cLayerCounts[layer] > 0)) return;
    const first = PHASE6C_LAYERS.find((layer) => phase6cLayerCounts[layer] > 0);
    if (first) setPhase6CLayers(new Set([first]));
  }, [hasPhase6CLayers, phase6cLayerCounts, phase6CLayers]);

  useEffect(() => {
    if (mapProvider !== "maplibre") return undefined;
    if (!containerRef.current || candidates.length === 0 || typeof WebGLRenderingContext === "undefined") {
      if (typeof WebGLRenderingContext === "undefined") setMapIssue("webgl");
      return undefined;
    }
    let map: MapLibreMap | null = null;
    let disposed = false;
    let loadTimer: ReturnType<typeof setTimeout> | undefined;
    const activateOfflineFallback = (issue: "style" | "resource") => {
      if (disposed) return;
      setMapIssue(issue);
      if (activeStyleUrl !== OFFLINE_MAP_STYLE) {
        setOfflineFallbackIssue((current) => current ?? issue);
      }
    };
    setMapReady(false);
    setMapIssue(null);
    layersReadyRef.current = false;
    try {
      const usingDefaultStyle = activeStyleUrl === DEFAULT_MAP_STYLE_URL;
      map = new maplibregl.Map({
        container: containerRef.current,
        style: activeStyleUrl === OFFLINE_MAP_STYLE ? offlineStyle : activeStyleUrl,
        center: [candidates[0]?.center.longitude ?? 0, candidates[0]?.center.latitude ?? 0],
        zoom: 3,
        attributionControl: {
          customAttribution: usingDefaultStyle
            ? [
                '<a href="https://openfreemap.org/">OpenFreeMap</a>',
                '<a href="https://www.openstreetmap.org/copyright">&copy; OpenStreetMap contributors</a>',
              ]
            : undefined,
        },
        renderWorldCopies: true,
      });
      mapRef.current = map;
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      loadTimer = setTimeout(() => {
        if (!layersReadyRef.current) activateOfflineFallback("style");
      }, 12_000);
      map.on("error", () => {
        activateOfflineFallback(layersReadyRef.current ? "resource" : "style");
      });
      map.on("load", () => {
        if (!map || disposed) return;
        if (loadTimer) clearTimeout(loadTimer);
        map.addSource("candidate-heatmap-source", { type: "geojson", data: points });
        map.addLayer({
          id: "candidate-heatmap",
          type: "heatmap",
          source: "candidate-heatmap-source",
          layout: { visibility: "none" },
          paint: {
            "heatmap-weight": ["get", "relative_weight"],
            "heatmap-intensity": 0.75,
            "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 0, 14, 8, 34],
            "heatmap-opacity": 0.55,
          },
        });
        if (phase6cPoints.features.length > 0) {
          map.addSource("phase6c-provider-points", { type: "geojson", data: phase6cPoints });
          for (const layer of PHASE6C_LAYERS) {
            map.addLayer({
              id: `phase6c-${layer}-points`,
              type: "circle",
              source: "phase6c-provider-points",
              filter: ["==", ["get", "layer"], layer],
              layout: { visibility: "none" },
              paint: {
                "circle-color": PHASE6C_LAYER_COLORS[layer],
                "circle-radius": ["interpolate", ["linear"], ["get", "rank"], 1, 10, PHASE6C_MAP_LAYER_LIMIT, 5],
                "circle-opacity": 0.82,
                "circle-stroke-color": "#0f1413",
                "circle-stroke-width": layer === "fused" ? 2.5 : 1.5,
              },
            });
          }
        }
        if (phase6cUncertainty.features.length > 0) {
          map.addSource("phase6c-provider-uncertainty", { type: "geojson", data: phase6cUncertainty });
          for (const layer of ["megaloc", "fused"] as const) {
            map.addLayer({
              id: `phase6c-${layer}-uncertainty-fill`,
              type: "fill",
              source: "phase6c-provider-uncertainty",
              filter: ["==", ["get", "layer"], layer],
              layout: { visibility: "none" },
              paint: { "fill-color": PHASE6C_LAYER_COLORS[layer], "fill-opacity": 0.05 },
            });
            map.addLayer({
              id: `phase6c-${layer}-uncertainty-line`,
              type: "line",
              source: "phase6c-provider-uncertainty",
              filter: ["==", ["get", "layer"], layer],
              layout: { visibility: "none" },
              paint: { "line-color": PHASE6C_LAYER_COLORS[layer], "line-width": 1, "line-opacity": 0.62, "line-dasharray": [2, 2] },
            });
          }
        }
        map.addSource("uncertainty", { type: "geojson", data: circles });
        map.addLayer({
          id: "uncertainty-fill",
          type: "fill",
          source: "uncertainty",
          layout: { visibility: "none" },
          paint: { "fill-color": "#5cc8b0", "fill-opacity": 0.07 },
        });
        map.addLayer({
          id: "uncertainty-line",
          type: "line",
          source: "uncertainty",
          layout: { visibility: "none" },
          paint: {
            "line-color": "#72dbc2",
            "line-width": 1.5,
            "line-opacity": 0.78,
            "line-dasharray": [3, 2],
          },
        });
        map.addSource("candidates", {
          type: "geojson",
          data: points,
          cluster: true,
          clusterMaxZoom: 8,
          clusterRadius: 48,
        });
        map.addLayer({
          id: "candidate-clusters",
          type: "circle",
          source: "candidates",
          filter: ["has", "point_count"],
          paint: {
            "circle-color": "#315d54",
            "circle-radius": ["step", ["get", "point_count"], 18, 4, 23, 10, 28],
            "circle-stroke-color": "#9ce1d0",
            "circle-stroke-width": 2,
          },
        });
        if (activeStyleUrl !== OFFLINE_MAP_STYLE) {
          map.addLayer({
            id: "candidate-cluster-count",
            type: "symbol",
            source: "candidates",
            filter: ["has", "point_count"],
            layout: {
              "text-field": "{point_count_abbreviated}",
              "text-font": [MAP_LABEL_FONT],
              "text-size": 12,
            },
            paint: { "text-color": "#f1f5f2" },
          });
        }
        map.addLayer({
          id: "candidate-points",
          type: "circle",
          source: "candidates",
          filter: ["!", ["has", "point_count"]],
          paint: {
            "circle-color": [
              "case",
              ["boolean", ["feature-state", "selected"], false],
              "#f1c96a",
              "#5cc8b0",
            ],
            "circle-radius": [
              "case",
              ["boolean", ["feature-state", "selected"], false],
              17,
              14,
            ],
            "circle-stroke-color": "#0f1413",
            "circle-stroke-width": 2,
          },
        });
        if (activeStyleUrl !== OFFLINE_MAP_STYLE) {
          map.addLayer({
            id: "candidate-ranks",
            type: "symbol",
            source: "candidates",
            filter: ["!", ["has", "point_count"]],
            layout: { "text-field": "{rank}", "text-font": [MAP_LABEL_FONT], "text-size": 11 },
            paint: { "text-color": "#0f1a18" },
          });
        }
        layersReadyRef.current = true;
        const hasBaseMap = activeStyleUrl !== OFFLINE_MAP_STYLE;
        setMapReady(hasBaseMap);
        setMapIssue(hasBaseMap ? null : offlineFallbackIssue ?? "style");
        const bounds = candidateCenterBounds(candidates);
        if (bounds) {
          if (candidates.length === 1) {
            map.jumpTo({ center: [bounds.west, bounds.south], zoom: 6 });
          } else {
            map.fitBounds(
              new LngLatBounds([bounds.west, bounds.south], [bounds.east, bounds.north]),
              { padding: 56, maxZoom: 8, duration: 0 },
            );
          }
        }
        applyView(map, viewRef.current, selectedRef.current);
        applyPhase6CLayers(map, phase6CLayersRef.current, viewRef.current);
        if (selectedRef.current) {
          map.setFeatureState(
            { source: "candidates", id: selectedRef.current },
            { selected: true },
          );
          previousSelectedRef.current = selectedRef.current;
        }
      });
      map.on("click", "candidate-clusters", (event) => {
        const feature = event.features?.[0];
        const clusterId = Number(feature?.properties?.cluster_id);
        if (!map || !Number.isFinite(clusterId) || feature?.geometry.type !== "Point") return;
        const source = map.getSource("candidates") as GeoJSONSource;
        void source.getClusterExpansionZoom(clusterId).then((zoom) => {
          if (!map || disposed || feature.geometry.type !== "Point") return;
          const coordinates = feature.geometry.coordinates as [number, number];
          const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
          map.easeTo({ center: coordinates, zoom, duration: reduced ? 0 : 250 });
        });
      });
      map.on("click", "candidate-points", (event) => {
        const candidateId: unknown = event.features?.[0]?.properties?.candidate_id;
        if (typeof candidateId === "string") onSelectRef.current?.(candidateId);
      });
      for (const layer of ["candidate-clusters", "candidate-points"] as const) {
        map.on("mouseenter", layer, () => {
          if (map) map.getCanvas().style.cursor = "pointer";
        });
        map.on("mouseleave", layer, () => {
          if (map) map.getCanvas().style.cursor = "";
        });
      }
      if (typeof ResizeObserver !== "undefined") {
        resizeObserverRef.current = new ResizeObserver(() => map?.resize());
        resizeObserverRef.current.observe(containerRef.current);
      }
    } catch {
      activateOfflineFallback("style");
    }
    return () => {
      disposed = true;
      if (loadTimer) clearTimeout(loadTimer);
      resizeObserverRef.current?.disconnect();
      resizeObserverRef.current = null;
      layersReadyRef.current = false;
      map?.remove();
      mapRef.current = null;
    };
  }, [activeStyleUrl, candidates, circles, mapProvider, offlineFallbackIssue, phase6cPoints, phase6cUncertainty, points, retryNonce]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !layersReadyRef.current) return;
    applyView(map, view, selectedCandidateId);
    applyPhase6CLayers(map, phase6CLayers, view);
    const previous = previousSelectedRef.current;
    if (previous && previous !== selectedCandidateId) {
      map.removeFeatureState({ source: "candidates", id: previous }, "selected");
    }
    if (selectedCandidateId) {
      map.setFeatureState({ source: "candidates", id: selectedCandidateId }, { selected: true });
      const selected = candidates.find((candidate) => candidate.id === selectedCandidateId);
      if (selected) {
        const longitude = longitudeNear(selected.center.longitude, map.getCenter().lng);
        map.easeTo({
          center: [longitude, selected.center.latitude],
          zoom: Math.max(map.getZoom(), 6),
          duration: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 250,
        });
      }
    }
    previousSelectedRef.current = selectedCandidateId;
  }, [candidates, phase6CLayers, selectedCandidateId, view]);

  const issueText = mapIssue === "webgl"
    ? t("result.mapWebglError")
    : mapIssue === "style"
      ? t("result.mapStyleError")
      : mapIssue === "resource"
        ? t("result.mapTileError")
        : null;

  return (
    <section className="map-section" aria-labelledby="map-heading">
      <div className="section-heading map-heading">
        <div>
          <p className="eyebrow">{t("result.mapTitle")}</p>
          <h2 id="map-heading">{t("result.mapDescription")}</h2>
        </div>
        <div className="map-view-controls" role="group" aria-label={t("result.mapViews")}>
          <button type="button" aria-pressed={view === "candidates"} onClick={() => setView("candidates")}>{t("result.mapCandidates")}</button>
          <button type="button" aria-pressed={view === "uncertainty"} onClick={() => setView("uncertainty")}>{t("result.mapUncertainty")}</button>
          {hasHeatmap ? <button type="button" aria-pressed={view === "evidence"} onClick={() => setView("evidence")}>{t("result.mapEvidence")}</button> : null}
        </div>
      </div>
      {hasPhase6CLayers ? (
        <div className="phase6c-map-layer-panel">
          <div className="phase6c-map-layer-controls" role="group" aria-label={t("phase6c.mapLayers")}>
            {PHASE6C_LAYERS.filter((layer) => phase6cLayerCounts[layer] > 0).map((layer) => (
              <button
                type="button"
                aria-pressed={phase6CLayers.has(layer)}
                onClick={() => setPhase6CLayers((current) => {
                  const next = new Set(current);
                  if (next.has(layer)) next.delete(layer);
                  else next.add(layer);
                  return next;
                })}
                key={layer}
              >
                <i aria-hidden="true" style={{ backgroundColor: PHASE6C_LAYER_COLORS[layer] }} />
                {t(PHASE6C_LAYER_LABELS[layer])} ({phase6cLayerCounts[layer]})
              </button>
            ))}
          </div>
          <p>{t("phase6c.mapLayerLimit")}</p>
        </div>
      ) : null}
      {mapProvider === "google" ? (
        <div className="map-provider-placeholder" role="status">{t("result.mapGoogleDisabled")}</div>
      ) : (
        <div className={`map-frame-wrap ${mapIssue && mapIssue !== "resource" ? "map-frame-wrap--error" : ""}`}>
          <div
            className="map-frame"
            ref={containerRef}
            data-testid="map-region"
            role="region"
            aria-label={t("result.mapTitle")}
            aria-busy={!mapReady && !mapIssue}
          />
          {!mapReady && !mapIssue ? <div className="map-overlay" role="status">{t("result.mapLoading")}</div> : null}
          {issueText ? <div className={`map-overlay map-overlay--issue ${mapIssue === "resource" ? "map-overlay--resource" : ""}`} role="status"><span>{issueText}</span><button type="button" onClick={() => { setOfflineFallbackIssue(null); setRetryNonce((value) => value + 1); }}>{t("result.mapRetry")}</button></div> : null}
        </div>
      )}
      {view === "evidence" ? <p className="map-semantics">{t("result.mapHeatmapSemantics")}</p> : null}
      <div className="text-map" aria-labelledby="text-map-heading">
        <h3 id="text-map-heading">{t("result.textMapTitle")}</h3>
        <ol>
          {candidates.map((candidate) => (
            <li key={candidate.id}>
              <button
                type="button"
                className="text-map-focus"
                aria-pressed={candidate.id === selectedCandidateId}
                onClick={() => onSelectCandidate?.(candidate.id)}
              >
                <strong>{candidate.label ?? t("result.candidate", { rank: candidate.rank })}</strong>
                <span>{displayedCoordinates(candidate)}</span>
                <span>{t("result.radius", { radius: displayedRadius(candidate.radius_km) })}</span>
              </button>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
