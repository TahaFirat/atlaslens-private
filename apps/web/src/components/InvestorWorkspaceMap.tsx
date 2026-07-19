import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl, {
  LngLatBounds,
  type Map as MapLibreMap,
  type MapOptions,
  type StyleSpecification,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { LocationHypothesis } from "../api/schemas";
import { longitudeNear, uncertaintyRegions } from "../map/geo";
import { useI18n } from "../i18n";

export const INVESTOR_OFFLINE_MAP = "atlaslens://offline";
export const DEFAULT_OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

const offlineStyle: StyleSpecification = {
  version: 8,
  name: "AtlasLens offline geographic context",
  sources: {},
  layers: [{ id: "background", type: "background", paint: { "background-color": "#e9ece8" } }],
};

function rasterStyle(tileUrl: string): StyleSpecification {
  return {
    version: 8,
    name: "AtlasLens OpenStreetMap raster",
    sources: {
      "osm-raster": {
        type: "raster",
        tiles: [tileUrl],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors",
      },
    },
    layers: [{ id: "osm-raster", type: "raster", source: "osm-raster" }],
  };
}

function pointCollection(hypotheses: LocationHypothesis[]) {
  return {
    type: "FeatureCollection" as const,
    features: hypotheses.map((item, index) => ({
      type: "Feature" as const,
      id: item.id,
      properties: { id: item.id, rank: item.rank ?? index + 1 },
      geometry: {
        type: "Point" as const,
        coordinates: [item.longitude, item.latitude] as [number, number],
      },
    })),
  };
}

function hypothesisBounds(hypotheses: LocationHypothesis[]): LngLatBounds | null {
  const first = hypotheses[0];
  if (!first) return null;
  const bounds = new LngLatBounds();
  const anchor = first.longitude;
  hypotheses.forEach((item) => {
    const longitude = longitudeNear(item.longitude, anchor);
    const radiusKm = Math.max(item.uncertainty_radius_m / 1000, 0.001);
    const latitudeDelta = Math.min(radiusKm / 111.32, 85);
    const longitudeScale = Math.max(Math.cos(item.latitude * Math.PI / 180), 0.08);
    const longitudeDelta = Math.min(radiusKm / (111.32 * longitudeScale), 179);
    bounds.extend([longitude - longitudeDelta, Math.max(-85, item.latitude - latitudeDelta)]);
    bounds.extend([longitude + longitudeDelta, Math.min(85, item.latitude + latitudeDelta)]);
  });
  return bounds;
}

function mapMotionDuration(): number {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? 0 : 180;
}

interface InvestorWorkspaceMapProps {
  hypotheses: LocationHypothesis[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function InvestorWorkspaceMap({ hypotheses, selectedId, onSelect }: InvestorWorkspaceMapProps) {
  const { locale } = useI18n();
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markerElementsRef = useRef(new Map<string, HTMLButtonElement>());
  const onSelectRef = useRef(onSelect);
  const selectedRef = useRef(selectedId);
  const [showUncertainty, setShowUncertainty] = useState(true);
  const showUncertaintyRef = useRef(showUncertainty);
  const [offline, setOffline] = useState(false);
  const [retryNonce, setRetryNonce] = useState(0);
  const points = useMemo(() => pointCollection(hypotheses), [hypotheses]);
  const uncertainty = useMemo(() => uncertaintyRegions(hypotheses.map((item, index) => ({
    id: item.id,
    rank: item.rank ?? index + 1,
    center: { latitude: item.latitude, longitude: item.longitude },
    radius_km: item.uncertainty_radius_m / 1000,
  })), 72), [hypotheses]);
  onSelectRef.current = onSelect;
  selectedRef.current = selectedId;
  showUncertaintyRef.current = showUncertainty;

  const fit = useCallback((animated = true) => {
    const map = mapRef.current;
    const bounds = hypothesisBounds(hypotheses);
    if (!map || !bounds) return;
    if (hypotheses.length === 1) {
      const only = hypotheses[0]!;
      map.easeTo({
        center: [only.longitude, only.latitude],
        zoom: only.uncertainty_radius_m > 100_000 ? 5 : only.uncertainty_radius_m > 25_000 ? 7 : 9,
        duration: animated ? mapMotionDuration() : 0,
      });
      return;
    }
    map.fitBounds(bounds, { padding: 72, maxZoom: 9, duration: animated ? mapMotionDuration() : 0 });
  }, [hypotheses]);

  useEffect(() => {
    if (!containerRef.current || hypotheses.length === 0 || typeof WebGLRenderingContext === "undefined") {
      setOffline(hypotheses.length > 0);
      return undefined;
    }
    let disposed = false;
    let resizeObserver: ResizeObserver | undefined;
    const markerElements = markerElementsRef.current;
    const configuredStyle = import.meta.env.VITE_MAP_STYLE_URL?.trim();
    const tileUrl = import.meta.env.VITE_MAP_TILE_URL?.trim() || DEFAULT_OSM_TILE_URL;
    const forceOffline = configuredStyle === INVESTOR_OFFLINE_MAP;

    const buildMap = (useOfflineStyle: boolean) => {
      const first = hypotheses[0];
      if (!first || !containerRef.current || disposed) return;
      const mapOptions = {
        container: containerRef.current,
        style: useOfflineStyle ? offlineStyle : rasterStyle(tileUrl),
        center: [first.longitude, first.latitude],
        zoom: 6,
        prefetchZoomDelta: 0,
        attributionControl: false,
      } as MapOptions;
      const map = new maplibregl.Map(mapOptions);
      mapRef.current = map;
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      map.addControl(new maplibregl.ScaleControl({ unit: "metric", maxWidth: 120 }), "bottom-left");
      let switchedToFallback = useOfflineStyle;
      map.on("error", () => {
        if (disposed || switchedToFallback) return;
        switchedToFallback = true;
        map.remove();
        buildMap(true);
      });
      map.on("load", () => {
        if (disposed) return;
        setOffline(useOfflineStyle);
        map.addSource("investor-uncertainty", { type: "geojson", data: uncertainty });
        map.addLayer({
          id: "investor-uncertainty-fill",
          type: "fill",
          source: "investor-uncertainty",
          layout: { visibility: showUncertaintyRef.current ? "visible" : "none" },
          paint: { "fill-color": "#176b5b", "fill-opacity": 0.13 },
        });
        map.addLayer({
          id: "investor-uncertainty-line",
          type: "line",
          source: "investor-uncertainty",
          layout: { visibility: showUncertaintyRef.current ? "visible" : "none" },
          paint: { "line-color": "#176b5b", "line-width": 1.5, "line-dasharray": [2, 2] },
        });
        map.addSource("investor-candidates", { type: "geojson", data: points });
        map.addLayer({
          id: "investor-candidate-markers",
          type: "circle",
          source: "investor-candidates",
          paint: {
            "circle-color": ["case", ["==", ["get", "rank"], 1], "#0f6b5d", "#ffffff"],
            "circle-radius": ["case", ["==", ["get", "rank"], 1], 18, 15],
            "circle-stroke-color": "#10221f",
            "circle-stroke-width": 2,
          },
        });
        map.addLayer({
          id: "investor-candidate-selected",
          type: "circle",
          source: "investor-candidates",
          filter: ["==", ["get", "id"], selectedRef.current ?? "__none__"],
          paint: {
            "circle-color": "transparent",
            "circle-radius": 21,
            "circle-stroke-color": "#f4a62a",
            "circle-stroke-width": 3,
          },
        });
        map.on("click", "investor-candidate-markers", (event) => {
          const id: unknown = event.features?.[0]?.properties?.id;
          if (typeof id === "string") onSelectRef.current(id);
        });
        map.on("mouseenter", "investor-candidate-markers", () => { map.getCanvas().style.cursor = "pointer"; });
        map.on("mouseleave", "investor-candidate-markers", () => { map.getCanvas().style.cursor = ""; });
        markerElements.clear();
        hypotheses.forEach((item, index) => {
          const rank = item.rank ?? index + 1;
          const marker = document.createElement("button");
          marker.type = "button";
          marker.className = `investor-numbered-marker${rank === 1 ? " investor-numbered-marker--primary" : ""}`;
          marker.textContent = String(rank);
          if (item.id === selectedRef.current) marker.classList.add("investor-numbered-marker--selected");
          marker.style.zIndex = item.id === selectedRef.current ? "4" : rank === 1 ? "3" : "1";
          marker.dataset.candidateRank = String(rank);
          marker.addEventListener("click", (event) => {
            event.stopPropagation();
            onSelectRef.current(item.id);
          });
          markerElements.set(item.id, marker);
          new maplibregl.Marker({ element: marker, anchor: "center" })
            .setLngLat([item.longitude, item.latitude])
            .addTo(map);
          marker.setAttribute("aria-label", `${locale === "tr" ? "Aday" : "Candidate"} ${rank}`);
        });
        fit(false);
        if (typeof ResizeObserver !== "undefined" && containerRef.current) {
          resizeObserver = new ResizeObserver(() => map.resize());
          resizeObserver.observe(containerRef.current);
        }
      });
    };

    try {
      buildMap(forceOffline);
    } catch {
      setOffline(true);
    }
    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      mapRef.current?.remove();
      mapRef.current = null;
      markerElements.clear();
    };
  }, [fit, hypotheses, locale, points, retryNonce, uncertainty]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getLayer("investor-candidate-selected")) return;
    map.setFilter("investor-candidate-selected", ["==", ["get", "id"], selectedId ?? "__none__"]);
    markerElementsRef.current.forEach((element, id) => {
      const selectedMarker = id === selectedId;
      element.classList.toggle("investor-numbered-marker--selected", selectedMarker);
      element.style.zIndex = selectedMarker ? "4" : element.classList.contains("investor-numbered-marker--primary") ? "3" : "1";
    });
    const selected = hypotheses.find((item) => item.id === selectedId);
    if (selected) map.easeTo({ center: [selected.longitude, selected.latitude], duration: mapMotionDuration() });
  }, [hypotheses, selectedId]);

  useEffect(() => {
    const map = mapRef.current;
    for (const layer of ["investor-uncertainty-fill", "investor-uncertainty-line"]) {
      if (map?.getLayer(layer)) map.setLayoutProperty(layer, "visibility", showUncertainty ? "visible" : "none");
    }
  }, [showUncertainty]);

  const labels = locale === "tr" ? {
    map: "Aday ve belirsizlik haritası",
    fit: "Sıfırla / adaylara sığdır",
    uncertainty: "Belirsizlik katmanı",
    offline: "Alt harita çevrimdışı",
    offlineHelp: "Adaylar ve belirsizlik geometrisi yerel olarak gösterilmeye devam eder.",
    retry: "Alt haritayı yeniden dene",
    empty: "Gösterilecek konum adayı yok.",
  } : {
    map: "Candidate and uncertainty map",
    fit: "Reset / fit candidates",
    uncertainty: "Uncertainty layer",
    offline: "Basemap offline",
    offlineHelp: "Candidates and uncertainty geometry remain available locally.",
    retry: "Retry basemap",
    empty: "No location candidate to map.",
  };

  return (
    <section className="investor-map-surface" aria-label={labels.map}>
      <div className="investor-map-toolbar" aria-label={locale === "tr" ? "Harita kontrolleri" : "Map controls"}>
        <button type="button" onClick={() => fit()} disabled={hypotheses.length === 0}>{labels.fit}</button>
        <button type="button" aria-pressed={showUncertainty} onClick={() => setShowUncertainty((current) => !current)}>{labels.uncertainty}</button>
      </div>
      <div ref={containerRef} className={`investor-map${offline ? " investor-map--offline" : ""}`} data-testid="investor-map" aria-label={labels.map}>
        <div className="investor-map__fallback-geometry" aria-hidden="true"><i /><i /><i /></div>
        {hypotheses.length === 0 ? <p className="investor-map__empty">{labels.empty}</p> : null}
      </div>
      {offline ? (
        <div className="investor-map-offline" role="status" data-testid="investor-map-offline">
          <span><strong>{labels.offline}</strong><small>{labels.offlineHelp}</small></span>
          <button type="button" onClick={() => setRetryNonce((value) => value + 1)}>{labels.retry}</button>
        </div>
      ) : null}
      <a className="investor-map-attribution" href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">© OpenStreetMap contributors</a>
    </section>
  );
}
