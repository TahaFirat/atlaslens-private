import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { InvestorMapWorkspace } from "../src/components/InvestorMapWorkspace";
import {
  DEFAULT_OSM_TILE_URL,
  InvestorWorkspaceMap,
} from "../src/components/InvestorWorkspaceMap";
import { I18nProvider } from "../src/i18n";
import {
  caseAuditEvents,
  caseAuditIntegrity,
  caseEvidence,
  caseHypothesis,
  caseMedia,
  completedAnalysis,
  investigationCase,
} from "./fixtures";

const maplibreMock = vi.hoisted(() => {
  type Handler = (event?: { features?: Array<{ properties?: Record<string, unknown> }> }) => void;

  class FakeMap {
    readonly options: Record<string, unknown>;
    readonly layers: Array<Record<string, unknown>> = [];
    readonly sources: Array<{ id: string; source: Record<string, unknown> }> = [];
    private readonly handlers = new Map<string, Handler[]>();

    constructor(options: Record<string, unknown>) {
      this.options = options;
      instances.push(this);
    }

    on(event: string, layerOrHandler: string | Handler, handler?: Handler) {
      const selected = typeof layerOrHandler === "function" ? layerOrHandler : handler;
      if (selected) this.handlers.set(event, [...(this.handlers.get(event) ?? []), selected]);
      return this;
    }

    emit(event: string) {
      for (const handler of this.handlers.get(event) ?? []) handler({});
    }

    addControl() {}
    addSource(id: string, source: Record<string, unknown>) { this.sources.push({ id, source }); }
    addLayer(layer: Record<string, unknown>) { this.layers.push(layer); }
    getLayer(id: string) { return this.layers.find((layer) => layer.id === id); }
    setFilter() {}
    setLayoutProperty() {}
    fitBounds() {}
    easeTo() {}
    resize() {}
    getCanvas() { return { style: { cursor: "" } }; }
    remove() {}
  }

  class FakeMarker {
    setLngLat() { return this; }
    addTo() { return this; }
  }

  class FakeBounds {
    extend() { return this; }
  }

  const instances: FakeMap[] = [];
  return { FakeBounds, FakeMap, FakeMarker, instances };
});

vi.mock("maplibre-gl", () => ({
  default: {
    Map: maplibreMock.FakeMap,
    Marker: maplibreMock.FakeMarker,
    NavigationControl: class NavigationControl {},
    ScaleControl: class ScaleControl {},
  },
  LngLatBounds: maplibreMock.FakeBounds,
}));

const rankOne = {
  ...caseHypothesis,
  id: "123e4567-e89b-42d3-a456-426614174011",
  rank: 1,
  locality_name: "Çankaya",
  region_name: "Ankara",
  created_at: "2026-07-15T10:03:02Z",
};
const rankTwo = {
  ...caseHypothesis,
  id: "123e4567-e89b-42d3-a456-426614174012",
  rank: 2,
  locality_name: "Altındağ",
  region_name: "Ankara",
  created_at: "2026-07-15T10:03:01Z",
};

function renderWorkspace(options: {
  hypotheses?: typeof caseHypothesis[];
  analysisLoading?: boolean;
  analysisFailed?: boolean;
  storageState?: typeof caseMedia.storage_state;
  retrievalContext?: Record<string, unknown>;
  caseRetrievalContext?: Record<string, unknown>;
} = {}) {
  return render(
    <I18nProvider initialLocale="en">
      <InvestorMapWorkspace
        caseRecord={investigationCase}
        media={[{ ...caseMedia, storage_state: options.storageState ?? "deleted_after_analysis" }]}
        selectedMedia={{ ...caseMedia, storage_state: options.storageState ?? "deleted_after_analysis" }}
        hypotheses={options.hypotheses ?? [rankOne]}
        evidence={options.caseRetrievalContext ? caseEvidence.map((item, index) => index === 0 ? { ...item, structured_payload: options.caseRetrievalContext! } : item) : caseEvidence}
        audit={caseAuditEvents}
        integrity={caseAuditIntegrity}
        analysis={{ ...completedAnalysis, retrieval_context: options.retrievalContext } as typeof completedAnalysis}
        analysisLoading={options.analysisLoading ?? false}
        analysisFailed={options.analysisFailed ?? false}
      />
    </I18nProvider>,
  );
}

describe("Phase 3E investor map-first workspace", () => {
  beforeEach(() => {
    maplibreMock.instances.splice(0);
    vi.stubGlobal("WebGLRenderingContext", undefined);
  });

  it("uses explicit backend rank for top-1 when API creation order differs", () => {
    renderWorkspace({ hypotheses: [rankTwo, rankOne] });

    expect(screen.getByRole("heading", { name: "Çankaya, Ankara" })).toBeVisible();
    const alternatives = screen.getByRole("heading", { name: "Alternative candidates" }).closest("section");
    expect(alternatives).not.toBeNull();
    const cards = within(alternatives as HTMLElement).getAllByRole("button");
    expect(cards[0]).toHaveTextContent("1Çankaya, Ankara");
    expect(cards[1]).toHaveTextContent("2Altındağ, Ankara");
    expect(screen.getByText(/Uncalibrated relative similarity; not confidence or probability/)).toBeVisible();
    expect(screen.queryByText(/probability/i, { selector: "strong" })).not.toBeInTheDocument();
  });

  it("shows loading, honest coverage abstention, and provider-unavailable states", () => {
    const loading = renderWorkspace({ hypotheses: [], analysisLoading: true });
    expect(screen.getByRole("status", { name: "" })).toHaveTextContent("Loading analysis");
    loading.unmount();

    const abstained = renderWorkspace({ hypotheses: [] });
    expect(screen.getByRole("heading", { name: "No location asserted" })).toBeVisible();
    expect(screen.getByText(/abstained instead of producing a false city/)).toBeVisible();
    abstained.unmount();

    renderWorkspace({ hypotheses: [], analysisFailed: true });
    expect(screen.getByRole("heading", { name: "Provider unavailable" })).toBeVisible();
    expect(screen.getByText(/did not invent a fallback/)).toBeVisible();
  });

  it("lets explicit out-of-coverage context suppress stale hypotheses", () => {
    renderWorkspace({
      hypotheses: [rankOne],
      retrievalContext: {
        analysis_scope: "generic_upload",
        coverage_status: "out_of_coverage",
        coverage_label: "Generic upload · unsupported coverage",
        result_semantics: "coverage_insufficient",
        abstained: true,
        abstention_reason: "reference_coverage_insufficient",
        similarity_semantics: "uncalibrated_relative_similarity",
      },
    });

    expect(screen.getByRole("heading", { name: "No location asserted" })).toBeVisible();
    expect(screen.getByText("Generic upload · unsupported coverage")).toBeVisible();
    expect(screen.getByText(/Reference coverage insufficient/)).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Çankaya, Ankara" })).not.toBeInTheDocument();
    expect(screen.getByText("No location candidate to map.")).toBeVisible();
  });

  it("uses materialized case-evidence coverage semantics when the live analysis omits them", () => {
    renderWorkspace({
      hypotheses: [rankOne],
      caseRetrievalContext: {
        analysis_scope: "ankara_reference_pilot",
        coverage_status: "pilot_eligible",
        coverage_label: "Contract coverage label",
        retrieval_provider: "megaloc_mapillary_faiss",
        retrieval_scope: "ankara_reference_collection",
        result_semantics: "ankara_reference_collection_visual_similarity_not_general_geolocation",
        abstained: false,
        abstention_reason: null,
        similarity_semantics: "cosine_similarity_not_confidence",
        supported_region: "Ankara pilot collection only",
        evidence_version: "fixture-evidence-v1",
        benchmark_version: "fixture-benchmark-v1",
      },
    });

    expect(screen.getByText("Contract coverage label")).toBeVisible();
    expect(screen.getByText("megaloc_mapillary_faiss")).toBeVisible();
    expect(screen.getByText("Visual similarity search within the Ankara reference collection.")).toBeVisible();
    expect(screen.getByText("Uncalibrated relative similarity; not confidence or probability")).toBeVisible();
  });

  it("labels a location-free pilot result as an Ankara reference rather than an unnamed claim", () => {
    renderWorkspace({
      hypotheses: [{ ...rankOne, locality_name: null, region_name: null, country_code: null }],
      caseRetrievalContext: {
        analysis_scope: "ankara_reference_pilot",
        coverage_status: "pilot_eligible",
        coverage_label: "Ankara reference pilot",
        retrieval_provider: "megaloc_mapillary_faiss",
        retrieval_scope: "ankara_reference_collection",
        result_semantics: "ankara_reference_collection_visual_similarity_not_general_geolocation",
        abstained: false,
        abstention_reason: null,
        similarity_semantics: "cosine_similarity_not_confidence",
        supported_region: "Ankara pilot collection only",
        evidence_version: "fixture-evidence-v1",
        benchmark_version: "fixture-benchmark-v1",
      },
    });

    expect(screen.getByRole("heading", { name: "Ankara pilot reference #1" })).toBeVisible();
    expect(screen.queryByText("Unnamed location hypothesis")).not.toBeInTheDocument();
  });

  it("keeps a deleted query private and exposes technical material only in the compact drawer", async () => {
    const user = userEvent.setup();
    renderWorkspace();

    const preview = screen.getByTestId("investor-query-preview");
    expect(within(preview).getByText("Image removed after analysis")).toBeVisible();
    expect(within(preview).getByText(/Raw pixels are not displayed/)).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Technical details" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Evidence and technical details/ }));
    expect(screen.getByRole("heading", { name: "Technical details" })).toBeVisible();
    expect(screen.queryByText("AtlasLens did not invent a fallback.")).not.toBeInTheDocument();
  });

  it("handles one and five overlapping candidates without changing their ranks", () => {
    const one = renderWorkspace({ hypotheses: [rankOne] });
    expect(screen.queryByRole("heading", { name: "Alternative candidates" })).not.toBeInTheDocument();
    one.unmount();

    const overlapping = Array.from({ length: 5 }, (_, index) => ({
      ...rankOne,
      id: `123e4567-e89b-42d3-a456-42661417402${index}`,
      rank: index + 1,
      locality_name: `Candidate ${index + 1}`,
      latitude: 39.92 + index * 0.001,
      longitude: 32.85 + index * 0.001,
      uncertainty_radius_m: 25_000 + index * 1_000,
    }));
    renderWorkspace({ hypotheses: overlapping });
    const alternatives = screen.getByRole("heading", { name: "Alternative candidates" }).closest("section");
    expect(within(alternatives as HTMLElement).getAllByRole("button")).toHaveLength(5);
    expect(screen.getByRole("heading", { name: "Candidate 1, Ankara" })).toBeVisible();
  });
});

describe("Phase 3E investor basemap", () => {
  beforeEach(() => {
    maplibreMock.instances.splice(0);
  });

  it("uses the centralized configurable OSM raster style with visible attribution", async () => {
    vi.stubGlobal("WebGLRenderingContext", class WebGLRenderingContext {});
    render(<I18nProvider initialLocale="en"><InvestorWorkspaceMap hypotheses={[rankOne]} selectedId={rankOne.id} onSelect={vi.fn()} /></I18nProvider>);
    await waitFor(() => expect(maplibreMock.instances).toHaveLength(1));
    const style = maplibreMock.instances[0]!.options.style as { sources: Record<string, { tiles: string[] }> };
    expect(style.sources["osm-raster"]?.tiles).toEqual([DEFAULT_OSM_TILE_URL]);
    expect(maplibreMock.instances[0]!.options.prefetchZoomDelta).toBe(0);
    expect(screen.getByRole("link", { name: "© OpenStreetMap contributors" })).toBeVisible();
    act(() => maplibreMock.instances[0]!.emit("load"));
    expect(screen.queryByTestId("investor-map-offline")).not.toBeInTheDocument();
  });

  it("falls back on resource failure, retains candidate geometry, and offers retry", async () => {
    vi.stubGlobal("WebGLRenderingContext", class WebGLRenderingContext {});
    render(<I18nProvider initialLocale="en"><InvestorWorkspaceMap hypotheses={[rankOne, rankTwo]} selectedId={rankOne.id} onSelect={vi.fn()} /></I18nProvider>);
    await waitFor(() => expect(maplibreMock.instances).toHaveLength(1));
    act(() => maplibreMock.instances[0]!.emit("error"));
    await waitFor(() => expect(maplibreMock.instances).toHaveLength(2));
    act(() => maplibreMock.instances[1]!.emit("load"));

    expect(await screen.findByText("Basemap offline")).toBeVisible();
    expect(screen.getByRole("button", { name: "Retry basemap" })).toBeVisible();
    expect(maplibreMock.instances[1]!.sources.map((source) => source.id)).toEqual(expect.arrayContaining(["investor-uncertainty", "investor-candidates"]));
    expect(screen.getByRole("link", { name: "© OpenStreetMap contributors" })).toBeVisible();
  });
});
