import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ResultMap, {
  DEFAULT_MAP_STYLE_URL,
  MAP_LABEL_FONT,
} from "../src/components/ResultMap";
import { I18nProvider } from "../src/i18n";
import { completedAnalysis } from "./fixtures";

const maplibreMock = vi.hoisted(() => {
  type Handler = (event?: unknown) => void;

  class FakeMap {
    readonly options: Record<string, unknown>;
    readonly layers: Array<Record<string, unknown>> = [];
    removed = 0;
    private readonly handlers = new globalThis.Map<string, Handler[]>();

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
    addSource() {}
    addLayer(layer: Record<string, unknown>) { this.layers.push(layer); }
    getLayer(id: string) { return this.layers.find((layer) => layer.id === id); }
    getSource() { return { getClusterExpansionZoom: () => Promise.resolve(6) }; }
    setFilter() {}
    setLayoutProperty() {}
    setFeatureState() {}
    removeFeatureState() {}
    jumpTo() {}
    fitBounds() {}
    easeTo() {}
    resize() {}
    getCenter() { return { lng: 0 }; }
    getZoom() { return 3; }
    getCanvas() { return { style: { cursor: "" } }; }
    remove() { this.removed += 1; }
  }

  const instances: FakeMap[] = [];
  return { FakeMap, instances };
});

vi.mock("maplibre-gl", () => ({
  default: {
    Map: maplibreMock.FakeMap,
    NavigationControl: class NavigationControl {},
  },
  LngLatBounds: class LngLatBounds {},
}));

describe("ResultMap fallback and selection", () => {
  beforeEach(() => {
    maplibreMock.instances.splice(0);
  });

  it("shows an in-frame WebGL fallback, retry action, and keyboard textual selection", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<I18nProvider initialLocale="en"><ResultMap candidates={completedAnalysis.candidates} onSelectCandidate={onSelect} /></I18nProvider>);

    expect(await screen.findByText(/WebGL map rendering is unavailable/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry basemap" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Textual map alternative" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Evidence" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /EXIF coordinate/ }));
    expect(onSelect).toHaveBeenCalledWith("candidate-exif");
  });

  it("uses the supported style/font stack and falls back offline only once after a resource failure", async () => {
    vi.stubEnv("VITE_MAP_PROVIDER", "maplibre");
    vi.stubEnv("VITE_MAP_STYLE_URL", "");
    vi.stubGlobal("WebGLRenderingContext", class WebGLRenderingContext {});

    render(<I18nProvider initialLocale="en"><ResultMap candidates={completedAnalysis.candidates} /></I18nProvider>);
    await waitFor(() => expect(maplibreMock.instances).toHaveLength(1));

    const onlineMap = maplibreMock.instances[0]!;
    expect(onlineMap.options.style).toBe(DEFAULT_MAP_STYLE_URL);
    act(() => onlineMap.emit("load"));

    for (const id of ["candidate-cluster-count", "candidate-ranks"]) {
      const layer = onlineMap.layers.find((candidate) => candidate.id === id);
      const layout = layer?.layout as Record<string, unknown> | undefined;
      expect(layout?.["text-font"]).toEqual([MAP_LABEL_FONT]);
    }

    act(() => onlineMap.emit("error"));
    await waitFor(() => expect(maplibreMock.instances).toHaveLength(2));
    expect(onlineMap.removed).toBe(1);

    const offlineMap = maplibreMock.instances[1]!;
    expect(offlineMap.options.style).toMatchObject({ name: "AtlasLens unavailable basemap" });
    act(() => offlineMap.emit("load"));
    expect(offlineMap.layers.some((layer) => layer.type === "symbol")).toBe(false);
    expect(await screen.findByText(/Some basemap resources could not load/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Textual map alternative" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /EXIF coordinate/ })).toBeInTheDocument();

    act(() => offlineMap.emit("error"));
    await act(async () => Promise.resolve());
    expect(maplibreMock.instances).toHaveLength(2);
  });
});
