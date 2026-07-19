import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { SceneSegmentationSummary } from "../src/api/schemas";
import { SceneAnalysisCard } from "../src/components/SceneAnalysisCard";
import { I18nProvider } from "../src/i18n";

const scene: SceneSegmentationSummary = {
  status: "completed",
  provider: "atlaslens-segformer-b2-v4",
  device: "cpu",
  inference_ms: 18.4,
  image_width: 640,
  image_height: 480,
  semantic_label_names_available: true,
  dominant_classes: [
    { class_id: 13, class_name: "Road", pixel_ratio: 0.341, percentage: 34.1 },
    { class_id: 2, class_name: "Sky", pixel_ratio: 0.173, percentage: 17.3 },
  ],
  scene_groups: [{ name: "road_surface", pixel_ratio: 0.341, percentage: 34.1 }],
  scene_tags: [{
    name: "road_heavy",
    strength: 0.74,
    strength_semantics: "deterministic_heuristic_not_probability",
    reason: "Road pixels exceed the reviewed descriptive threshold",
  }],
  warnings: [],
};

describe("SceneAnalysisCard", () => {
  it("renders pixel coverage and deterministic scene evidence without geographic claims", () => {
    render(<I18nProvider initialLocale="en"><SceneAnalysisCard scene={scene} /></I18nProvider>);

    expect(screen.getByRole("heading", { name: "Visible scene components" })).toBeInTheDocument();
    expect(screen.getByText("Road")).toBeInTheDocument();
    expect(screen.getAllByText("34.1%")).toHaveLength(2);
    expect(screen.getByText("Road surface")).toBeInTheDocument();
    expect(screen.getByText(/Heuristic strength: 0.74.*not a probability/)).toBeInTheDocument();
    expect(screen.queryByText("74%")).not.toBeInTheDocument();
    expect(screen.getByText(/does not predict a country, city, climate, or architectural origin/i)).toBeInTheDocument();
  });

  it("uses generic class IDs and withholds semantic groups and tags when exact labels are unavailable", () => {
    render(<I18nProvider initialLocale="en"><SceneAnalysisCard scene={{ ...scene, semantic_label_names_available: false }} /></I18nProvider>);

    expect(screen.getByText("Class 13")).toBeInTheDocument();
    expect(screen.queryByText("Road", { exact: true })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Coarse scene groups" })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Descriptive scene tags" })).not.toBeInTheDocument();
    expect(screen.getByText(/Generic class IDs are shown/)).toBeInTheDocument();
  });
});
