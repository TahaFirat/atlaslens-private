import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Phase4AssessmentPanel } from "../src/components/Phase4AssessmentPanel";
import type { Phase4AssessmentView } from "../src/phase4-view";

const assessment: Phase4AssessmentView = {
  classification: "geometry_supported",
  relative_rank_score: 0.81234,
  score_semantics: "uncalibrated_relative_rank",
  reranker_version: "phase4-v1",
  score_breakdown: [
    { feature: "geometry_inliers", raw_value: 0.78, weight: 0.2, contribution: 0.11, reason_code: "geometry_support" },
  ],
  source_diversity: 2,
  contributing_retrieval_hit_ids: ["hit-1", "hit-2"],
  map_observations: [{ clue: "road", map_feature: "highway", status: "neutral", reliability: 0, query_radius_km: 2, provider: "fixture", limitation: "No comparable scene clue" }],
  geometry_results: [{ reference_id: "hit-1", provider: "opencv_orb_homography", status: "supported", query_keypoints: 80, reference_keypoints: 77, raw_matches: 40, filtered_matches: 23, inliers: 18, inlier_ratio: 0.78, query_coverage: 0.31, reference_coverage: 0.28, residual_error_px: 0.8, robust_model_type: "homography", limitations: ["not geographic proof"], runtime_ms: 20 }],
  contradictions: ["Text clue conflicts with region"],
  reference_attributions: [{ reference_id: "hit-1", source: "Wikimedia Commons", license: "CC BY-SA 4.0", attribution: "Example Author", display_allowed: false }],
  limitations: ["Geometry is visual consistency, not geographic proof"],
};

describe("Phase4AssessmentPanel", () => {
  it("renders transparent diagnostics without probability or percentage wording", async () => {
    const user = userEvent.setup();
    render(<Phase4AssessmentPanel assessment={assessment} />);
    expect(screen.getByText("0.812")).toBeInTheDocument();
    expect(screen.getByText(/not a probability/i)).toBeInTheDocument();
    expect(screen.queryByText(/81%/)).not.toBeInTheDocument();
    expect(screen.getByText(/Reference display denied by policy/)).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    await user.click(screen.getByText("Score breakdown"));
    expect(screen.getByText("geometry inliers")).toBeVisible();
  });

  it("renders Turkish safety wording", () => {
    render(<Phase4AssessmentPanel assessment={assessment} locale="tr" />);
    expect(screen.getByRole("heading", { name: "Aday değerlendirmesi" })).toBeInTheDocument();
    expect(screen.getByText(/olasılık veya konum garantisi değildir/)).toBeInTheDocument();
  });

  it("renders explicit abstention without inventing evidence", () => {
    const abstained = { ...assessment, classification: "abstained" as const, geometry_results: [], limitations: ["Insufficient independent support"] };
    render(<Phase4AssessmentPanel assessment={abstained} />);
    expect(screen.getByText("Insufficient independent support")).toBeInTheDocument();
    expect(screen.getByText("abstained")).toBeInTheDocument();
  });

  it("keeps distinct diagnostic codes console-safe when translations match", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const duplicateTranslation = {
      ...assessment,
      limitations: [
        "model_prediction_is_unverified",
        "phase5.model_prediction_is_unverified",
      ],
    };

    render(<Phase4AssessmentPanel assessment={duplicateTranslation} />);

    expect(screen.getAllByText("The model hypothesis is unverified.")).toHaveLength(2);
    expect(consoleError).not.toHaveBeenCalled();
    consoleError.mockRestore();
  });
});
