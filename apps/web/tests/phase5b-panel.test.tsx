import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { Phase5BAssessment, Phase5BDiagnostics } from "../src/api/schemas";
import { Phase5BAssessmentPanel, Phase5BDiagnosticsPanel } from "../src/components/Phase5BAssessmentPanel";

const assessment: Phase5BAssessment = {
  classification: "multi_source_supported",
  relative_rank_score: 0.7234,
  score_semantics: "uncalibrated_relative_rank",
  reranker_version: "phase5b-v1",
  score_breakdown: [
    { feature: "place_support", raw_value: 0.8, weight: 0.2, contribution: 0.16, reason_code: "phase5b.feature.place_support" },
  ],
  provider_diversity: 2,
  source_diversity: 2,
  place_matches: [{
    matched_entity: "PRIVATE_RAW_INPUT_MUST_NOT_RENDER",
    normalized_name: "Istanbul",
    country_code: "TR",
    region: "Marmara",
    center: { latitude: 41.0082, longitude: 28.9784 },
    match_type: "city",
    text_similarity: 0.8,
    ambiguity_count: 1,
    evidence_strength: 0.75,
    source: "GeoNames",
    dataset_version: "fixture-v1",
    license: "CC BY 4.0",
  }],
  retrieval_matches: [{
    reference_id: "reference-1",
    provider: "faiss",
    source: "Wikimedia Commons",
    distance: 0.2,
    relative_similarity: 0.8,
    center: { latitude: 41.01, longitude: 28.98 },
    geographic_cluster: "cluster-1",
    license: "CC BY-SA 4.0",
    attribution: "Fixture Author",
    display_allowed: false,
  }],
  map_observations: [],
  supports: ["phase5b.support.place", "phase5b.support.retrieval"],
  contradictions: [],
  movement_reasons: ["phase5b.rank_increased.evidence_support"],
  limitations: ["phase5b.relative_rank_is_not_probability"],
};

const diagnostics: Phase5BDiagnostics = {
  reranker_version: "phase5b-v1",
  providers: [{ provider_id: "licensed-index", provider_type: "retrieval", status: "succeeded", duration_ms: 12, offline: true }],
  reference_index: { status: "ready", index_id: "fixture", embedding_provider: "siglip2", embedding_version: "fixture-v1", dimension: 768, image_count: 12 },
  partial_failures: [],
};

describe("Phase5B evidence panels", () => {
  it("shows raw relative evidence without percentage claims or sensitive matched input", async () => {
    const user = userEvent.setup();
    render(<Phase5BAssessmentPanel assessment={assessment} locale="en" headingId="phase5b-heading" />);

    expect(screen.getByText("0.723")).toBeInTheDocument();
    expect(screen.getByText(/not a probability or a guarantee/i)).toBeInTheDocument();
    expect(screen.queryByText(/72%/)).not.toBeInTheDocument();
    expect(screen.getByText("Istanbul")).toBeInTheDocument();
    expect(screen.queryByText("PRIVATE_RAW_INPUT_MUST_NOT_RENDER")).not.toBeInTheDocument();
    expect(screen.getByText(/Reference display denied/)).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    await user.click(screen.getByText("Weighted evidence"));
    expect(screen.getAllByText("Place-name support").length).toBeGreaterThan(0);
  });

  it("renders Turkish safety language and bounded diagnostics", () => {
    render(<><Phase5BAssessmentPanel assessment={assessment} locale="tr" headingId="phase5b-heading-tr" /><Phase5BDiagnosticsPanel diagnostics={diagnostics} locale="tr" /></>);

    expect(screen.getByRole("heading", { name: "Birleşik kanıt değerlendirmesi" })).toBeInTheDocument();
    expect(screen.getByText(/olasılık veya konum garantisi değildir/)).toBeInTheDocument();
    expect(screen.getByText(/12 görüntü/)).toBeInTheDocument();
    expect(screen.getByText(/çevrimdışı/)).toBeInTheDocument();
  });
});
