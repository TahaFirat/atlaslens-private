import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Analysis } from "../src/api/schemas";
import { Phase6BAnalysisCard } from "../src/components/Phase6BAnalysisCard";
import { NoSignal } from "../src/components/NoSignal";
import { I18nProvider } from "../src/i18n";
import { completedAnalysis } from "./fixtures";

const phase6bAnalysis: Analysis = {
  ...completedAnalysis,
  model_predictions: {
    geoclip: {
      provider: "geoclip",
      model_id: "GeoCLIP-1.2.0",
      model_revision: "reviewed-geoclip-revision",
      source_family: "mp16_family",
      status: "completed",
      device: "cuda",
      duration_ms: 112,
      score_semantics: "similarity",
      candidates: [{ candidate_id: "geoclip-1", latitude: 41, longitude: 29, raw_score: 0.8, provider_rank: 1, sample_support: 3 }],
      warnings: [],
    },
    osv5m: {
      provider: "osv5m",
      model_id: "osv5m/baseline",
      model_revision: "reviewed-osv-revision",
      source_family: "osv5m_family",
      status: "completed",
      device: "cuda",
      duration_ms: 184,
      score_semantics: "direct_regression",
      candidates: [{ candidate_id: "osv5m-1", latitude: 41.0082, longitude: 28.9784, raw_score: null, provider_rank: 1, sample_support: 1 }],
      warnings: [],
    },
    plonk: {
      provider: "plonk",
      model_id: "nicolas-dufour/PLONK_YFCC",
      model_revision: "reviewed-plonk-revision",
      source_family: "yfcc_family",
      status: "completed",
      device: "cuda",
      duration_ms: 220,
      score_semantics: "sample_density",
      candidates: [{ candidate_id: "plonk-1", latitude: 41.1, longitude: 29.1, raw_score: null, provider_rank: 1, sample_support: 12 }],
      warnings: [],
    },
  },
  fusion: {
    version: "phase6b-v1",
    source_families: ["mp16_family", "osv5m_family", "yfcc_family"],
    agreement_summary: { provider_count: 3, independent_family_count: 3, same_family_duplicate_support: 1, geographic_disagreement: false, ocr_agreement: true, ocr_contradiction: false },
    candidate_clusters: [{
      cluster_id: "phase6b-cluster-1",
      latitude: 41.03,
      longitude: 29.02,
      radius_km: 90,
      provider_count: 3,
      independent_family_count: 3,
      same_family_duplicate_support: 1,
      spread_km: 42,
      members: [
        { provider: "geoclip", model_id: "GeoCLIP-1.2.0", source_family: "mp16_family", provider_rank: 1 },
        { provider: "osv5m", model_id: "osv5m/baseline", source_family: "osv5m_family", provider_rank: 1 },
        { provider: "plonk", model_id: "nicolas-dufour/PLONK_YFCC", source_family: "yfcc_family", provider_rank: 1 },
      ],
      contributions: [{ name: "independent_model_agreement", contribution: 0.14, reason: "GeoCLIP and PLONK YFCC support locations within 42 km" }],
      ocr_agreement: true,
      ocr_contradiction: false,
    }],
  },
  ocr: {
    provider: "paddleocr",
    status: "completed",
    fallback_used: false,
    detections: [{ redacted_text: "İstanbul Caddesi", normalized_text: "istanbul caddesi", confidence: 0.83, confidence_semantics: "uncalibrated_ocr_engine_score", script: "latin", provider: "paddleocr", profile: "standard", sensitive_content: false }],
    place_evidence: [{ normalized_name: "İstanbul", country_code: "TR", match_type: "city", evidence_strength: 0.75, source: "GeoNames" }],
  },
  cloud_assist: {
    allowed: true,
    triggered: true,
    provider: "openai",
    model: "gpt-5.6-luna",
    prompt_version: "openai-geo-review-v1",
    reason_code: "strong_model_disagreement",
    status: "completed",
    cache_hit: false,
    estimated_cost_usd: 0.0012,
    cost_basis: "usage_based",
    budget: { calls_today: 1, estimated_month_spend_usd: 0.3, configured_monthly_budget_usd: 4.5, remaining_budget_usd: 4.2 },
    review: {
      decision: "support_candidate",
      selected_candidate_ids: ["candidate-exif"],
      observed_clues: [{ type: "road", observation: "The visible road layout is consistent with the supplied candidate summary.", supports_candidate_ids: ["candidate-exif"], contradicts_candidate_ids: [] }],
      uncertainty_reason: "The clue is compatible but not uniquely identifying.",
    },
  },
};

describe("Phase 6B frontend evidence", () => {
  it("keeps old Phase 6A analyses backward compatible", () => {
    const { container } = render(<I18nProvider initialLocale="en"><Phase6BAnalysisCard analysis={completedAnalysis} /></I18nProvider>);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders specializations, independent versus correlated support, OCR, and cloud review semantics", () => {
    render(<I18nProvider initialLocale="en"><Phase6BAnalysisCard analysis={phase6bAnalysis} /></I18nProvider>);

    expect(screen.getByRole("heading", { name: "Model evidence and agreement" })).toBeInTheDocument();
    expect(screen.getByText("PLONK YFCC specialization · mixed scenes")).toBeInTheDocument();
    expect(screen.getByText("Unverified model point: 41.0082, 28.9784")).toBeInTheDocument();
    expect(screen.getByText("Independent families supporting the top region").parentElement).toHaveTextContent("3");
    expect(screen.getByText("Same-family duplicate support").parentElement).toHaveTextContent("1");
    expect(screen.getByText(/Same-family agreement is correlated evidence/)).toBeInTheDocument();
    expect(screen.getByText("İstanbul Caddesi")).toBeInTheDocument();
    expect(screen.getByText(/Recognized text is untrusted evidence/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "AI visual review" })).toBeInTheDocument();
    expect(screen.getByText(/not a verified fact or an independent location source/)).toBeInTheDocument();
    expect(screen.getByText("gpt-5.6-luna")).toBeInTheDocument();
    expect(screen.getByText(/\$0\.0012/)).toBeInTheDocument();
    expect(screen.getByText(/server-side daily and monthly budgets/)).toBeInTheDocument();
    expect(screen.getByText(/Calls recorded today: 1/)).toBeInTheDocument();
    expect(screen.getByText(/Estimated monthly spend:/)).toHaveTextContent("$0.30");
    expect(screen.getByText("The visible road layout is consistent with the supplied candidate summary.")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("sk-");
    expect(document.body.textContent).not.toContain("0.8");
  });

  it("keeps hard-case review diagnostics visible when the pipeline abstains", () => {
    const abstained: Analysis = {
      ...phase6bAnalysis,
      candidates: [],
      abstention: { abstained: true, reason_code: "insufficient_evidence", message_key: "reason.insufficient_evidence" },
    };
    render(<I18nProvider initialLocale="en"><NoSignal analysis={abstained} deleting={false} onDelete={() => undefined} /></I18nProvider>);
    expect(screen.getByRole("heading", { name: "Not enough evidence to estimate a location" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "AI visual review" })).toBeInTheDocument();
    expect(screen.getByText(/not a verified fact or an independent location source/)).toBeInTheDocument();
  });
});
