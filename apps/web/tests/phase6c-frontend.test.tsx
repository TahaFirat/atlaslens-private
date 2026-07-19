import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AtlasLensApp } from "../src/App";
import type {
  Analysis,
  Phase6CAnalysisSummary,
  Phase6BModelPredictionSummary,
  SystemIntelligenceModelCard,
  SystemIntelligenceResponse,
} from "../src/api/schemas";
import { Phase6CAnalysisCard } from "../src/components/Phase6CAnalysisCard";
import ResultMap from "../src/components/ResultMap";
import { I18nProvider } from "../src/i18n";
import { PHASE6C_MAP_LAYER_LIMIT, phase6CMapPoints } from "../src/map/phase6c";
import { completedAnalysis, createFakeClient } from "./fixtures";

function modelCard(index: number): SystemIntelligenceModelCard {
  return {
    model_id: `model-${index}`,
    display_name: index === 0 ? "GeoCLIP" : `Runtime model ${index}`,
    runtime_model_id: `runtime-${index}`,
    repository_url: index === 0 ? "https://example.test/model?token=must-not-render" : "https://example.test/model",
    purpose: "Bounded geographic evidence",
    enabled: index !== 1,
    available: index === 0,
    status: index === 0 ? "ready" : index === 1 ? "disabled" : "unavailable",
    installed: index < 3,
    weights_available: index < 2,
    worker_reachable: index === 0,
    model_loaded: index === 0,
    load_verified: index === 0,
    real_inference_verified: index === 0,
    device: index === 0 ? "cuda" : null,
    execution_mode: index === 9 ? "not_integrated" : "isolated_worker",
    source_revision: index === 0 ? "<PRIVATE_SOURCE_REVISION>" : `source-${index}`,
    model_revision: index === 0 ? "sk-secret-must-not-render" : `revision-${index}`,
    license: "MIT",
    last_success_at: index === 0 ? "2026-07-14T18:00:00Z" : null,
    last_latency_ms: index === 0 ? 80 : null,
    error_code: index > 1 ? "worker_unavailable" : null,
    current_participation: index === 0 ? "primary" : index === 9 ? "not_integrated" : index === 1 ? "disabled" : "candidate",
  };
}

function intelligence(overrides: Partial<SystemIntelligenceResponse["reference_index"]> = {}): SystemIntelligenceResponse {
  return {
    active_pipeline_version: "phase6c-v1",
    models: Array.from({ length: 10 }, (_, index) => modelCard(index)),
    reference_index: {
      index_id: "turkiye_megaloc_reference_index",
      enabled: true,
      status: "ready",
      reason_code: "ready",
      index_version: "turkiye-megaloc-v1",
      descriptor_version: "megaloc-8448-v1",
      count: 6,
      sequences: 6,
      countries: 1,
      provinces: 6,
      images_per_province: { Ankara: 1, "<PRIVATE_PROVINCE_LABEL>": 99 },
      source_distribution: { kartaview: 6 },
      attributions: [
        "© Grab and KartaView Contributors",
        "<PRIVATE_ATTRIBUTION_PATH>",
        "<PRIVATE_LINUX_ATTRIBUTION_PATH>",
      ],
      built_at: "2026-07-14T18:00:00Z",
      disk_usage_bytes: 209_628,
      leakage_status: "passed",
      leakage_audit: {
        status: "passed",
        audit_version: "atlaslens-leakage-audit-v1",
        audit_fingerprint: "c".repeat(64),
        source_report_sha256: "d".repeat(64),
        checked_reference_count: 6,
        descriptor_checked_count: 6,
        excluded_reference_count: 0,
        pre_index_excluded_reference_count: 1,
      },
      duplicates: 0,
      excluded: 1,
      health: "healthy",
      ...overrides,
    },
  };
}

function phase6cSummary(): Phase6CAnalysisSummary {
  return {
    schema_version: "atlaslens-phase6c-analysis-v1",
    pipeline_version: "phase6c-v1",
    fusion_version: "phase6c-v1",
    reference_index_version: "turkiye-megaloc-v1",
    evaluation_run_id: null,
    hierarchical_candidates: [{
      candidate_id: "hierarchy-1",
      latitude: 39,
      longitude: 35,
      search_level: "turkiye_refinement",
      provider_rank: 1,
      provider_score: 0.321234,
      score_semantics: "raw_cosine_similarity_not_confidence",
      grid_resolution_km: 75,
    }],
    megaloc_matches: [{
      reference_id: "opaque-reference",
      rank: 1,
      latitude: 38.7,
      longitude: 35.4,
      similarity: 0.876543,
      similarity_semantics: "cosine_similarity_not_confidence",
      confidence: null,
      uncertainty_radius_m: 50,
      source: "kartaview",
      source_family: "kartaview-sequence",
      source_sequence_id: "opaque-sequence",
      source_url: "https://example.test/raw-reference-must-not-render",
      captured_at: null,
      license: "CC BY-SA",
      attribution: "© Grab and KartaView Contributors",
    }],
    g3_scores: [],
    providers: [
      { provider_id: "geoclip_hierarchical", status: "completed", source_revision: "source", model_revision: "model", duration_ms: 120, candidates_produced: 8, reason_code: null },
      { provider_id: "megaloc", status: "skipped", source_revision: "source", model_revision: "model", duration_ms: 0, candidates_produced: 0, reason_code: "reference_index_unavailable" },
      { provider_id: "g3", status: "disabled", source_revision: null, model_revision: null, duration_ms: 0, candidates_produced: 0, reason_code: "not_integrated" },
    ],
    fusion_candidates: [{
      cluster_id: "phase6c-cluster-1",
      latitude: 38.8,
      longitude: 35.5,
      uncertainty_radius_km: 41,
      relative_rank_score: 0.765432,
      score_semantics: "uncalibrated_relative_rank_not_probability",
      confidence: null,
      calibrated: false,
      source_family_count: 2,
      independent_source_family_count: 2,
      correlated_source_family_count: 0,
      provider_count: 2,
      spread_km: 18,
      publication_eligible: true,
      publication_basis: "at_least_two_independent_source_families",
      pre_diversity_rank: 1,
      final_rank: 1,
      rank_movement: 0,
      movement_reasons: ["stable_rank"],
      members: [{
        evidence_kind: "geoclip_hierarchical",
        provider: "geoclip_hierarchical",
        model_id: "geoclip",
        model_revision: "model",
        candidate_id: "hierarchy-1",
        source_family: "geoclip",
        correlation_group: "mp16",
        score_semantics: "raw_cosine_similarity_not_confidence",
        raw_value: 0.321234,
        provider_rank: 1,
        sample_support: 1,
        provenance: "reviewed runtime",
      }],
      contributions: [{ name: "rank", source: "fusion", raw_value: 1, normalized_value: 1, weight: 1, contribution: 1, reason: "bounded rank", independent: true, correlation_group: "mp16" }],
    }],
    publication_candidate_count: 1,
    turkiye_signal_count: 2,
    turkiye_signal_groups: ["hierarchy", "retrieval"],
    leakage_audit: { status: "passed", audit_version: "phase6c-leakage-v1", report_fingerprint: "a".repeat(64), references_checked: 6, references_excluded: 1, reason_code: null },
    reference_attributions: ["© Grab and KartaView Contributors"],
    ablations: [],
    cache_fingerprint: `phase6c-v1:${"b".repeat(64)}`,
  };
}

describe("Phase 6C operator and analysis UI", () => {
  it("renders safe model/index status with truthful disabled states and aggregate attribution", async () => {
    vi.stubEnv("VITE_ENABLE_OPERATOR_UI", "true");
    const { client } = createFakeClient({ clientOverrides: { getSystemIntelligence: vi.fn(() => Promise.resolve(intelligence())) } });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);

    await user.click(await screen.findByRole("button", { name: "System status" }));
    expect(await screen.findByRole("heading", { name: "System Intelligence / Model & Data Status" })).toBeInTheDocument();
    expect(screen.getByText("phase6c-v1")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Türkiye MegaLoc reference index" })).toBeInTheDocument();
    expect(screen.getByText("References").parentElement).toHaveTextContent("6");
    expect(screen.getByText("Leakage audit").parentElement).toHaveTextContent("Passed");
    expect(screen.getByText("© Grab and KartaView Contributors")).toBeInTheDocument();
    expect(screen.getAllByText("Disabled").length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toContain("C:\\Users");
    expect(document.body.textContent).not.toContain("private-attribution");
    expect(document.body.textContent).not.toContain("/mnt/private");
    expect(document.body.textContent).not.toContain("sk-secret");
    expect(document.body.textContent).not.toContain("token=must-not-render");
  });

  it("shows unavailable operator status without inferring readiness", async () => {
    vi.stubEnv("VITE_ENABLE_OPERATOR_UI", "true");
    const getSystemIntelligence = vi.fn(() => Promise.reject(new Error("errors.api_not_found")));
    const { client } = createFakeClient({ clientOverrides: { getSystemIntelligence } });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await user.click(await screen.findByRole("button", { name: "System status" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/API route was not found/);
    expect(screen.getByText(/No model is inferred to be ready/)).toBeInTheDocument();
    expect(screen.queryByText("Ready")).not.toBeInTheDocument();
  });

  it("blocks MegaLoc presentation when leakage has not passed", async () => {
    vi.stubEnv("VITE_ENABLE_OPERATOR_UI", "true");
    const response = intelligence({ leakage_status: "not_run", leakage_audit: null, health: "degraded" });
    const { client } = createFakeClient({ clientOverrides: { getSystemIntelligence: vi.fn(() => Promise.resolve(response)) } });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await user.click(await screen.findByRole("button", { name: "System status" }));
    expect(await screen.findByText(/MegaLoc must not participate until the audit passes/)).toHaveTextContent("Not run");
  });

  it("reports every provider outcome and keeps raw scores and source URLs out of the diagnostics card", () => {
    const analysis: Analysis = { ...completedAnalysis, phase6c: phase6cSummary() };
    render(<I18nProvider initialLocale="en"><Phase6CAnalysisCard analysis={analysis} /></I18nProvider>);
    expect(screen.getByRole("heading", { name: "Provider runs and fusion" })).toBeInTheDocument();
    expect(screen.getByText("geoclip_hierarchical").parentElement).toHaveTextContent("120 ms");
    expect(screen.getByText("megaloc").parentElement).toHaveTextContent("Reference index unavailable");
    expect(screen.getByText("g3").parentElement).toHaveTextContent("Not integrated");
    expect(screen.getByText((_content, element) => element?.tagName === "SPAN" && element.textContent?.includes("Independent groups: 2") === true)).toBeInTheDocument();
    expect(screen.getByText("© Grab and KartaView Contributors")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("0.876543");
    expect(document.body.textContent).not.toContain("raw-reference-must-not-render");
    expect(document.body.textContent).not.toContain("phase6c-v1:bbbb");
  });

  it("bounds provider overlay points and exposes layer toggles with bounded uncertainty", async () => {
    const phase6c = phase6cSummary();
    phase6c.hierarchical_candidates = Array.from({ length: 20 }, (_, index) => ({ ...phase6c.hierarchical_candidates[0]!, candidate_id: `h-${index}`, provider_rank: index + 1 }));
    phase6c.megaloc_matches = Array.from({ length: 20 }, (_, index) => ({ ...phase6c.megaloc_matches[0]!, reference_id: `r-${index}`, rank: index + 1 }));
    const prediction: Phase6BModelPredictionSummary = {
      provider: "geoclip",
      model_id: "GeoCLIP",
      model_revision: "model",
      source_family: "mp16_family",
      status: "completed",
      device: "cuda",
      duration_ms: 10,
      score_semantics: "similarity",
      candidates: Array.from({ length: 20 }, (_, index) => ({ candidate_id: `g-${index}`, latitude: 30 + index / 100, longitude: 35, raw_score: null, provider_rank: index + 1, sample_support: 1 })),
      warnings: [],
    };
    const points = phase6CMapPoints(phase6c, { geoclip: prediction });
    expect(points.features.filter((feature) => feature.properties.layer === "geoclip")).toHaveLength(PHASE6C_MAP_LAYER_LIMIT);
    expect(points.features.filter((feature) => feature.properties.layer === "hierarchy")).toHaveLength(PHASE6C_MAP_LAYER_LIMIT);
    expect(points.features.filter((feature) => feature.properties.layer === "megaloc")).toHaveLength(PHASE6C_MAP_LAYER_LIMIT);

    const user = userEvent.setup();
    render(<I18nProvider initialLocale="en"><ResultMap candidates={completedAnalysis.candidates} phase6c={phase6c} modelPredictions={{ geoclip: prediction }} /></I18nProvider>);
    expect(screen.getByRole("button", { name: `Fused (1)` })).toHaveAttribute("aria-pressed", "true");
    const hierarchy = screen.getByRole("button", { name: `Hierarchy (${PHASE6C_MAP_LAYER_LIMIT})` });
    expect(hierarchy).toHaveAttribute("aria-pressed", "false");
    await user.click(hierarchy);
    expect(hierarchy).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/Each layer is capped at 12 points/)).toBeInTheDocument();
  });
});
