import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AtlasLensApp } from "../src/App";
import type { Analysis, DatasetQAReport, EvaluationReportSummary } from "../src/api/schemas";
import { analysisId, completedAnalysis, createFakeClient, jpegFile, systemIntelligenceState } from "./fixtures";

async function submit(analysis: Analysis, locale: "en" | "tr" = "en") {
  const user = userEvent.setup();
  const { client } = createFakeClient({ analysis });
  render(<AtlasLensApp client={client} initialLocale={locale} />);
  await screen.findByText(locale === "tr" ? "Sunucu yetenekleri" : "Server capabilities");
  await user.upload(screen.getByTestId("file-input"), jpegFile());
  await user.click(screen.getByRole("checkbox", { name: locale === "tr" ? /sahibiyim/ : /own this image/i }));
  await user.click(screen.getByRole("button", { name: locale === "tr" ? "Fotoğrafı analiz et" : "Analyze photo" }));
  return user;
}

const evaluation: EvaluationReportSummary = {
  report_id: "eval-geoclip-fixed",
  provider_id: "geoclip-global-v1",
  model_revision: "1.2.0",
  evaluation_fingerprint: "fixed-evaluation-fingerprint",
  image_count: 30,
  calibration_state: "uncalibrated",
  country_top1: { numerator: 18, denominator: 30, value: 0.6 },
  country_top5: { numerator: 24, denominator: 30, value: 0.8 },
  region_top1: { numerator: 12, denominator: 30, value: 0.4 },
  city_top1: { numerator: 0, denominator: 0, value: null },
  recall_top1: { "200_km": { numerator: 14, denominator: 30, value: 14 / 30 } },
  mean_error_km: 810,
  median_error_km: 143.1,
  p95_error_km: 2400,
  abstention: { numerator: 0, denominator: 30, value: 0 },
  provider_failure: { numerator: 1, denominator: 30, value: 1 / 30 },
  latency_median_ms: 72,
  latency_p95_ms: 120,
  uncertainty_coverage: { numerator: 20, denominator: 30, value: 2 / 3 },
  geographic_distribution: { Europe: 10, Asia: 10, Americas: 10 },
  scene_distribution: { urban: 15, rural: 15 },
  exclusions: {},
  limitations: ["uncalibrated_fixed_slice"],
};

const qaReport: DatasetQAReport = {
  summary: {
    report_id: "qa-safe-report",
    schema_version: 1,
    dataset_fingerprint: "dataset-fingerprint-safe",
    dataset_type: "geolocation",
    created_at: "2026-07-12T09:00:00Z",
    scanned_images: 30,
    scanned_masks: 0,
    error_count: 1,
    warning_count: 2,
  },
  checks: { image_integrity: "passed", coordinate_validity: "failed" },
  distributions: { country: { TR: 12, DE: 8 } },
  issues: [{ code: "invalid_coordinate", severity: "error", asset_key: "asset-opaque-17", field: "latitude", message_key: "qa.invalid_coordinate", safe_metrics: {} }],
  outputs: ["qa-report.json"],
  limitations: ["country_consistency_unavailable"],
};

describe("Phase 5C product workspace", () => {
  it("marks simulated results prominently in English and Turkish without changing real results", async () => {
    const simulated: Analysis = {
      ...completedAnalysis,
      result_classification: "simulated",
      simulation: { scenario_id: "safe-development-scenario", warning_key: "warning.simulated_development_result", watermark: "SIMULATED DEVELOPMENT RESULT" },
    };
    const user = await submit(simulated);
    expect(await screen.findByText("SIMULATED DEVELOPMENT RESULT")).toBeInTheDocument();
    expect(screen.getByText(/not a real geolocation prediction/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "TR" }));
    expect(screen.getByText("SİMÜLE EDİLMİŞ GELİŞTİRME SONUCU")).toBeInTheDocument();
    expect(screen.queryByText("safe-development-scenario")).not.toBeInTheDocument();
  });

  it("lists private history summaries, opens detail, and disables rerun without a retained source", async () => {
    const listAnalyses = vi.fn(() => Promise.resolve({
      items: [{
        id: analysisId,
        created_at: "2026-07-12T09:00:00Z",
        expires_at: "2026-07-12T10:00:00Z",
        status: "completed" as const,
        analysis_mode: "local_only" as const,
        result_classification: "real" as const,
        image_sha256: "a".repeat(64),
        image_width: 1200,
        image_height: 800,
        provider_ids: ["exif"],
        primary_label: "Safe result label",
        candidate_count: 1,
        evidence_count: 2,
        runtime_ms: 24,
        warning_count: 0,
        source_retained: false,
        deletion_state: "active" as const,
      }],
      total: 1,
      limit: 12,
      offset: 0,
    }));
    const { client } = createFakeClient({ clientOverrides: { listAnalyses } });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await user.click(screen.getByRole("button", { name: "History" }));
    expect(await screen.findByRole("heading", { name: "Recent analyses" })).toBeInTheDocument();
    expect(screen.getByText("Safe result label")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Rerun" })).toBeDisabled();
    expect(screen.queryByText(/authorized-photo/i)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Open analysis" }));
    expect(await screen.findByRole("heading", { name: "EXIF coordinate" })).toBeInTheDocument();
  });

  it("renders fixed-fingerprint evaluation metrics and excludes internal code formatting", async () => {
    const { client } = createFakeClient({ clientOverrides: {
      listEvaluations: vi.fn(() => Promise.resolve({ reports: [evaluation] })),
      getEvaluation: vi.fn(() => Promise.resolve(evaluation)),
    } });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await user.click(screen.getByRole("button", { name: "Evaluation" }));
    expect(await screen.findByRole("heading", { name: "Evaluation reports" })).toBeInTheDocument();
    expect(await screen.findByText("fixed-evaluation-fingerprint")).toBeInTheDocument();
    expect(screen.getByText("60% (18/30)")).toBeInTheDocument();
    expect(screen.getByText("Uncalibrated fixed slice")).toBeInTheDocument();
    expect(screen.queryByText("uncalibrated_fixed_slice")).not.toBeInTheDocument();
  });

  it("renders read-only dataset QA and system/model unavailable states safely", async () => {
    vi.stubEnv("VITE_ENABLE_OPERATOR_UI", "true");
    const system = systemIntelligenceState({
      model_id: "atlaslens-custom-geolocation",
      display_name: "AtlasLens custom geolocation",
      enabled: true,
      status: "not_installed",
      current_participation: "candidate",
      execution_mode: "isolated_worker",
      device: "cuda",
    });
    const { client } = createFakeClient({ clientOverrides: {
      listDatasetQaReports: vi.fn(() => Promise.resolve({ reports: [qaReport.summary] })),
      getDatasetQaReport: vi.fn(() => Promise.resolve(qaReport)),
      getSystemIntelligence: vi.fn(() => Promise.resolve(system)),
    } });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await user.click(screen.getByRole("button", { name: "Dataset QA" }));
    expect(await screen.findByRole("heading", { name: "Dataset quality" })).toBeInTheDocument();
    expect(await screen.findByText("dataset-fingerprint-safe")).toBeInTheDocument();
    expect(screen.getByText("Qa invalid coordinate")).toBeInTheDocument();
    expect(screen.queryByText("qa.invalid_coordinate")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "System status" }));
    const customModel = (await screen.findByText("AtlasLens custom geolocation")).closest("article");
    expect(customModel).not.toBeNull();
    expect(screen.getByText("Model not installed")).toBeInTheDocument();
    expect(customModel).toHaveTextContent("Isolated worker");
    expect(customModel).toHaveTextContent("cuda");
  });

  it("does not expose or call operator workspaces without the explicit UI switch", async () => {
    vi.stubEnv("VITE_ENABLE_OPERATOR_UI", "false");
    const listDatasetQaReports = vi.fn(() => Promise.resolve({ reports: [qaReport.summary] }));
    const getSystemIntelligence = vi.fn(() => Promise.resolve(systemIntelligenceState()));
    const { client } = createFakeClient({ clientOverrides: { listDatasetQaReports, getSystemIntelligence } });

    render(<AtlasLensApp client={client} initialLocale="en" />);
    await screen.findByText("Server capabilities");

    expect(screen.queryByRole("button", { name: "Dataset QA" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "System status" })).not.toBeInTheDocument();
    expect(listDatasetQaReports).not.toHaveBeenCalled();
    expect(getSystemIntelligence).not.toHaveBeenCalled();
  });

  it("keeps the real result free of simulation marks and fake confidence percentages", async () => {
    await submit(completedAnalysis);
    expect(await screen.findByRole("heading", { name: "EXIF coordinate" })).toBeInTheDocument();
    expect(screen.queryByText("SIMULATED DEVELOPMENT RESULT")).not.toBeInTheDocument();
    expect(screen.queryByText(/95% source support/i)).not.toBeInTheDocument();
    expect(screen.getAllByText("Source-derived support").length).toBeGreaterThan(0);
    await waitFor(() => expect(document.querySelectorAll("[id='main-content']")).toHaveLength(1));
  });
});
