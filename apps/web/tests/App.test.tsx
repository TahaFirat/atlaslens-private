import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AtlasLensApp } from "../src/App";
import { ApiError } from "../src/api/client";
import type { Analysis, AnalysisAccepted } from "../src/api/schemas";
import { capabilities, completedAnalysis, createFakeClient, jpegFile, modelAnalysis, noSignalAnalysis, systemIntelligenceState } from "./fixtures";

async function prepareAndSubmit(clientOptions: Parameters<typeof createFakeClient>[0] = {}, locale: "en" | "tr" = "en") {
  const user = userEvent.setup();
  const fake = createFakeClient(clientOptions);
  render(<AtlasLensApp client={fake.client} initialLocale={locale} />);
  await screen.findByText(locale === "en" ? "Server capabilities" : "Sunucu yetenekleri");
  await user.upload(screen.getByTestId("file-input"), jpegFile());
  await user.click(screen.getByRole("checkbox", { name: /own this image|görüntünün sahibiyim/i }));
  await user.click(screen.getByRole("button", { name: locale === "en" ? "Analyze photo" : "Fotoğrafı analiz et" }));
  return { user, ...fake };
}

describe("AtlasLens upload and capability workflow", () => {
  it("renders complete English and Turkish entry states with a visible switch", async () => {
    const { client } = createFakeClient();
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    expect(await screen.findByRole("heading", { name: "Locate the evidence, not just the image" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "TR" }));
    expect(screen.getByRole("heading", { name: "Yalnızca görüntüyü değil, kanıtı konumlandırın" })).toBeInTheDocument();
    expect(document.documentElement).toHaveAttribute("lang", "tr");
  });

  it("validates upload content before enabling analysis", async () => {
    const { client } = createFakeClient();
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await screen.findByText("Server capabilities");
    await user.upload(screen.getByTestId("file-input"), new File(["plain"], "plain.jpg", { type: "image/jpeg" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("contents do not match");
    expect(screen.getByRole("button", { name: "Analyze photo" })).toBeDisabled();
  });

  it("keeps authorization and cloud consent separate", async () => {
    const { client } = createFakeClient();
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await screen.findByText("Server capabilities");
    await user.upload(screen.getByTestId("file-input"), jpegFile());
    await user.click(screen.getByRole("radio", { name: /Cloud assisted/ }));
    const analyze = screen.getByRole("button", { name: "Analyze photo" });
    const cloudAssist = screen.getByRole("checkbox", { name: /Use cloud assistance only when local evidence is weak/ });
    expect(cloudAssist).not.toBeChecked();
    expect(analyze).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: /I own this image/ }));
    expect(analyze).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: /I consent/ }));
    expect(analyze).toBeEnabled();
    expect(cloudAssist).not.toBeChecked();
  });

  it("keeps hard-case cloud review off by default and resets it outside cloud mode", async () => {
    const { client } = createFakeClient();
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="tr" />);
    await screen.findByText("Sunucu yetenekleri");
    await user.click(screen.getByRole("radio", { name: /Bulut destekli/ }));
    const assist = screen.getByRole("checkbox", { name: /^Düşük güven durumunda bulut desteği kullan/ });
    expect(assist).not.toBeChecked();
    expect(screen.getByText(/API anahtarı sunucuda kalır/)).toBeInTheDocument();
    await user.click(assist);
    expect(assist).toBeChecked();
    await user.click(screen.getByRole("radio", { name: /Yalnızca yerel/ }));
    expect(screen.queryByRole("checkbox", { name: /^Düşük güven durumunda bulut desteği kullan/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: /Bulut destekli/ }));
    expect(screen.getByRole("checkbox", { name: /^Düşük güven durumunda bulut desteği kullan/ })).not.toBeChecked();
  });

  it("explains missing cloud capability without requesting a browser key", async () => {
    const unavailableCapabilities = {
      ...capabilities,
      providers: {
        ...capabilities.providers,
        cloud_vision: {
          ...capabilities.providers.cloud_vision,
          available: false,
          reason_code: "missing_api_key",
        },
      },
    };
    const { client } = createFakeClient({ capabilityState: unavailableCapabilities });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await screen.findByText("Server capabilities");
    await user.click(screen.getByRole("radio", { name: /Cloud assisted/ }));
    expect(screen.getByRole("alert")).toHaveTextContent("no configured cloud provider");
    expect(screen.getByRole("alert")).toHaveTextContent("never need to paste an API key");
  });

  it("shows global model operational status before upload", async () => {
    const { client } = createFakeClient();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    expect(await screen.findByText("Global visual prediction")).toBeInTheDocument();
    expect(screen.getByText(/GeoCLIP · revision 1.2.0 · cuda/)).toBeInTheDocument();
    expect(screen.getByText(/Ready.*Server-local/)).toBeInTheDocument();
  });

  it("distinguishes a model that is not installed from a generic unavailable provider", async () => {
    const unavailableCapabilities = {
      ...capabilities,
      providers: {
        ...capabilities.providers,
        global_geolocation: {
          ...capabilities.providers.global_geolocation!,
          available: false,
          operational_status: "not_installed" as const,
          reason_code: "model_not_installed",
          device: null,
        },
      },
    };
    const { client } = createFakeClient({ capabilityState: unavailableCapabilities });
    render(<AtlasLensApp client={client} initialLocale="en" />);
    expect(await screen.findByText(/Model not installed.*Server-local/)).toBeInTheDocument();
    expect(screen.queryByText(/Provider failed.*Server-local/)).not.toBeInTheDocument();
  });

  it("does not show a green operational indicator for a failed model", async () => {
    const failedCapabilities = {
      ...capabilities,
      providers: {
        ...capabilities.providers,
        global_geolocation: {
          ...capabilities.providers.global_geolocation!,
          operational_status: "failed" as const,
          reason_code: "provider_initialization_failed",
        },
      },
    };
    const { client } = createFakeClient({ capabilityState: failedCapabilities });
    render(<AtlasLensApp client={client} initialLocale="en" />);
    const label = await screen.findByText("Global visual prediction");
    const providerRow = label.closest("li");
    expect(providerRow).not.toBeNull();
    expect(providerRow?.querySelector(".provider-dot--on")).toBeNull();
    expect(screen.getByText(/Provider failed.*Server-local/)).toBeInTheDocument();
  });

  it("labels an isolated worker lifecycle failure in English and Turkish", async () => {
    vi.stubEnv("VITE_ENABLE_OPERATOR_UI", "true");
    const { client } = createFakeClient({
      clientOverrides: {
        getSystemIntelligence: vi.fn(() => Promise.resolve(systemIntelligenceState({
          model_id: "osv5m",
          display_name: "OSV-5M",
          enabled: true,
          available: false,
          status: "worker_unreachable",
          execution_mode: "isolated_worker",
          current_participation: "candidate",
        }))),
      },
    });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await user.click(await screen.findByRole("button", { name: "System status" }));
    expect(await screen.findByText("Worker unreachable")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "TR" }));
    expect(screen.getByText("Çalışan sürece ulaşılamıyor")).toBeInTheDocument();
  });

  it("opens the file picker from the keyboard-accessible drop zone", async () => {
    const { client } = createFakeClient();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await screen.findByText("Server capabilities");
    const input = screen.getByTestId("file-input");
    const click = vi.spyOn(input, "click");
    fireEvent.keyDown(screen.getByRole("button", { name: "Drop a photo here or choose a file" }), { key: "Enter" });
    expect(click).toHaveBeenCalledOnce();
  });

  it("freezes source and consent controls while an upload is in flight", async () => {
    let finish: ((value: AnalysisAccepted) => void) | undefined;
    const pending = new Promise<AnalysisAccepted>((resolve) => { finish = resolve; });
    const { client } = createFakeClient({ clientOverrides: { createAnalysis: vi.fn(() => pending) } });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} initialLocale="en" />);
    await screen.findByText("Server capabilities");
    await user.upload(screen.getByTestId("file-input"), jpegFile());
    await user.click(screen.getByRole("checkbox", { name: /own this image/i }));
    await user.click(screen.getByRole("button", { name: "Analyze photo" }));
    expect(screen.getByTestId("file-input")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Drop a photo here or choose a file" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("radio", { name: /Local only/ })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: /own this image/i })).toBeDisabled();
    act(() => finish?.({ id: completedAnalysis.id, status: "queued", status_url: `/api/v1/analyses/${completedAnalysis.id}`, events_url: `/api/v1/analyses/${completedAnalysis.id}/events`, delete_url: `/api/v1/analyses/${completedAnalysis.id}` }));
  });
});

describe("AtlasLens live and terminal analysis states", () => {
  it("shows real server progress and the polling fallback when SSE disconnects", async () => {
    const processing: Analysis = {
      ...completedAnalysis,
      status: "processing",
      progress: { stage: "quality", percent: 62, message_key: "progress.quality" },
      candidates: [],
    };
    await prepareAndSubmit({ analysis: processing, disconnectStream: true });
    expect(await screen.findByRole("progressbar", { name: "Measuring image quality" })).toHaveAttribute("aria-valuenow", "62");
    expect(await screen.findByText(/Live stream disconnected/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel and delete" })).toBeEnabled();
  });

  it("localizes the additive segmentation progress stage", async () => {
    const processing: Analysis = {
      ...completedAnalysis,
      status: "processing",
      progress: { stage: "scene_segmentation", percent: 70, message_key: "progress.scene_segmentation" },
      candidates: [],
    };
    await prepareAndSubmit({ analysis: processing }, "tr");
    expect(await screen.findByRole("progressbar", { name: "Görünür sahne bileşenleri özetleniyor" })).toHaveAttribute("aria-valuenow", "70");
  });

  it("presents honest no-signal abstention", async () => {
    await prepareAndSubmit({ analysis: noSignalAnalysis });
    expect(await screen.findByRole("heading", { name: "Not enough evidence to estimate a location" })).toBeInTheDocument();
    expect(screen.getByText(/did not invent one/)).toBeInTheDocument();
    expect(screen.getByText(/No supported geolocation signal/)).toBeInTheDocument();
  });

  it("renders candidates, uncertainty map, quality, provenance, and unknown evidence safely", async () => {
    const { user } = await prepareAndSubmit({ analysis: completedAnalysis });
    expect(await screen.findByRole("heading", { name: "EXIF coordinate" })).toBeInTheDocument();
    expect(screen.getAllByText("Source-derived support").length).toBeGreaterThan(0);
    expect(screen.queryByText("95% source support estimate")).not.toBeInTheDocument();
    expect(screen.getByText(/EXIF coordinates can be stale/)).toBeInTheDocument();
    expect(screen.getByTestId("map-region")).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Textual map alternative" })).toBeInTheDocument();
    expect(screen.getAllByText("Additional evidence").length).toBeGreaterThan(0);
    expect(screen.queryByText("future_safe_label")).not.toBeInTheDocument();
    expect(screen.getByText("Image quality")).toBeInTheDocument();
    await user.click(screen.getByText("Advanced diagnostics and raw relative scores"));
    expect(screen.getAllByText(/exif · local/).length).toBeGreaterThan(0);
  });

  it("renders Phase 6A cluster, scene, coordinate-name, and uncalibrated confidence additions", async () => {
    const hybrid: Analysis = {
      ...modelAnalysis,
      scene_analysis: {
        status: "completed",
        provider: "atlaslens-segformer-b2-v4",
        device: "cpu",
        inference_ms: 18.4,
        image_width: 640,
        image_height: 480,
        semantic_label_names_available: true,
        dominant_classes: [{ class_id: 13, class_name: "Road", pixel_ratio: 0.341, percentage: 34.1 }],
        scene_groups: [{ name: "road_surface", pixel_ratio: 0.341, percentage: 34.1 }],
        scene_tags: [],
        warnings: [],
      },
      candidates: [{
        ...modelAnalysis.candidates[0]!,
        model_prediction: null,
        geoclip_cluster: {
          cluster_id: "cluster-1",
          source: "geoclip",
          member_count: 3,
          member_ranks: [1, 3, 8],
          max_raw_similarity: 0.18,
          mean_raw_similarity: 0.12,
          raw_score_type: "uncalibrated_gallery_softmax",
          cluster_support: 0.71,
          score_semantics: "uncalibrated_relative_rank",
        },
        confidence_assessment: { label: "high", score: null, calibrated: false, basis: ["dense_candidate_cluster", "strong_ocr_match"] },
        reverse_geocode: {
          country: "Türkiye",
          country_code: "TR",
          region: "Marmara",
          city: "İstanbul",
          district: null,
          display_name: "İstanbul, Türkiye",
          provider: "offline-fixture",
          dataset_version: "fixture-v1",
          license: "test-only",
        },
      }],
    };
    const { user } = await prepareAndSubmit({ analysis: hybrid });

    expect(await screen.findByRole("heading", { name: "Visible scene components" })).toBeInTheDocument();
    expect(screen.getAllByText("3 nearby GeoCLIP candidates").length).toBeGreaterThan(0);
    expect(screen.getAllByText("High").length).toBeGreaterThan(0);
    expect(screen.getByText("Not calibrated")).toBeInTheDocument();
    expect(screen.queryByText("71%")).not.toBeInTheDocument();
    expect(screen.getByText("İstanbul, Türkiye")).toBeInTheDocument();
    expect(screen.getByText(/only names this coordinate/i)).toBeInTheDocument();
    await user.click(screen.getByText("Advanced diagnostics and raw relative scores"));
    expect(screen.getByRole("heading", { name: "GeoCLIP cluster diagnostics" })).toBeInTheDocument();
    expect(screen.getByText(/relative ranking diagnostics, not probabilities/i)).toBeInTheDocument();
  });

  it("renders top-k model hypotheses without presenting scores as probabilities", async () => {
    const { user } = await prepareAndSubmit({ analysis: modelAnalysis });
    expect((await screen.findAllByText("Uncalibrated relative support")).length).toBeGreaterThan(0);
    expect(document.querySelectorAll(".candidate-summary")).toHaveLength(3);
    expect(screen.getByText(/Raw relative score:/)).not.toBeVisible();
    await user.click(screen.getByText("Advanced diagnostics and raw relative scores"));
    expect(screen.getByText("Unverified, uncalibrated model hypothesis")).toBeInTheDocument();
    expect(screen.getByText(/Raw relative score:/)).toBeInTheDocument();
    expect(screen.getAllByText(/not a probability/).length).toBeGreaterThan(0);
    expect(screen.getByText("cuda · float32")).toBeInTheDocument();
    expect(screen.getByText("184 ms")).toBeInTheDocument();
    expect(screen.getByText("Global visual-model hypothesis")).toBeInTheDocument();
    expect(screen.getByText("The fixed coordinate gallery limits geographic resolution.")).toBeInTheDocument();
    expect(screen.queryByText("fixed_gallery_limits_geographic_resolution")).not.toBeInTheDocument();
    expect(screen.getByText("Offline place label")).toBeInTheDocument();
    expect(screen.getByText("GeoNames · atlaslens-geonames-v1")).toBeInTheDocument();
  });

  it("renders model installation failure without exposing raw provider codes", async () => {
    await prepareAndSubmit({
      analysis: {
        ...noSignalAnalysis,
        abstention: { abstained: true, reason_code: "model_not_installed", message_key: "reason.model_not_installed" },
        warnings: ["provider.global_prediction.model_not_installed"],
      },
    });
    expect(await screen.findByText(/The global prediction model is not installed\./)).toBeInTheDocument();
    expect(screen.getByText("Global model prediction was skipped because the model is not installed.")).toBeInTheDocument();
    expect(screen.queryByText("provider.global_prediction.model_not_installed")).not.toBeInTheDocument();
  });

  it("renders safe OCR, segmentation, and reverse-geocoding warnings without raw codes", async () => {
    await prepareAndSubmit({
      analysis: {
        ...completedAnalysis,
        warnings: [
          "provider.ocr.timeout",
          "provider.segmentation.unavailable",
          "provider.reverse_geocoding.timeout",
          "provider.reverse_geocoding.cache_unavailable",
          "warning.confidence_not_calibrated",
        ],
      },
    });
    expect(await screen.findByText(/Local text extraction timed out/)).toBeInTheDocument();
    expect(screen.getByText(/Scene segmentation was unavailable/)).toBeInTheDocument();
    expect(screen.getByText(/Coordinate naming timed out/)).toBeInTheDocument();
    expect(screen.getByText(/naming continued without cache/)).toBeInTheDocument();
    expect(screen.getByText(/Confidence labels are qualitative/)).toBeInTheDocument();
    expect(screen.queryByText("provider.segmentation.unavailable")).not.toBeInTheDocument();
  });

  it("surfaces safe API errors and unknown codes", async () => {
    await prepareAndSubmit({ createError: new ApiError("a_new_server_condition") });
    expect(await screen.findByRole("alert")).toHaveTextContent("unrecognized condition");
    expect(screen.getByRole("alert")).toHaveTextContent("a_new_server_condition");
  });

  it("shows safe structured diagnostics without internal details", async () => {
    const problem = {
      type: "about:blank",
      title: "Invalid request",
      status: 422,
      code: "authorization_required",
      message_key: "errors.authorization_required",
      request_id: "request-safe-123",
      retry_after_seconds: null,
    };
    await prepareAndSubmit({ createError: new ApiError(problem.message_key, problem, "client_error", 422) });
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Image authorization acknowledgement is required");
    expect(alert).toHaveTextContent("authorization_required");
    expect(alert).toHaveTextContent("request-safe-123");
  });

  it("deletes a completed analysis and confirms it is gone", async () => {
    const { user, deleteAnalysis } = await prepareAndSubmit({ analysis: completedAnalysis });
    const deleteButton = await screen.findByRole("button", { name: "Delete analysis" });
    await user.click(deleteButton);
    await waitFor(() => expect(deleteAnalysis).toHaveBeenCalledWith(completedAnalysis.id));
    expect(await screen.findByText("Analysis deleted from the server.")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "EXIF coordinate" })).not.toBeInTheDocument();
  });

  it("shows a safe failed state", async () => {
    const failed: Analysis = {
      ...completedAnalysis,
      status: "failed",
      candidates: [],
      failure: { code: "provider_timeout", message_key: "errors.provider_timeout", retryable: true },
    };
    await prepareAndSubmit({ analysis: failed });
    expect(await screen.findByRole("heading", { name: "The analysis could not be completed" })).toBeInTheDocument();
    expect(screen.getByText(/optional provider timed out/i)).toBeInTheDocument();
  });

  it("labels completed output with warnings as a partial result", async () => {
    const partial: Analysis = { ...completedAnalysis, warnings: ["provider_timeout"] };
    await prepareAndSubmit({ analysis: partial });
    expect(await screen.findByText("Partial result")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Warnings and contradictions" })).toBeInTheDocument();
  });
});
