import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiClient } from "../src/api/client";

const accepted = {
  id: "1950c61c-0e02-4b77-b6d7-343dd5c76ce0",
  status: "queued",
  status_url: "/api/v1/analyses/1950c61c-0e02-4b77-b6d7-343dd5c76ce0",
  events_url: "/api/v1/analyses/1950c61c-0e02-4b77-b6d7-343dd5c76ce0/events",
  delete_url: "/api/v1/analyses/1950c61c-0e02-4b77-b6d7-343dd5c76ce0",
} as const;

class FakeXMLHttpRequest {
  static instances: FakeXMLHttpRequest[] = [];
  static status = 202;
  static responseText = JSON.stringify(accepted);
  static outcome: "load" | "error" = "load";
  static failSetup = false;

  readonly headers = new Map<string, string>();
  readonly upload: { onprogress: ((event: ProgressEvent) => void) | null } = { onprogress: null };
  method = "";
  url = "";
  body: Document | XMLHttpRequestBodyInit | null = null;
  status = 0;
  responseText = "";
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;

  constructor() {
    FakeXMLHttpRequest.instances.push(this);
  }

  open(method: string, url: string) {
    if (FakeXMLHttpRequest.failSetup) throw new Error("setup details must not escape");
    this.method = method;
    this.url = url;
  }

  setRequestHeader(name: string, value: string) {
    this.headers.set(name.toLowerCase(), value);
  }

  send(body: Document | XMLHttpRequestBodyInit | null) {
    this.body = body;
    this.status = FakeXMLHttpRequest.status;
    this.responseText = FakeXMLHttpRequest.responseText;
    queueMicrotask(() => {
      if (FakeXMLHttpRequest.outcome === "error") this.onerror?.();
      else this.onload?.();
    });
  }
}

function input() {
  return {
    file: new File([Uint8Array.from([0xff, 0xd8, 0xff])], "licensed.jpg", { type: "image/jpeg" }),
    mode: "local_only" as const,
    cloudConsent: false,
    authorizationAcknowledged: true,
  };
}

function requestInstance(): FakeXMLHttpRequest {
  const request = FakeXMLHttpRequest.instances[0];
  if (!request) throw new Error("Expected an XMLHttpRequest instance");
  return request;
}

beforeEach(() => {
  FakeXMLHttpRequest.instances = [];
  FakeXMLHttpRequest.status = 202;
  FakeXMLHttpRequest.responseText = JSON.stringify(accepted);
  FakeXMLHttpRequest.outcome = "load";
  FakeXMLHttpRequest.failSetup = false;
  vi.stubGlobal("XMLHttpRequest", FakeXMLHttpRequest);
});

afterEach(() => vi.unstubAllGlobals());

describe("real analysis API client", () => {
  it("still emits POST when crypto idempotency APIs are unavailable", async () => {
    vi.stubGlobal("crypto", {});

    await expect(apiClient.createAnalysis(input())).resolves.toEqual(accepted);

    const request = requestInstance();
    expect(request.method).toBe("POST");
    expect(request.url).toBe("/api/v1/analyses");
    expect(request.headers.has("idempotency-key")).toBe(false);
  });

  it("sends the exact multipart contract without setting Content-Type", async () => {
    vi.stubGlobal("crypto", { randomUUID: () => "test-idempotency-key" });

    await expect(apiClient.createAnalysis(input())).resolves.toEqual(accepted);

    const request = requestInstance();
    expect(request.status).toBe(202);
    expect(request.body).toBeInstanceOf(FormData);
    const form = request.body as FormData;
    expect(form.get("image")).toBeInstanceOf(File);
    expect(form.get("analysis_mode")).toBe("local_only");
    expect(form.get("cloud_processing_consent")).toBe("false");
    expect(form.get("allow_cloud_assist")).toBe("false");
    expect(form.get("authorization_acknowledged")).toBe("true");
    expect([...form.keys()].sort()).toEqual([
      "allow_cloud_assist",
      "analysis_mode",
      "authorization_acknowledged",
      "cloud_processing_consent",
      "image",
    ]);
    expect(request.headers.get("idempotency-key")).toBe("test-idempotency-key");
    expect(request.headers.has("content-type")).toBe(false);
  });

  it("sends explicit request-level cloud assistance without exposing configuration", async () => {
    await expect(apiClient.createAnalysis({ ...input(), mode: "cloud_assisted", cloudConsent: true, allowCloudAssist: true })).resolves.toEqual(accepted);

    const form = requestInstance().body as FormData;
    expect(form.get("cloud_processing_consent")).toBe("true");
    expect(form.get("allow_cloud_assist")).toBe("true");
    expect([...form.keys()].some((key) => /key|secret|token/i.test(key))).toBe(false);
  });

  it("preserves structured safe API errors", async () => {
    FakeXMLHttpRequest.status = 422;
    FakeXMLHttpRequest.responseText = JSON.stringify({
      type: "about:blank",
      title: "Invalid request",
      status: 422,
      code: "authorization_required",
      message_key: "errors.authorization_required",
      request_id: "request-safe-123",
      retry_after_seconds: null,
    });

    const error = await apiClient.createAnalysis(input()).catch((reason: unknown) => reason);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      kind: "client_error",
      status: 422,
      problem: { code: "authorization_required", request_id: "request-safe-123" },
    });
  });

  it("classifies transport failure as backend unavailable", async () => {
    FakeXMLHttpRequest.outcome = "error";

    const error = await apiClient.createAnalysis(input()).catch((reason: unknown) => reason);
    expect(error).toMatchObject({ kind: "backend_unreachable", message: "errors.backend_unreachable" });
  });

  it("converts synchronous request setup failures without leaking details", async () => {
    FakeXMLHttpRequest.failSetup = true;

    const error = await apiClient.createAnalysis(input()).catch((reason: unknown) => reason);
    expect(error).toMatchObject({ kind: "request_setup", message: "errors.request_setup" });
    expect(String(error)).not.toContain("setup details");
  });

  it("degrades to polling when EventSource setup fails synchronously", () => {
    class FailingEventSource {
      constructor() {
        throw new Error("event source internals must not escape");
      }
    }
    vi.stubGlobal("EventSource", FailingEventSource);
    const onDisconnect = vi.fn();

    const subscription = apiClient.subscribeAnalysis(accepted.id, {
      onEvent: vi.fn(),
      onDisconnect,
    });

    expect(onDisconnect).toHaveBeenCalledOnce();
    expect(() => subscription.close()).not.toThrow();
  });

  it("reports live stream readiness only after EventSource opens", () => {
    class FakeEventSource {
      static instance: FakeEventSource | undefined;
      onopen: (() => void) | null = null;
      onerror: (() => void) | null = null;
      constructor() { FakeEventSource.instance = this; }
      addEventListener() {}
      removeEventListener() {}
      close() {}
    }
    vi.stubGlobal("EventSource", FakeEventSource);
    const onOpen = vi.fn();
    const onDisconnect = vi.fn();
    const subscription = apiClient.subscribeAnalysis(accepted.id, { onEvent: vi.fn(), onOpen, onDisconnect });
    expect(onOpen).not.toHaveBeenCalled();
    FakeEventSource.instance?.onopen?.();
    expect(onOpen).toHaveBeenCalledOnce();
    FakeEventSource.instance?.onerror?.();
    expect(onDisconnect).toHaveBeenCalledOnce();
    subscription.close();
  });

  it("validates the additive history, provider, evaluation, and dataset-QA endpoints", async () => {
    const ratio = { numerator: 1, denominator: 2, value: 0.5 };
    const evaluation = {
      report_id: "evaluation-1",
      provider_id: "geoclip-global-v1",
      model_revision: "1.2.0",
      evaluation_fingerprint: "fixed-fingerprint",
      image_count: 2,
      calibration_state: "uncalibrated",
      country_top1: ratio,
      country_top5: ratio,
      region_top1: ratio,
      city_top1: ratio,
      recall_top1: { "200_km": ratio },
      mean_error_km: 10,
      median_error_km: 8,
      p95_error_km: 20,
      abstention: ratio,
      provider_failure: ratio,
      latency_median_ms: 30,
      latency_p95_ms: 50,
      uncertainty_coverage: ratio,
      geographic_distribution: { TR: 2 },
      scene_distribution: { urban: 2 },
      exclusions: {},
      limitations: [],
    };
    const qaSummary = {
      report_id: "qa-1",
      schema_version: 1,
      dataset_fingerprint: "dataset-fingerprint",
      dataset_type: "geolocation",
      created_at: "2026-07-12T10:00:00Z",
      scanned_images: 2,
      scanned_masks: 0,
      error_count: 0,
      warning_count: 0,
    };
    const systemIntelligence = {
      active_pipeline_version: "phase6c-v1",
      models: Array.from({ length: 10 }, (_, index) => ({
        model_id: `model-${index}`,
        display_name: `Model ${index}`,
        runtime_model_id: null,
        repository_url: null,
        purpose: "Bounded test status",
        enabled: index === 0,
        available: index === 0,
        status: index === 0 ? "ready" : "disabled",
        installed: index === 0,
        weights_available: index === 0,
        worker_reachable: null,
        model_loaded: index === 0,
        load_verified: index === 0,
        real_inference_verified: index === 0,
        device: index === 0 ? "cuda" : null,
        execution_mode: index === 9 ? "not_integrated" : "in_process",
        source_revision: null,
        model_revision: null,
        license: null,
        last_success_at: null,
        last_latency_ms: null,
        error_code: null,
        current_participation: index === 0 ? "primary" : index === 9 ? "not_integrated" : "disabled",
      })),
      reference_index: {
        index_id: "turkiye_megaloc_reference_index",
        enabled: true,
        status: "ready",
        reason_code: "ready",
        index_version: "index-v1",
        descriptor_version: "descriptor-v1",
        count: 6,
        sequences: 6,
        countries: 1,
        provinces: 6,
        images_per_province: { Ankara: 1 },
        source_distribution: { kartaview: 6 },
        attributions: ["© Grab and KartaView Contributors"],
        built_at: "2026-07-14T18:00:00Z",
        disk_usage_bytes: 100,
        leakage_status: "passed",
        leakage_audit: {
          status: "passed",
          audit_version: "atlaslens-leakage-audit-v1",
          audit_fingerprint: "a".repeat(64),
          source_report_sha256: "b".repeat(64),
          checked_reference_count: 6,
          descriptor_checked_count: 6,
          excluded_reference_count: 0,
          pre_index_excluded_reference_count: 1,
        },
        duplicates: 0,
        excluded: 1,
        health: "healthy",
      },
    };
    const payloads = [
      { items: [], total: 0, limit: 12, offset: 0 },
      accepted,
      { providers: [{ provider_id: "geoclip-global-v1", provider_type: "global_geolocation", mode: "primary", available: true, status: "ready", classification: "real", weights_available: true, usable: true, execution_mode: "in_process", source_revision: "reviewed-source", load_error: null, key_configured: null, budget_available: null }] },
      { models: [{ model_id: "geoclip", model_version: "1.2.0", mode: "primary", verified: true, status: "verified" }] },
      systemIntelligence,
      { reports: [evaluation] },
      evaluation,
      { reports: [qaSummary] },
      { summary: qaSummary, checks: { integrity: "passed" }, distributions: {}, issues: [], outputs: [], limitations: [] },
    ];
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify(payloads.shift()), { status: 200, headers: { "Content-Type": "application/json" } })));
    vi.stubGlobal("fetch", fetchMock);

    await expect(apiClient.listAnalyses({ limit: 12, search: "safe label" })).resolves.toMatchObject({ total: 0 });
    await expect(apiClient.rerunAnalysis(accepted.id)).resolves.toMatchObject({ id: accepted.id });
    await expect(apiClient.getProviders()).resolves.toMatchObject({ providers: [{ status: "ready" }] });
    await expect(apiClient.getModels()).resolves.toMatchObject({ models: [{ verified: true }] });
    await expect(apiClient.getSystemIntelligence()).resolves.toMatchObject({ active_pipeline_version: "phase6c-v1", reference_index: { leakage_status: "passed" } });
    await expect(apiClient.listEvaluations()).resolves.toMatchObject({ reports: [{ report_id: "evaluation-1" }] });
    await expect(apiClient.getEvaluation("evaluation-1")).resolves.toMatchObject({ evaluation_fingerprint: "fixed-fingerprint" });
    await expect(apiClient.listDatasetQaReports()).resolves.toMatchObject({ reports: [{ report_id: "qa-1" }] });
    await expect(apiClient.getDatasetQaReport("qa-1")).resolves.toMatchObject({ checks: { integrity: "passed" } });
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("search=safe+label");
    expect(String(fetchMock.mock.calls[1]?.[0])).toContain(`/analyses/${accepted.id}/rerun`);
  });
});
