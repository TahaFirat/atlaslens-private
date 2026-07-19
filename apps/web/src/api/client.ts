import {
  acceptedSchema,
  analysisHistoryPageSchema,
  analysisSchema,
  capabilitiesSchema,
  caseAdjudicationSchema,
  caseAuditEventPageSchema,
  caseAuditIntegritySchema,
  caseEvidencePageSchema,
  caseHypothesisPageSchema,
  caseMaterializationResultSchema,
  caseMediaPageSchema,
  caseMediaSchema,
  casePageSchema,
  datasetQaReportListSchema,
  datasetQaReportSchema,
  evaluationReportListSchema,
  evaluationReportSummarySchema,
  eventSchema,
  investigationCaseSchema,
  modelStatusResponseSchema,
  operatorHypothesisResultSchema,
  problemSchema,
  providerStatusResponseSchema,
  systemIntelligenceResponseSchema,
  type Analysis,
  type AnalysisAccepted,
  type AnalysisEvent,
  type AnalysisHistoryPage,
  type AnalysisMode,
  type AdjudicationDecision,
  type Capabilities,
  type CaseAdjudication,
  type CaseAuditEventPage,
  type CaseAuditIntegrity,
  type CaseEvidencePage,
  type CaseHypothesisPage,
  type CaseMaterializationResult,
  type CaseMedia,
  type CaseMediaPage,
  type CasePage,
  type CasePurpose,
  type CaseSensitivity,
  type CaseStatus,
  type DatasetQAReport,
  type DatasetQAReportList,
  type EvaluationReportList,
  type EvaluationReportSummary,
  type InvestigationCase,
  type ModelStatusResponse,
  type OperatorHypothesisResult,
  type ProblemDetails,
  type ProviderStatusResponse,
  type SystemIntelligenceResponse,
} from "./schemas";
import type { ZodType } from "zod";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export type ApiErrorKind =
  | "backend_unreachable"
  | "api_not_found"
  | "client_error"
  | "server_error"
  | "invalid_response"
  | "request_setup"
  | "upload_cancelled"
  | "request_error";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly problem?: ProblemDetails,
    readonly kind: ApiErrorKind = "request_error",
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function parseProblem(response: Response): Promise<ProblemDetails | undefined> {
  try {
    const parsed = problemSchema.safeParse(await response.json());
    return parsed.success ? parsed.data : undefined;
  } catch {
    return undefined;
  }
}

async function assertResponse(response: Response): Promise<Response> {
  if (response.ok) return response;
  const problem = await parseProblem(response);
  throw httpError(response.status, problem);
}

function httpError(status: number, problem?: ProblemDetails): ApiError {
  if (problem?.code === "backend_unreachable" || status === 502) {
    return new ApiError("errors.backend_unreachable", problem, "backend_unreachable", status);
  }
  if (status === 404) {
    return new ApiError(problem?.message_key ?? "errors.api_not_found", problem, "api_not_found", status);
  }
  if (status >= 500) {
    return new ApiError(problem?.message_key ?? "errors.server_error", problem, "server_error", status);
  }
  return new ApiError(problem?.message_key ?? "errors.request_error", problem, "client_error", status);
}

async function safeFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch {
    throw new ApiError("errors.backend_unreachable", undefined, "backend_unreachable");
  }
}

async function parsedJson<T>(response: Response, schema: ZodType<T>): Promise<T> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new ApiError("errors.invalidResponse", undefined, "invalid_response", response.status);
  }
  const parsed = schema.safeParse(payload);
  if (!parsed.success) throw new ApiError("errors.invalidResponse", undefined, "invalid_response", response.status);
  return parsed.data;
}

function absoluteUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

function idempotencyKey(): string | undefined {
  try {
    const cryptoApi = globalThis.crypto;
    if (!cryptoApi) return undefined;
    if (typeof cryptoApi.randomUUID === "function") return cryptoApi.randomUUID();
    if (typeof cryptoApi.getRandomValues !== "function") return undefined;

    const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
    bytes[6] = ((bytes[6] ?? 0) & 0x0f) | 0x40;
    bytes[8] = ((bytes[8] ?? 0) & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
  } catch {
    return undefined;
  }
}

function setupError(): ApiError {
  return new ApiError("errors.request_setup", undefined, "request_setup");
}

export interface CreateAnalysisInput {
  file: File;
  mode: AnalysisMode;
  cloudConsent: boolean;
  allowCloudAssist?: boolean;
  authorizationAcknowledged: boolean;
  onUploadProgress?: (percent: number) => void;
}

export interface AnalysisSubscription {
  close: () => void;
}

export interface AnalysisListFilters {
  limit?: number;
  offset?: number;
  status?: "queued" | "processing" | "completed" | "failed" | "deleted";
  classification?: "real" | "simulated";
  provider?: string;
  search?: string;
  createdFrom?: string;
  createdTo?: string;
}

export interface CaseListFilters {
  limit?: number;
  offset?: number;
  status?: CaseStatus;
  purpose?: CasePurpose;
  sensitivity?: CaseSensitivity;
}

export interface CreateCaseInput {
  title: string;
  description?: string | null;
  purpose: CasePurpose;
  purpose_detail?: string | null;
  source_context: string;
  sensitivity: CaseSensitivity;
  authorization_attested: true;
  retention_policy: string;
  created_by_actor_id: string;
}

export interface UpdateCaseInput {
  actor_id: string;
  expected_version: number;
  title?: string;
  description?: string | null;
  purpose?: CasePurpose;
  purpose_detail?: string | null;
  source_context?: string;
  status?: CaseStatus;
  sensitivity?: CaseSensitivity;
  retention_policy?: string;
}

export interface CreateCaseMediaInput {
  actor_id: string;
  media_type: "image";
  source_type: "upload" | "source_url" | "external_archive" | "other";
  original_filename_display: string;
  mime_type: "image/jpeg" | "image/png" | "image/webp";
  byte_size: number;
  sha256: string;
  captured_at?: string | null;
  source_url?: string | null;
  archive_url?: string | null;
  source_description?: string | null;
  authorization_attested: true;
  storage_state: "ephemeral" | "deleted_after_analysis" | "unavailable" | "externally_managed";
}

export interface CreateAdjudicationInput {
  actor_id: string;
  decision: AdjudicationDecision;
  rationale: string;
  supersedes_adjudication_id?: string | null;
}

export interface CreateOperatorHypothesisInput {
  media_id: string;
  analysis_id?: string | null;
  supersedes_hypothesis_id?: string | null;
  actor_id: string;
  latitude: number;
  longitude: number;
  uncertainty_radius_m: number;
  country_code?: string | null;
  region_name?: string | null;
  locality_name?: string | null;
  supporting_evidence_ids: string[];
  rationale: string;
  decision: AdjudicationDecision;
}

export interface AtlasLensApiClient {
  getCapabilities: () => Promise<Capabilities>;
  createAnalysis: (input: CreateAnalysisInput) => Promise<AnalysisAccepted>;
  listAnalyses: (filters?: AnalysisListFilters) => Promise<AnalysisHistoryPage>;
  rerunAnalysis: (id: string) => Promise<AnalysisAccepted>;
  getAnalysis: (id: string) => Promise<Analysis>;
  deleteAnalysis: (id: string) => Promise<void>;
  getProviders: () => Promise<ProviderStatusResponse>;
  getModels: () => Promise<ModelStatusResponse>;
  getSystemIntelligence: () => Promise<SystemIntelligenceResponse>;
  listEvaluations: () => Promise<EvaluationReportList>;
  getEvaluation: (id: string) => Promise<EvaluationReportSummary>;
  listDatasetQaReports: () => Promise<DatasetQAReportList>;
  getDatasetQaReport: (id: string) => Promise<DatasetQAReport>;
  createCase: (input: CreateCaseInput) => Promise<InvestigationCase>;
  listCases: (filters?: CaseListFilters) => Promise<CasePage>;
  getCase: (id: string) => Promise<InvestigationCase>;
  updateCase: (id: string, input: UpdateCaseInput) => Promise<InvestigationCase>;
  createCaseMedia: (caseId: string, input: CreateCaseMediaInput) => Promise<CaseMedia>;
  listCaseMedia: (caseId: string) => Promise<CaseMediaPage>;
  linkCaseAnalysis: (
    caseId: string,
    analysisId: string,
    mediaId: string,
    actorId: string,
  ) => Promise<CaseMedia>;
  materializeCaseAnalysis: (
    caseId: string,
    analysisId: string,
    mediaId: string,
    actorId: string,
  ) => Promise<CaseMaterializationResult>;
  listCaseEvidence: (caseId: string) => Promise<CaseEvidencePage>;
  listCaseHypotheses: (caseId: string) => Promise<CaseHypothesisPage>;
  adjudicateCaseHypothesis: (
    caseId: string,
    hypothesisId: string,
    input: CreateAdjudicationInput,
  ) => Promise<CaseAdjudication>;
  createOperatorHypothesis: (
    caseId: string,
    input: CreateOperatorHypothesisInput,
  ) => Promise<OperatorHypothesisResult>;
  listCaseAuditEvents: (caseId: string) => Promise<CaseAuditEventPage>;
  getCaseAuditIntegrity: (caseId: string) => Promise<CaseAuditIntegrity>;
  subscribeAnalysis: (
    id: string,
    handlers: { onEvent: (event: AnalysisEvent) => void; onOpen?: () => void; onDisconnect: () => void },
  ) => AnalysisSubscription;
}

function createAnalysis(input: CreateAnalysisInput): Promise<AnalysisAccepted> {
  return new Promise((resolve, reject) => {
    try {
      if (!(input.file instanceof File)) {
        reject(new ApiError("errors.file_required", undefined, "request_error"));
        return;
      }

      const form = new FormData();
      form.set("image", input.file);
      form.set("analysis_mode", input.mode);
      form.set("cloud_processing_consent", String(input.cloudConsent));
      form.set("allow_cloud_assist", String(input.allowCloudAssist ?? false));
      form.set("authorization_acknowledged", String(input.authorizationAcknowledged));

      const request = new XMLHttpRequest();
      request.open("POST", absoluteUrl("/api/v1/analyses"));
      request.setRequestHeader("Accept", "application/json, application/problem+json");
      const key = idempotencyKey();
      if (key) request.setRequestHeader("Idempotency-Key", key);
      request.upload.onprogress = (event) => {
        if (event.lengthComputable) input.onUploadProgress?.(Math.round((event.loaded / event.total) * 100));
      };
      request.onerror = () => reject(new ApiError("errors.backend_unreachable", undefined, "backend_unreachable"));
      request.onabort = () => reject(new ApiError("errors.uploadCancelled", undefined, "upload_cancelled"));
      request.onload = () => {
        let payload: unknown;
        try {
          payload = JSON.parse(request.responseText) as unknown;
        } catch {
          reject(new ApiError("errors.invalidResponse", undefined, "invalid_response", request.status));
          return;
        }
        if (request.status < 200 || request.status >= 300) {
          const problem = problemSchema.safeParse(payload);
          reject(httpError(request.status, problem.success ? problem.data : undefined));
          return;
        }
        const accepted = acceptedSchema.safeParse(payload);
        if (!accepted.success) {
          reject(new ApiError("errors.invalidResponse", undefined, "invalid_response", request.status));
          return;
        }
        input.onUploadProgress?.(100);
        resolve(accepted.data);
      };
      request.send(form);
    } catch {
      reject(setupError());
    }
  });
}

async function caseJson<T>(
  path: string,
  schema: ZodType<T>,
  init: RequestInit = {},
  useIdempotency = false,
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json, application/problem+json");
  if (init.body !== undefined) headers.set("Content-Type", "application/json");
  if (useIdempotency) {
    const key = idempotencyKey();
    if (key) headers.set("Idempotency-Key", key);
  }
  const response = await assertResponse(
    await safeFetch(absoluteUrl(path), { ...init, headers, cache: "no-store" }),
  );
  return parsedJson(response, schema);
}

function pagedCasePath(path: string, filters: CaseListFilters): string {
  const query = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined) query.set(key, String(value));
  });
  return query.size ? path + "?" + query.toString() : path;
}

export const apiClient: AtlasLensApiClient = {
  async getCapabilities() {
    const response = await assertResponse(await safeFetch(absoluteUrl("/api/v1/capabilities"), { headers: { Accept: "application/json" } }));
    return parsedJson(response, capabilitiesSchema);
  },

  createAnalysis,

  async listAnalyses(filters = {}) {
    const query = new URLSearchParams();
    if (filters.limit !== undefined) query.set("limit", String(filters.limit));
    if (filters.offset !== undefined) query.set("offset", String(filters.offset));
    if (filters.status) query.set("status", filters.status);
    if (filters.classification) query.set("classification", filters.classification);
    if (filters.provider) query.set("provider", filters.provider);
    if (filters.search) query.set("search", filters.search);
    if (filters.createdFrom) query.set("created_from", filters.createdFrom);
    if (filters.createdTo) query.set("created_to", filters.createdTo);
    const suffix = query.size ? `?${query.toString()}` : "";
    const response = await assertResponse(
      await safeFetch(absoluteUrl(`/api/v1/analyses${suffix}`), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      }),
    );
    return parsedJson(response, analysisHistoryPageSchema);
  },

  async rerunAnalysis(id) {
    const response = await assertResponse(
      await safeFetch(absoluteUrl(`/api/v1/analyses/${encodeURIComponent(id)}/rerun`), {
        method: "POST",
        headers: { Accept: "application/json" },
      }),
    );
    return parsedJson(response, acceptedSchema);
  },

  async getAnalysis(id) {
    const response = await assertResponse(
      await safeFetch(absoluteUrl(`/api/v1/analyses/${encodeURIComponent(id)}`), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      }),
    );
    return parsedJson(response, analysisSchema);
  },

  async deleteAnalysis(id) {
    await assertResponse(
      await safeFetch(absoluteUrl(`/api/v1/analyses/${encodeURIComponent(id)}`), {
        method: "DELETE",
        headers: { Accept: "application/json" },
      }),
    );
  },

  async getProviders() {
    const response = await assertResponse(
      await safeFetch(absoluteUrl("/api/v1/providers"), { headers: { Accept: "application/json" }, cache: "no-store" }),
    );
    return parsedJson(response, providerStatusResponseSchema);
  },

  async getModels() {
    const response = await assertResponse(
      await safeFetch(absoluteUrl("/api/v1/models"), { headers: { Accept: "application/json" }, cache: "no-store" }),
    );
    return parsedJson(response, modelStatusResponseSchema);
  },

  async getSystemIntelligence() {
    const response = await assertResponse(
      await safeFetch(absoluteUrl("/api/v1/system-intelligence"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      }),
    );
    return parsedJson(response, systemIntelligenceResponseSchema);
  },

  async listEvaluations() {
    const response = await assertResponse(
      await safeFetch(absoluteUrl("/api/v1/evaluations"), { headers: { Accept: "application/json" }, cache: "no-store" }),
    );
    return parsedJson(response, evaluationReportListSchema);
  },

  async getEvaluation(id) {
    const response = await assertResponse(
      await safeFetch(absoluteUrl(`/api/v1/evaluations/${encodeURIComponent(id)}`), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      }),
    );
    return parsedJson(response, evaluationReportSummarySchema);
  },

  async listDatasetQaReports() {
    const response = await assertResponse(
      await safeFetch(absoluteUrl("/api/v1/datasets/qa"), { headers: { Accept: "application/json" }, cache: "no-store" }),
    );
    return parsedJson(response, datasetQaReportListSchema);
  },

  async getDatasetQaReport(id) {
    const response = await assertResponse(
      await safeFetch(absoluteUrl(`/api/v1/datasets/qa/${encodeURIComponent(id)}`), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      }),
    );
    return parsedJson(response, datasetQaReportSchema);
  },

  async createCase(input) {
    return caseJson(
      "/api/v1/cases",
      investigationCaseSchema,
      { method: "POST", body: JSON.stringify(input) },
      true,
    );
  },

  async listCases(filters = {}) {
    return caseJson(pagedCasePath("/api/v1/cases", filters), casePageSchema);
  },

  async getCase(id) {
    return caseJson("/api/v1/cases/" + encodeURIComponent(id), investigationCaseSchema);
  },

  async updateCase(id, input) {
    return caseJson("/api/v1/cases/" + encodeURIComponent(id), investigationCaseSchema, {
      method: "PATCH",
      body: JSON.stringify(input),
    });
  },

  async createCaseMedia(caseId, input) {
    return caseJson(
      "/api/v1/cases/" + encodeURIComponent(caseId) + "/media",
      caseMediaSchema,
      { method: "POST", body: JSON.stringify(input) },
      true,
    );
  },

  async listCaseMedia(caseId) {
    return caseJson(
      "/api/v1/cases/" + encodeURIComponent(caseId) + "/media?limit=100&offset=0",
      caseMediaPageSchema,
    );
  },

  async linkCaseAnalysis(caseId, analysisId, mediaId, actorId) {
    const path = "/api/v1/cases/" + encodeURIComponent(caseId)
      + "/analyses/" + encodeURIComponent(analysisId) + "/link";
    return caseJson(
      path,
      caseMediaSchema,
      { method: "POST", body: JSON.stringify({ media_id: mediaId, actor_id: actorId }) },
      true,
    );
  },

  async materializeCaseAnalysis(caseId, analysisId, mediaId, actorId) {
    const path = "/api/v1/cases/" + encodeURIComponent(caseId)
      + "/analyses/" + encodeURIComponent(analysisId) + "/materialize-evidence";
    return caseJson(
      path,
      caseMaterializationResultSchema,
      { method: "POST", body: JSON.stringify({ media_id: mediaId, actor_id: actorId }) },
      true,
    );
  },

  async listCaseEvidence(caseId) {
    return caseJson(
      "/api/v1/cases/" + encodeURIComponent(caseId) + "/evidence?limit=100&offset=0",
      caseEvidencePageSchema,
    );
  },

  async listCaseHypotheses(caseId) {
    return caseJson(
      "/api/v1/cases/" + encodeURIComponent(caseId) + "/hypotheses?limit=100&offset=0",
      caseHypothesisPageSchema,
    );
  },

  async adjudicateCaseHypothesis(caseId, hypothesisId, input) {
    const path = "/api/v1/cases/" + encodeURIComponent(caseId)
      + "/hypotheses/" + encodeURIComponent(hypothesisId) + "/adjudications";
    return caseJson(
      path,
      caseAdjudicationSchema,
      { method: "POST", body: JSON.stringify(input) },
      true,
    );
  },

  async createOperatorHypothesis(caseId, input) {
    return caseJson(
      "/api/v1/cases/" + encodeURIComponent(caseId) + "/operator-hypotheses",
      operatorHypothesisResultSchema,
      { method: "POST", body: JSON.stringify(input) },
      true,
    );
  },

  async listCaseAuditEvents(caseId) {
    return caseJson(
      "/api/v1/cases/" + encodeURIComponent(caseId) + "/audit-events?limit=100&offset=0",
      caseAuditEventPageSchema,
    );
  },

  async getCaseAuditIntegrity(caseId) {
    return caseJson(
      "/api/v1/cases/" + encodeURIComponent(caseId) + "/audit-integrity",
      caseAuditIntegritySchema,
    );
  },

  subscribeAnalysis(id, handlers) {
    let source: EventSource;
    try {
      source = new EventSource(absoluteUrl(`/api/v1/analyses/${encodeURIComponent(id)}/events`));
    } catch {
      handlers.onDisconnect();
      return { close() {} };
    }
    const eventNames = ["progress", "heartbeat", "completed", "failed", "deleted"] as const;
    const listeners = eventNames.map((name) => {
      const listener = (raw: MessageEvent<string>) => {
        try {
          const event = eventSchema.safeParse(JSON.parse(raw.data) as unknown);
          if (event.success) {
            handlers.onEvent(event.data);
            if (["completed", "failed", "deleted"].includes(event.data.event_type)) source.close();
          }
        } catch {
          // Ignore malformed events; polling remains the source-of-truth fallback.
        }
      };
      source.addEventListener(name, listener as EventListener);
      return [name, listener] as const;
    });
    source.onopen = () => handlers.onOpen?.();
    source.onerror = () => {
      handlers.onDisconnect();
      source.close();
    };
    return {
      close() {
        listeners.forEach(([name, listener]) => source.removeEventListener(name, listener as EventListener));
        source.close();
      },
    };
  },
};
