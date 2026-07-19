import { vi } from "vitest";
import type { AtlasLensApiClient, CreateAnalysisInput } from "../src/api/client";
import type {
  Analysis,
  AnalysisAccepted,
  AnalysisEvent,
  Capabilities,
  CaseAdjudication,
  CaseAuditEvent,
  CaseAuditIntegrity,
  CaseEvidenceRecord,
  CaseMedia,
  InvestigationCase,
  LocationHypothesis,
  SystemIntelligenceModelCard,
  SystemIntelligenceResponse,
} from "../src/api/schemas";

export const analysisId = "123e4567-e89b-42d3-a456-426614174000";
export const caseId = "223e4567-e89b-42d3-a456-426614174001";
export const mediaId = "323e4567-e89b-42d3-a456-426614174002";
export const hypothesisId = "423e4567-e89b-42d3-a456-426614174003";
export const evidenceId = "523e4567-e89b-42d3-a456-426614174004";
export const adjudicationId = "623e4567-e89b-42d3-a456-426614174005";

export const investigationCase: InvestigationCase = {
  id: caseId,
  workspace_id: "local-default",
  title: "Sentetik kıyı doğrulama vakası",
  description: "Yalnızca açıkça sentetik test verisi.",
  purpose: "journalism",
  purpose_detail: null,
  source_context: "AtlasLens test paketi tarafından üretilen sentetik görsel.",
  status: "under_review",
  sensitivity: "conflict_related",
  authorization_attested: true,
  retention_policy: "analysis_metadata_only",
  created_at: "2026-07-15T10:00:00Z",
  updated_at: "2026-07-15T10:05:00Z",
  closed_at: null,
  created_by_actor_id: "local-analyst",
  version: 1,
  media_count: 1,
  evidence_count: 2,
  hypothesis_count: 1,
  adjudication_count: 0,
  latest_adjudication_decision: null,
};

export const caseMedia: CaseMedia = {
  id: mediaId,
  case_id: caseId,
  analysis_id: analysisId,
  media_type: "image",
  source_type: "upload",
  original_filename_display: "synthetic-coast-fixture.png",
  mime_type: "image/png",
  byte_size: 128,
  sha256: "b".repeat(64),
  captured_at: null,
  received_at: "2026-07-15T10:01:00Z",
  source_url: null,
  archive_url: null,
  source_description: "Explicitly synthetic fixture.",
  authorization_attested: true,
  storage_state: "deleted_after_analysis",
  created_at: "2026-07-15T10:01:00Z",
  analysis_status_at_link: "completed",
  analysis_created_at: "2026-07-15T10:01:00Z",
  analysis_expires_at: "2026-07-15T11:01:00Z",
  materialized_at: "2026-07-15T10:02:00Z",
};

export const caseEvidence: CaseEvidenceRecord[] = [
  {
    id: evidenceId,
    case_id: caseId,
    media_id: mediaId,
    analysis_id: analysisId,
    evidence_type: "metadata",
    provider: "synthetic-metadata-fixture",
    provider_family: "metadata",
    summary: "Sentetik fixture için güvenli meta veri özeti.",
    structured_payload: { fixture: true },
    provenance: { fixture_kind: "explicitly_synthetic" },
    observed_at: "2026-07-15T10:02:00Z",
    created_at: "2026-07-15T10:02:00Z",
    immutable_source_hash: "c".repeat(64),
  },
  {
    id: "723e4567-e89b-42d3-a456-426614174006",
    case_id: caseId,
    media_id: mediaId,
    analysis_id: analysisId,
    evidence_type: "visual_clue",
    provider: "synthetic-visual-fixture",
    provider_family: "visual",
    summary: "Sentetik görsel ipucu; gerçek OSINT sonucu değildir.",
    structured_payload: { fixture: true },
    provenance: { fixture_kind: "explicitly_synthetic" },
    observed_at: "2026-07-15T10:02:01Z",
    created_at: "2026-07-15T10:02:01Z",
    immutable_source_hash: "d".repeat(64),
  },
];

export const caseAdjudication: CaseAdjudication = {
  id: adjudicationId,
  case_id: caseId,
  hypothesis_id: hypothesisId,
  actor_id: "local-analyst",
  decision: "needs_more_evidence",
  rationale: "Sentetik fixture tek başına doğrulama için yeterli değil.",
  created_at: "2026-07-15T10:04:00Z",
  supersedes_adjudication_id: null,
};

export const caseHypothesis: LocationHypothesis = {
  id: hypothesisId,
  case_id: caseId,
  media_id: mediaId,
  analysis_id: analysisId,
  origin: "model",
  latitude: 0.25,
  longitude: 0.5,
  uncertainty_radius_m: 25000,
  country_code: null,
  region_name: "Synthetic region",
  locality_name: null,
  rank: 1,
  confidence_label: null,
  calibration_state: "uncalibrated",
  supporting_evidence_ids: [evidenceId],
  model_family_groups: ["synthetic-model-family"],
  created_at: "2026-07-15T10:03:00Z",
  supersedes_hypothesis_id: null,
  adjudications: [],
  adjudication_count: 0,
  latest_adjudication: null,
};

export const caseAuditEvents: CaseAuditEvent[] = [
  {
    id: "823e4567-e89b-42d3-a456-426614174007",
    case_id: caseId,
    sequence_number: 1,
    event_type: "case_created",
    actor_id: "local-analyst",
    actor_type: "operator",
    payload: { purpose: "journalism" },
    created_at: "2026-07-15T10:00:00Z",
    previous_event_hash: null,
    event_hash: "e".repeat(64),
  },
  {
    id: "923e4567-e89b-42d3-a456-426614174008",
    case_id: caseId,
    sequence_number: 2,
    event_type: "analysis_materialized",
    actor_id: "atlaslens-system",
    actor_type: "system",
    payload: { fixture: true },
    created_at: "2026-07-15T10:02:00Z",
    previous_event_hash: "e".repeat(64),
    event_hash: "f".repeat(64),
  },
];

export const caseAuditIntegrity: CaseAuditIntegrity = {
  case_id: caseId,
  valid: true,
  checked_event_count: 2,
  first_invalid_sequence: null,
  reason: null,
  integrity_scope: "tamper_evident_application_history",
  legally_certified_evidence: false,
};

export const capabilities: Capabilities = {
  supported_formats: ["jpeg", "png", "webp"],
  max_upload_bytes: 20 * 1024 * 1024,
  max_decoded_pixels: 40_000_000,
  enabled_analysis_modes: ["local_only", "cloud_assisted"],
  providers: {
    exif: { provider_id: "exif", enabled: true, available: true, execution_boundary: "local", reason_code: null },
    quality: { provider_id: "quality", enabled: true, available: true, execution_boundary: "local", reason_code: null },
    ocr: { provider_id: "ocr", enabled: false, available: false, execution_boundary: "local", reason_code: "not_installed" },
    cloud_vision: { provider_id: "cloud_vision", enabled: true, available: true, execution_boundary: "cloud", reason_code: null },
    global_geolocation: {
      provider_id: "geoclip-global-v1",
      enabled: true,
      available: true,
      execution_boundary: "local",
      reason_code: null,
      installed: true,
      verified: true,
      operational_status: "ready",
      model_name: "GeoCLIP",
      model_revision: "1.2.0",
      device: "cuda",
      calibration_state: "uncalibrated",
    },
  },
  retention: { keep_uploads: false, ttl_seconds: 3600, originals_deleted_after_analysis: true },
  version: "0.1.0-test",
};

export function systemIntelligenceState(firstModel: Partial<SystemIntelligenceModelCard> = {}): SystemIntelligenceResponse {
  return {
    active_pipeline_version: "legacy-v1",
    models: Array.from({ length: 10 }, (_, index) => ({
      model_id: `runtime-model-${index}`,
      display_name: `Runtime model ${index}`,
      runtime_model_id: null,
      repository_url: null,
      purpose: "Safe runtime status",
      enabled: false,
      available: false,
      status: "disabled" as const,
      installed: null,
      weights_available: null,
      worker_reachable: null,
      model_loaded: null,
      load_verified: null,
      real_inference_verified: null,
      device: null,
      execution_mode: "in_process" as const,
      source_revision: null,
      model_revision: null,
      license: null,
      last_success_at: null,
      last_latency_ms: null,
      error_code: null,
      current_participation: "disabled" as const,
      ...(index === 0 ? firstModel : {}),
    })),
    reference_index: {
      index_id: "turkiye_megaloc_reference_index",
      enabled: false,
      status: "disabled",
      reason_code: "phase6c_disabled",
      index_version: null,
      descriptor_version: null,
      count: 0,
      sequences: 0,
      countries: null,
      provinces: null,
      images_per_province: null,
      source_distribution: null,
      attributions: [],
      built_at: null,
      disk_usage_bytes: 0,
      leakage_status: "not_run",
      leakage_audit: null,
      duplicates: null,
      excluded: null,
      health: "disabled",
    },
  };
}

export const completedAnalysis: Analysis = {
  id: analysisId,
  status: "completed",
  analysis_mode: "local_only",
  created_at: "2026-07-10T10:00:00Z",
  expires_at: "2026-07-10T11:00:00Z",
  progress: { stage: "completed", percent: 100, message_key: "progress.completed" },
  image: {
    format: "jpeg",
    width: 1200,
    height: 800,
    megapixels: 0.96,
    sha256: "a".repeat(64),
    orientation_normalized: true,
    exif_present: true,
  },
  quality: { blur_score: 0.8, brightness_score: 0.7, contrast_score: 0.65, resolution_score: 0.9, warnings: [] },
  evidence: [
    {
      id: "evidence-exif",
      type: "exif",
      label: "gps_coordinates",
      display_value: "GPS metadata available",
      confidence: 0.95,
      confidence_basis: "Embedded GPS metadata parsed successfully",
      source: "exif",
      sensitive: false,
      provenance: {
        provider_id: "exif",
        provider_kind: "metadata",
        provider_version: "1.0.0",
        execution_boundary: "local",
        model_name: null,
        output_schema_version: "1.0",
      },
    },
    {
      id: "evidence-future",
      type: "future_signal",
      label: "future_safe_label",
      display_value: "Safe future evidence",
      confidence: 0.2,
      confidence_basis: "Test-only forward compatibility",
      source: "future_provider",
      sensitive: false,
      provenance: {
        provider_id: "future_provider",
        provider_kind: "future",
        provider_version: "0.1.0",
        execution_boundary: "local",
        model_name: null,
        output_schema_version: "1.0",
      },
    },
  ],
  candidates: [
    {
      id: "candidate-exif",
      rank: 1,
      center: { latitude: 41.0082, longitude: 28.9784 },
      geometry: { type: "Point", coordinates: [28.9784, 41.0082] },
      radius_km: 0.25,
      uncertainty_basis: "GPS metadata precision floor",
      confidence: 0.95,
      confidence_kind: "source_reliability",
      confidence_basis: "Valid embedded GPS metadata",
      granularity: "exact_metadata",
      country_code: "TR",
      label: "EXIF coordinate",
      source: "exif",
      evidence_ids: ["evidence-exif"],
      evidence_summary: "Embedded GPS metadata supplies the location candidate.",
      provenance: [
        {
          provider_id: "exif",
          provider_kind: "metadata",
          provider_version: "1.0.0",
          execution_boundary: "local",
          model_name: null,
          output_schema_version: "1.0",
        },
      ],
      verification_status: "metadata_only",
      verified: false,
    },
  ],
  abstention: null,
  warnings: [],
  timings_ms: { total: 24 },
  fusion_policy_version: "phase1-v1",
  result_classification: "real",
  failure: null,
};

export const noSignalAnalysis: Analysis = {
  ...completedAnalysis,
  image: { ...completedAnalysis.image!, exif_present: false },
  evidence: [],
  candidates: [],
  abstention: { abstained: true, reason_code: "no_geolocation_signal", message_key: "reason.no_geolocation_signal" },
};

export const modelAnalysis: Analysis = {
  ...completedAnalysis,
  image: { ...completedAnalysis.image!, exif_present: false },
  evidence: [
    {
      id: "evidence-model",
      type: "global_model_prediction",
      label: "evidence.global_model_prediction",
      display_value: "evidence.global_model_prediction_present",
      confidence: null,
      confidence_basis: "phase5.uncalibrated_model_score_not_confidence",
      source: "geoclip-global-v1",
      sensitive: false,
      provenance: {
        provider_id: "geoclip-global-v1",
        provider_kind: "global_geolocation",
        provider_version: "1.0.0",
        execution_boundary: "local",
        model_name: "GeoCLIP",
        output_schema_version: "1.0",
      },
    },
  ],
  candidates: [1, 2, 3].map((rank) => ({
    id: `candidate-model-${rank}`,
    rank,
    center: { latitude: 35 + rank * 3, longitude: rank === 3 ? -179.4 : 173 + rank * 2 },
    geometry: { type: "Point" as const, coordinates: [rank === 3 ? -179.4 : 173 + rank * 2, 35 + rank * 3] },
    radius_km: 750 + rank * 100,
    uncertainty_basis: "phase5.topk_geodesic_dispersion_with_750km_floor",
    confidence: null,
    confidence_kind: "uncalibrated_score" as const,
    confidence_basis: "phase5.no_confidence_before_calibration",
    granularity: "broad_area" as const,
    country_code: null,
    label: null,
    source: "geoclip-global-v1",
    evidence_ids: ["evidence-model"],
    evidence_summary: "evidence.global_model_prediction_unverified",
    provenance: [
      {
        provider_id: "geoclip-global-v1",
        provider_kind: "global_geolocation",
        provider_version: "1.0.0",
        execution_boundary: "local" as const,
        model_name: "GeoCLIP",
        output_schema_version: "1.0",
      },
    ],
    verification_status: "unverified_model" as const,
    verified: false,
    model_prediction: {
      provider_id: "geoclip-global-v1",
      model_name: "GeoCLIP",
      model_revision: "1.2.0",
      implementation_revision: "official-runtime-v1",
      device: "cuda",
      dtype: "float32",
      raw_score: 0.2 / rank,
      score_type: "uncalibrated_gallery_softmax" as const,
      normalization_method: "softmax_over_fixed_gallery" as const,
      calibration_state: "uncalibrated" as const,
      original_rank: rank,
      inference_ms: 184,
      external_transfer: false as const,
      limitations: ["fixed_gallery_limits_geographic_resolution", "model_prediction_is_unverified"],
      place_label: {
        country: "Example country",
        region: "Example region",
        city: "Example place",
        distance_to_place_km: 12.5,
        source: "GeoNames",
        dataset_version: "atlaslens-geonames-v1",
        license: "CC BY 4.0",
      },
    },
  })),
  abstention: null,
};

export function jpegFile(name = "authorized-photo.jpg"): File {
  return new File([Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0, 0, 0, 0, 0])], name, { type: "image/jpeg" });
}

export function createFakeClient(options: {
  analysis?: Analysis;
  analysisSequence?: Analysis[];
  capabilityState?: Capabilities;
  createError?: Error;
  disconnectStream?: boolean;
  clientOverrides?: Partial<AtlasLensApiClient>;
} = {}) {
  const deleteAnalysis = vi.fn(() => Promise.resolve());
  let analysisCall = 0;
  const accepted: AnalysisAccepted = {
    id: analysisId,
    status: "queued",
    status_url: `/api/v1/analyses/${analysisId}`,
    events_url: `/api/v1/analyses/${analysisId}/events`,
    delete_url: `/api/v1/analyses/${analysisId}`,
  };
  const client: AtlasLensApiClient = {
    getCapabilities: vi.fn(() => Promise.resolve(options.capabilityState ?? capabilities)),
    createAnalysis: vi.fn((input: CreateAnalysisInput): Promise<AnalysisAccepted> => {
      input.onUploadProgress?.(64);
      if (options.createError) return Promise.reject(options.createError);
      return Promise.resolve(accepted);
    }),
    listAnalyses: vi.fn(() => Promise.resolve({ items: [], total: 0, limit: 12, offset: 0 })),
    rerunAnalysis: vi.fn(() => Promise.resolve(accepted)),
    getAnalysis: vi.fn(() => {
      const sequence = options.analysisSequence;
      const value = sequence?.[Math.min(analysisCall, sequence.length - 1)] ?? options.analysis ?? completedAnalysis;
      analysisCall += 1;
      return Promise.resolve(value);
    }),
    deleteAnalysis,
    getProviders: vi.fn(() => Promise.resolve({ providers: [] })),
    getModels: vi.fn(() => Promise.resolve({ models: [] })),
    getSystemIntelligence: vi.fn(() => Promise.resolve(systemIntelligenceState())),
    listEvaluations: vi.fn(() => Promise.resolve({ reports: [] })),
    getEvaluation: vi.fn(() => Promise.reject(new Error("errors.api_not_found"))),
    listDatasetQaReports: vi.fn(() => Promise.resolve({ reports: [] })),
    getDatasetQaReport: vi.fn(() => Promise.reject(new Error("errors.api_not_found"))),
    createCase: vi.fn(() => Promise.resolve(investigationCase)),
    listCases: vi.fn(() => Promise.resolve({
      items: [],
      total: 0,
      limit: 100,
      offset: 0,
      ordering: "updated_at_desc_id_desc" as const,
    })),
    getCase: vi.fn(() => Promise.resolve(investigationCase)),
    updateCase: vi.fn(() => Promise.resolve(investigationCase)),
    createCaseMedia: vi.fn(() => Promise.resolve(caseMedia)),
    listCaseMedia: vi.fn(() => Promise.resolve({
      items: [caseMedia],
      total: 1,
      limit: 100,
      offset: 0,
      ordering: "created_at_asc_id_asc" as const,
    })),
    linkCaseAnalysis: vi.fn(() => Promise.resolve(caseMedia)),
    materializeCaseAnalysis: vi.fn(() => Promise.resolve({
      case_id: caseId,
      media_id: mediaId,
      analysis_id: analysisId,
      created: true,
      evidence_created: 2,
      hypotheses_created: 1,
      already_materialized: false,
      evidence_ids: caseEvidence.map((item) => item.id),
      hypothesis_ids: [hypothesisId],
    })),
    listCaseEvidence: vi.fn(() => Promise.resolve({
      items: caseEvidence,
      total: caseEvidence.length,
      limit: 100,
      offset: 0,
      ordering: "created_at_asc_id_asc" as const,
    })),
    listCaseHypotheses: vi.fn(() => Promise.resolve({
      items: [caseHypothesis],
      total: 1,
      limit: 100,
      offset: 0,
      ordering: "created_at_asc_id_asc" as const,
    })),
    adjudicateCaseHypothesis: vi.fn(() => Promise.resolve(caseAdjudication)),
    createOperatorHypothesis: vi.fn(() => Promise.resolve({
      hypothesis: {
        ...caseHypothesis,
        id: "a23e4567-e89b-42d3-a456-426614174009",
        origin: "operator_correction" as const,
        rank: null,
        calibration_state: "not_applicable" as const,
        model_family_groups: [],
        latest_adjudication: caseAdjudication,
        adjudications: [caseAdjudication],
        adjudication_count: 1,
      },
      adjudication: caseAdjudication,
    })),
    listCaseAuditEvents: vi.fn(() => Promise.resolve({
      items: caseAuditEvents,
      total: caseAuditEvents.length,
      limit: 100,
      offset: 0,
      ordering: "sequence_number_asc" as const,
      integrity_scope: "tamper_evident_application_history" as const,
    })),
    getCaseAuditIntegrity: vi.fn(() => Promise.resolve(caseAuditIntegrity)),
    subscribeAnalysis: vi.fn((_id: string, handlers: { onEvent: (event: AnalysisEvent) => void; onOpen?: () => void; onDisconnect: () => void }) => {
      queueMicrotask(() => handlers.onOpen?.());
      if (options.disconnectStream) queueMicrotask(handlers.onDisconnect);
      return { close: vi.fn() };
    }),
  };
  Object.assign(client, options.clientOverrides);
  return { client, deleteAnalysis };
}
