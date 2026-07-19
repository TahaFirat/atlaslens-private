import { describe, expect, it } from "vitest";
import {
  analysisSchema,
  capabilitiesSchema,
  caseHypothesisPageSchema,
  geoClipClusterSummarySchema,
  phase4AssessmentSchema,
  phase5bAssessmentSchema,
  providerStatusResponseSchema,
  sceneSegmentationSummarySchema,
  uncalibratedConfidenceSummarySchema,
} from "../src/api/schemas";
import { capabilities, completedAnalysis, modelAnalysis } from "./fixtures";

describe("runtime contract validation", () => {
  it("accepts the API hypothesis-page ordering contract", () => {
    const page = {
      items: [],
      total: 0,
      limit: 100,
      offset: 0,
      ordering: "created_at_asc_id_asc",
    };

    expect(caseHypothesisPageSchema.safeParse(page).success).toBe(true);
    expect(caseHypothesisPageSchema.safeParse({
      ...page,
      ordering: "rank_asc_created_at_asc_id_asc",
    }).success).toBe(false);
  });

  it("accepts bounded additive Phase 6B provider capabilities", () => {
    expect(capabilitiesSchema.safeParse({
      ...capabilities,
      providers: {
        ...capabilities.providers,
        osv5m: {
          provider_id: "osv5m",
          enabled: true,
          available: false,
          execution_boundary: "local",
          reason_code: "worker_not_installed",
          weights_available: false,
          usable: false,
          execution_mode: "isolated_worker",
          source_revision: "reviewed-source-revision",
          load_error: null,
          key_configured: null,
          budget_available: null,
        },
      },
    }).success).toBe(true);
  });

  it("accepts each additive Phase 6B worker lifecycle state", () => {
    const states = [
      "dependencies_installed",
      "weights_prepared",
      "worker_unreachable",
      "model_load_failed",
      "inference_not_verified",
    ] as const;

    for (const status of states) {
      expect(capabilitiesSchema.safeParse({
        ...capabilities,
        providers: {
          ...capabilities.providers,
          osv5m: {
            provider_id: "osv5m",
            enabled: true,
            available: false,
            execution_boundary: "local",
            operational_status: status,
          },
        },
      }).success).toBe(true);
      expect(providerStatusResponseSchema.safeParse({
        providers: [{
          provider_id: "osv5m",
          provider_type: "global_geolocation",
          mode: "primary",
          available: false,
          status,
          classification: "real",
        }],
      }).success).toBe(true);
    }
  });

  it("accepts a contract-shaped analysis and rejects invalid candidate uncertainty", () => {
    expect(analysisSchema.safeParse(completedAnalysis).success).toBe(true);
    expect(analysisSchema.safeParse({
      ...completedAnalysis,
      provider_comparisons: [{
        provider_id: "fixture-shadow",
        mode: "shadow",
        status: "succeeded",
        runtime_ms: 4,
        candidate_overlap: 50,
        ranking_impact: "none",
      }],
    }).success).toBe(true);
    const invalid = {
      ...completedAnalysis,
      candidates: [{ ...completedAnalysis.candidates[0], radius_km: 0 }],
    };
    expect(analysisSchema.safeParse(invalid).success).toBe(false);
  });

  it("accepts null confidence with bounded uncalibrated model diagnostics", () => {
    const parsed = analysisSchema.safeParse(modelAnalysis);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.evidence[0]?.confidence).toBeNull();
      expect(parsed.data.candidates[0]?.confidence).toBeNull();
    }
    const invalid = {
      ...modelAnalysis,
      candidates: [{
        ...modelAnalysis.candidates[0],
        model_prediction: { ...modelAnalysis.candidates[0]?.model_prediction, raw_score: 1.1 },
      }],
    };
    expect(analysisSchema.safeParse(invalid).success).toBe(false);
  });

  it("accepts uncalibrated Phase 4 detail and rejects probability/path leakage", () => {
    const assessment = {
      classification: "geometry_supported",
      relative_rank_score: 0.71,
      score_semantics: "uncalibrated_relative_rank",
      reranker_version: "phase4-v1",
      score_breakdown: [
        {
          feature: "geometry_support",
          raw_value: 0.8,
          weight: 0.2,
          contribution: 0.16,
          reason_code: "rerank.geometry_support",
        },
      ],
      source_diversity: 2,
      contributing_retrieval_hit_ids: ["hit-a", "hit-b"],
      map_observations: [],
      geometry_results: [],
      contradictions: [],
      reference_attributions: [],
      limitations: ["uncalibrated"],
    };

    expect(phase4AssessmentSchema.safeParse(assessment).success).toBe(true);
    expect(
      phase4AssessmentSchema.safeParse({
        ...assessment,
        score_semantics: "calibrated_probability",
      }).success,
    ).toBe(false);
    expect(
      phase4AssessmentSchema.safeParse({
        ...assessment,
        absolute_path: "C:/private/reference.jpg",
      }).success,
    ).toBe(false);
  });

  it("accepts bounded Phase 5B evidence and rejects probability or private-field leakage", () => {
    const assessment = {
      classification: "multi_source_supported",
      relative_rank_score: 0.72,
      score_semantics: "uncalibrated_relative_rank",
      reranker_version: "phase5b-v1",
      score_breakdown: [
        {
          feature: "place_support",
          raw_value: 0.8,
          weight: 0.2,
          contribution: 0.16,
          reason_code: "phase5b.feature.place_support",
        },
      ],
      provider_diversity: 2,
      source_diversity: 2,
      place_matches: [
        {
          matched_entity: "sensitive-input-token",
          normalized_name: "Istanbul",
          country_code: "TR",
          region: "Istanbul",
          center: { latitude: 41.0082, longitude: 28.9784 },
          match_type: "city",
          text_similarity: 0.8,
          ambiguity_count: 1,
          evidence_strength: 0.75,
          source: "GeoNames",
          dataset_version: "fixture-v1",
          license: "CC BY 4.0",
        },
      ],
      retrieval_matches: [],
      map_observations: [],
      supports: ["phase5b.support.place"],
      contradictions: [],
      movement_reasons: ["phase5b.rank_increased.evidence_support"],
      limitations: ["phase5b.relative_rank_is_not_probability"],
    } as const;

    expect(phase5bAssessmentSchema.safeParse(assessment).success).toBe(true);
    expect(phase5bAssessmentSchema.safeParse({ ...assessment, reranker_version: "phase6a-v1" }).success).toBe(true);
    expect(phase5bAssessmentSchema.safeParse({ ...assessment, reranker_version: "phase6b-v1" }).success).toBe(true);
    expect(
      phase5bAssessmentSchema.safeParse({ ...assessment, score_semantics: "calibrated_probability" }).success,
    ).toBe(false);
    expect(
      phase5bAssessmentSchema.safeParse({ ...assessment, raw_ocr: "private payload" }).success,
    ).toBe(false);
  });

  it("accepts bounded Phase 6A scene, cluster, and explicitly uncalibrated confidence summaries", () => {
    const scene = {
      status: "completed",
      provider: "atlaslens-segformer-b2-v4",
      device: "cpu",
      inference_ms: 18.4,
      image_width: 640,
      image_height: 480,
      semantic_label_names_available: true,
      dominant_classes: [{ class_id: 13, class_name: "Road", pixel_ratio: 0.341, percentage: 34.1 }],
      scene_groups: [{ name: "road_surface", pixel_ratio: 0.341, percentage: 34.1 }],
      scene_tags: [{ name: "road_heavy", strength: 0.74, strength_semantics: "deterministic_heuristic_not_probability", reason: "Road pixels exceed the reviewed descriptive threshold" }],
      warnings: [],
    };
    const cluster = {
      cluster_id: "geoclip-cluster-1",
      source: "geoclip",
      member_count: 3,
      member_ranks: [1, 3, 8],
      max_raw_similarity: 0.18,
      mean_raw_similarity: 0.12,
      raw_score_type: "uncalibrated_gallery_softmax",
      cluster_support: 0.71,
      score_semantics: "uncalibrated_relative_rank",
    };
    const confidence = { label: "medium", score: null, calibrated: false, basis: ["dense_candidate_cluster"] };

    expect(sceneSegmentationSummarySchema.safeParse(scene).success).toBe(true);
    expect(geoClipClusterSummarySchema.safeParse(cluster).success).toBe(true);
    expect(uncalibratedConfidenceSummarySchema.safeParse(confidence).success).toBe(true);
    expect(analysisSchema.safeParse({
      ...completedAnalysis,
      scene_analysis: scene,
      candidates: [{
        ...completedAnalysis.candidates[0],
        geoclip_cluster: cluster,
        confidence_assessment: confidence,
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
    }).success).toBe(true);
  });

  it("rejects fake Phase 6A probability, inconsistent clusters, and inline mask payloads", () => {
    expect(uncalibratedConfidenceSummarySchema.safeParse({ label: "high", score: 0.91, calibrated: true, basis: ["fixture"] }).success).toBe(false);
    expect(geoClipClusterSummarySchema.safeParse({
      cluster_id: "cluster-1",
      source: "geoclip",
      member_count: 2,
      member_ranks: [1],
      max_raw_similarity: 0.2,
      mean_raw_similarity: 0.1,
      raw_score_type: "relative",
      cluster_support: 0.4,
      score_semantics: "uncalibrated_relative_rank",
    }).success).toBe(false);
    expect(sceneSegmentationSummarySchema.safeParse({
      status: "completed",
      provider: "fixture",
      device: "cpu",
      inference_ms: 1,
      image_width: 2,
      image_height: 2,
      semantic_label_names_available: false,
      dominant_classes: [],
      scene_groups: [],
      scene_tags: [],
      warnings: [],
      mask: [[1, 2], [3, 4]],
    }).success).toBe(false);
  });

  it("accepts bounded optional Phase 6B evidence while old responses remain valid", () => {
    const phase6b = {
      ...completedAnalysis,
      model_predictions: {
        geoclip: {
          provider: "geoclip",
          model_id: "geoclip-1.2.0",
          model_revision: "reviewed-revision",
          source_family: "mp16_family",
          status: "completed",
          device: "cpu",
          duration_ms: 25,
          score_semantics: "similarity",
          candidates: [{ candidate_id: "geoclip-1", latitude: 41.0, longitude: 29.0, raw_score: null, provider_rank: 1, sample_support: 1 }],
          warnings: [],
        },
      },
      phase5b_diagnostics: {
        reranker_version: "phase6b-v1",
        providers: [],
        reference_index: null,
        partial_failures: [],
      },
      fusion: {
        version: "phase6b-v1",
        source_families: ["mp16_family"],
        agreement_summary: { provider_count: 1, independent_family_count: 1, same_family_duplicate_support: 0, geographic_disagreement: false, ocr_agreement: true, ocr_contradiction: false },
        candidate_clusters: [{
          cluster_id: "phase6b-cluster-1",
          latitude: 41.0,
          longitude: 29.0,
          radius_km: 100,
          relative_rank_score: 0.4,
          score_semantics: "uncalibrated_relative_rank_not_probability",
          provider_count: 1,
          independent_family_count: 1,
          same_family_duplicate_support: 0,
          spread_km: 0,
          members: [{ provider: "geoclip", model_id: "geoclip-1.2.0", candidate_id: "geoclip-1", source_family: "mp16_family", provider_rank: 1, sample_support: 1 }],
          contributions: [{ name: "within_provider_rank", raw_value: 1, weight: 0.2, contribution: 0.2, reason: "Top-ranked GeoCLIP cluster" }],
          ocr_agreement: true,
          ocr_contradiction: false,
        }],
      },
      ocr: {
        provider: "paddleocr",
        status: "completed",
        fallback_used: false,
        detections: [{ redacted_text: "İstanbul Caddesi", confidence: 0.8, script: "latin", provider: "paddleocr", profile: "standard" }],
        place_evidence: [{ matched_entity: "İstanbul", normalized_name: "İstanbul", country_code: "TR", region: "Marmara", center: { latitude: 41.0082, longitude: 28.9784 }, match_type: "city", text_similarity: 0.9, ambiguity_count: 1, evidence_strength: 0.7, source: "reviewed-gazetteer", dataset_version: "fixture-v1", license: "test-only" }],
        reason_code: null,
      },
      cloud_assist: {
        allowed: false,
        triggered: false,
        provider: "openai",
        model: "gpt-5.6-luna",
        prompt_version: "openai-geo-review-v1",
        reason_code: "consent_false",
        status: "skipped",
        cache_hit: false,
        estimated_cost_usd: null,
        budget: { calls_today: 0, estimated_month_spend_usd: 0, configured_monthly_budget_usd: 4.5, remaining_budget_usd: 4.5 },
        review: null,
        warnings: [],
      },
    };

    expect(analysisSchema.safeParse(completedAnalysis).success).toBe(true);
    const parsed = analysisSchema.safeParse(phase6b);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.model_predictions?.geoclip?.candidates[0]?.raw_score).toBeNull();
    }
    expect(analysisSchema.safeParse({ ...phase6b, cloud_assist: { ...phase6b.cloud_assist, api_key: "must-not-escape" } }).success).toBe(false);
  });
});
