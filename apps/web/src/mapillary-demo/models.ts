import { z } from "zod";
import type { operations } from "../api/generated";

const sha256 = z.string().regex(/^[0-9a-f]{64}$/);
type GeneratedCandidate = operations["queryMapillaryDemo"]["responses"][200]["content"]["application/json"]["candidates"][number];
type GeneratedQuery = operations["queryMapillaryDemo"]["responses"][200]["content"]["application/json"];
type GeneratedStatus = operations["getMapillaryDemoStatus"]["responses"][200]["content"]["application/json"];

function contractSchema<TContract>() {
  return <TSchema extends z.ZodType<TContract>>(schema: TSchema) => schema;
}

export const mapillaryDemoCandidateSchema = contractSchema<GeneratedCandidate>()(z.object({
  rank: z.number().int().positive().max(20),
  cosine_similarity: z.number().min(-1).max(1),
  cosine_distance: z.number().min(0).max(2),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
  mapillary_image_id: z.string().min(1).max(128),
  contributor: z.string().min(1).max(240).nullable(),
  source_url: z.string().url(),
  license_identifier: z.literal("CC-BY-SA-4.0"),
  license_url: z.literal("https://creativecommons.org/licenses/by-sa/4.0/"),
  capture_date: z.string().min(1).max(40).nullable(),
  confidence: z.null(),
  confidence_semantics: z.literal("uncalibrated_unavailable"),
  similarity_semantics: z.literal("cosine_similarity_not_confidence"),
  uncertainty_radius_m: z.number().positive(),
  uncertainty_semantics: z.literal("presentation_radius_not_accuracy_or_probability"),
  experimental_status: z.literal("private_technical_demo_not_production"),
}).strict());

export const mapillaryDemoStatusSchema = contractSchema<GeneratedStatus>()(z.object({
  state: z.enum(["disabled", "not_ready", "active"]),
  enabled: z.boolean(),
  available: z.boolean(),
  reason_code: z.string().min(1).max(120).nullable(),
  city: z.string().min(1).max(160).nullable(),
  image_count: z.number().int().nonnegative(),
  model_id: z.literal("gberton/MegaLoc"),
  model_version: z.string().min(1).max(200).nullable(),
  model_artifact_sha256: sha256.nullable(),
  descriptor_dimension: z.literal(8_448),
  index_version: z.string().min(1).max(200).nullable(),
  index_checksum: sha256.nullable(),
  attribution_url: z.literal("https://www.mapillary.com/"),
  license_identifier: z.literal("CC-BY-SA-4.0"),
  license_url: z.literal("https://creativecommons.org/licenses/by-sa/4.0/"),
  experimental_status: z.literal("private_technical_demo_not_production"),
  coverage_status: z.literal("limited_pilot"),
  coverage_label: z.literal("Ankara reference pilot"),
  retrieval_provider: z.literal("megaloc_mapillary_faiss"),
  retrieval_scope: z.literal("ankara_reference_collection"),
  result_semantics: z.literal("ankara_reference_collection_visual_similarity_not_general_geolocation"),
  similarity_semantics: z.literal("cosine_similarity_not_confidence"),
  supported_region: z.literal("Ankara pilot collection only"),
  evidence_version: z.literal("phase3b3-mapillary-ankara-pilot-v1"),
  benchmark_version: z.literal("phase3b3-mapillary-benchmark-v1"),
  limitations: z.array(z.string().min(1).max(300)).min(1).max(12),
}).strict());

export const mapillaryDemoQuerySchema = contractSchema<GeneratedQuery>()(z.object({
  status: z.enum(["completed", "abstained", "failed"]),
  reason_code: z.string().min(1).max(120).nullable(),
  analysis_scope: z.enum(["generic_upload", "ankara_reference_pilot"]),
  coverage_status: z.enum(["pilot_eligible", "insufficient", "provider_unavailable"]),
  coverage_label: z.literal("Ankara reference pilot"),
  retrieval_provider: z.literal("megaloc_mapillary_faiss"),
  retrieval_scope: z.literal("ankara_reference_collection"),
  result_semantics: z.literal("ankara_reference_collection_visual_similarity_not_general_geolocation"),
  abstained: z.boolean(),
  abstention_reason: z.string().min(1).max(120).nullable(),
  similarity_semantics: z.literal("cosine_similarity_not_confidence"),
  supported_region: z.literal("Ankara pilot collection only"),
  evidence_version: z.literal("phase3b3-mapillary-ankara-pilot-v1"),
  benchmark_version: z.literal("phase3b3-mapillary-benchmark-v1"),
  city: z.string().min(1).max(160).nullable(),
  index_version: z.string().min(1).max(200).nullable(),
  candidates: z.array(mapillaryDemoCandidateSchema).max(10),
  confidence: z.null(),
  confidence_semantics: z.literal("uncalibrated_unavailable"),
  experimental_status: z.literal("private_technical_demo_not_production"),
  limitations: z.array(z.string().min(1).max(300)).min(1).max(12),
}).strict().superRefine((value, context) => {
  if (value.reason_code !== value.abstention_reason) {
    context.addIssue({ code: z.ZodIssueCode.custom, message: "abstention reason aliases must agree" });
  }
  if (value.status === "completed" && (value.abstained || value.candidates.length === 0 || value.reason_code !== null)) {
    context.addIssue({ code: z.ZodIssueCode.custom, message: "completed demo query requires candidates" });
  }
  if (value.status !== "completed" && (!value.abstained || value.candidates.length !== 0 || value.reason_code === null)) {
    context.addIssue({ code: z.ZodIssueCode.custom, message: "non-completed demo query must abstain" });
  }
}));

export type MapillaryDemoCandidate = z.infer<typeof mapillaryDemoCandidateSchema>;
export type MapillaryDemoStatus = z.infer<typeof mapillaryDemoStatusSchema>;
export type MapillaryDemoQuery = z.infer<typeof mapillaryDemoQuerySchema>;
