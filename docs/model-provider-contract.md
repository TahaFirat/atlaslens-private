# Model and provider contract

Version: `phase1-provider-contract-v1` (frozen 2026-07-10).

## Phase 6A additive scene and hybrid contracts

Phase 6A preserves every frozen provider method. `SegFormerSceneProvider` is an
optional local evidence adapter following the existing `ProviderOutcome` and
descriptor conventions. It accepts the validated normalized image handle and
returns either a bounded `SegmentationResult` or a safe skipped/failed outcome.
Startup and status checks do not download or eagerly load weights.

The runtime contract requires a prepared Hugging Face directory containing
safe-serialized weights, processor config, and
`atlaslens-segmentation-deployment-v1` metadata. The metadata binds checkpoint
SHA-256/size, selected `ema` or `student` source, state-dict adapter, SegFormer
variant, base model, classifier count, image size, epoch, best mIoU, label source,
and semantic-label availability. Runtime load is local-only with remote code
disabled. A single lazy model instance uses `eval()`, `inference_mode()`, a bounded
semaphore, and CPU/CUDA selection.

`SegmentationResult` contains original dimensions, inference time, class IDs,
class names, pixel counts reduced to ratios, optional reviewed scene groups/tags,
and safe warnings. Generic `class_N` labels are permitted only with
`semantic_label_names_available=false`; in that state scene groups and tags must
be empty. Scene-tag strength is
`deterministic_heuristic_not_probability`. No output field may name or score a
country, city, region, or coordinate.

GeoCLIP may now produce up to 100 bounded internal hypotheses; the configured
Phase 6A default is 50. This expands only the private candidate-generation input.
The existing public list remains bounded to five. Each raw hypothesis continues
to retain WGS84 coordinates, original rank, provider identity, raw similarity,
score type, and uncalibrated semantics.

`GeoClipCandidateClusterer` consumes only these raw hypotheses and produces
dateline/pole-safe geodesic clusters. Cluster support is an
`uncalibrated_relative_rank`; it is not copied into candidate confidence.
`CachedClusterReverseGeocoder` consumes cluster centroids and may add a local
GeoNames label. Reverse labels are ranking-neutral and cannot create a prediction.

`Phase6AHybridEvidenceEngine` consumes clustered GeoCLIP candidates, real OCR
public-place evidence, narrow explicit language consistency, segmentation/quality
context, and provider diagnostics. Its `phase6a-v1` configuration keeps GeoCLIP
dominant, ignores weak OCR, applies explicit contradiction penalties, and gives
segmentation no geographic weight. Every non-zero contribution carries a reason.
The qualitative confidence contract is always:

```json
{
  "label": "low | medium | high | very_high",
  "score": null,
  "calibrated": false,
  "basis": ["candidate_consensus"]
}
```

The label is an evidence-quality category, not a probability or accuracy claim.
Optional segmentation, OCR, or reverse naming may fail independently; a valid
GeoCLIP result remains available with safe provider diagnostics. No DINOv2, new
FAISS corpus, geometric verification, or confidence calibration is part of this
contract.

## Phase 5C canonical inference adapter

Phase 5C does not change the frozen provider methods. It adds an internal
`GeolocationInferenceProvider` adapter so GeoCLIP, a reviewed custom artifact and
the development mock can be coordinated without replacing their native managers
or public schemas.

The canonical request contains a normalized private image handle or bounded
immutable bytes, image dimensions, optional EXIF, analysis mode, requested Top-K,
deadline, cancellation and a safe trace ID. The canonical result contains:

- provider, model, implementation and runtime revisions;
- `real` or `simulated` classification and deployment mode;
- bounded finite WGS84 candidates with source rank and raw score;
- declared score type, normalization and calibration state;
- safe duration, warnings, limitations, failure and provenance; and
- no product confidence, retrieval-hit fabrication, or correctness assertion.

`ProviderEnsembleCoordinator` may invoke independent eligible providers
concurrently on immutable input. It preserves every source outcome. Shadow results
are diagnostic only and cannot affect ranking; simulated results are excluded from
real fusion, evaluation and promotion. Provider absence/failure remains visible and
does not erase valid EXIF or another successful real provider.

The committed mock is disabled by default, development/test only, deterministic,
contained under `dev_fixtures/inference`, and production-refused. Its analysis
classification and simulation metadata must survive persistence/API/frontend
rendering and carry a visible warning/watermark.

Custom model execution requires a separately registered
`atlaslens-trained-artifact-v1` receipt. Registration/verification binds artifact
hash and size, manifest identity, preprocessing/output contract, training lineage,
evaluation gates and approved license. Generic pickle/checkpoint/remote-code input
is forbidden. The initial executable contract is reviewed ONNX
`top_k_coordinates`; safetensors requires a repository-known architecture.

Deployment modes are `disabled`, `shadow`, `candidate`, and `primary`. Successful
artifact verification enters shadow. Only `shadow -> candidate -> primary` is
permitted, using an identity-bound promotion report. Missing paired evaluation,
sample/subgroup gates, schema/lineage/license/safety checks, latency limits, runtime
isolation for primary, or explicit operator approval fails closed. Deployment mode
does not change raw scores into confidence or calibrated probability.

## Phase 5B additive contract

The frozen Phase 1 interfaces remain valid. Phase 5B registers concrete optional
implementations without changing their required methods:

- `RapidOCRProvider` is local, lazy and killable; `OCRResult` may add redacted
  blocks and canonical `PlaceEvidenceSummary` records. Raw text is ephemeral.
- `Siglip2EmbeddingProvider` implements the existing `EmbeddingProvider` with a
  pinned verified snapshot, 768-dimensional normalized float32 vectors, identical
  query/reference preprocessing, bounded reuse, CUDA/CPU selection and no health
  check load/download.
- `FaissRetrievalProvider.search` queries a verified local index from immutable
  image bytes and returns typed licensed hits. A hit is evidence, not a correct
  location assertion.
- `Phase5BEvidenceReranker` consumes typed source-specific hypotheses and emits
  optional `Candidate.phase5b_assessment` plus `Analysis.phase5b_diagnostics`.
  Scores are deterministic uncalibrated relative ranks, never probabilities.

Provider absence is a visible `skipped`/partial-failure state. No production path
may substitute mocks, hash vectors, random coordinates or fixture answers.

## Boundary

Public API schemas and provider adapter schemas are distinct. Provider output is
validated and sanitized before conversion to persisted evidence or candidates.
Provider inputs use opaque local image handles with redacted representations;
cloud adapters receive a separate metadata-stripped derivative and cannot access
the original upload through their request type.

## Common types

Each provider exposes a stable descriptor containing `id`, `kind`, `version`,
`execution_boundary` (`local` or `cloud`), `criticality`, availability, and a
safe unavailable reason code. Invocation context contains only analysis ID,
request ID, mode, consent, deadline, and cancellation signal.

Provider outcomes are one of:

- `succeeded`: validated value plus safe run metadata.
- `abstained`: provider intentionally found no defensible signal.
- `skipped`: disabled, unavailable, missing dependency/secret, or privacy denied.
- `failed`: safe error code, retryability, attempts, and duration.

Allowed failure codes include `timeout`, `unavailable`, `disabled`,
`missing_dependency`, `missing_secret`, `upstream_auth`, `rate_limited`,
`transient_upstream`, `invalid_output`, `unsupported_input`, `privacy_denied`,
and `internal_provider_error`. Adapters never return stack traces or raw upstream
bodies. Only transient connection, rate-limit, timeout, and upstream 5xx failures
may be retried, with a maximum of two retries and bounded exponential backoff.

## Required interfaces

- `ExifProvider.extract(original_handle) -> ProviderOutcome[ExifResult]`
- `ImageQualityProvider.analyze(normalized_handle) -> ProviderOutcome[QualityResult]`
- `OCRProvider.extract(normalized_handle) -> ProviderOutcome[OCRResult]`
- `VisionClueProvider.extract(cloud_safe_derivative) -> ProviderOutcome[VisionClueResult]`
- `CandidateProvider.propose(context) -> ProviderOutcome[CandidateBatch]`
- `GlobalGeolocationProvider.predict(normalized_handle)` — implemented by the
  explicitly installed local GeoCLIP adapter in Phase 5.
- `RetrievalProvider.search(normalized_handle)` — interface only in Phase 1.
- `CandidateFusionService.fuse(evidence, batches) -> FusionResult`

Historically, Phase 1 implemented EXIF/quality and forbade global/retrieval
implementations. Phase 5 and the bounded Phase 5B decisions explicitly supersede
that historical restriction while preserving the interface and API contracts.

## Evidence and provenance

Persisted evidence contains a stable ID, extensible type, localization-safe
label, sanitized display value, source-support confidence and basis, sensitive
flag, and provenance. Provenance includes provider ID/kind/version, execution
boundary, optional configured model name, and output schema version.

Never persist or log provider prompts, private reasoning, raw provider response,
raw OCR, original filename, local path, secret, or cloud request payload.

## Candidate and uncertainty policy

- Latitude is `[-90, 90]`; longitude is `[-180, 180]`; radius is positive;
  numeric confidence is `[0, 1]`, while an uncalibrated Phase 5 model may use
  `null`; every evidence reference resolves.
- `confidence_kind` is `source_reliability`, `uncalibrated_score`, or later
  `calibrated_probability`. Phase 1 does not claim calibrated probability.
- EXIF produces `exact_metadata`, `metadata_only`, high source support, an
  uncertainty basis, and a warning that metadata may be stale or altered.
- Vision-only hypotheses number at most five, reference visible clue evidence,
  remain `unverified_model`, never use exact granularity, have confidence at
  most `0.40`, and radius at least `25 km`.
- Quality evidence never creates a geographic candidate.
- Candidate geometry is canonical center plus radius; GeoJSON point/polygon is
  transport data and uses longitude-first coordinate order.

## Deterministic fusion

Policy `phase1-fusion-v1` validates all inputs, rejects invalid provider output,
deduplicates comparable hypotheses with antimeridian-safe haversine distance,
preserves all evidence/provenance, never boosts confidence because duplicates
agree, ranks metadata above unverified vision, and uses stable identifiers as
tie-breakers. Contradictions remain visible. Zero valid candidates produces a
completed result with an explicit abstention.

## Timeouts, privacy, and tests

The orchestrator owns provider deadlines and cancellation. Optional-provider
failure degrades the job and readiness remains healthy. Cloud invocation is
impossible unless mode, consent, and configured server key all pass.

Contract tests cover invalid outputs, timeouts/cancellation, bounded retry,
local-only cloud exclusion, metadata stripping, missing capabilities, vision
caps/floors, evidence integrity, coordinate bounds, deterministic ordering,
EXIF priority, abstention, future evidence types, and sensitive-log sentinels.

## Phase 4 assessment contract

Phase 4 consumes bounded `RetrievalHit` values without changing the Phase 3
interfaces. `phase4-v1` separates raw feature values, normalized weights and
contributions, uses deterministic stable ties, and labels its result
`uncalibrated_relative_rank`. Missing optional features are omitted and remaining
weights are normalized; no random or placeholder confidence is generated.

Map observations state the clue, mapped feature, status, reliability, radius,
provider and limitation. Missing map data is neutral/unknown by default. Geometry
reports bounded diagnostics and is exposed as `geometry_supported`, never geographic
proof. Optional `Candidate.phase4_assessment` preserves hit IDs, contradictions,
attribution and limitations; existing candidate payloads remain valid without it.

## Phase 5 global prediction contract

`GlobalPredictionResult` carries provider/model/implementation revisions, device,
dtype, local-only transfer status, measured inference duration and one to 100 stable
internal hypotheses. Each hypothesis carries contiguous rank, original gallery rank, finite
WGS84 coordinates, raw score, score type, normalization method, calibration state
and explicit limitations.

The GeoCLIP adapter declares `uncalibrated_gallery_softmax` normalized over the
pinned fixed gallery. This is ordering evidence only. It must never be copied into
`Candidate.confidence`, described as accuracy, or treated as geographic proof.
Before compatible held-out calibration, model-only candidates use:

- nullable confidence (`null`);
- `confidence_kind=uncalibrated_score`;
- `verification_status=unverified_model`;
- at least a `750 km` radius, widened by robust Top-K dispersion;
- broad granularity and explicit model/gallery limitations.

EXIF metadata remains independent and rank-prioritized by its own policy. OCR,
retrieval, map constraints, geometry and GeoNames are optional enrichments; none is
required to execute the global provider, and none silently converts its raw score to
probability. Missing/failed GeoCLIP uses a safe provider code and cannot erase valid
EXIF evidence.

Installation, startup and inference are separate trust boundaries. Startup and
health never download or load the model. Explicit install stages pinned official
artifacts outside Git; verify checks hashes and receipts; runtime is offline and
loads the model once under concurrency and timeout bounds.

## Phase 5 evaluation and calibration contract

Evaluation providers consume only explicit licensed manifest assets. Production
provider code is adapted through the typed `GlobalGeolocationProvider`; deterministic
test providers remain test-only. Metrics retain numerator/denominator, provider
failures, abstentions and exclusions. Top-5 oracle distance is a diagnostic upper
bound, not product accuracy.

A calibration artifact is compatible only when provider ID, exact model revision,
feature schema and distinct fit/validation/locked-test fingerprints match. A
preliminary artifact returns no display probability. No automatic fitting, upload
ingestion or cross-revision fallback is allowed. Regional/scene breakdowns are
evaluation outputs, not hard-coded regional experts.
