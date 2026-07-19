# API usage

The frozen source of truth is `packages/contracts/openapi.yaml`. All private
analysis responses should be treated as non-cacheable.

## Endpoints

- `GET /api/v1/health`: process liveness only.
- `GET /api/v1/ready`: database and required local pipeline readiness; optional
  OCR/cloud provider failure does not make local-only mode unready.
- `GET /api/v1/capabilities`: formats, limits, modes, provider availability,
  operational model status/revision/device/calibration state where applicable,
  retention policy, and version. It never loads or downloads the model.
- `GET /api/v1/providers`: safe provider availability, mode and reason codes; no
  cache paths, secrets or sensitive diagnostics.
- `POST /api/v1/analyses`: multipart image plus mode, cloud consent, and ownership
  acknowledgement; returns `202` and resource URLs. `Idempotency-Key` is optional.
- `GET /api/v1/analyses`: paginated safe TTL-bound history with optional status,
  classification, provider, text and creation-time filters. It shares the
  unauthenticated single-user/local API boundary and must not be publicly exposed.
- `GET /api/v1/analyses/{analysis_id}`: current or terminal safe result.
- `POST /api/v1/analyses/{analysis_id}/rerun`: creates a new local-only analysis
  only when its source was explicitly retained; cloud rerun requires re-upload and
  consent.
- `GET /api/v1/analyses/{analysis_id}/events`: real progress/heartbeat/terminal SSE.
- `DELETE /api/v1/analyses/{analysis_id}`: cancels/deletes and acknowledges
  deletion idempotently where practical.
- `GET /api/v1/models`, `/api/v1/evaluations[/{report_id}]`, and
  `/api/v1/datasets/qa[/{report_id}]`: read-only operator views, disabled by
  default and refused for unauthenticated production configuration.

## SSE framing

Event names are `progress`, `heartbeat`, `completed`, `failed`, and `deleted`.
The `data` JSON follows the `AnalysisEvent` schema. Clients must reconnect with
bounded backoff and fall back to polling the analysis resource. A terminal event
closes the stream; consumers must not infer progress with artificial timers.

## Contract invariants

- Candidate latitude, longitude, nullable confidence, radius, evidence references, and
  provenance are validated.
- Candidate ranks are stable and sorted.
- GeoJSON coordinates use longitude then latitude.
- Unknown evidence `type` or warning codes must render through a safe localized
  fallback rather than failing the result.
- A completed analysis may contain zero candidates with an explicit abstention.
- Problems use safe machine-readable codes and message keys, never stack traces.

# Phase 6A additive hybrid diagnostics

Phase 6A does not rename an endpoint or remove an existing response field. Older
clients may ignore the following optional additions, and newer clients must accept
them as absent when a provider is disabled or unavailable:

- `Analysis.scene_analysis` is a bounded segmentation summary. It includes
  provider/device/timing, original image dimensions, dominant class IDs and pixel
  proportions, optional reviewed scene groups/tags, and safe warning codes. It
  never includes a mask, tensor, image blob, or location prediction.
- `Candidate.geoclip_cluster` contains raw member count/ranks, maximum/mean gallery
  similarity, and cluster support. Both the raw similarity and support are
  uncalibrated relative ordering signals.
- `Candidate.reverse_geocode` contains only a name for the candidate coordinate,
  plus source, dataset version, and license. The lookup does not create or improve
  a candidate score.
- `Candidate.confidence_assessment` contains `low`, `medium`, `high`, or
  `very_high`, always with `score: null`, `calibrated: false`, and a list of
  qualitative evidence bases. It is not a probability.
- `Candidate.phase5b_assessment` remains the compatible assessment envelope;
  `reranker_version=phase6a-v1` identifies the hybrid score breakdown. Every raw
  feature, normalized weight, contribution, support, contradiction, movement, and
  limitation remains explicit.

The public candidate collection remains bounded to five even though the local
GeoCLIP pipeline requests an internal configurable Top-K (50 by default). Raw
internal hypotheses are retained only for bounded processing/diagnostics and are
not exposed as 50 independent location claims.

`GET /api/v1/capabilities` and `GET /api/v1/providers` report the optional
segmentation, OCR, reverse-label, and GeoCLIP states through the existing provider
status system. Status inspection does not download a model. Segmentation and
reverse-geocoding failure are partial failures; they do not erase a valid GeoCLIP
candidate or produce a fake successful result.

The current prepared artifact has the reviewed exact 124-entry mapping and reports
`semantic_label_names_available=true`. Clients must still treat class
IDs/proportions/groups as scene description, never geographic proof. The schema
continues to accept the generic-label unavailable state for older persisted rows or
another artifact and must not invent names in that state.

# Phase 6B additive analysis fields

`Analysis.model_predictions`, `Analysis.fusion`, `Analysis.ocr`, and
`Analysis.cloud_assist` are optional. Older responses and persisted rows therefore
remain valid. Provider entries include bounded status, model/source revision,
device, duration, source family and declared raw-score semantics. Fusion exposes
relative-rank contributions and agreement counts, not a probability. Missing
optional providers are explicit skipped/disabled/failed outcomes.

Capability/provider diagnostics use the exact local-runtime states
`not_installed`, `dependencies_installed`, `weights_prepared`,
`worker_unreachable`, `model_load_failed`, `inference_not_verified`, `ready`, and
`disabled`. `ready` requires retained real-load and real-inference proof, not only
an installed package or weight directory. Public responses omit worker executable,
artifact, verification-receipt and log paths, ports, PIDs, raw OCR and generated
verification text. Private operator CLI diagnostics may expose local paths and live
health details.

Submitting `allow_cloud_assist=true` is accepted only with `analysis_mode` set to
`cloud_assisted` and the existing explicit cloud consent. It still does not force a
call: the backend key, hard-case trigger, cache and local budget policy decide. The
response reports used/not-used/cache/cost status without a key, prompt, raw OCR,
raw provider response or private cache key. A local-only request never invokes this
provider.

# Phase 4 additive candidate diagnostics

`Candidate.phase4_assessment` is optional and backward compatible. When present it
contains an `uncalibrated_relative_rank`, versioned score breakdown, independent
source count, retrieval-hit IDs, bounded map/geometry summaries, contradictions,
licensed-reference attribution/display policy, and limitations. It does not replace
the candidate contract, expose reference assets, or assert geographic proof.

# Phase 5 additive model diagnostics

`Candidate.model_prediction` is optional. It records provider/model/implementation
revision, device, raw score and score type, original rank, normalization method,
calibration state, inference duration and limitations. Model hypotheses remain
unverified and distinct from retrieval/map/geometry support.

For an uncalibrated model candidate:

- `confidence` is `null` and must not be formatted as a percentage;
- `confidence_kind` remains `uncalibrated_score`;
- `radius_km` is at least `750` and may be widened by Top-K dispersion;
- the raw `uncalibrated_gallery_softmax` is not probability or accuracy;
- model absence/failure remains distinct from an evidence-based abstention.

The multipart endpoint, accepted response, analysis URLs and SSE framing are
unchanged. Progress may additionally include global prediction, normalization and
reranking stages. GeoNames attribution is returned with resolved labels where the
optional offline artifact is installed.

Model/gazetteer mutation, benchmark execution and acceptance remain local operator
CLI surfaces. Phase 5C adds only safe read-only model/report HTTP summaries. Benchmark
JSON/CSV/Markdown reports preserve denominators, failures, abstentions and exclusions
and do not expose server paths or truth coordinates. Acceptance reports are
direct-provider smoke reports; they do not imply that the API/frontend product gate
ran. Model diagnostics may include an optional offline place-label block with country,
region, city, label distance, GeoNames source, dataset version and license. Actual
runtime/benchmark verification remains pending final handoff.

# Phase 5C additive analysis classification

`Analysis.result_classification` is `real` by default. A deterministic development
simulation sets it to `simulated`, includes explicit simulation metadata, and must
be visibly watermarked by every client. Optional provider comparisons are safe
diagnostics; a shadow provider cannot affect final ranking. None of these fields
changes candidate confidence semantics or asserts a correct location.

History summaries contain classification, status, safe image characteristics,
provider IDs/revisions, candidate/abstention summary, duration and warnings. They
do not contain image blobs, filenames, raw OCR or local paths. All analysis/history
responses remain private and `no-store`.
