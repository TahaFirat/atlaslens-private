# AtlasLens privacy model

## Product Phase 3C1 optional NVIDIA cloud vision

- NVIDIA vision reasoning is disabled by default and does not replace the
  default `openai` cloud-vision selection. An operator must explicitly select
  `nvidia`, enable it, and configure a backend-only key. Every invocation still
  requires `cloud_assisted` mode and separate request-level consent. A key alone
  never activates the provider, and `local_only` performs no NVIDIA call. Both
  backend settings and Vite startup reject populated frontend-prefixed
  OpenAI/NVIDIA credential variables without printing their names or values.
- The provider receives only in-memory, metadata-free JPEG derivatives produced
  from the already normalized private upload. Each derivative is at most 180,000
  bytes; Phase 3C1 deliberately does not use NVIDIA's separate asset-upload
  surface. AtlasLens does not persist these NVIDIA derivatives, prompts or raw
  responses, and does not enroll the upload in a corpus, training set, tuning set
  or comparison set.
- The exact hosted model is `qwen/qwen3.5-397b-a17b` at the allowlisted official
  `https://integrate.api.nvidia.com/v1` origin. Redirects and arbitrary endpoints
  are refused. The image's visible text is untrusted input; no tools, browsing,
  candidate coordinates or separate raw OCR transcript are supplied.
- Strict output permits bounded visible clues, at most five unverified location
  hypotheses, or explicit abstention. Every hypothesis has a positive uncertainty
  radius and confidence semantics fixed to
  `uncalibrated_model_self_assessment`; it is not a probability or measured
  accuracy. The existing provider adapter further floors presentation radius at
  25 km and caps the imported self-assessment at 0.35.
- The one-image blind-query CLI uses a fresh in-memory execution, zero transport
  retries and a sanitized report. It omits coordinates, local path, original
  filename, content hash, raw clue text, prompt, request ID and raw response. It
  does not retain the image or response and provides no batch, city-label,
  ground-truth or tuning argument.
- AtlasLens retention controls cannot recall a request already transmitted to
  NVIDIA. Before enabling the service, the operator must review NVIDIA's current
  data processing, retention, regional availability, account and model terms.
  The pinned hosted endpoint is scheduled for retirement on 2026-07-27.

## Product Phase 2 case workspace

- Case APIs persist investigation metadata: purpose, source context, sensitivity,
  sanitized filename display, byte size, SHA-256, optional source/archive URLs,
  bounded analysis provenance, analyst rationale and audit events. These fields
  can themselves be sensitive and belong only on the trusted local workstation.
- The case tables do not store image blobs, temporary paths, prompts, API keys or
  raw OCR text. OCR materialization uses a redacted summary plus bounded confidence
  semantics and provider provenance. Sensitive-key filtering rejects raw-media,
  raw-OCR, path, prompt, token, password and secret-shaped payload keys.
- Case upload reuses the existing validated analysis endpoint in `local_only`
  mode. Image bytes keep the existing temporary-storage and cleanup lifecycle; a
  case-media storage state reports whether bytes are ephemeral, deleted,
  unavailable or externally managed. Linking a previously consented analysis
  does not resend it to any provider.
- The media SHA-256 supplied with metadata must match the completed analysis before
  evidence can be materialized. The hash is retained for provenance and duplicate
  relationship checks, so it can correlate identical files even after bytes are
  gone.
- `retention_policy` is a bounded recorded label in Phase 2, not an automated
  retention engine. There is no case delete API yet. Case metadata, normalized
  evidence, adjudications and audit events remain in the local database until an
  operator applies a future approved lifecycle process or removes the database
  under an appropriate backup policy.
- The fixed `local-default` workspace does not provide authentication, tenant
  isolation or multi-user authorization. Actor IDs are client-supplied local
  metadata. Do not expose the case endpoints on an untrusted or public network.
- Adjudications and audit events are append-only at the API and database-trigger
  boundary. Hash verification detects changed payloads, hashes or sequence order,
  but a privileged administrator can replace the database/application and there
  is no external timestamp, key signature or legal evidence certification.
- Conflict-related cases show a sensitive-use warning and keep model hypotheses
  unverified pending analyst action. Phase 2 adds no face/person search, real-time
  tracking, tactical targeting, Türkiye corpus, cloud GPU or new provider.

## Phase 6B local ensemble and optional cloud review

- GeoCLIP, SegFormer and any explicitly prepared OSV-5M/PLONK/PaddleOCR worker
  process only the temporary normalized upload. Phase 6B does not enroll uploads
  into training, a reference gallery or an evaluation set. Worker messages are
  bounded and do not persist original filenames, raw pixels, tensors or paths.
- Worker lifecycle metadata, artifact paths, PIDs, ports and logs are private
  operator state. Public readiness exposes only bounded status/reason/revision
  fields. The RapidOCR verification receipt is bound to artifact hashes and records
  no verification image, path, original filename or recognized text.
- OCR output is untrusted image data. Raw recognized text is never logged or sent
  to a map/network lookup. Only normalized public-place evidence and narrow
  script/language summaries may reach fusion. Prompt-looking text cannot change
  provider policy or cloud tools because no tools are enabled.
- Phase 6B persists bounded provider/fusion/OCR/cloud summaries, not image blobs,
  prompts, raw OCR or raw cloud responses. The cloud cache key is stored in a
  private database column and excluded from the API. Existing TTL and delete
  behavior still applies.
- OpenAI hard-case review is disabled by default. It requires cloud-assisted mode,
  request-level consent and `allow_cloud_assist=true` in addition to a backend-only
  key and local budget. The provider receives one 768-pixel low-detail JPEG
  derivative whose metadata was stripped. Candidate coordinates are not supplied
  as arbitrary exact-coordinate generation instructions, and output is bounded to
  existing candidate IDs or reject-all.
- Cache and budget ledgers contain counters, bounded identifiers and cost estimates;
  never API keys, images, prompts, OCR text or raw responses. API keys are accepted
  only from the server's untracked environment. Frontend-prefixed key variables are
  rejected.
- OpenAI output is optional review evidence, not ground truth. Model agreement,
  scene labels, OCR and every Phase 6B score remain uncalibrated. They do not justify
  private-residence inference or disclosure beyond the existing uncertainty policy.

## Phase 6A local scene and hybrid evidence

- SegFormer-B2 inference is local to the configured AtlasLens API server. Runtime
  reads only an operator-prepared private model directory and never downloads a
  model. The normalized image payload and logits remain in memory. Masks, tensors,
  raw pixels, checkpoint contents, and model-cache paths are not persisted or
  returned through the API.
- Segmentation output is a bounded aggregate of reviewed class names and pixel
  proportions. It is scene description, not a city/country predictor. The exact
  124-entry mapping was restored as metadata only; masks and weights were not sent
  anywhere and scene labels receive zero direct geographic weight.
- The private GeoCLIP orchestration may inspect 50 raw candidates by default, but
  the public result remains bounded. Raw hypotheses are not written as 50
  independent claims. Candidate coordinates and cached normalized cache keys remain
  private analysis data.
- Reverse geocoding uses the existing local GeoNames artifact only. Prediction-time
  naming sends no coordinate to Nominatim or another public service. The persistent
  SQLite cache contains normalized coordinate keys and place labels and must inherit
  the private gazetteer directory's access and retention policy. A label never
  changes rank.
- OCR keeps the existing local/redaction boundary. Only canonical public-place
  matches and narrow script/language summaries enter the hybrid reranker. Raw OCR,
  prompts, original filenames, and arbitrary recognized strings are not added to a
  reverse/geocoder network query.
- Phase 6A candidate scores and confidence labels are uncalibrated. A qualitative
  `high` or `very_high` label is not permission to infer a private residence or to
  reveal more coordinate precision than the uncertainty policy permits.

Staging the official NVIDIA base model is a separate explicit operator network
action. API startup and analysis remain offline with respect to Hugging Face. The
trusted training checkpoint and generated safe-serialized model are Git-ignored
private artifacts; AtlasLens does not upload them.

## Phase 5C history, simulation, artifacts, and reports

- Analysis history is a paginated view over the same TTL-bound result records.
  It stores safe hashes/dimensions, status, classification, provider identities,
  summaries, timings and warnings; it does not create an image gallery or retain
  original filenames, raw OCR, prompts, local paths, or image blobs. The history
  endpoint shares the application's unauthenticated single-user/local boundary and
  must not be exposed to an untrusted network. Its content hash can correlate
  repeated identical uploads and is therefore private analysis metadata.
- A rerun is possible only when an operator explicitly set `KEEP_UPLOADS=true`
  and the original local-only source still exists. It creates a new analysis from
  a private clone. Cloud-assisted analyses require a new upload and fresh consent.
- The deterministic development mock is off by default, refused in production,
  persisted as `simulated`, visibly watermarked, and excluded from evaluation.
  A simulated coordinate is never represented as observed or model-predicted fact.
- Custom trained artifacts are operator-supplied local files. Registration copies
  only a hash/size/lineage/license-verified artifact into the private model cache;
  it does not upload or download weights. Runtime receives immutable normalized
  image bytes and does not enroll uploads into training or evaluation.
- Evaluation and dataset-QA HTTP views expose validated summaries from configured
  private roots. They omit source paths, exact truth/GPS values, raw EXIF/OCR and
  source images. The optional QA contact sheet is metadata-free but still contains
  pixels and must remain access-controlled.
- The Phase 5C operator API is read-only and unauthenticated. Configuration
  refuses it in production; authentication and multi-user authorization remain a
  later production concern. Do not expose a development instance publicly.

## Phase 5B OCR and retrieval boundary

- RapidOCR receives only the normalized local temporary image in a killable local
  worker. Raw recognized text has a redacting representation, is never logged, and
  is cleared after in-memory place lookup. Persisted snippets are redacted; Phase
  5B candidate evidence contains only canonical public-place entities.
- Image text is untrusted data, never an instruction. Network map queries are built
  from validated feature enums and candidate coordinates—not arbitrary OCR text.
- SigLIP2 and FAISS run locally after explicit installation. Query uploads are not
  added to the reference corpus, training set, or evaluation set.
- Reference SQL stores opaque keys and license/provenance metadata, not image blobs,
  filenames, or absolute paths. Display is denied unless the source permits it.
- Optional Overpass requests disclose a candidate vicinity and server IP. They are
  disabled by default, bounded, cached, rate-limited, and require a contactable
  operator User-Agent. Browser tile requests separately disclose the viewer IP and
  viewed viewport to the selected provider.
- Retrieval similarity, OCR strength, map support and GeoCLIP gallery scores are
  uncalibrated evidence—not permission to infer an exact private residence.

## Defaults

- `local_only` is selected by default. The image is processed by the configured
  AtlasLens server and is not sent to a cloud analysis provider.
- Uploaded files are not used for training by AtlasLens.
- `KEEP_UPLOADS=false` deletes the original after the analysis cleanup path.
- Results and safe metadata expire after the configured TTL and may be deleted
  earlier by the user.
- Cloud processing never activates automatically because a key exists.

## Cloud-assisted mode

Cloud processing requires a user-selected cloud mode, a separate explicit
consent acknowledgement, and a configured server-side key. AtlasLens sends only
a normalized, resized, metadata-stripped derivative needed for clue extraction.
The configured provider receives that derivative under its own data terms. The
original EXIF block and separate OCR output are not attached. Deleting an
analysis cannot recall a request that has already reached the provider.

## Sensitive metadata and outputs

EXIF can reveal a precise coordinate and may include timestamps or device data.
AtlasLens persists only the safe normalized analysis fields needed for the
temporary result; it does not log exact coordinates. Raw OCR is redacted before
persistence and never logged. Original filenames and internal paths are not
persisted as product data.

The web client rounds candidate centers according to uncertainty before display,
clipboard copy, or an external OpenStreetMap link. Those actions never expose
more coordinate precision than the user can see. The trusted local analysis API
still carries typed candidate centers and must not be exposed as a public
multi-user service without Phase 6 identity and authorization controls.

Map style/tile requests can reveal the viewer's IP address and the viewed map
region to the selected tile provider. Production operators must choose a
provider and retention policy appropriate to their jurisdiction and traffic, or
self-host tiles.

## Accuracy and user expectations

Coordinates and estimates may be wrong. EXIF indicates what metadata reports,
not proof that the depicted scene was captured there; metadata may be stale or
altered. An uncertainty circle is an estimate, not a guaranteed boundary.
AtlasLens presents unverified visual hypotheses broadly and abstains when no
defensible geographic signal is available.

## Deletion and retention limits

Deletion removes the database record and any retained temporary artifacts known
to the application. It does not promise forensic secure erasure from storage
media, backups outside Phase 1 scope, map-provider logs, or an already-sent cloud
request. Production backup and erasure guarantees must be defined in Phase 6.

## Phase 4 reference and map processing

Licensed reference images are resolved only for bounded in-memory comparison by an
opaque key beneath an operator-configured root. Database/API/log payloads contain no
reference path, signed URL, credential, image blob, or descriptor. Reference display
is denied unless manifest metadata explicitly permits it, and attribution remains
attached even when display is denied.

Map constraints are optional and disabled by default. A configured remote provider
would receive a bounded query location; it therefore requires an operator privacy and
terms review. Exact query coordinates are not logged or used as cache keys. Phase 4
scores and geometry diagnostics are uncalibrated and cannot establish where a person
or image is located.

## Phase 5 local model, labels, and evaluation

GeoCLIP inference in `local_only` mode runs on the configured AtlasLens server.
After explicit installation it uses only the private local model cache and sends no
image, embedding, tensor, score, or coordinate to Hugging Face, PyPI, GeoNames, or
OpenAI. Model installation is a separate operator network action and must not be
confused with analysis-time processing.

The provider copies the bounded normalized file into a private immutable memory
payload before native inference. A timed-out native call never retains or reopens
the pipeline file, so normal deletion and TTL rules still apply. The in-memory
payload can remain until that non-cancellable call returns; a permanently hung call
requires process restart, and killable worker isolation is Phase 6. Images, tensors,
gallery values and exact predictions are not training data and are not logged.
Model diagnostics expose only safe status, revision, device, duration and limitation
codes; they do not expose cache paths or usernames.

GeoNames installation is also an explicit operator network action. Prediction-time
label resolution is read-only and offline and sends no coordinates to a public
reverse-geocoding service. Labels include GeoNames source/version/license metadata;
coordinate-only output is used when no suitable place is available.

Benchmarking can contain sensitive ground-truth locations and is therefore an
admin-only filesystem workflow, not an upload/API feature. Manifests require a
separate licensed root and never ingest user analyses automatically. Public reports
omit local paths and exact truth coordinates, but operators must still protect the
manifest, source assets and detailed per-image artifacts. Calibration fingerprints
identify datasets without exposing their contents.

An uncalibrated model's raw gallery score is not user confidence. Public confidence
is null and the radius remains broad (at least 750 km) until a compatible held-out
artifact genuinely supports stronger semantics. Runtime and calibration acceptance
are pending; no accuracy guarantee is documented.
