# Decision log

## 2026-07-20 - Phase 3F validates the minimum REST Pod contract before create

- Context: Run `d63e2c8b65f725bb5ba4f026c6ab193b` made its only POST and
  received HTTP 400 with `provider_error_invalid_json_shape`, no Pod ID, and a
  restored 0/0/0/0 inventory. That historical diagnostic proves the body was
  valid JSON but not an object; it did not retain enough metadata to distinguish
  an array from a scalar or to recover a provider field path. The exact outgoing
  request is reconstructable from the committed adapter and included
  `volumeInGb=0`, a null `templateId`, `locked=false`, and availability priority.
  RunPod's official REST contract documents a 20 GiB default Pod volume, a
  separate optional `networkVolumeId`, and custom priority for honoring exact
  GPU ID order.
- Decision: Reduce the Phase 3F request to the official bounded GPU fields, use a
  20 GiB Pod volume mounted at `/workspace`, omit network-volume/template/null,
  CPU-only and Serverless fields, and validate field set, JSON types, ranges,
  exact selected GPU ID, one boolean-false non-interruptible GPU, disks, and the
  named secret reference offline before POST. Treat HTTP 400 as capacity only
  when a strict provider code/message allowlist says no capacity, while retaining
  the existing 404/409/422 allocation status policy. Classify schema/field errors
  as `RUNPOD_CREATE_PAYLOAD_INVALID` and unknown 400 responses as
  `RUNPOD_CREATE_BAD_REQUEST_UNKNOWN`.
- Alternatives: Continue sending zero Pod storage and null placeholders; treat
  every 400 as a capacity race; persist raw provider bodies; resolve the named
  secret locally; retry create.
- Consequences: Dry-run exposes field names and JSON types with zero API calls,
  zero creates, and no values. Future array validation responses retain only safe
  field path/type/message-class metadata; scalar/text/empty responses retain only
  length, digest, content type, and allowlist class. The historical 400's exact
  array/scalar subtype and provider-reported field remain honestly unavailable.
- Target phase: Product Phase 3F bounded RunPod pilot only.

## 2026-07-19 - Phase 3F binds a 201 Pod ID before validating create fields

- Context: Run `c1b3f58f9eddc945db24399c7173e493` ended with
  `create_response_interruptible`, a null receipt Pod ID and restored empty
  inventory. The adapter could reach that code only after HTTP 201, a JSON object,
  a valid `id` and a matching `name`; however, the ID was returned to the session
  only after all response fields passed, so the atomic receipt missed the cleanup
  target. The old sanitized state did not retain response keys or the JSON type of
  `interruptible`, so that historical type cannot be reconstructed honestly.
- Decision: Classify create HTTP status before Pod fields and parse only HTTP 201
  as success. Map capacity, auth, permission, rate and provider statuses to stable
  secret-free codes with no retry. On 201, bind the valid Pod ID atomically before
  all other validation, carry it on any post-ID exception into the one-Pod cleanup,
  reject boolean `true`, and use an authenticated Pod GET to require exact boolean
  `false` without truthiness conversion. Retain only sanitized status, top-level
  key names, presence flags and JSON type metadata for future diagnostics.
- Alternatives: Parse error objects as Pods; bind only after complete validation;
  coerce strings/nulls to booleans; retry create; persist raw provider responses.
- Consequences: A successful-looking 201 can no longer lose its termination target,
  while non-201 errors never trigger Pod-field validation. Raw responses, Pod IDs,
  credentials and account-private values remain excluded from diagnostics.
- Target phase: Product Phase 3F bounded RunPod pilot only.

## 2026-07-19 - Phase 3F treats advertised RunPod stock as unconfirmed capacity

- Context: RunPod's authenticated GPU detail response can advertise High, Medium
  or Low stock and a finite on-demand price while returning
  `availableGpuCounts=null`. Treating the nullable field as malformed blocked the
  bounded pilot even though the provider still advertised stock.
- Decision: A list-valued count remains authoritative and must contain one. A null
  count is eligible only as explicitly unconfirmed capacity when stock, price,
  memory and cloud gates pass. Prefer A5000, then L4, then RTX 3090, then cheaper
  eligible alternatives within the fallback tier. Prefer an eligible Secure
  variant over Community. Revalidate the exact GPU ID immediately before create,
  permit one non-interruptible one-GPU REST attempt, never retry another GPU or
  Pod, and classify a sanitized allocation race as
  `GPU_CAPACITY_RACE_NO_POD` only after all inventories are restored.
- Alternatives: Reject all nullable counts; treat null as confirmed capacity;
  retry another GPU after an allocation race; choose globally by lowest price.
- Consequences: Live readiness can be true with `capacity_confirmed=false`, while
  paid execution remains fail-closed at the second detail query and single-create
  boundary. No Serverless endpoint or network volume is introduced.
- Target phase: Product Phase 3F bounded RunPod pilot only.

## 2026-07-19 - Product Phase 3E makes limited retrieval coverage explicit and map-first

- Context: The private index contains 29 Ankara references and 11 locked holdouts.
  Its exact nearest-neighbor search previously ran for any accepted demo query,
  so an arbitrary image could receive Ankara-only matches that looked like a
  general geolocation result. The guided investor page also selected the first
  case hypothesis by persistence order and presented a tile-free case dashboard
  instead of a focused spatial review surface.
- Decision: Default the isolated query seam to `generic_upload` and abstain for
  insufficient reference coverage before storage, decode, inference or search.
  Admit candidates only for an explicit `ankara_reference_pilot` route; reserve
  coarse-provider routing for a separately real and validated provider. Bind the
  retained walkthrough to the exact checksum-locked holdout identity and fail
  closed on reference ID/hash/sequence/perceptual or selected-contributor overlap.
  Carry additive coverage, provider, scope, result, abstention, similarity and
  evidence-version semantics in query responses and analysis evidence, flattening
  them into the existing bounded case-evidence JSON. Preserve the case API's
  ordering and database schema; the investor client sorts by explicit rank, so no
  DB migration or breaking case-API migration is required. Replace the critical
  route with a 100dvh map-first workspace. Use configurable viewport-only OSM
  raster tiles with visible attribution and zero zoom prefetch; retain markers,
  uncertainty and an honest retryable local fallback when tiles fail.
- Alternatives: Treat Ankara similarity as Türkiye-wide geolocation; invent a
  confidence/margin threshold from the small holdout; silently use a simulated
  coarse provider; reorder the public case API; retain the old dashboard; force
  offline tiles or create a tile archive; copy a third-party product surface.
- Consequences: Generic uploads can no longer fabricate an Ankara prediction from
  limited coverage. The explicit walkthrough remains real local retrieval, but is
  labeled as visual similarity within the Ankara reference collection and never
  as probability, calibrated confidence or nationwide accuracy. Identical pixels
  are invariant to filename, case, locale and demo presentation context at the
  provider boundary. The selected query is clean against the locked reference
  gates; corpus-wide contributor and spatial independence are still not claimed.
  The browser can use a readable online basemap without an API key and degrades
  without losing candidate geometry. Automated browser tests substitute a
  controlled tile fixture and never contact the real OSM tile service.
- Target phase: Product Phase 3E honest geolocation and map-first reset only.

## 2026-07-18 - Product Phase 3D binds the generated case to one offline browser contract

- Context: The first live investor rehearsal reached healthy API and proxy case
  detail endpoints but the guided workspace failed. The canonical demo writer
  admitted the retained public Mapillary attribution page while the response
  schema rejected its required `pKey` query, the launcher discarded the CLI's
  generated case ID, and the hand-written frontend hypothesis-page ordering
  literal differed from the generated/backend response. Real browser validation
  also showed that the default case map attempted an online basemap and that a
  deterministic 404 was retried before its recovery view appeared.
- Decision: Share one bounded HTTPS metadata-URL validator across case writes and
  responses. Keep ordinary URLs query/fragment-free and admit only the exact
  credential-free `https://www.mapillary.com/app/?pKey=<public-id>` attribution
  form. Parse the private-demo preparation receipt, validate one RFC UUID, bind it
  into trusted runtime state, verify every workspace endpoint through the proxy,
  and print `?demo=investor&lang=tr&caseId=<UUID>`. The frontend consumes only that
  validated ID; missing/malformed routes require explicit selection from existing
  case APIs, and a valid missing case gets an immediate 404-specific retry/back
  view. Align hypothesis pagination with the generated
  `created_at_asc_id_asc` contract and force the existing offline map style in the
  investor launcher. Gate the flow with real API + real Vite Playwright from an
  outside-repository working directory and always use the official verified stop.
  Bind launcher/listener roots to the same CIM creation-time snapshot used for
  descendant discovery, revalidate every root before the first stop, and exclude
  stale Windows parent-PID links when a process predates its verified parent.
- Alternatives: Restore a production hard-coded UUID/title; infer a case by title;
  weaken URL validation for arbitrary query strings; change backend ordering to
  match a stale handwritten client literal; allow online tiles in the offline
  private flow; mock case responses in the integrated browser gate; terminate
  unverified processes after a failed test.
- Consequences: The exact launcher URL loads and survives refresh; old links fail
  safely with explicit recovery; the real attributed pilot URL round-trips without
  admitting credentials or general queries; NVIDIA/cloud remains disabled; and the
  final browser gate makes no non-loopback request. Existing public case APIs,
  geolocation inference, ranking, fusion, thresholds, pilot data and uncertainty
  semantics are unchanged. A recycled root cannot lend ownership to helpers, and
  an unrelated older process with a stale parent PID never enters the owned tree.
- Target phase: Product Phase 3D live investor-demo repair only.

## 2026-07-18 - Product Phase 3D maps the private pilot into the canonical case workflow

- Context: The attributed Phase 3B3 Ankara pilot and its local MegaLoc/FAISS
  query were intentionally isolated from the normal analysis domain, while the
  Phase 2 case workspace already provided evidence materialization, operator
  adjudication and hash-chained audit history. An investor-ready private demo
  needed one reliable offline launch and one coherent product route without
  embedding a result, depending on NVIDIA, weakening the pilot gates or creating
  another demo-only API/domain.
- Decision: Keep the frozen Phase 3B3 bundle and exact publication, source-policy
  and split-lock checks. Select only a retained, hash-verified, fully attributed
  non-reference query; send image bytes, never sidecar coordinates, to the
  loopback MegaLoc worker; and map the real query or honest abstention into the
  canonical `Analysis -> case/media -> evidence/hypotheses` materialization path.
  Use deterministic case/analysis/media IDs in a dedicated disposable SQLite
  runtime, preserve null/uncalibrated confidence, and apply the canonical 25 km
  minimum radius to unverified retrieval-model hypotheses. Launch only the
  required MegaLoc worker, API and web in that order. Disable downloads and
  cloud providers, remove inherited cloud secrets from child environments, bind
  to loopback, reject occupied ports, require identity-matched readiness and
  stop only PID/start-time/command/tree-verified owned processes. Delete the
  exact reparse-safe private runtime root after normal stop or fully verified
  failed-start cleanup.
- Alternatives: Hard-code Ankara or a prepared success payload; feed retained
  ground truth to inference; use a simulated provider as real evidence; create a
  parallel investor-demo case API; require NVIDIA or internet; start every local
  model worker; silently choose alternate ports; terminate unrelated listeners;
  retain a sensitive demo database after shutdown; change fusion, ranking or
  thresholds.
- Consequences: The Turkish route combines source authority, actual local
  analysis, provider/source state, candidate uncertainty, evidence, operator
  adjudication, audit integrity and explicit limitations. Optional NVIDIA
  failure does not fail the local case. The result remains a private technical
  demonstration of the 29-reference/11-holdout Ankara pilot, not Türkiye-wide
  accuracy, calibrated probability, public/commercial clearance, production
  service readiness or legally certified evidence. No geolocation fusion,
  ranking, threshold, public API or normal migration behavior changes.
- Target phase: Product Phase 3D investor-ready private demo hardening only.

## 2026-07-18 - Product Phase 3C2 aligns the active Qwen non-thinking response contract

- Context: The first two isolated 122B operator calls reached accepted 2xx
  response paths but produced strict-schema and empty/invalid-content failures.
  The API returns text and does not document `response_format` or `json_schema`.
  The active profile disabled thinking while retaining thinking-mode sampling.
- Decision: Bind the active `qwen/qwen3.5-122b-a10b` profile to documented
  non-thinking sampling (`temperature=0.7`, `top_p=0.8`) with seed `0`, retry
  zero and the existing 2,048-token cap. Do not send undocumented structured-
  output fields. Request exactly one JSON object with no prose or Markdown. The
  parser accepts one bare object or one complete lowercase `json` fence only;
  it never salvages prose, multiple/trailing objects, empty/null/list content or
  `reasoning_content`. Keep the deprecated 397B profile historically strict.
- Consequences: Mock coverage and all backend/frontend/static/security gates
  pass. One committed-adapter synthetic call made one POST, returned HTTP 200,
  and produced a typed abstention with zero hypotheses. This is response-contract
  compatibility evidence, not accuracy, availability or production evidence.
  A new operator blind validation is technically possible only with later
  explicit authorization; the prior operator images were not reused.
- Target phase: Product Phase 3C2 response-contract repair only. Retrieval,
  ranking, thresholds, automatic fallback and operator-image reuse remain out
  of scope.

## 2026-07-18 - Product Phase 3C2 recommends an explicit NVIDIA prototype model without automatic fallback

- Context: The Phase 3C1 hosted `qwen/qwen3.5-397b-a17b` path repeatedly ended
  in a read timeout and is scheduled for retirement on 2026-07-27. An evaluated
  Phi hosted route returned HTTP 410. A synthetic vision probe for
  `qwen/qwen3.5-122b-a10b` returned HTTP 200 in 41.499 seconds with temperature
  `0.6` and top-p `0.95`, but one synthetic response is not availability,
  accuracy, calibration or production evidence.
- Decision: Recommend `qwen/qwen3.5-122b-a10b` only as the disabled-by-default
  free prototype. Continue to require explicit NVIDIA/model selection,
  `cloud_assisted` mode and per-request cloud consent. Use retry zero and a total
  timeout no greater than 120 seconds. A documented HTTP 202 may be followed
  only by bounded status polling for that same request. Never automatically
  resend an image to another model or treat a low-quality result as permission
  to fall back. Keep `qwen/qwen3.5-397b-a17b` deprecated, unreliable and
  explicit-selection-only; do not offer the HTTP-410 Phi route.
- Consequences: The existing privacy boundary remains unchanged: metadata-free
  in-memory JPEG derivatives are at most 180,000 bytes each, the asset-upload
  API is not used, and images, derivatives, EXIF/GPS, raw OCR, request payloads
  and raw responses are not retained. Model availability can still change and
  abstention or a fail-closed error remains valid. The committed implementation
  and automated test gates passed. Two isolated frozen blind calls each made one
  POST with no poll or retry and reached accepted non-202 2xx paths, but neither
  produced a valid public result: one failed structured schema validation and
  one had invalid/empty model content. This blocks private-demo readiness without
  weakening the fail-closed or privacy result. Any compatibility investigation
  must use synthetic/non-private input first and requires new authorization
  before another operator-image validation.
- Target phase: Product Phase 3C2 NVIDIA vision fallback only. Retrieval,
  reranking, corpus/index use, prompt tuning, accuracy claims and production
  scaling remain out of scope.

## 2026-07-18 - Product Phase 3C1 adds an explicitly selected NVIDIA vision provider

- Context: AtlasLens has an existing optional cloud-vision seam, but introducing
  a second hosted provider must not change the OpenAI default, weaken the
  request-level cloud authorization boundary, or turn model self-assessment into
  calibrated location evidence. The authorized Phase 3C1 scope is a bounded
  NVIDIA reasoning experiment, not retrieval, training, evaluation tuning or a
  production-provider migration.
- Decision: Keep `openai` as `CLOUD_VISION_PROVIDER` by default and keep NVIDIA
  disabled. NVIDIA runs only when the operator explicitly selects `nvidia`,
  enables it with a backend-only key, and the request independently selects
  `cloud_assisted` mode with explicit consent. Pin the transport to
  `qwen/qwen3.5-397b-a17b` at the exact official
  `https://integrate.api.nvidia.com/v1` origin, refuse redirects and arbitrary
  endpoints, and support only the documented bounded completion/status-poll
  flow. Send metadata-free in-memory JPEG derivatives of at most 180,000 bytes
  each; do not use the separate NVCF asset-upload surface. Accept only a strict
  evidence schema with positive uncertainty, explicit uncalibrated confidence
  semantics and abstention. Keep the one-image blind-query command stateless,
  set transport retries to zero, and omit coordinates, image identity, raw clue
  text, prompts, request IDs and raw responses from its report.
- Alternatives: Replace or silently fall back from the default provider; activate
  cloud use merely because a key exists; accept arbitrary endpoints or redirects;
  upload oversized assets; persist inputs or raw responses; request chain of
  thought; batch/tune against operator images; treat a model score as calibrated
  confidence or an accuracy result.
- Consequences: A consented request discloses only the bounded derivative to
  NVIDIA under NVIDIA's current service/data terms; local deletion cannot recall
  a request already sent. Results remain optional, unverified model evidence and
  may abstain. No model, dataset, image, index or runtime artifact is added. The
  pinned hosted endpoint is scheduled for retirement on 2026-07-27, so continued
  use after that date requires a new official-endpoint/model/terms review and an
  explicit follow-up decision. This phase makes no accuracy, calibration,
  production-readiness or commercial-clearance claim.
- Target phase: Product Phase 3C1 NVIDIA vision reasoning only. Phase 6C,
  retrieval/reranking changes, corpus use, evaluation tuning and production
  scaling remain out of scope.

## 2026-07-18 - Product Phase 3B3 isolates an attributed Mapillary private demo

- Context: Phase 3B1/3B2 provide rights-gated ingestion, real offline MegaLoc and
  FAISS infrastructure, but no first-party production corpus. A separately
  authorized, bounded Mapillary pilot can test real retrieval wiring only if the
  source remains attributed, revocable, private and distinct from user evidence,
  first-party data and production serving.
- Decision: Use only the official Mapillary Graph API with a named runtime token;
  never scrape pages or persist signed media URLs. Audit all five versioned AOIs
  before downloading one selected city, keep stable image IDs and complete
  CC-BY-SA attribution, lock sequence/duplicate/adjacent-frame-separated
  reference and holdout sets, reuse the exact Phase 3B2 MegaLoc artifact and
  Phase 3B1 publication flow, and publish only a checksum/model/policy/split-bound
  `IndexFlatIP` bundle. Expose it through a separate disabled-by-default,
  non-production, loopback-only API/UI that returns raw similarity/distance and
  positive presentation uncertainty with no fabricated confidence. Retain at
  most 20 attributed demo images and delete other raw/materialized media only
  through the contained dry-run-first cleanup workflow.
- Alternatives: Scrape Mapillary pages; acquire every audited region; call the
  pilot Kayseri after Ankara wins; download originals or replacement weights;
  tune on the locked holdout/operator images; mix Mapillary with user evidence or
  the future first-party corpus; activate Phase 6C, public serving, commercial
  use or cloud infrastructure; treat a CC-BY-SA label, `GO_WITH_ATTRIBUTION` or a
  successful smoke as professional legal/production clearance.
- Consequences: The private pilot has 29 Ankara references and 11 holdout queries,
  which is real execution evidence but far below the 1,500/100 target. Its 1 km
  Recall@1 is 0.454545, median error is 7.12 km, p90 is 10.38 km and no calibrated
  abstention threshold exists. The outcome is therefore partial, not a Türkiye-
  wide accuracy or production-readiness claim. Attribution and the private source
  partition remain mandatory; public distribution and production activation are
  false, while commercial use remains `LEGAL_REVIEW_REQUIRED`.
- Target phase: Product Phase 3B3 private technical demo only. Stop after local
  documentation/validation; no cap expansion, new acquisition, holdout tuning,
  RunPod, publication or commercial deployment is authorized.

## 2026-07-17 - Product Phase 3B2 separates technical smoke from production approval

- Context: Phase 3B1 provides a rights-gated corpus/index pipeline but has no
  approved production descriptor or real first-party corpus. A pre-existing local
  MegaLoc source and weight artifact can support bounded offline engineering, and
  phone/dashcam capture needs explicit GPS, privacy, provenance and revocation
  controls before any media can enter that pipeline.
- Decision: Reuse the Phase 3B1 descriptor contract, registry, checkpoints,
  manifest admission and FAISS backend. Add an offline-only MegaLoc adapter that
  binds exact source/license/receipt/weight identities and preprocessing metadata,
  keeps injected backends test-only, and treats a real smoke as technical evidence
  rather than approval. Add a resumable capture importer for EXIF GPS, GPX, CSV
  and video metadata with bounded interpolation, distance sampling and fail-closed
  route checks. Quarantine all new media as privacy-pending; publish only after an
  explicit human approval, require an explicit manually redacted copy when
  redaction is claimed, and physically withdraw published media on revocation.
- Alternatives: Download or implicitly load a replacement model; infer production
  approval from an MIT label or a successful smoke; enable the synthetic provider
  in production; silently approve captures; claim automatic redaction accuracy;
  create a second corpus/index path; activate Phase 6C or provision RunPod.
- Consequences: The adapter, six capture commands and synthetic
  capture-to-FAISS query are executable offline without changing existing fusion,
  ranking, weights or thresholds. The source license identifies MIT and the local
  receipt labels the weight MIT, but professional weight-authority, production-use
  and training-lineage review remain unresolved. The Türkiye provider therefore
  remains disabled and fail-closed `not_ready`; real capture and commercial
  release are not authorized. No real media, download or cloud resource is used.
- Target phase: Product Phase 3B2 implementation and technical pilot readiness
  only. The next action is professional model-rights and first-party capture-policy
  approval followed by a local readiness recheck; it does not begin acquisition or
  infrastructure provisioning.

## 2026-07-16 - Product Phase 3B1 separates governed corpus building from serving

- Context: Phase 3A selected rights-controlled first-party/partner imagery, but
  no real corpus or approved production descriptor exists. AtlasLens still needs
  executable ingestion/index infrastructure that can be validated without data
  acquisition, model downloads, cloud provisioning or changes to live ranking.
- Decision: Add a Phase 3A-schema-compatible, fail-closed corpus pipeline with
  explicit checkpoints and atomic publication; keep locked holdout rows out of
  reference descriptors/indexes; require every benchmark query to match one
  admitted holdout asset and bind the holdout plus split-lock hashes into index
  metadata; require an explicitly rights-approved production descriptor registry;
  use the already locked FAISS dependency plus an exact test backend; and expose a
  separate disabled-by-default Turkiye index capability. The capability reports
  distance/rank and provenance only and does not enter the existing provider
  fusion/ranking path.
- Alternatives: Reuse the deterministic fixture provider in production; trust
  caller-supplied holdout metadata without split-lock binding; activate
  MegaLoc/Phase 6C; build an index before rights admission; permit partial index
  publication; hide missing artifacts behind a simulated ready state; provision
  a Pod before a real corpus and production descriptor are approved.
- Consequences: Synthetic tests can exercise manifest rejection, leakage,
  interruption/resume, exact/FAISS publication, locked metrics, revocation and
  cloud packaging without making an accuracy claim. Real execution remains
  blocked on licensed imagery, production descriptor approval, locked holdout and
  explicit user/cloud budget approval. No current ranking, fusion weight,
  threshold, model or default provider behavior changes.
- Target phase: Product Phase 3B1 implementation only. Phase 3B2, real acquisition
  and any RunPod setup remain unstarted and separately gated.

## 2026-07-16 - Product Phase 3A audit closes planning-control ambiguities

- Context: The first Phase 3A planning draft combined metadata with index space,
  staging with rollback space, omitted explicit street-versus-aerial counts and
  auditable Tier 2 road/terrain/season strata, and used a validator filename that
  differed from the authorized deliverable. RunPod storage documentation also
  required a more precise distinction between its volume-disk encryption toggle
  and broader DPA representations.
- Decision: Keep strategy A and the existing bounded scenario totals, but split
  every storage component, declare all current references as street imagery with
  zero initial aerial patches, require the four Tier 2 manifest strata, strengthen
  holdout independence, version the source policy `2026-07-16`, and make
  `scripts/phase3/validate_phase3_plan.py` the canonical fail-closed validator.
  Require application-layer encryption for selected RunPod network storage while
  professional review reconciles product docs, DPA, subprocessors and transfer.
- Consequences: The Recommended MVP remains 50,000 references, 240 locked queries,
  120.20992 GB cloud space, 11.25 expected GPU-hours and a $25 hard cap. Planning
  validation becomes stricter without acquiring data, provisioning infrastructure,
  changing application/geolocation behavior or activating Phase 6C.
- Target phase: Product Phase 3A planning closure only; Phase 3B remains subject to
  explicit approval and all rights, privacy, security, leakage and budget gates.

## 2026-07-15 - Product Phase 3A selects a rights-controlled Türkiye corpus

- Context: Product Phase 2 is complete, Phase 6C remains disabled, and the
  existing six-reference index cannot support broad Türkiye candidate recall.
  A larger corpus requires explicit imagery, metadata, code, weight, cloud,
  privacy, leakage, deletion and cost decisions before any acquisition.
- Decision: Select first-party AtlasLens capture as the initial production
  source, with partner imagery admitted only under an executed source-specific
  agreement. Keep open, share-alike, research-only and commercial-provider
  sources in independently governed partitions; reject blocked, unknown and
  permission-pending records from acquisition-ready manifests. Google imagery is
  never a persistent index source. Use a 50,000-reference/240-query Recommended
  MVP across all 81 provinces, with a separate dense Kayseri-Ankara-Sivas tier
  excluded from the national holdout. If separately approved later, use one
  on-demand RunPod Secure Cloud A5000 with a 150 GB Standard volume, 16 maximum
  GPU-hours and a $25 hard cap.
- Alternatives: Persistent Mapillary/KartaView indexing without clarification;
  research datasets as production corpus; Google Street View; one mixed index;
  Community Cloud, savings plans, uncontrolled marketplace capacity, or an
  uncapped local/cloud run.
- Consequences: Product Phase 3A adds planning documents, fail-closed JSON
  controls and a standard-library offline validator only. It creates no account,
  billing, Pod, volume, token, download, imagery, model, descriptor or index; it
  does not read operator images, activate Phase 6C, or change product/geolocation
  behavior. Professional licensing/privacy review and explicit user approval are
  required before Product Phase 3B.
- Target phase: Product Phase 3A planning closure. The only recommended next
  phase is Product Phase 3B rights-approved acquisition, corpus assembly and a
  locked offline benchmark; this decision does not begin it.

## 2026-07-15 - Product Phase 2 wraps immutable analyses with case history

- Context: AtlasLens had a validated single-image analysis path but no durable
  investigation container for multiple media, provenance, analyst decisions or
  later reopening. The newly authorized Product Phase 2 numbering is distinct
  from older technical component milestones retained in this repository.
- Decision: Preserve `/api/v1/analyses` and add a fixed `local-default` case
  workspace seam. Store image metadata and link an accepted analysis; only a
  completed persisted analysis can be normalized idempotently into immutable
  evidence and model-origin hypotheses. Require SHA-256 media/analysis agreement.
  Append adjudications, and represent operator corrections as new hypotheses with
  positive uncertainty and a new adjudication. Add transactional canonical
  SHA-256 audit chaining plus SQLite/PostgreSQL update/delete guards. Keep raw
  image bytes and raw OCR out of the new tables.
- Alternatives: Duplicate the upload/model pipeline inside cases; copy mutable
  analysis JSON into a case row; overwrite model candidates after review; claim
  the local workspace identifier provides tenant isolation; use an editable event
  log without a verification function.
- Consequences: Existing analysis clients remain compatible, while one local
  analyst workspace can reopen cases and review provenance/history. Case metadata
  currently has no delete endpoint or automated retention executor. Actor IDs are
  not authenticated identities. The audit chain is tamper-evident application
  history, not externally anchored, administrator-proof or legally certified.
  Multi-user collaboration, authentication, tenant isolation and distributed
  sequencing remain out of scope.
- Target phase: Product Phase 2 case and evidence investigation workspace. It
  does not activate Phase 6C, alter geolocation ranking/confidence, add a Türkiye
  corpus, download models/data or provision cloud GPU.

## 2026-07-15 - Phase 1 closes as two local, scope-separated commits

- Context: The repository had no HEAD and the sanitized 484-file baseline was
  already reviewed and staged separately from the relocatable runtime changes.
  Git closure must not broaden Phase 1, track private/heavy state or imply remote
  backup.
- Decision: Configure the user-authorized identity in repository-local Git
  configuration only. Commit the reviewed baseline first as
  `c2bfc221bfc542ab94ef9e9e2d1900bd759b7895`
  (`chore: establish sanitized AtlasLens baseline`), then commit only the
  explicit Phase 1 runtime, validation-test and documentation paths as
  `build: restore relocatable Windows development runtime`. Keep the
  pre-existing `asda.html` untracked and untouched. Do not create a remote,
  push, open a pull request, start Phase 2 or provision cloud GPU. According to
  the user, the external credential remains only in ignored `.env`; closure
  never reads or records its value.
- Alternatives: One mixed commit; broad `git add`; global identity changes;
  rewriting the baseline; remote publication during local closure.
- Consequences: Local history now separates the trusted source baseline from the
  Windows runtime restoration and remains independently auditable. This is not a
  remote backup. No model, dataset, database, environment, cache, user media,
  geolocation algorithm or Phase 6C activation change enters the runtime commit.
- Target phase: Phase 1 Git closure only; Phase 2 remains unstarted.


## 2026-07-15 - Windows application runtime may live outside the repository

- Context: The active checkout is on a capacity-constrained D: drive. Its existing
  API console-script trampoline contains stale absolute paths, while locked Python
  packages can be rebuilt safely on C: without moving or duplicating model and
  dataset artifacts.
- Decision: Keep the repository as the source of truth and add an optional
  `-RuntimeRoot`. It selects `api-venv`, uv cache and npm cache below that
  root; omission preserves the repository-local fallback. Resolve source paths
  from the checkout, invoke Alembic, Uvicorn, Ruff, mypy and pytest through the
  selected Python modules, and enforce offline model/dataset guards at startup.
  Provide an explicit env-isolated validation mode and deterministic smoke exit.
  Cleanup is limited to processes created by the launcher and checks creation
  time as well as the Windows parent PID.
- Alternatives: Move existing models/environments blindly; duplicate weights on
  C:; keep using stale console-script trampolines; rewrite dependency locks;
  hard-code a user profile or a single checkout drive.
- Consequences: Frozen application dependencies are relocatable, C/D disk gates
  are enforced, and normal development remains a single command. Existing worker
  environments and weight paths stay in place. A full local analysis may expand
  the Windows-managed pagefile; acceptance must recheck disk after inference and
  may restart only the API to release in-process model memory. No ranking,
  retrieval, confidence, threshold, candidate, provider-routing, or production
  geolocation behavior changes.
- Target phase: Phase 1 repository-safety/runtime restoration; durable Windows
  development behavior.


## 2026-07-14 - Reference retrieval requires a safe leakage attestation

- Context: A checksum-valid reference index does not prove separation from the
  private holdout, and operator model cards must not read or expose truth-bearing
  evaluation reports, local paths, credentials, OCR text, or coordinates.
- Decision: Project a completed leakage report into an atomic, index-bound
  `leakage-attestation.json` containing only versions, opaque fingerprints and
  aggregate checked/excluded counts. Require complete real-descriptor coverage,
  zero residual exclusions and exact index/descriptor/count binding. Only a
  `passed` attestation makes the index and MegaLoc retrieval usable. Expose model
  and index state through the existing development-only operator API gate, with
  unknown runtime fields left null. Checked Phase 6C fusion and OCR configuration
  files fail startup when Phase 6C is enabled and are bound into cache identity.
- Alternatives: Trust index checksums alone; read the holdout report at API
  runtime; expose filesystem diagnostics; treat missing audits as ready; silently
  substitute default Phase 6C policies after a configuration error.
- Consequences: Retrieval fails closed until an operator creates a clean safe
  attestation, while the UI can report real readiness, provenance and coverage
  without receiving evaluation truth or private host details.
- Target phase: Phase 6C candidate recall and retrieval activation.

## 2026-07-14 - OCR fallback budget and place-token independence are explicit

- Context: The target baseline gave PaddleOCR 45 seconds inside a 46-second
  enclosing OCR deadline. A full Paddle timeout therefore left RapidOCR no
  usable fallback budget even though both engines were operational. OCR engine
  scores and fuzzy or ambiguous gazetteer matches also do not establish location
  confidence or an independent geographic vote.
- Decision: Bound the preferred local OCR chain to 45 seconds, cap the first
  Paddle attempt at 28 seconds and reserve 16 seconds for RapidOCR, with shared
  cancellation and per-attempt deadlines. Define a separate `phase6c-v1` crop
  policy with one full-image pass, at most six SegFormer text-region crops and at
  most two rotated passes. Redact before typed evidence, preserve Turkish
  dotted/dotless-I semantics, and admit OCR as independent fusion evidence only
  for an exact or alias match to a specific, unique, country-attributed public
  place. Keep fuzzy, ambiguous, broad, countryless and sensitive text neutral;
  allow specific country disagreement to be represented as contradiction.
- Alternatives: Let Paddle consume the enclosing deadline; run both engines and
  unlimited crops concurrently; treat every gazetteer result or OCR engine score
  as independent location support.
- Consequences: RapidOCR can actually run after a stalled Paddle attempt while
  total work stays bounded and cancellable. Crop, engine and gazetteer provenance
  remains explicit, raw OCR is not logged or persisted, and OCR confidence stays
  an uncalibrated engine diagnostic rather than a location probability. The crop
  planner requires ephemeral region boxes; current segmentation summaries alone
  do not fabricate them.
- Target phase: Phase 6C OCR policy `phase6c-v1`, preserving Phase 6B response
  models and provider identifiers.

## 2026-07-14 - Phase 6C separates source identity from correlation

- Context: Phase 6C must retain geographically different recall modes while
  combining GeoCLIP, direct regression, sampled predictions, retrieval, OCR and
  verification signals whose raw values are not comparable. Distinct model names
  do not prove independent evidence; OSV-5M and PLONK-OSV share OSV5M training
  signal, while G3 reuses GeoCLIP's MP-16 location representation.
- Decision: Keep source family and correlation group as separate typed fields in
  `phase6c-v1`. Never average raw provider values. Use only within-provider rank,
  within-batch support, geodesic spread and bounded corroboration contributions;
  count OSV-5M plus PLONK-OSV as one independence group and G3 plus GeoCLIP as
  one independence group. Retain recall-only
  candidates, require at least two independent groups for publication eligibility,
  keep confidence null/uncalibrated, and apply a bounded diversity-aware Top-K.
- Alternatives: Treat every checkpoint as an independent family; normalize unlike
  provider outputs onto one scale; drop every one-family candidate before later
  verification.
- Consequences: Correlated street-view models cannot manufacture consensus, raw
  semantics remain auditable, and uncommon geographic modes can survive ranking
  without being mislabeled as calibrated or independently corroborated.
- Target phase: Phase 6C fusion policy `phase6c-v1`.

## 2026-07-14 - Holdout truth crosses only the post-prediction scoring boundary

- Context: Phase 6C needs repeatable evaluation of a private user-owned image,
  while its city label must not influence provider routing, prompts, retrieval,
  candidate generation, or model output.
- Decision: Keep truth in an evaluation-only JSON manifest. Project only opaque
  IDs and image paths, invoke inference through a strict subprocess protocol, and
  load the truth-bearing schema only after every prediction finishes. The real
  Phase 6C HTTP adapter accepts only a plain loopback origin, disables environment
  proxy inheritance, uses a sanitized upload filename, forces local-only/no-cloud
  request fields, and maps only validated real terminal output. Audit any reference
  manifest against SHA-256, perceptual/crop/transform, geometric,
  source/capture-family and optional caller-supplied real descriptor evidence.
- Alternatives: Pass a labelled record to an in-process provider; add the target
  to a reference corpus; synthesize descriptor vectors when MegaLoc is absent.
- Consequences: Production inference cannot import the target metadata through
  this workflow, exclusions are explicit, and an absent descriptor artifact is
  reported rather than simulated. Process isolation is a data-flow boundary, not
  an operating-system filesystem sandbox.
- Target phase: Phase 6C evaluation foundation.

## 2026-07-14 - Operator UI and public basemap failures fail closed

- Context: Vite development exposed operator navigation while the API correctly
  kept unauthenticated operator catalogs disabled, and the Liberty style plus an
  implicit MapLibre font stack produced recurring console/resource errors.
- Decision: Show dataset-QA/provider navigation only behind the explicit
  `VITE_ENABLE_OPERATOR_UI=true` switch while preserving the backend 404 gate.
  Use OpenFreeMap Positron as the token-free development default, request its
  supported `Noto Sans Regular` glyphs for AtlasLens symbol layers, and fall back
  once to the existing tile-free style after a style/resource failure.
- Alternatives: Enable operator APIs by default; fake an empty QA response;
  suppress MapLibre errors; retry a failing public glyph endpoint indefinitely.
- Consequences: Safe defaults no longer issue operator requests, remote map
  failures stop after one attempt, and candidate/text alternatives remain usable.
  Operators must explicitly enable both UI and API gates for local dashboards.
- Target phase: Phase 6B product-shell maintenance; no Phase 6C model work.

## 2026-07-14 - Phase 6B local providers require real-inference readiness proof

- Context: Pinned source, installed dependencies and present weights prove only
  preparation. The API must not advertise a model as operational until the exact
  runtime can load it and produce a structurally valid result, and incompatible
  research environments must not destabilize the existing Python 3.12/CUDA API.
- Decision: Run OSV-5M and PLONK in isolated Python 3.10 CPU workers and PaddleOCR
  in an isolated Python 3.12 CPU worker, all bound to loopback. Retain RapidOCR's
  killable API-side process boundary. A manager owns exact port/process/protocol/
  provider/revision checks and performs real offline inference by default. Expose
  the state ladder `not_installed`, `dependencies_installed`, `weights_prepared`,
  `worker_unreachable`, `model_load_failed`, `inference_not_verified`, `ready`,
  and `disabled`. Require successful import, reviewed weights, load and real
  inference for `ready`; treat current model residency separately from retained
  verification proof.
- Accepted scope: OSV-5M baseline, PLONK YFCC and PaddleOCR passed real worker
  inference. RapidOCR passed hash-bound real inference and a full Paddle-disabled
  HTTP fallback with a completed, persisted analysis. RapidOCR enablement is
  independent of Paddle worker enablement, and its configured device must match
  the verified ONNX Runtime receipt. PLONK OSV and iNaturalist remain prepared-only
  and are not readiness claims. No predicted
  coordinate was judged correct and no benchmark or calibration claim follows.
- Acceptance repair: The first fallback smoke exposed that RapidOCR construction
  was incorrectly coupled to Paddle worker enablement; the second exposed an
  untracked local `RAPIDOCR_DEVICE=cuda` setting that did not match the verified
  ONNX Runtime CPU receipt. Provider-request gates were separated, a regression
  test was added, and the local setting was corrected to `cpu` before the passing
  full HTTP smoke.
- Alternatives: Import research stacks into the API environment; mark weights as
  ready; trust an arbitrary listener; auto-download at startup/request time; claim
  all PLONK variants from one successful specialization.
- Consequences: Existing APIs/providers/frontend remain compatible, unavailable or
  failed workers degrade explicitly, and private paths/PIDs stay out of public
  payloads. The loopback manager is a bounded developer/single-host mechanism, not
  a production service supervisor. No dataset was downloaded or committed.
- Target phase: Phase 6B activation only. Phase 6C has not started.

## 2026-07-14 - Phase 6B bootstrap distinguishes source, environment and readiness

- Context: Python 3.10 installation exposed an empty `--no-checkout` worktree,
  attempted pip installation of the source-only OSV-5M repository, an obsolete
  `hf.exe` assumption, and a Hub downgrade incompatible with Transformers 5.13.1.
- Decision: Always force-checkout the exact pinned source commit; install OSV's
  official requirements rather than the repository; pin worker
  `huggingface-hub==1.23.0`; use `snapshot_download` through the worker Python API;
  run `uv pip check`; and probe the official `models.huggingface`/`plonk` imports.
  Recognize OSV `.bin` weights but report an installed artifact as `prepared`, not
  `ready`, until the API has a reviewed worker transport.
- Consequences: Interrupted setup is resumable, dependency conflicts fail before
  multi-gigabyte downloads, and diagnostics cannot overstate runtime readiness.
  PLONK remains CPU-only in the current worker and all model weights remain an
  explicit operator download.
- Target phase: Phase 6B Windows bootstrap maintenance; no Phase 6C work.

## 2026-07-14 - Phase 6B is additive and capability-disabled providers stay honest

- Context: The target Python 3.12/CUDA environment already runs GeoCLIP and
  SegFormer, while official OSV-5M/PLONK research stacks require a different
  interpreter and PaddleOCR has a separate native dependency/weight lifecycle.
- Decision: Preserve every existing provider interface and API. Normalize new
  models behind AtlasLens-owned typed contracts and a single-heavy-model scheduler.
  Keep OSV-5M, PLONK and PaddleOCR behind reviewed isolated-worker seams; when no
  worker/artifact is bound, expose a concrete `not_installed` capability instead of
  importing incompatible stacks, downloading at request time or emitting fallback
  coordinates. Exactly one scene-routed PLONK specialization may run per request.
- Alternatives: Upgrade/downgrade the main environment; auto-download weights on
  startup; represent a test double as a production model; run every PLONK variant.
- Consequences: GeoCLIP-only and Phase 6A behavior remain operational. The current
  host honestly reports OSV-5M, PLONK and PaddleOCR unavailable until an explicit
  worker acceptance; weights-only bootstrap does not imply runtime readiness.
- Target phase: Phase 6B only; fleet worker management is later production work.

## 2026-07-14 - Phase 6B fusion counts source families, not raw-score votes

- Context: GeoCLIP similarity, OSV direct regression and PLONK sample density have
  incompatible score semantics. OSV-5M and PLONK-OSV also inherit correlated
  street-view training signal.
- Decision: Never average raw provider scores. Cluster coordinates geodesically and
  rank with bounded, named contributions from provider rank, density, spread, OCR
  and independent source families. Classify OSV-5M plus PLONK-OSV as one family;
  same-family support receives only its separately bounded contribution. Require
  at least two independent families before Phase 6B replaces the preserved public
  candidate batch. Keep confidence null/uncalibrated.
- Alternatives: Weighted average of model scores; count every checkpoint as an
  independent vote; always return the highest Phase 6B cluster.
- Consequences: Correlated models cannot manufacture consensus, every movement is
  explainable, and absence of optional providers cannot weaken the real baseline.
- Target phase: Phase 6B fusion policy `phase6b-v1`.

## 2026-07-14 - OpenAI is a consented hard-case candidate reviewer

- Context: Optional cloud reasoning may help interpret ambiguous visual clues, but
  it introduces privacy, prompt-injection, cost and false-precision risks.
- Decision: Disable it by default. Require cloud-assisted mode, request-level
  consent, backend-only key, hard-case trigger, cache/budget admission and one call
  per analysis. Send one metadata-free 768-pixel low-detail derivative. Use
  `gpt-5.6-luna`, reasoning `none`, bounded structured output, no tools and no
  high-detail retry. Give the model candidate IDs without arbitrary coordinate
  creation; it can only apply bounded candidate adjustments or reject all.
- Alternatives: Browser key; automatic cloud fallback; original image/metadata;
  free-form answer or exact new coordinates; unconditional calls.
- Consequences: Local-only behavior is unchanged, usage is locally cached/metered,
  and cloud output remains review evidence rather than verified ground truth.
- Target phase: Phase 6B optional hard-case assistance.

## 2026-07-14 - Exact Mapillary labels are restored as metadata only

- Context: Phase 6A safely exported the 124-output checkpoint with generic names
  because no reviewed exact mapping was then available. A repository-contained
  exact 124-entry inverse mapping subsequently passed architecture, count, required
  class-name and checksum-preservation validation.
- Decision: Relabel only `config.json` and deployment metadata from
  `assets/mapillary/segformer_mapillary_config.json`; never load or rewrite model
  tensors. Require 124 exact inverse IDs, known key classes, compatible SegFormer
  architecture and classifier count. Record the mapping source and semantic flag.
- Alternatives: Guess labels by index; download Mapillary imagery; retrain; rewrite
  the safetensors file.
- Consequences: Real class names and centralized scene groups are available while
  the safetensors SHA remains
  `a364afc012b98cc494d1d95ac81b9db1d37e7fd09eaa021a23fcbea74fc16e8b`.
  Pixel labels remain descriptive and carry no direct geographic weight.
- Target phase: Phase 6B metadata repair; no Mapillary dataset ingestion.

## 2026-07-14 - Phase 6A keeps geography, scene description and naming separate

- Context: The existing GeoCLIP path generated broad coordinates, while the
  supplied SegFormer-B2 checkpoint could describe pixels but had no reviewed
  geographic reference database. Reverse geocoding could make coordinates usable
  but must not manufacture evidence.
- Decision: GeoCLIP remains the only Phase 6A geographic candidate generator.
  Preserve up to 50 private canonical candidates, geodesically cluster them, and
  publish at most five. SegFormer runs as an optional local scene-description
  provider with zero geographic weight. Local GeoNames only names cluster
  centroids and never changes score or rank. Real sufficiently strong OCR
  public-place matches may support or contradict a cluster; contradicted-only
  clusters are suppressed. Qualitative confidence is explicitly uncalibrated and
  always carries `score=null` and `calibrated=false`.
- Alternatives: Treat scene classes as country/style rules; let reverse-geocoded
  city names strengthen candidates; expose gallery similarity as probability;
  replace the existing provider and frontend architecture.
- Consequences: Phase 6A is explainable and additive, custom/non-GeoCLIP and
  simulated paths retain their prior behavior, and missing optional evidence
  degrades safely. Accuracy calibration, DINOv2/new FAISS work and Phase 6B remain
  excluded.
- Target phase: Phase 6A only.

## 2026-07-14 - Trusted SegFormer checkpoint becomes a strict safe artifact

- Context: `last_checkpoint.pt` contains training state plus separate student and
  EMA dictionaries. Its modular SegFormer parameter names do not directly match
  the installed Transformers legacy implementation, and no exact Mapillary Vistas
  v2 label configuration accompanied it.
- Decision: Inspect the trusted checkpoint on CPU with weights-only loading,
  select EMA, adapt only the reviewed deterministic parameter-name mapping,
  require exact target keys and tensor shapes, strict-load against the pinned
  official NVIDIA SegFormer-B2 base, and save safetensors plus bounded deployment
  metadata atomically. Keep the checkpoint and prepared model ignored. Infer 124
  outputs but use generic `class_N` names and disable semantic grouping until the
  exact label map is separately reviewed.
- Alternatives: Load the pickle checkpoint at API startup; use student weights;
  silently map guessed Mapillary labels; allow remote model code; commit large
  weights.
- Consequences: Runtime consumes only a prepared local safe artifact, preparation
  is idempotent, and the API cannot imply unknown class semantics. The official
  base and user checkpoint remain operator-reviewed local artifacts, not bundled
  dependencies.
- Target phase: Phase 6A artifact boundary; exact labels require a new artifact
  receipt, not an in-place relabel.

## 2026-07-14 - GPU stages are sequential and reverse-name cache failure is neutral

- Context: The target has one 8 GB CUDA device, so concurrent GeoCLIP and
  SegFormer loading risks avoidable memory pressure. A real HTTP smoke also found
  that another local process could lock the shared reverse-name SQLite cache and
  raise `sqlite3.OperationalError` even though the read-only GeoNames resolver was
  healthy.
- Decision: Run SegFormer only after GeoCLIP/retrieval consumption and serialize
  its native inference. Bound reverse naming by cluster count and timeout. Treat
  cache initialization/read/write failure as a safe partial failure: continue with
  uncached local resolution, retain raw coordinates and emit a bounded warning.
  Cache or naming status never enters the rank score.
- Alternatives: Run both GPU models concurrently; fail the entire analysis when a
  cache is locked; silently omit candidates; use a public reverse-geocoding API in
  local-only mode.
- Consequences: The repeated real HTTP smoke completed with GeoCLIP, SegFormer and
  local GeoNames while reporting cache degradation. SQLite remains a single-host
  cache, not distributed coordination; worker isolation and production storage are
  later production work.
- Target phase: Phase 6A runtime resilience; distributed execution remains in the
  broader Phase 6 program.

## 2026-07-12 - Phase 5C completion keeps execution, evaluation and display bounded

- Context: Product completion required concurrent evidence work, a finite custom
  model handoff and useful coordinate actions without weakening privacy or
  treating an uncalibrated score as probability.
- Decision: Start OCR, real inference and local retrieval concurrently after
  immutable preprocessing, cancel/gather every sibling before upload cleanup and
  isolate simulated inference from real retrieval. Pass only bytes whose exact
  hash was verified in the same read to the reviewed ONNX adapter; serialize one
  custom native invocation and retain its slot if cancellation cannot stop the
  native thread. Run GeoCLIP and a verified custom provider independently on the
  same frozen benchmark manifest, recompute both summaries from per-image rows,
  reject simulated reports and write a paired comparison. Round clipboard and
  external-map coordinates to the same uncertainty-derived precision shown in
  the UI.
- Consequences: A finished artifact can be registered, smoked, benchmarked in
  shadow and promoted without application rewrites or fake fallback. Parallelism
  cannot retain an upload path after cancellation, benchmark summary edits fail
  closed, and UI actions do not reveal hidden precision. Native process isolation,
  authenticated production operator APIs and fleet rollout remain Phase 6.
- Target phase: Phase 5C only.

## 2026-07-12 - Phase 5C extends the product shell without replacing Phase 5B

- Context: The custom model is still training and Phase 5B is blocked on external
  OCR/embedding/gazetteer/index artifacts. Wave 1 confirmed that GeoCLIP, provider
  outcomes, FAISS seams, Phase 5B reranking and the MapLibre result already exist,
  while history, custom-model lifecycle, report dashboards and general dataset QA
  are missing.
- Decision: Implement Phase 5C as additive adapters, safe summary/report APIs and
  frontend screens. Preserve all existing endpoints, candidate semantics and
  provider interfaces. Do not recreate retrieval/reranking/map architecture, weaken
  Phase 5B gates, claim accuracy improvement or start Phase 6.
- Consequences: Product completeness can progress independently of model training;
  unavailable evidence remains visible and neutral. The frozen scope and schema
  additions are recorded in `docs/phase-5c-product-plan.md`.
- Target phase: Phase 5C only.

## 2026-07-12 - Mock and trained-model execution are separate trust domains

- Context: A deterministic mock is needed for product development, while a future
  artifact is untrusted local input and its score cannot become confidence.
- Decision: Refuse mock configuration in production, classify and watermark every
  simulated analysis, exclude it from evaluation and keep fixtures outside
  production data. Manage trained artifacts separately from GeoCLIP with exact
  hashes, safe formats/adapters, immutable receipts and deployment modes
  `disabled`, `shadow`, `candidate`, `primary`. Reject pickle-family artifacts and
  remote code. Promotion fails closed on missing paired evaluation, lineage,
  compatibility, safety or operator approval.
- Consequences: Development can exercise the complete UI without fake production
  coordinates. A finished model can be registered and evaluated without silent
  fallback or automatic promotion; live shadow output never changes user ranking.
- Target phase: Phase 5C, durable safety rule.

## 2026-07-12 - History and operator reports remain privacy-bounded

- Context: Investigation history and evaluation/QA dashboards are useful, but
  original uploads, filenames, raw OCR, source paths and exact training GPS are
  sensitive. Production operator identity remains Phase 6.
- Decision: History lists only existing TTL-bound analysis summaries and reuses
  detail/delete behavior; rerun requires an explicitly retained source. Evaluation
  and dataset QA APIs are read-only views over prebuilt private-root reports and
  are unavailable to unauthenticated production clients. QA is read-only and emits
  opaque-key reports; contact sheets are explicit metadata-free local artifacts.
- Consequences: Phase 5C gains history and diagnostics without turning uploads into
  a permanent gallery or exposing private datasets. Operator mutations remain CLI
  only.
- Target phase: Phase 5C; production identity remains Phase 6.

## 2026-07-11 - Phase 5B remains blocked until real evidence gates pass

- Context: Provider code, fixed evaluation and upload integration are implemented,
  but the target lacks RapidOCR/ONNX, forward GeoNames, SigLIP2 and the reviewed
  10,000-image index. The session download/approval limit rejected further
  external installation.
- Decision: Keep GeoCLIP operational, register unavailable evidence providers with
  safe diagnostics, and mark Phase 5B blocked. Fixture-backed ranking tests prove
  wiring only. Do not compute candidate metrics, weaken dataset gates, use another
  download route, or begin Phase 6.
- Consequences: The public API remains additive; migration `0004` persists provider
  and index diagnostics. Completion requires real installs, five-mode reports and
  the unchanged measurable-improvement calculation.
- Target phase: Phase 5B.

## 2026-07-11 - Timed-out retrieval never owns an upload path

- Context: Cancelling `asyncio.to_thread` cannot kill native model inference.
- Decision: The upload-facing FAISS adapter copies the bounded normalized image to
  immutable bytes, and SigLIP2 exposes an additive byte encoder. Background native
  work may retain bytes/its single slot, never the temporary upload path.
- Consequences: Normal cleanup remains safe; a permanently hung native call still
  requires process restart. Killable shared model workers remain Phase 6.
- Target phase: Phase 5B local retrieval.

## 2026-07-11 - Phase 5B uses independent evidence before deterministic fusion

- Context: A real upload returned five valid GeoCLIP points, but four were a tight
  backfilled cluster. Every candidate was model-only with one provider, no OCR
  place, retrieval or map evidence, and the same 750 km radius.
- Decision: Preserve the GeoCLIP provider and normalizer. Add source-specific EXIF,
  GeoCLIP, OCR-place, gazetteer and licensed-retrieval hypotheses, suppress repeated
  source/capture-family support, cluster geodesically, then apply a versioned
  `phase5b-v1` deterministic evidence reranker. Missing evidence is neutral;
  contradictions reduce rank; honest abstention remains valid.
- Consequences: Five upstream rows are no longer presented as five independent
  proofs. Raw features and contributions remain inspectable and uncalibrated.
- Target phase: Phase 5B only.

## 2026-07-11 - RapidOCR and SigLIP2 are the Phase 5B local evidence models

- Context: Tesseract is disabled/unavailable on the target and the current OCR
  contract has no boxes or confidence. Production embedding providers are disabled.
- Decision: Explicitly stage and receipt RapidOCR 3.9.1 with ONNX Runtime GPU 1.27.0
  and CPU fallback, plus SigLIP2 B/16-384 revision
  `f775b65a79762255128c981547af89addcfe0f88`. Runtime is offline and never silently
  downloads. Query and reference embeddings share deterministic preprocessing and
  normalized float32 dimension 768.
- Consequences: The earlier Phase 6 deferral for SigLIP2 is superseded only for this
  bounded Phase 5B evidence-recovery scope. Neither model is operational until its
  explicit install, checksum, CUDA/CPU/offline and real-image gates pass.
- Target phase: Phase 5B; production fleet/model serving remains Phase 6.

## 2026-07-11 - Reference acquisition is reviewed per record

- Context: KartaView is CC BY-SA under current terms, while Wikimedia Commons has
  no corpus-wide license. OSV-5M remains conflicting and Mapillary requires a token
  and additional platform review.
- Decision: Permit only operator-approved KartaView and per-file Commons records
  carrying exact source, coordinate provenance, license, attribution, capture
  family and display policy. KartaView acquisition is blocked pending explicit
  operator terms/share-alike approval. Commons uses an explicit CC0 1.0 or CC BY /
  CC BY-SA 2.0-4.0 per-file allowlist, rejects unresolved restrictions and retains
  the exact version. No Google, Mapillary or OSV-5M imagery enters the index.
- Consequences: The 10,000-record gate is never reached by weakening legal,
  duplicate, geographic or evaluation-overlap validation. Images remain private
  operator artifacts and paths never enter SQL/API.
- Target phase: Phase 5B reference index.

## 2026-07-11 - MapLibre uses a labeled development style and marker-first uncertainty

- Context: The current empty style has no sources and attribution is disabled;
  fitting five 750 km polygons creates a blank, over-zoomed result.
- Decision: Keep MapLibre. Use configurable OpenFreeMap Liberty for documented
  development, visible OSM attribution and explicit asynchronous errors; require an
  approved/self-hosted style for production. Fit candidate centers, cluster markers,
  and make uncertainty/heatmap opt-in. Google remains an optional official-JS-API
  renderer only and is disabled without a restricted operator key.
- Consequences: The default development result shows roads, cities, borders and
  water without ingesting map imagery. Tile availability is not silently treated as
  geolocation evidence or an SLA.
- Target phase: Phase 5B frontend/map repair.

## 2026-07-11 - Valid GeoCLIP Top-K survives diversity normalization

- Context: The official installed runtime returned a two-value tuple with finite
  float32 CPU GPS `[20,2]` and score `[20]` tensors. AtlasLens then applied its
  25 km diversity radius to the complete valid set and rejected the concentrated
  result when fewer than three candidates survived. A generic handler mislabeled
  that policy rejection as malformed model output.
- Decision: Keep the official package and provider boundary unchanged. Normalize
  Torch, NumPy and Python tuple outputs in one adapter function; handle valid batch
  and Top-1 shapes; convert to native floats; filter malformed rows independently;
  preserve latitude/longitude order and original rank. Apply the configured
  diversity pass first, then backfill only genuinely distinct source rows using a
  one-metre floor. Emit bounded structured subreason/type/shape/dtype/device/count
  diagnostics without raw values or paths. Explicitly retain the supported slow
  processor with `use_fast=False`.
- Consequences: Real CUDA, CPU and network-disabled inference now return five
  hypotheses for the authorized smoke image. Exact duplicate coordinates still
  cannot be fabricated into the minimum count. Gallery scores remain uncalibrated,
  candidate confidence remains null, and capabilities expose optional
  installed/verified status without changing existing required fields. No
  correctness claim follows from Top-K presence or the six-image smoke benchmark.
- Target phase: Phase 5 operational repair; process isolation and production-scale
  evaluation remain outside this change.

## 2026-07-11 — Native inference timeout owns an immutable memory payload

- Context: Python cannot safely kill a native PyTorch call running in a thread.
  Waiting for that thread after a timeout would make cancellation and upload
  deletion unbounded; letting it retain the pipeline file would violate cleanup.
- Decision: Read the bounded normalized image into an immutable in-memory payload
  before native inference. On timeout or cancellation, return immediately while
  retaining the provider semaphore until the detached native call completes. The
  pipeline can delete its file without racing a background reader.
- Consequences: Request and deletion deadlines remain bounded and no background
  task owns a file. A permanently hung native call keeps the single local inference
  slot unavailable until process restart; killable worker isolation is Phase 6.
- Target phase: Phase 5 local runtime; process isolation remains Phase 6.

## 2026-07-11 — Evaluation truth and calibration promotion fail closed

- Context: Manifest-supplied perceptual hashes, missing source coordinates, stale
  attribution, or a self-declared calibration state could create convincing but
  invalid benchmark and confidence output.
- Decision: Recompute a bounded image hash and reject near duplicates across
  splits; the provisional Commons acquisition recipe requires official camera
  coordinates, creator/credit, license URL and attribution requirement. A
  calibrated artifact requires separate fingerprints, exact compatibility,
  non-trivial split counts, class support and named validation/test calibration
  metrics before it can emit probability.
- Consequences: The six-image recipe can support only a preliminary benchmark,
  never calibration. Missing official metadata aborts acquisition rather than
  filling truth or attribution from guesses. Runtime confidence remains null.
- Target phase: Phase 5 evaluation safety; production dataset governance remains
  Phase 6.

## 2026-07-11 — GeoCLIP 1.2.0 is the Phase 5 operational baseline

- Context: The live local-only pipeline has no global provider and universally
  abstains without EXIF. Official GeoCLIP provides image-to-Top-K GPS; OSV-5M's
  baseline provides one coordinate through a heavier research stack; SigLIP2 is an
  encoder rather than a coordinate predictor.
- Decision: Pin the official `geoclip==1.2.0` wheel and its SHA-256, explicitly stage
  a pinned `openai/clip-vit-large-patch14` snapshot, verify every installed artifact,
  and force offline inference. Treat gallery softmax as an uncalibrated relative
  score. Hold OSV-5M and SigLIP2 outside the operational gate.
- Consequences: The requested Windows host can receive a real local Top-K provider
  without a 259 GB dataset, but installation is large and explicit. Official bundled
  `.pth` files are loaded only after hash verification and with `weights_only=True`;
  model/data provenance and deployment-use limitations remain visible.
- Target phase: Phase 5 operational prediction.

## 2026-07-11 — Model predictions are not retrieval hits

- Context: Phase 4 hypotheses require licensed retrieval-hit IDs. Creating fake hits
  from GeoCLIP coordinates would corrupt provenance.
- Decision: Add a distinct typed global-prediction domain and a model-origin path
  through deterministic Phase 4 ranking. Public candidates gain optional model
  diagnostics and `model_only`; contributing retrieval IDs remain empty unless real
  retrieval occurred. Model-only confidence is null and uncertainty is at least
  750 km until held-out calibration.
- Consequences: Existing contracts remain additive and EXIF behavior is unchanged;
  frontend code must stop assuming every confidence value is a percentage.
- Target phase: Phase 5; calibration remains data-gated.

## 2026-07-11 — GeoNames labels are an explicit offline attributed artifact

- Context: Broad coordinates are more usable with country/region/place labels, but
  public reverse-geocoding per prediction leaks query locations and creates an
  operational dependency.
- Decision: Install official GeoNames `cities15000`, country and admin files only by
  explicit CLI action over an allowlisted HTTPS origin. Build a verified read-only
  local SQLite artifact, retain CC BY 4.0 and `GeoNames` attribution, and fall back
  honestly to coordinate-only labels.
- Consequences: Prediction remains offline and label failure does not erase model
  coordinates. Operators must preserve attribution and review source updates.
- Target phase: Phase 5 local artifact; production distribution review in Phase 6.

## 2026-07-11 — Evaluation is licensed, held out, and separate from uploads

- Context: GeoCLIP's MP-16 membership is not published, a small web slice may leak,
  and raw gallery softmax cannot become probability without independent data.
- Decision: Require an explicit root-contained manifest, license allowlist, source
  attribution, hashes, capture-family/split controls and distinct fingerprints.
  Preserve failures, abstentions, exclusions and denominators in JSON/CSV/Markdown.
  Preliminary calibration returns no display confidence and cross-revision artifacts
  are rejected.
- Consequences: No evaluation asset or benchmark result is bundled or fabricated;
  user uploads never enter training/evaluation automatically. Runtime benchmark and
  calibration status remain pending until reviewed data and the target host pass.
- Target phase: Phase 5; dataset governance/release automation in Phase 6.

## 2026-07-11 — OSV-5M and SigLIP2 are deferred to Phase 6 only

- Context: GeoCLIP is the selected Phase 5 coordinate provider. OSV-5M retains
  checkpoint/runtime/dataset-license concerns, while SigLIP2 is an encoder rather
  than a standalone GPS predictor.
- Decision: Do not install, download, register, or use either as a Phase 5 fallback.
  Reconsider them only in Phase 6 through separate license, artifact, benchmark and
  operational review.
- Consequences: Phase 5 remains bounded and cannot hide GeoCLIP failure behind an
  unverified alternate provider.
- Target phase: Phase 6 review only.

## 2026-07-11 — Optional evidence cannot promote geometry into geographic proof

- Context: Phase 4 can find local visual consistency, but its reference location,
  map completeness and retrieval model are not calibrated geographic evidence.
- Decision: Keep retrieval, hypothesis, map, geometry, rerank and final-assessment
  types separate. Public classification uses `geometry_supported`; the adapter adds
  an explicit non-proof limitation and the UI never renders relative rank as percent.
  Optional providers fail partially and honest abstention remains valid.
- Consequences: Phase 4 is explainable and conservative; calibration, accuracy and
  threshold selection remain Phase 5.
- Target phase: Phase 4, durable safety rule.

## 2026-07-11 — SQLite analysis transactions are locally serialized

- Context: The initial shared-cache in-memory repair removed unsafe shared-connection
  use, but full E2E exposed `database table is locked` when an EXIF worker commit and
  SSE read overlapped.
- Decision: Serialize the analysis repository's short transactions with one async
  lock while retaining normal SQLAlchemy sessions/connections. Do not hide the HTTP
  error or weaken the console assertion.
- Consequences: Local SQLite avoids transient SSE/poll/delete 500s. Cross-process and
  distributed coordination remains Phase 6; PostgreSQL behavior is unchanged.
- Target phase: Mandatory repair gate, durable local behavior.

## 2026-07-11 — Phase 4 assessments are an additive candidate extension

- Context: Phase 4 must expose relative reranking, map observations, geometry
  diagnostics, contradictions and attribution without redefining retrieval hits
  or breaking the Phase 1 candidate contract.
- Decision: Add optional `Candidate.phase4_assessment` to OpenAPI and Pydantic.
  It carries an explicit `uncalibrated_relative_rank` score semantic, versioned
  score breakdown, source diversity, contributing retrieval-hit IDs, bounded map
  and geometry summaries, contradictions, attributions/display policy and
  limitations. Existing candidate fields and endpoints remain compatible.
- Alternatives: Replace `Candidate`; publish retrieval hits as locations; reuse
  confidence as a probability; expose raw descriptors or reference paths.
- Consequences: Existing analyses may return `null`; Phase 4-aware clients render
  transparent details when present. Geometry is classified as support, not proof;
  the legacy `geometrically_verified` enum remains for compatibility but is not
  assigned merely because a Phase 4 matcher found inliers. Generated frontend
  types and strict runtime validators are updated with the contract.
- Target phase: Phase 4; calibration remains Phase 5.

## 2026-07-11 — Submission repair preserves the multipart contract

- Context: A browser displayed the generic Turkish analysis error after submit,
  while the available backend log contained no POST. Canonical direct and Vite-
  proxied multipart requests were independently verified as `202`, and generated
  no-EXIF jobs completed with honest abstention.
- Decision: Treat the server endpoint and OpenAPI multipart contract as frozen.
  Repair the frontend pre-send boundary: guard optional idempotency-key crypto,
  normalize synchronous browser API exceptions into localized `ApiError`s,
  retain safe problem code/request ID, and keep polling after SSE degradation.
- Evidence: `crypto.randomUUID()` currently executes before XHR `send()`. When it
  is unavailable or throws, no POST can exist; the native error bypasses
  `messageCode` and becomes `errors.unknown`. Current Chromium supports the API,
  explaining why the existing E2E succeeds. The historical console trace was not
  retained, so the repair adds a deterministic regression for this exact path.
- Alternatives: Change multipart fields or backend endpoint without evidence;
  require `randomUUID`; generate an insecure pseudo-random key; hide the failure
  behind the same generic message.
- Consequences: Older/restricted browser contexts can still submit without a
  fabricated key, known failures become actionable, and API compatibility is
  preserved. Windows startup must apply migrations before Uvicorn.
- Target phase: Phase 4 mandatory repair gate; durable client behavior.

## 2026-07-10 — Retrieval hits remain separate from geolocation candidates

- Context: Phase 3 needs nearest reference images but expressly excludes
  reranking, map constraints, geometric verification, calibration, and benchmark
  claims.
- Decision: Implement retrieval in an isolated backend package. A hit contains
  cosine distance, embedding provider specification, and licensed reference
  metadata. It is never converted into a final candidate by Phase 3 and has no
  confidence, radius, or verification claim.
- Alternatives: Feed raw nearest neighbors directly into Phase 1 fusion; expose
  a new frontend/API result before verification exists.
- Consequences: Existing API/frontend behavior stays unchanged; Phase 4 gets a
  typed retrieval seam for reranking and verification.
- Target phase: Durable.

## 2026-07-10 — Exact FAISS sidecar for the Phase 3 local implementation

- Context: The current demo targets a few thousand images while future releases
  may reach ten million.
- Decision: Use normalized float32 embeddings in
  `IndexIDMap2(IndexFlatIP)` with explicit int64 IDs and incremental
  `add_with_ids`. Persist atomic snapshots and keep SQLAlchemy metadata separate.
  Qdrant is an unavailable interface adapter; Milvus is future work.
- Alternatives: Premature IVF/PQ tuning; embedding blobs in the analysis table;
  a network vector database required for local development.
- Consequences: Simple, exact and testable local retrieval with no current-scale
  performance claims; future backends replace adapters, not contracts.
- Target phase: Phase 3 local adapter; backend choice may change later.

## 2026-07-10 — No production embedding is fabricated in Phase 3

- Context: The requested current providers are a disabled placeholder plus future
  SigLIP2 and CLIP, and no model/download approval was granted.
- Decision: Production descriptors are explicitly unavailable and never emit
  hash/random/demo vectors. The CLI refuses index creation without an operational
  provider. Deterministic embedding adapters are confined to tests while real
  FAISS behavior is exercised.
- Alternatives: Hash pixels into misleading vectors; silently download a model;
  bundle an unreviewed checkpoint.
- Consequences: Retrieval storage/indexing is operational and honest, while
  image-to-embedding creation awaits a reviewed provider.
- Target phase: Phase 3.

## 2026-07-10 — Dataset ingestion is local-manifest only

- Context: Official sources have materially different attribution, terms and GPS
  semantics; OSV-5M primary sources conflict on the CC BY-SA version.
- Decision: Implement no dataset downloader. Accept only local user-provided
  files with row-level source/license metadata. Hold OSV-5M; conditionally review
  Mapillary and KartaView; allowlist Wikimedia Commons per file.
- Alternatives: Bulk scrape/API mirroring; infer one license for Commons;
  normalize GPS precision from decimal digits.
- Consequences: Smaller legal/privacy surface and no proprietary corpus in the
  repository; production import requires a later source-specific review.
- Target phase: Durable ingestion policy.

## 2026-07-10 — Canonical port ownership is part of readiness

- Context: Browser requests to `localhost:5173/api/v1/capabilities` returned 404.
  Live inspection proved port 5173 belonged to a different Vite application and
  port 8000 to a Uvicorn service identifying itself as NexusOSINT. The proxy
  forwarded correctly but reached the wrong backend.
- Decision: Keep canonical local defaults 5173/8000, preflight both ports, reject
  unrelated listeners, verify the AtlasLens health/capability identity after
  startup, load proxy configuration from the root environment, and expose clear
  port/backend/API/response diagnostics. Never silently attach to or terminate a
  foreign service.
- Alternatives: Treat every 404 as a missing Vite proxy; silently choose random
  ports; terminate unrelated processes automatically.
- Consequences: Startup fails early with actionable ownership information until
  the conflicting service is stopped or explicit alternative ports are chosen.
- Target phase: Durable.

## 2026-07-10 — Phase 2 global models are explicit offline installations

- Context: GeoCLIP is a useful broad research baseline, but its constructor can
  auto-download a large CLIP backbone and a separately explicit license for the
  released weights/gallery was not established.
- Decision: Add registry, manifest, cache, list/install/verify CLI and disabled
  adapter behavior, but keep GeoCLIP off and non-operational until artifacts,
  hashes, dependencies, license review, and target-device measurements pass.
  Health checks never load or download models.
- Alternatives: Auto-download at startup; present a deterministic test adapter as
  production inference; omit the extension architecture.
- Consequences: Phase 2 remains honest and startup-safe; global no-EXIF inference
  is unavailable by default on the inspected host.
- Target phase: Phase 2, revisited only after explicit artifact/legal approval.

## 2026-07-10 — Tesseract TSV is the default local OCR boundary

- Context: Phase 1 OCR returned only redacted lines. Phase 2 needs boxes, engine
  confidence, and optional script/language hints without a new heavyweight ML
  stack.
- Decision: Extend the optional Tesseract subprocess adapter to TSV/OSD, normalize
  boxes, redact before constructing provider output, and keep raw OCR ephemeral.
  Operator-installed binaries/language data are capability-detected and never
  silently downloaded.
- Alternatives: EasyOCR/PyTorch default; RapidOCR/ONNX default; cloud OCR.
- Consequences: Small application dependency surface and graceful absence, with
  third-party Windows binary provenance still requiring operator review.
- Target phase: Durable default; additional OCR adapters may be registered later.

## 2026-07-10 — OpenAPI-first monorepo and strict Wave 2 ownership

- Context: The workspace was empty and implementation must run in parallel.
- Decision: Use `apps/web`, `services/api`, `packages/contracts`, `infra`,
  `scripts`, and `docs`; freeze OpenAPI before Wave 2 and enforce directory
  ownership because worktrees have no initial Git history.
- Alternatives: A single-language monolith; implementation before contract;
  creating an artificial baseline commit solely for worktrees.
- Consequences: Parallel changes stay reviewable; contract changes require lead
  integration.
- Target phase: Durable.

## 2026-07-10 — WGS84 and explicit uncertainty/provenance semantics

- Context: Future providers must work globally and must not imply false precision.
- Decision: Canonical coordinates are WGS84. Every candidate includes evidence
  references, provider provenance, a positive radius and basis, a source-support
  confidence and basis, and a verification status. GeoJSON always uses
  `[longitude, latitude]`.
- Alternatives: Country-specific schemas; bare point plus probability.
- Consequences: More verbose payloads, but globally extensible and auditable.
- Target phase: Durable; confidence becomes calibrated only in Phase 5.

## 2026-07-10 — Deterministic Phase 1 fusion and honest abstention

- Context: Phase 1 sources are heterogeneous and not calibrated for statistical
  score fusion.
- Decision: Validate, deduplicate by deterministic spatial policy, never boost
  confidence from duplicate providers, rank valid EXIF above unverified vision,
  preserve contradictions, and abstain when no valid candidate remains. Persist
  `fusion_policy_version`.
- Alternatives: Weighted average; random/demo confidence; always returning a
  best guess.
- Consequences: Sparse but honest results; calibration is deferred to Phase 5.
- Target phase: Phase 1 policy, replaceable through a documented versioned migration.

## 2026-07-10 — Cloud processing uses a distinct safe derivative

- Context: Images, EXIF, OCR, and prompts can reveal sensitive information.
- Decision: `local_only` is the default. Cloud invocation requires selected
  `cloud_assisted` mode, explicit consent, and a server-side key. Only a
  normalized, downscaled, metadata-stripped derivative crosses the boundary;
  prompts, raw responses, images, EXIF, and OCR are not logged.
- Alternatives: Browser-held API key; send original; implicit cloud fallback.
- Consequences: Additional transform step and capability messaging; much smaller
  privacy and secret-handling surface.
- Target phase: Durable.

## 2026-07-10 — Ephemeral storage and one-hour record TTL

- Context: Original images and GPS-bearing results are sensitive.
- Decision: `KEEP_UPLOADS=false`, originals are removed in cleanup paths, result
  records expire after 3600 seconds by default, and startup/periodic cleanup is
  abstracted behind `RetentionCleanupService`.
- Alternatives: Permanent gallery; image blobs in the database.
- Consequences: Analyses are intentionally temporary and deletion is first-class.
- Target phase: Default through Phase 6 unless a reviewed product policy changes it.

## 2026-07-10 — SQLite default, PostgreSQL/PostGIS readiness, in-process jobs

- Context: Phase 1 must be zero-friction locally yet prepare for later spatial work.
- Decision: SQLite is the default; Compose uses PostgreSQL with PostGIS. Phase 1
  uses a bounded in-process non-durable queue behind `JobQueue`.
- Alternatives: PostgreSQL required locally; Celery/Kafka in Phase 1.
- Consequences: Simple local startup; restarts lose queued work. Phase 3's fixed
  retrieval scope left the queue unchanged, so multi-replica coordination remains
  deferred to Phase 6.
- Target phase: Temporary through Phase 1.

## 2026-07-10 — Map-first accessible bilingual UX

- Context: Results require spatial context without hiding uncertainty or excluding
  mobile and keyboard users.
- Decision: Use a dark neutral, map-first layout with textual alternatives,
  visible uncertainty, provenance and warnings; complete English/Turkish
  dictionaries; WCAG 2.2 AA target; polling fallback for SSE.
- Alternatives: Generic admin dashboard; map-only output; English-only UI.
- Consequences: UI tests must cover unknown evidence types, no-signal, reduced
  motion, keyboard upload, and approximately 390 px layout.
- Target phase: Durable.
