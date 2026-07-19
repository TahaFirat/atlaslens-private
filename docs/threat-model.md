# AtlasLens threat model

## Product Phase 2 additive threats

| Threat | Product Phase 2 control | Residual risk / later boundary |
|---|---|---|
| Unauthenticated case access or actor spoofing | Fixed local workspace seam, high-entropy UUIDs, relationship checks, safe errors and explicit documentation that actor IDs are metadata | No identity, role policy or tenant isolation; bind only to a trusted local environment until a later authentication phase |
| Analysis linked under false media metadata | Completed analysis SHA-256 must equal case-media SHA-256 before materialization; an analysis can link to only one media record | Client metadata other than the persisted hash is not independently attested; external source claims need analyst verification |
| Model output overwritten during review | Evidence and hypotheses have no update/delete API; operator corrections and every adjudication are new rows | A privileged database/application administrator remains outside this application-level boundary |
| Audit payload, hash or order tampering | Canonical serialization, monotonic per-case sequence, SHA-256 previous/event hashes, transactional append, verifier, and SQLite/PostgreSQL update/delete triggers | No external timestamp, signature, transparency log or administrator-proof storage; not legally certified evidence |
| Concurrent audit sequence collision | Local repository transactions are serialized and the database has a unique case/sequence constraint | Multi-process/distributed sequencing is unsupported and requires a production database coordination design |
| Raw media, OCR or secret copied into case history | Metadata-only media, redacted OCR summary, bounded flat payloads and sensitive-key rejection | Free-text title/source/rationale can still contain sensitive information; operator training and future DLP/field policy are needed |
| False confidence or confirmed-location language | Original calibration state and provenance are preserved, uncertainty must be positive, uncalibrated output is not converted to percentages, and operator correction cannot claim model calibration | Analysts can still enter an incorrect coordinate or over-trust rank; formal calibration and review governance remain future work |
| Conflict-related case used for tactical targeting | Visible sensitive-use warning, hypotheses remain unverified until adjudication, audit records decisions, and no real-time/person-search feature is added | Documentation/UI controls are not an organizational authorization system; high-risk deployment requires policy, identity and oversight |
| Case metadata retained indefinitely | Media bytes retain existing TTL/delete behavior and storage state is explicit | The recorded retention label is not enforced and no case delete API exists in Phase 2 |

Product Phase 2 changes no geolocation provider, fusion weight, candidate threshold
or confidence rule. Phase 6C remains disabled by default, and no model, dataset,
Türkiye reference corpus or cloud infrastructure was added.

## Assets and trust boundaries

Sensitive assets include uploaded pixels, exact GPS candidates, EXIF timestamps,
OCR text, normalized cloud derivatives, analysis URLs, API keys, temporary paths,
and logs. Trust boundaries exist between browser and API, request parsing and
image decoders, local and cloud providers, API and database/storage, map tile
provider and browser, and unauthenticated clients sharing bearer-like analysis IDs.

## Threats and implemented controls

| Threat | Phase 1 control | Residual risk / later phase |
|---|---|---|
| Malicious or unsupported image | Signature/MIME/decoder agreement, JPEG/PNG/WebP allowlist, malformed and multiframe rejection | Native decoder isolation is later hardening |
| Decompression bomb / CPU exhaustion | Compressed-byte, decoded-pixel and per-dimension limits; Pillow bomb handling; bounded queue | Strong isolation and fleet quotas belong to Phase 5/6 |
| Path traversal | Server-generated random names in a private temp root; no filename interpolation | Storage adapter review remains mandatory |
| Oversized upload | Chunk-count actual bytes and reject above configured limit | Reverse-proxy limit required in production |
| Cloud cost abuse | Explicit mode/consent/key gate, separate rate limit, bounded queue, timeout/retry and max hypotheses | Distributed quotas require Phase 6 |
| Stolen API key | Server-only environment secret; never returned or logged; no browser key input | Secret manager and rotation belong to Phase 6 |
| Sensitive logs | Allowlisted structured fields and redaction; no request bodies, coordinates, OCR, paths, prompts, or keys | Operational sink policy requires Phase 6 |
| False precision | Positive radius and basis, confidence semantics, EXIF authenticity warning, unverified vision cap/floor, abstention | Calibration and evaluation are Phase 5 |
| Re-identification / private address inference | Provider instructions prohibit identity and private-home address inference; concise visible categories only | Policy/eval expansion belongs to Phase 5 |
| Analysis enumeration | High-entropy UUIDs, generic not-found, no full IDs in logs; Phase 5C history returns only safe TTL-bound summaries for the single-user local product | The history list is unauthenticated and must not be publicly exposed; identity/access control is outside Phase 5C |
| Long-lived uploads/results | `KEEP_UPLOADS=false`, cleanup in `finally`, startup/periodic TTL cleanup, delete endpoint | Secure physical erasure depends on infrastructure |
| Cross-origin abuse | Exact CORS allowlist, no wildcard credentials, security headers | CSRF/identity policy revisited with auth |
| Parser exception / stack leak | Provider/decoder exception normalization and safe problem responses | Process isolation later |
| SSE exhaustion | Per-client cap, heartbeat, finite terminal close, bounded/coalesced events and disconnect cleanup | Multi-replica coordination later |
| Deletion race | Queue cancellation/tombstone check and no late result write; artifact cleanup | Already-sent cloud request cannot be recalled |
| OCR injection / PII | Unicode normalization, PII redaction before persistence, output length limit, UI escaping, no raw logs | Redaction is heuristic and imperfect |
| OCR-to-query instruction injection | Only canonical gazetteer entities and fixed map-feature enums can form research inputs; OCR strings are data, never prompts/commands | Gazetteer false positives remain possible and are ambiguity-weighted |
| Map privacy / availability | Configurable style, textual result fallback, tests without tiles, deployment disclosure | Production tile provider and self-hosting decision pending |
| Duplicate-source manipulation | Content-hash suppression, per-source caps, independent-source accounting | Dataset identity quality still requires review |
| Reference path escape | Opaque keys, root containment, traversal/symlink-escape checks, byte-only results | Filesystem replacement races require operator-controlled roots |
| Remote map SSRF/abuse | HTTPS host allowlist, validated public-IP pinning, direct TLS/SNI verification, no proxy/redirect, byte/time bounds, contactable User-Agent, rate limit, cache and circuit breaker | Network provider remains opt-in; endpoint terms/capacity need operator review |
| Reference license violation | Per-record allowlist, exact version/source/attribution/display policy, human approval before download and validation before indexing | Commons non-copyright/personality rights still require operator judgment |
| Retrieval ranking manipulation | Exact/perceptual duplicate rejection plus provider/source/capture-family/content-hash caps | Coordinated semantically similar uploads can evade compact duplicate signals |
| Map API-key leakage | MapLibre needs no application key; Google seam is disabled without an operator key and requires referrer restriction | Browser keys remain observable and must rely on provider restrictions/quotas |
| Native retrieval timeout owns upload | Provider copies bounded immutable bytes before background inference; timeout can retain bytes but not a temporary path | A hung native thread can retain memory/slot until process restart |
| Repeated-pattern geometry | Mutual ratio filtering, descriptor-uniqueness, coverage, degeneracy and residual checks | Geometry remains visual support, never geographic proof |
| SQLite request races | Repository transactions are serialized locally; SSE/polling remain independent at the HTTP layer | Distributed storage coordination remains Phase 6 |
| Compromised or substituted model artifact | Official HTTPS/PyPI and pinned Hugging Face sources; wheel and CLIP weight SHA-256; verified receipt; private cache; atomic promotion | Upstream compromise, transitive packages, signing/SBOM and revocation remain Phase 6 release concerns |
| Unsafe model deserialization | CLIP requires safetensors; official GeoCLIP state dictionaries use `torch.load(..., weights_only=True, map_location="cpu")`; only allowlisted wheel members are extracted | `weights_only` reduces but does not eliminate native/runtime supply-chain risk |
| Partial model installation/cache poisoning | Unique staging, exclusive target semantics, hash verification, per-asset receipt, cleanup and explicit verify command; no startup download | Cache ACLs and multi-user host policy require operator/Phase 6 controls |
| Model resource denial of service | Lazy singleton load lock, bounded semaphore, request-bounded timeout, provider-owned in-memory payload and safe CUDA-OOM failure; a detached native call retains the slot until completion | A permanently hung native call requires process restart; killable workers, fleet quotas and GPU scheduling are Phase 6 |
| Invalid/NaN model output | Finite WGS84/score validation, antimeridian normalization, bounded Top-K and stable geodesic deduplication | Broad model can still be semantically wrong; calibration is pending |
| False model probability/precision | Gallery softmax labeled uncalibrated, nullable confidence, model-only radius floor of 750 km, unverified wording and abstention | Real held-out calibration/coverage remains pending |
| Malicious gazetteer archive | Official HTTPS allowlist and redirect checks, byte bounds, exact archive member validation, safe SQLite build, receipt/hash verification | GeoNames source/update governance and release notices remain operator responsibilities |
| Benchmark traversal or leakage | Explicit root containment, bounded content verification, recomputed perceptual hash with near-duplicate cross-split checks, license allowlist, capture-family controls and locked fingerprints | Cropped/semantic duplicates may evade a compact hash; MP-16 membership is unavailable, so web-image overlap cannot be proven absent |
| Private upload enters evaluation/training | Runtime uploads and benchmark manifests are separate; no automatic ingestion path; reports omit paths and truth coordinates | Operators must keep manually curated evaluation roots access-controlled |

## Phase 6A additive threats

| Threat | Phase 6A control | Residual risk / later boundary |
|---|---|---|
| Training checkpoint executes unsafe content | Only the explicitly trusted local checkpoint is accepted; inspection/preparation map tensors to CPU, use the restricted weights-only loader, validate the expected structure, and never accept checkpoint upload through HTTP | A compromised trusted host or unsupported PyTorch serialization still requires operator provenance review and process sandboxing |
| EMA/student or architecture silently mismatched | EMA is selected before student; classifier size is inferred; the reviewed name adapter changes keys only; exact target key set and tensor shapes are checked before strict load | A structurally compatible but semantically wrong operator checkpoint can still produce poor output |
| Prepared model/cache substitution | Deployment metadata binds source SHA-256/size, weight source, adapter, base model and label metadata; safe serialization and atomic publication are required; model/checkpoint directories are Git-ignored | Cache ACLs, signing, revocation and host attestation remain later production work |
| Fabricated Mapillary semantics | Exact labels must come from a repository/operator configuration; otherwise stable generic `class_N` labels set semantic availability false and scene rules remain off | An incorrect operator-supplied mapping requires human provenance and version review |
| Segmentation becomes a geographic stereotype | SegFormer returns scene proportions only and has zero geographic reranking weight; no road, vegetation, building, climate, architecture, country or city rule exists | A future reference comparison needs licensed data, evaluation and a separate approved phase |
| GPU memory exhaustion from concurrent models | One lazy segmentation instance, serialized inference, safe CUDA autocast, CPU fallback, and sequential GeoCLIP/SegFormer scheduling on the single-GPU path | Native-call process isolation, GPU quotas and multi-tenant scheduling remain later Phase 6 work |
| Raw similarity or reranker score shown as probability | GeoCLIP and cluster support remain explicitly uncalibrated; qualitative confidence always has `score=null` and `calibrated=false`; UI text states the limitation | Users may still over-trust rank labels; labelled calibration/evaluation is deferred |
| Reverse geocoding leaks coordinates or creates evidence | Prediction-time provider is the existing offline GeoNames resolver; only top clusters are named; normalized cache keys are private; names have zero ranking weight and failures preserve raw coordinates | Cache access/erasure and production database policy require operational review |
| Optional provider failure corrupts a valid result | Segmentation, OCR and reverse naming return existing typed skipped/failed outcomes; warnings are safe; valid GeoCLIP output survives without fabricated fallback | A permanently hung native call can retain memory/slot until process restart |

## Phase 5C additive threats

| Threat | Phase 5C control | Residual risk / later boundary |
|---|---|---|
| Mock output mistaken for real evidence | Mock is disabled by default, contained to development/test, refused in production, classified `simulated`, visibly watermarked and rejected by evaluation catalogs | Screenshots can be stripped of context; operators must not present simulations as real results |
| Arbitrary custom-model code/deserialization | Separate strict manifest; reviewed ONNX coordinate adapter; repository-known safetensors architecture only; pickle/checkpoint/remote-code rejection; no startup download | Native ONNX/runtime vulnerabilities and host compromise still require dependency, sandbox and release controls |
| Artifact substitution after registration | Private cache, canonical manifest, exact SHA-256/size/identity, atomic receipts, explicit re-verification and identity-bound promotion | Host administrators can replace processes/files; signing, revocation and fleet attestation remain outside Phase 5C |
| Unreviewed model becomes primary | Verified artifacts enter shadow; only `shadow -> candidate -> primary`; paired report, sample/regression/latency/subgroup/schema/lineage/license/operator gates fail closed; primary also requires runtime isolation | The operator approval and benchmark governance process remains organizational |
| Shadow output influences real result | Coordinator retains source results separately and excludes shadow output from user ranking/fusion | Timing/resource contention is still possible; robust process/fleet isolation is later work |
| History becomes a persistent private gallery | History reuses TTL-bound result records and contains no blob/path/name/raw OCR; rerun requires an explicitly retained local-only source; normal default deletes uploads | The content hash can correlate identical uploads; `KEEP_UPLOADS=true` increases exposure until TTL/delete and is intended only for controlled use |
| Report path traversal or private dataset leakage | Safe report IDs, private-root containment, symlink rejection, bounded schema reads, opaque asset keys, no exact GPS/source paths | Report/contact-sheet roots and local detailed benchmark artifacts still require OS access control |
| Unauthenticated operator endpoint exposed | Operator reports/models are disabled by default and production configuration refuses enablement | Development/test instances have no identity layer and must remain bound to trusted local networks |
| Dataset QA mutates or destroys source data | Scanner/reporting is read-only, output must be outside source roots, and `--apply` is rejected | Human repair workflows are outside the scanner and require separate backup/review |
| Clipboard/external-map precision leakage | Candidate centers are rounded from their positive uncertainty before display, copy and OSM-link construction; all three use the same visible precision | The trusted local API retains typed centers and is not a public multi-user boundary |

## Security headers and response behavior

API responses use request IDs, `Cache-Control: no-store` for private analysis
data, MIME sniffing protection, referrer/frame/permissions restrictions where
applicable, and safe errors without tracebacks. HSTS is enabled only at a TLS
terminating production boundary. SQL echo and body logging remain disabled.

## Abuse defaults

The example configuration limits creates to 12 per minute, cloud creates to 3
per minute, queue depth to 8, and SSE connections per client to 4. These are
single-process Phase 1 controls, not production-wide guarantees.

Phase 5 model and evaluation controls are local engineering boundaries, not a
production sandbox. Model cache ownership, signed releases, dependency scanning,
GPU tenant isolation and organizational incident response remain Phase 6 gates.
