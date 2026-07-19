# Phase 3E inference lineage and coverage audit

## Scope and result

This report covers the private Mapillary/MegaLoc investor walkthrough and the
isolated `/api/v1/mapillary-demo/query` seam. It contains no image bytes, private
coordinates, raw OCR, credentials, contributor identities, source asset IDs, or
operator filenames.

The Ankara bias came from routing, not from a location prompt: every accepted
query was previously searched against a reference index containing only the
bounded Ankara pilot. The worker received normalized image bytes and the selected
device. It did not receive the upload filename, case title, case ID, locale, demo
route, source context, operator note, known city, or ground truth. The old route
nevertheless made the Ankara-only gallery implicitly eligible for a generic
upload, so its top matches could be mistaken for general geolocation.

Phase 3E makes eligibility explicit. A generic upload defaults to
`generic_upload` and returns `reference_coverage_insufficient` before AtlasLens
temporary storage, decoding, descriptor inference, or FAISS search. The retained
walkthrough calls the same runtime with the explicit
`ankara_reference_pilot` scope. Its output semantics are **visual similarity
inside the Ankara reference collection**, not Turkiye-wide geolocation,
probability, confidence, verification, or calibrated accuracy.

## Executable lineage

| Stage | Truth boundary |
|---|---|
| CLI receipt | The private-demo CLI emits the generated case/media/analysis IDs plus `analysis_scope`, coverage label, retrieval scope, result semantics, null-confidence semantics, and non-retention state. No location coordinates or source identity are emitted. |
| Locked query selection | The selected retained query must be an exact raw/normalized identity match for a record in the checksum-bound locked holdout inventory. An unknown or drifted holdout fails closed. |
| Leakage gate | Query/reference image ID, raw hash, normalized hash, sequence/capture-run proxy, and perceptual distance at or below the locked threshold fail closed. The selected walkthrough contributor must also be absent from the reference contributor set. |
| Case and media | Case title, source context, display filename, locale, route state, and attribution are created or projected after query inference. They are not arguments to the runtime query. The processing copy remains deleted after analysis. |
| Provider request | `MapillaryDemoRuntime.query` accepts only image bytes plus the routing scope. The scope is consumed by the coverage gate. MegaLoc `describe` receives the bytes and device only. The loopback transport adds protocol/request bookkeeping, which is not inference context. |
| Descriptor | MegaLoc returns one bounded, validated descriptor. It is ephemeral and is deleted from the runtime variable after search; it is not persisted in the case or API. |
| FAISS | The checksum-bound publication opens only when descriptor, source-policy, selection-lock, model, index, attribution, and reference-inventory identities agree. The underlying `IndexFlatIP` search is exact. |
| Ranking | Corpus search orders by cosine distance and stable reference identity, then assigns contiguous ranks. The demo response now fails closed if rank is non-contiguous or distance order changes. Case candidates and materialized hypotheses iterate in that order. |
| API lineage | Query responses and case evidence carry coverage/routing/result/similarity semantics. Case evidence uses bounded scalar fields. Hypotheses retain positive uncertainty and uncalibrated state; cosine similarity is never copied into confidence. |

The semantic provider-input fingerprint is therefore a function of the normalized
pixel payload and execution device, not presentation context. Automated tests run
identical pixels under different filenames, case-title fields, case IDs, locales,
and demo-entry fields and assert identical provider-side pixel fingerprints and
candidate behavior. Transport request IDs are intentionally unique and are not
part of this semantic fingerprint.

## Aggregate retained-publication audit

A read-only local audit opened the checksum-bound publication and emitted only
counts and booleans:

- 29 reference records and 11 locked holdout records were present.
- The selected walkthrough query matched its locked holdout identity exactly.
- Cross-split image-ID, raw-hash, normalized-hash, and sequence/capture-run-proxy
  overlap were all absent.
- No cross-split perceptual pair was within the prohibited distance of four.
- Contributor overlap exists somewhere across the complete reference and holdout
  sets. Contributor identity is therefore **not** claimed as a corpus-wide
  independence guarantee.
- The selected walkthrough query's contributor did not overlap the reference
  contributor set. Runtime preparation now fails closed if that selected-query
  condition changes.

Spatial proximity between some holdout queries and references is intentional: the
pilot benchmark requires geographic positives. This is not spatial independence
and is not presented as such. Coordinates and positive-reference mappings are
truth-bearing evaluation metadata; they do not enter the query descriptor request
or inferred case evidence. Reference and holdout descriptors are built separately,
and only reference provenance is admitted to the FAISS index. The live query
descriptor is neither added to the index nor retained.

## Coverage and abstention contract

The additive response/evidence contract carries:

- `analysis_scope`, `coverage_status`, and `coverage_label`;
- `retrieval_provider` and `retrieval_scope`;
- `result_semantics`, `supported_region`, and evidence/benchmark versions;
- `abstained` and `abstention_reason`;
- `similarity_semantics=cosine_similarity_not_confidence`.

The fail-closed behavior is:

- generic upload with only the Ankara gallery: coverage abstention without image
  storage, decoding, provider inference, or search;
- explicit pilot with an unavailable worker/index: provider-unavailable abstention;
- empty, failed, timed-out, or unordered pilot output: explicit abstention/failure
  with no candidates;
- no reference match: explicit abstention, retained as coverage evidence in the
  case without a location hypothesis;
- query/reference or locked-holdout identity drift: case preparation fails before
  inference.

No similarity or margin threshold was invented. The existing pilot benchmark is
not sufficient to calibrate a general-geolocation acceptance threshold. A real,
independent coarse provider is not silently simulated: until a separately
validated provider supplies an eligible Ankara route, generic uploads remain
abstained. No low-information classifier was fabricated from the two operator
images; those images were not read or used for tuning, thresholds, prompts,
corpus construction, or indexing.

## Remaining limits

The explicit walkthrough can demonstrate local, attributed visual retrieval
inside one small Ankara collection. It cannot establish the query's city, measure
Turkiye-wide accuracy, calibrate confidence, prove geographic correctness, or
support arbitrary uploads. Corpus-wide contributor independence and spatial
independence are not claimed. Expanding the supported region requires a separately
approved, licensed, leakage-audited multi-region reference corpus and independent
evaluation/calibration data; changing UI wording or similarity thresholds cannot
substitute for that data phase.
