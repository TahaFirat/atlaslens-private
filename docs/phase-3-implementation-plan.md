# Phase 3 implementation plan

## Scope

Add a modular, offline-first visual retrieval foundation without changing the
existing analysis API, frontend, providers, candidate fusion, or geolocation
claims. Phase 3 accepts only user-supplied, locally available images with an
explicit manifest. It does not download datasets, perform reranking, apply map
constraints, run geometric verification/LightGlue, calibrate scores, or publish
benchmarks.

## Architecture

- New isolated `atlaslens_api.retrieval` package.
- Existing `RetrievalProvider` remains unchanged as the deferred image-facing
  facade. Phase 3 exposes the requested typed embedding-to-neighbors boundary
  through `NearestNeighborEngine`; binding images to that facade waits for an
  approved operational embedding provider.
- `EmbeddingProvider` produces finite normalized float32 vectors with a stable
  provider/version/dimension/metric specification.
- Runtime providers are deliberately unavailable placeholders for disabled,
  future SigLIP2, and future CLIP. Deterministic embeddings exist only in tests.
- `ImageMetadataRepository` stores reference metadata in SQLAlchemy; no image
  blob, source path, filename, or raw image bytes are stored.
- `FaissImageIndex` uses `IndexIDMap2(IndexFlatIP)` and explicit int64 IDs.
  Normalized inner-product results are exposed as cosine distance `1 - score`,
  never confidence or correctness probability.
- Qdrant implements an unavailable interface adapter; Milvus remains a future
  adapter behind the same protocol.
- A retrieval service joins FAISS IDs to metadata and returns Top-K neighbors in
  deterministic `(distance, image_id)` order. It does not create candidates.

## Data and import

Manifest CSV headers are:

`image_path,latitude,longitude,country,region,city,license,source,Notes`

The importer validates the complete UTF-8/UTF-8-BOM file before mutation,
rejects duplicate headers, non-finite or out-of-range coordinates, missing or
non-regular files, blank license/source, unsupported images, excessive fields,
and conflicting duplicate metadata. Matching duplicate content is skipped
idempotently. Relative paths resolve below an explicit input root.
`Notes` is validated but deliberately not persisted. Country/region/city may be
blank and are never inferred. `capture_type` defaults to `user_provided`.

Image identity is UUIDv5 derived from SHA-256 content; the database also keeps
the lowercase SHA-256. Same content/provider/version is an idempotent duplicate
only when metadata agrees; conflicting metadata is rejected.

## Incremental and persistence behavior

All rows are validated and hashed before embedding. New vectors are appended
with `add_with_ids`; existing vectors are neither re-embedded nor used to
rebuild the index. FAISS persistence writes a temporary snapshot and atomically
replaces the active artifact. Metadata rows use pending/active state so failed
appends can be removed by ID and the restored index snapshot persisted. Verification
checks provider spec, count, IDs, checksum, and metadata consistency.

This exact flat index is intentionally for a few thousand images. Protocols,
batch APIs and int64 IDs preserve a path to Qdrant/Milvus or trained FAISS
indexes at ten-million scale without claiming current scale performance.

## CLI

Install a single `atlas` console entry point:

- `atlas embeddings create --manifest ... --input-root ... --index-dir ...`
- `atlas embeddings verify --index-dir ...`
- `atlas embeddings info --index-dir ...`
- `atlas embeddings remove --index-dir ... --yes`

No command downloads a dataset or model. `create` fails safely and leaves no
artifacts when the selected production embedding provider is unavailable.

## Dataset policy

- OSV-5M: hold; primary sources conflict between CC BY-SA 4.0 and inherited
  CC BY-SA 2.0, and its datasheet discourages surveillance/OSINT usage.
- Mapillary: conditional only after current API/commercial-terms review and
  complete creator/attribution provenance.
- Wikimedia Commons: per-file allowlist only; no corpus-wide license assumption.
- KartaView: conditional CC BY-SA 4.0 with required attribution and terms review.

Phase 3 implements no downloader for these sources.

## Verification

Run real FAISS tests for CSV validation, duplicate/conflict behavior,
incremental append, known-vector nearest neighbors, stable ties, reload,
metadata persistence, invalid coordinates, empty manifests, provider
unavailability, diagnostics, migration, and absence of blob/path columns. Then
rerun the existing backend, frontend, OpenAPI, build and E2E suites and scan for
fake data, benchmark claims, reranking, map constraints and geometric
verification.
