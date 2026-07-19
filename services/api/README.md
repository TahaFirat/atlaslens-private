# AtlasLens API

FastAPI service for the privacy-conscious Phase 1 vertical slice. It validates
JPEG/PNG/WebP uploads, extracts EXIF GPS and real image-quality measurements,
optionally runs locally discovered Tesseract OCR, optionally invokes a
consent-gated OpenAI vision-clue adapter, then deterministically returns
provenance-backed candidates or an explicit abstention.

## Run and verify

From this directory:

```powershell
uv sync --all-groups --frozen
uv run alembic upgrade head
uv run uvicorn atlaslens_api.main:app --host 127.0.0.1 --port 8000
```

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
```

The SQLite default and temporary storage are relative to the process working
directory. PostgreSQL URLs are normalized to the psycopg driver and use the
same SQLAlchemy models/migration.

## Phase 5C additive surfaces

Phase 5C preserves the existing analysis create/detail/delete/events contract and
adds safe provider status, paginated TTL-bound history, retained-source local-only
rerun, model status, and read-only evaluation/dataset-QA report catalogs. History
does not persist image blobs, filenames or raw OCR. Rerun returns a conflict unless
`KEEP_UPLOADS=true` retained the source; cloud-assisted work requires re-upload and
fresh consent.

The custom artifact registry is separate from GeoCLIP. It accepts an exact local
manifest and safe reviewed adapter, verifies hashes/lineage/license, and enters
shadow mode before any gated promotion:

```powershell
uv run --no-sync atlas models register-local --manifest C:\private\model\manifest.yaml
uv run --no-sync atlas models verify atlaslens-custom-geolocation
uv run --no-sync atlas models info atlaslens-custom-geolocation
uv run --no-sync atlas models test atlaslens-custom-geolocation --image C:\licensed\smoke.jpg --device cpu
```

No model is downloaded at startup. Missing ONNX Runtime, unverified artifacts,
unknown adapters, and rejected/pending licenses remain safely unavailable; no
fixture prediction is substituted. See `../../docs/custom-model-handoff.md`.

The deterministic simulation is disabled by default and requires
`APP_ENV=development` or `test`; production refuses
`ENABLE_MOCK_INFERENCE=true`. Simulated analyses are classified/watermarked and
excluded from evaluation.

Evaluation/model/dataset-QA HTTP views are disabled by default. On a trusted local
development machine, set `OPERATOR_API_ENABLED=true` and private report roots.
Production refuses the unauthenticated operator API. Dataset QA remains a local,
read-only CLI:

```powershell
uv run --no-sync atlas dataset qa --images C:\licensed\images --output C:\private\reports\dataset-qa\run-1
uv run --no-sync atlas dataset qa-report --output C:\private\reports\dataset-qa\run-1
```

These commands create diagnostics, not benchmark or accuracy claims. See
`../../docs/dataset-qa.md` and `../../docs/evaluation-dashboard.md`.

## Phase 3 retrieval boundary

The isolated `atlaslens_api.retrieval` package stores licensed reference-image
metadata in SQLAlchemy and normalized embeddings in an exact local FAISS index.
It never stores image blobs or source paths. A retrieval hit is only a nearest
reference image with cosine distance and provenance; it is not a geolocation
candidate, confidence value, or claim that the reference location is correct.

The manifest schema is:

```text
image_path,latitude,longitude,country,region,city,license,source,Notes
```

`image_path` resolves below `--input-root`; `Notes` is validated but not stored.
The entire manifest is validated before metadata or index mutation. Commands do
not download images, datasets, or models:

```powershell
uv run atlas embeddings create --manifest manifest.csv --input-root licensed-images --index-dir data/retrieval
uv run atlas embeddings verify --index-dir data/retrieval
uv run atlas embeddings info --index-dir data/retrieval
uv run atlas embeddings remove --index-dir data/retrieval --yes
```

The disabled, future SigLIP2, and future CLIP production descriptors are
intentionally unavailable, so non-empty `create` currently fails without
artifacts. Installing an approved embedding implementation is a future explicit
step; deterministic embeddings exist only in tests. FAISS is operational behind
the index protocol, Qdrant is an unavailable adapter boundary, and Milvus remains
a future adapter. Phase 3 performs no reranking, geometric verification, map
constraints, calibration, dataset download, or benchmark.

## Existing analysis boundary

The `JobQueue`, repository, storage, provider, fusion, and retention services
are replaceable interfaces. The Phase 1 queue is bounded and in-process: jobs
do not survive a process restart. Global geolocation is an interface only and
belongs to Phase 2. No retrieval hit is connected to the existing analysis API
or candidate fusion. Reranking and geometric verification remain outside Phase 3.

Originals are deleted before a terminal event by default. `KEEP_UPLOADS=true`
retains the random-key original only until explicit deletion or TTL cleanup.
Cloud processing requires `cloud_assisted`, explicit consent, and a configured
server-side key; it receives only a resized, metadata-stripped JPEG derivative.
