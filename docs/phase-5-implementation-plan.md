# Phase 5 implementation plan

Status: in progress. Phase 4 baseline reverified on 2026-07-11 before edits:
backend 123 passed/2 skipped, strict mypy 55 files, Ruff, frontend 30 tests,
ESLint, TypeScript, API drift and production build passed.

## Operational provider decision

The mandatory provider is the official `geoclip==1.2.0` PyPI wheel backed by the
official `VicenteVivan/geo-clip` implementation. The wheel is pinned by SHA-256.
Its hard-coded `openai/clip-vit-large-patch14` dependency is explicitly staged at
a recorded Hugging Face revision during `atlas models install geoclip`; API startup,
health, capabilities and inference never download artifacts. Runtime forces offline
Hub behavior and verifies the installation manifest before lazy loading.

The official GeoCLIP `.pth` artifacts are accepted only from the verified wheel and
loaded with `weights_only=True` and `map_location="cpu"` through the AtlasLens
adapter. Upstream output is a softmax over the fixed 100K gallery and is exposed as
`uncalibrated_gallery_softmax`, never as geographic probability or accuracy.

Frozen manifest facts:

- GeoCLIP wheel: `geoclip-1.2.0-py3-none-any.whl`
- implementation revision: `7a1a23b49648a5872a771cfda28490a17ab17d15`
- wheel SHA-256: `c29ae90b01cf177ffa50707686fc8d0bd104bc5ee8537b6c7480ed4e1a0a5a2d`
- CLIP repository: `openai/clip-vit-large-patch14`
- CLIP revision: `32bd64288804d66eefd0ccbe215aa642df71cc41`
- CLIP `model.safetensors` SHA-256:
  `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`

These are source-manifest facts, not evidence of successful target-host install.
The installer records hashes for named GeoCLIP wheel assets in its receipt.

OSV-5M is deferred to Phase 6: its official baseline returns one coordinate, uses a pickle
checkpoint and repository/Hydra stack, and has no native Top-K/confidence output. It
must not delay GeoCLIP. SigLIP2 is also deferred to Phase 6 as an
embedding/retrieval option, not a GPS predictor. Neither is downloaded, registered,
or used as a Phase 5 fallback.

## Frozen additive contract

- `GlobalPredictionResult` and `GlobalPredictionHypothesis` are provider-domain
  values distinct from retrieval hits and final candidates.
- `Candidate.confidence` becomes nullable while remaining present; existing EXIF and
  cloud values remain numeric. An uncalibrated model candidate uses `null` rather
  than a fabricated confidence.
- Optional `Candidate.model_prediction` carries provider/model revisions, device,
  raw score and score type, original rank, normalization, calibration state,
  inference duration and limitations.
- Provider capabilities gain optional installed/verified/operational
  status/model/device fields.
- Phase 4 assessment gains `model_only`; model hypotheses have no invented retrieval
  hit IDs.
- Existing endpoints and multipart fields remain unchanged. Frontend types are
  regenerated from OpenAPI and runtime validation stays strict.

## Implementation ownership

- Agent D: GeoCLIP runtime, manifest/cache/installer/verifier and model CLI.
- Agent E: live provider registration, candidate normalization/reranking,
  uncertainty, offline gazetteer and backend integration.
- Agent F: existing-design provider readiness, progress, Top-K/map/diagnostics,
  bilingual errors and browser tests.
- Agent G: benchmark manifest/metrics/reports and calibration abstraction.
- Main: shared contracts, dependencies/lock, conflict resolution, security review,
  actual installation, real-model/browser acceptance and final state.

## Sequential gates

1. Baseline: complete.
2. Operational prediction: install/verify official artifacts, CUDA and CPU smoke,
   offline restart, real non-EXIF CLI and web Top-K, five-image acceptance.
3. Evaluation/calibration: only after gate 2; no dataset download without a reviewed
   manifest. Missing legal assets means benchmark remains incomplete and calibration
   uncalibrated/preliminary, never fabricated.
4. Product hardening: all regressions, contracts, security scans and documentation.

## Safety and uncertainty policy

- Model-only confidence is null and radius is at least 750 km, widened by robust
  Top-K geographic dispersion.
- Coordinates are validated finite WGS84, antimeridian normalized and geodesically
  deduplicated with stable original ranks.
- GeoCLIP is optional for service readiness but, when verified installed, is invoked
  for every valid normal local-only analysis independently of OCR, retrieval, map or
  geometry availability.
- Provider failure is a specific diagnostic/abstention reason; it never becomes a
  generic internal error and never erases valid EXIF evidence.
- User uploads remain temporary and never enter model training, galleries,
  acceptance sets or benchmarks automatically.

## Evaluation policy

The preferred clean tier is newly captured, operator-owned/contributed imagery with
explicit evaluation permission and no prior publication. A provisional diversity
tier may use individually reviewed Wikimedia Commons files, but overlap with the
undisclosed MP-16 membership remains unknown. The target 120-image workflow slice is
not enough for production probability calibration. Calibration stays uncalibrated
unless held-out calibration, validation and locked-test requirements genuinely pass.

The reproducible provisional recipe currently names six Commons files spanning six
continents. Acquisition revalidates official camera coordinates, creator/credit,
license URL and attribution requirements, then writes ignored metadata-free local
derivatives and a hashed manifest. The loader recomputes a perceptual hash and
rejects near duplicates across splits. This six-image, landmark-heavy test split can
measure only preliminary behavior and cannot promote a calibration artifact.

A syntactically calibrated artifact is not sufficient: Phase 5 requires exact
provider/model/feature/split fingerprint and count compatibility, at least 50
records in every fit/validation/test split, positive and negative event support,
and named validation/test Brier and calibration-error metrics. No such artifact is
created or loaded by this phase.

Phase 6 production identity, distributed execution, SLOs and fleet scaling remain
out of scope.

## Operator commands

```powershell
cd services\api
uv run atlas models install geoclip
uv run atlas models verify geoclip
uv run atlas models info geoclip
uv run atlas models test geoclip --image C:\licensed\photo.jpg --device auto
uv run atlas models benchmark geoclip --image C:\licensed\photo.jpg --runs 5 --device auto

uv run atlas gazetteer install
uv run atlas gazetteer verify
uv run atlas gazetteer info

uv run atlas benchmark validate --manifest C:\eval\manifest.csv --asset-root C:\eval\images --allow-license CC0-1.0
uv run atlas benchmark run --manifest C:\eval\manifest.csv --asset-root C:\eval\images --allow-license CC0-1.0 --provider geoclip --output C:\eval\report --split test
uv run atlas benchmark report --output C:\eval\report

uv run atlas acceptance run --directory C:\eval\acceptance --provider geoclip --output C:\eval\acceptance-report --device auto
```

GeoNames uses the official `download.geonames.org/export/dump/` source, schema
`atlaslens-geonames-v1`, CC BY 4.0 and attribution `GeoNames`. The installer
records the actual downloaded SHA-256 values and sizes in its receipt; there are
no fabricated or hard-coded upstream hashes.

## Verified target-host state

- implementation regression: **220 backend passed, 2 host-privilege symlink
  skips; 38 frontend passed; Ruff, strict mypy (85 files), ESLint, TypeScript,
  generated API drift, production build, OpenAPI parse and fresh migration passed**;
- target Windows model install/verify: **passed** for GeoCLIP 1.2.0 and both pinned
  revisions without reinstalling or redownloading artifacts;
- real CUDA / CPU / network-disabled restart inference: **passed**, five native
  WGS84 hypotheses in each run;
- real API and frontend EXIF-free Top-K: **passed**, five model-only candidates,
  map/uncertainty/SSE/cleanup and a clean browser console;
- reviewed acceptance: **passed**, six of six EXIF-free Commons derivatives returned
  non-empty real candidate lists;
- reviewed smoke benchmark: **completed**, six of six inferences succeeded; this is
  a preliminary landmark-heavy slice, not a production benchmark claim;
- calibration artifact/state: **uncalibrated; no accepted artifact** because the
  held-out promotion count and independence gates are not met.

The operational repair changed only AtlasLens normalization and diagnostics. The
official package, installed artifacts and provider boundary remain intact. Timing
diagnostics are target-host observations, not product performance claims.
