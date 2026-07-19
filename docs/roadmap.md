# AtlasLens roadmap

AtlasLens has exactly six main implementation phases. Phase 1 is a trustworthy
foundation and is not yet a competitive global geolocation engine.

## Phase 1 — Foundation and functional vertical slice

- Purpose: Prove the private upload-to-result lifecycle with real EXIF/quality,
  optional providers, abstention, map UX, deletion, and engineering controls.
- Inputs: User-owned JPEG/PNG/WebP; optional Tesseract/OpenAI configuration.
- Deliverables: Versioned API, React client, FastAPI pipeline, SQLite/PostGIS-ready
  persistence, tests, CI, Docker/DX, privacy and threat documentation.
- Exit criteria: All applicable Phase 1 repository, runtime, vertical-slice,
  optional-cloud, safety, quality, and documentation checks are verified.
- Dependencies: Python, Node; Docker only for the full Compose path.

## Phase 2 — Global baseline models and evidence extraction

- Purpose: Add measurable worldwide visual geolocation signal beyond EXIF.
- Inputs: Approved model candidates, evaluation samples, license/privacy review,
  and the Phase 1 provider contract.
- Deliverables: One or more licensed baseline providers, richer local evidence,
  resource controls, model cards, and reproducible evaluation harnesses.
- Exit criteria: Real global benchmark results, provenance-complete candidates,
  documented coverage/bias, graceful abstention, and no Phase 1 privacy regression.
- Dependencies: Phase 1 complete; explicit model/data approval.

## Phase 3 — Licensed retrieval and geospatial indexing

- Status: Completed on 2026-07-10 for the local exact-FAISS/admin boundary;
  production embedding providers remain disabled pending explicit approval.
- Purpose: Retrieve legally usable reference imagery and index it geospatially.
- Inputs: Approved datasets/APIs, license register, baseline embeddings, PostGIS,
  and operational storage/queue requirements.
- Deliverables: Local licensed-manifest validation, versioned embedding and
  metadata contracts, incremental exact FAISS index, Top-K retrieval service,
  diagnostics, and Qdrant/Milvus extension boundaries. No dataset downloader or
  image blob store is included.
- Exit criteria: Existing application regressions pass; real FAISS append/search/
  reload tests pass; metadata/license provenance is retained; unavailable
  embedders fail safely; no benchmark, reranking, map constraint, or geometric
  verification claim is made.
- Dependencies: Phases 1–2 and dataset license approval.

## Phase 4 — Reranking, map constraints, and geometric verification

- Status: Completed on 2026-07-11 for the bounded local/offline architecture.
- Purpose: Combine independent signals and reject visually plausible but
  geometrically or geographically inconsistent hypotheses.
- Inputs: Baseline candidates, retrieval hits, road/map constraints, and reference
  image geometry with known licensing.
- Deliverables: Versioned rerankers, constraint engine, feature/geometric checks,
  contradiction evidence, and explainable verification statuses.
- Exit criteria: Existing flows remain compatible; deterministic clustering and
  relative reranking, optional bounded constraints, licensed reference resolution,
  generated-fixture geometry, contradiction/abstention, and transparent UI pass.
  Benchmark improvement and calibration are explicitly Phase 5 work.
- Dependencies: Phases 2–3.

## Phase 5 — Benchmarking, calibration, regional experts, and safety hardening

- Status: Phase 5B accuracy recovery is blocked. GeoCLIP and the fixed baseline
  pass; OCR/place, SigLIP2/FAISS upload retrieval, evidence reranking and map UX
  seams are implemented, but the real OCR/embedding/10k-index/combined-metric
  gates remain unrun and may not be promoted from fixture-backed tests.
- Purpose: Turn source scores into evaluated uncertainty while addressing regions,
  bias, adversarial inputs, and misuse.
- Inputs: Representative benchmark slices, Phase 4 pipeline traces, threat model,
  abuse research, and regional license-reviewed data.
- Deliverables: Explicit offline GeoCLIP management, broad Top-K candidates,
  optional offline GeoNames labels, manifest-driven evaluation, coverage/error
  and regional breakdowns, revision-bound calibration boundary, red-team suite,
  policy enforcement, and updated model documentation.
- Exit criteria: Real non-EXIF predictions pass CLI and browser acceptance on the
  target host; actual evaluation denominators/results are recorded on reviewed
  assets; calibration is honestly marked calibrated, preliminary, or uncalibrated;
  and critical safety findings are closed. Implementation tests alone do not pass
  these runtime gates.
- Dependencies: Phases 1–4 and governance approval.

### Phase 5C - Product completion while the custom model trains

- Status: Completed on 2026-07-12 for the bounded model-independent product
  shell. Phase 5B's real OCR/retrieval/combined-accuracy gates remain blocked and
  are not waived by this completion.
- Purpose: Complete truthful product workflows independently of custom-model
  training without weakening the blocked Phase 5B real-artifact gates.
- Deliverables: Canonical inference adapters/coordinator; production-refused,
  watermarked development simulation; strict local custom-artifact lifecycle with
  shadow/candidate/primary modes; TTL-bound history and retained-source rerun;
  provider/model status; read-only evaluation and dataset-QA reports; and compact
  investigation screens using the existing frontend design.
- Exit criteria: Existing EXIF/GeoCLIP/delete flows and API contracts remain
  compatible; simulated output is unmistakable and excluded from evaluation;
  unsafe/missing custom artifacts fail closed; history and reports preserve their
  privacy boundaries; backend/frontend/contracts/migrations/security checks pass;
  and unavailable real dependencies are reported honestly.
- Non-goals: No new trained weights, dataset download, benchmark claim,
  calibration claim, automatic promotion, production identity, distributed queue,
  model-serving fleet, SLO, or Phase 6 work.
- Dependencies: Existing Phase 5/5B interfaces, an operator-supplied reviewed
  artifact for real custom inference, and licensed held-out data for promotion.

## Phase 6 — Production deployment, observability, scaling, and release readiness

- Purpose: Operate the validated system reliably for authorized users.
- Inputs: Phase 5 release candidate, SLO/capacity/security/privacy requirements,
  chosen cloud and tile vendors, and legal review.
- Deliverables: AuthN/AuthZ, distributed queues/limits, secret management,
  deployment automation, observability, backup/erasure policy, runbooks, load and
  disaster tests, release gates.
- Exit criteria: SLO/load/security/privacy sign-off, rollback and incident drills,
  cost/capacity approval, and production release readiness.
- Dependencies: Phases 1–5 and organizational release approval.

### Phase 6A - Hybrid geolocation evidence and SegFormer-B2

- Status: Completed on 2026-07-14 for the bounded single-host pre-release
  extension. Phase 6B later restored the exact reviewed label mapping; that does
  not change the Phase 6A geographic responsibility boundary.
- Scope: A bounded pre-release extension of the existing single-host analysis
  pipeline, not completion of the Phase 6 production program.
- Deliverables: Trusted-checkpoint inspection and safe deployment preparation;
  optional lazy local SegFormer-B2 scene evidence; GeoCLIP internal Top-50/public
  Top-5 compatibility; dateline/pole-safe geodesic clustering; cached offline
  reverse naming; optional existing OCR/language evidence; explainable
  `phase6a-v1` reranking; qualitative uncalibrated confidence; additive API and
  frontend presentation.
- Responsibility boundary: GeoCLIP proposes geographic coordinates. SegFormer
  describes scene pixels only. Reverse geocoding names existing coordinates only.
  OCR contributes only real sufficiently strong public-place evidence. Missing
  optional providers remain visible and neutral.
- Artifact state: The current trusted checkpoint selects EMA, infers 124 outputs,
  and prepares against the official NVIDIA SegFormer-B2 Cityscapes base. The
  reviewed exact 124-entry Mapillary mapping is now restored in configuration;
  weights were not changed and segmentation remains descriptive only.
- Exit criteria: Checkpoint/artifact identity, real scene inference, real GeoCLIP
  clustering and final API smoke, graceful unavailable providers, compatibility,
  backend/frontend/contracts/migrations/security checks, and exact runbook
  commands must be recorded in `PROJECT_STATE.md`. Unit fixtures alone do not
  satisfy the runtime gates.
- Non-goals: No confidence calibration, DINOv2, new reference corpus, fake FAISS
  path, geographic scene-style rules, Phase 6B, load/SLO claim, distributed GPU
  fleet, production identity, or release-readiness claim.

### Phase 6B - Multi-model ensemble and bounded cloud review

- Status: Completed on 2026-07-14 for the bounded implementation and installed-host
  runtime. This is not Phase 6 production readiness. OSV-5M baseline, PLONK YFCC
  and PaddleOCR now pass real inference through isolated local CPU workers;
  RapidOCR has hash-bound real-inference proof and a completed Paddle-disabled
  HTTP fallback. PLONK OSV and iNaturalist remain prepared-only, while GeoCLIP
  and SegFormer remain real.
- Scope: Additive geographic provider contracts for OSV-5M and scene-routed PLONK,
  PP-OCR with reviewed fallback, source-family-aware explainable fusion, optional
  hard-case-only OpenAI candidate review, capability/cost UI, persistence,
  diagnostics, weights-only bootstrap and manifest-driven evaluation.
- Exit evidence: Exact Mapillary names restored without changing safetensors;
  GeoCLIP, SegFormer, OSV-5M baseline, PLONK YFCC, PaddleOCR and RapidOCR
  inference were exercised; full local pipeline smokes completed with both the
  primary and fallback OCR routes; worker identity/readiness, scheduler/OOM/correlation/fusion/cloud
  consent/cache/budget/API/migration/frontend tests passed. All fields remain
  optional and uncalibrated.
- Non-goals: No training dataset, full OSV/YFCC/iNaturalist/Mapillary download,
  retrieval gallery, DINOv2/FAISS corpus, raw-score averaging, calibrated
  confidence, Phase 6C, distributed serving, load/SLO or release claim.
- Next gate: Phase 6C has not started. A later explicitly approved phase may accept
  the remaining PLONK variants, run a
  representative licensed evaluation, and address production identity, durable
  queues, service supervision, observability and SLOs. Phase 6B itself stops here.

The OSV-5M training/imagery dataset remains blocked by its recorded license and
ethical review; Phase 6B permits only an explicit pinned pretrained-weight adapter
and does not download that dataset. The 2026-07-11 decision still permits only the
pinned bounded SigLIP2 B/16-384 retrieval implementation inside Phase 5B; it is not
a standalone GPS predictor or an automatic download. Distributed model serving,
large-scale indexing, SLOs and release operations remain later production work.
