# Phase 1 implementation plan

Status: frozen for implementation on 2026-07-10.

## Objective and boundary

Deliver one real, privacy-conscious vertical slice: a user uploads JPEG, PNG, or
WebP; the API validates and decodes it; extracts safe metadata, EXIF GPS, and
quality measurements; optionally invokes configured OCR or consent-gated cloud
vision; deterministically synthesizes provenance-backed candidates or abstains;
streams real progress; renders uncertainty and evidence; and deletes the
analysis. No global model, dataset, retrieval index, map constraint, regional
expert, calibration system, or geometric verification is implemented in Phase 1.

## Reconciled discovery

- The workspace was empty, so the prescribed monorepo structure is the least
  disruptive starting point.
- Windows has Python 3.12, Node 24, npm 11, uv, and Git. PowerShell script
  execution is restricted, so documented commands use `npm.cmd` and explicit
  `-ExecutionPolicy Bypass`. Docker, a WSL distribution, CUDA, and Tesseract are
  absent; none is required for the default local-only application.
- WGS84 is the canonical global coordinate system. API `center` fields use
  latitude/longitude; GeoJSON uses longitude/latitude ordering.
- Confidence is a versioned source-support score in Phase 1, not a calibrated
  probability. Spatial uncertainty does not establish metadata authenticity.
- Local-only means processing on the AtlasLens server without a cloud analysis
  provider; it does not mean the image never leaves the browser.
- Cloud providers receive only a normalized, resized, metadata-stripped
  derivative. Raw OCR and EXIF are never sent alongside it.

## Frozen shared contracts

- Public HTTP and SSE contract: `packages/contracts/openapi.yaml`.
- Provider and fusion contract: `docs/model-provider-contract.md`.
- Stable progress and error codes cross the API; the web app localizes them.
- Evidence types and provider identifiers are extensible strings so future
  providers degrade safely in old clients.
- Every candidate has provenance, evidence references, positive radius,
  confidence semantics, uncertainty basis, and verification status.
- Completed jobs may contain no candidates and an explicit abstention object.

Contract changes during Wave 2 require delivery-lead review and an entry in
`docs/decision-log.md`; implementers must not edit shared contracts silently.

## Wave 2 ownership

| Owner | Exclusive write scope | Deliverable |
|---|---|---|
| Backend agent | `services/api/**` | FastAPI, persistence, migrations, job/SSE pipeline, providers, cleanup, tests |
| Frontend agent | `apps/web/**` | Responsive EN/TR React UI, API validation, MapLibre uncertainty, unit and E2E tests |
| DX/CI agent | `infra/**`, `scripts/**`, `.github/**` | Dockerfiles, cross-platform scripts, CI and validation helpers |
| Delivery lead | Root files, `packages/contracts/**`, `docs/**` | Contracts, integration, Compose, documentation, fixes, final verification |

Worktrees are unavailable because the directory began without Git history. The
non-overlapping ownership above is mandatory.

## Integration sequence

1. Review all Wave 2 changes against the frozen contract and security boundary.
2. Generate frontend API types from OpenAPI and verify no manually duplicated
   incompatible response models exist.
3. Install from deterministic lockfiles; migrate the SQLite database.
4. Run backend lint, type checks, and tests; frontend lint, type checks, tests,
   and production build; contract validation; generated-type drift check.
5. Start API and web, run a generated EXIF vertical slice, verify a no-EXIF
   abstention, SSE terminal event, and deletion.
6. Run Playwright smoke without depending on live map tiles.
7. Validate Compose when Docker is available; otherwise record the exact
   environmental blocker without converting it to a pass.
8. Perform read-only adversarial Wave 3 review, fix critical/high findings, and
   rerun affected verification.

## Acceptance risks and mitigations

- Native image parser/CPU risk: compressed-byte, decoded-pixel, per-dimension,
  animation, format, and decompression-bomb checks; future process isolation is
  documented for Phase 5/6.
- Sensitive data: allowlisted logs, random temporary paths, no raw request-body
  logging, TTL cleanup, `finally` cleanup, no-store responses, and sentinel log
  tests.
- Cloud abuse: server-side mode/consent/key triple gate, separate rate limit,
  bounded concurrency, timeout, retry, and hypothesis count.
- False precision: EXIF is labeled metadata-only and potentially stale; vision
  confidence is capped at 0.40 with at least 25 km radius and remains unverified.
- SSE exhaustion: bounded subscriptions, heartbeat, finite lifetime, terminal
  close, and disconnect cleanup.

