# AtlasLens web

React/Vite client for the Phase 1 AtlasLens API. Local-only analysis still uploads
the image to the configured AtlasLens server, but the server does not invoke a
cloud analysis provider. Cloud-assisted analysis is separately selected and
consent-gated.

Phase 6C keeps the existing visual system and adds compact analysis/history,
evaluation, dataset-QA and candidate-recall workspaces. Simulated results always
carry a visible watermark; raw provider scores are categorical diagnostics, not
percent confidence. Candidate coordinates are rounded for normal display while
explicit copy/open actions remain available. Map loading/error/retry and a full
textual fallback preserve access when the configured basemap is unavailable.

History and Evaluation navigation is present in the normal workspace. Dataset QA
and the read-only `System Intelligence / Model & Data Status` view appear only when
`VITE_ENABLE_OPERATOR_UI=true`. This frontend flag grants no access: the API must
also run as development/test with `OPERATOR_API_ENABLED=true`. Production API
configuration refuses the unauthenticated operator surface.

The system view reads `GET /api/v1/system-intelligence` and displays only safe
model lifecycle state plus aggregate reference-index counts, coverage, attribution,
and leakage-audit status. It never renders local paths, secrets, reference image
URLs, reference identifiers, audit hashes, or exact reference coordinates. An
index is shown as usable only when its leakage status is `passed`.

Completed Phase 6C analyses include bounded provider run outcomes, fusion and
ablation summaries, and optional GeoCLIP, hierarchy, OSV, PLONK, MegaLoc, and
fused map overlays. Each overlay is capped at 12 points; fused uncertainty is
visible in the uncertainty view. Raw provider values remain diagnostics and are
not displayed as confidence.

```powershell
npm.cmd install
npm.cmd run generate:api
npm.cmd run dev
```

`VITE_API_BASE_URL` selects the API origin. In local development it can remain
empty and Vite proxies `/api` to `VITE_DEV_API_TARGET`. `VITE_MAP_STYLE_URL`
selects a MapLibre style. When unset, the result map uses the OpenFreeMap Positron
development style. Candidate overlays and the textual map automatically remain
available on the bounded tile-free fallback if a style or resource fails.
Production operators must select a tile/style provider whose license, terms,
privacy behavior, and capacity fit the deployment.

Validation commands are `npm.cmd run check:api`, `npm.cmd run lint`,
`npm.cmd run typecheck`, `npm.cmd test`, `npm.cmd run build`, and—while the real
API dependencies are available—`npm.cmd run test:e2e`.
