# Phase 3B3 private Mapillary Türkiye demo runbook

This is a private, non-production technical demonstration. It is disabled by
default, loopback-only and isolated from Phase 6C, the production Türkiye index,
user evidence and any future first-party corpus. It does not establish Türkiye-
wide accuracy, calibrated confidence, public-distribution permission, production
readiness or commercial/legal clearance.

The pilot decision is **PRIVATE DEMO PARTIAL — COVERAGE OR ACCURACY
INSUFFICIENT**. Stop after local review. Do not expand acquisition, tune against
the holdout/operator images, provision cloud infrastructure or publish the demo
without a new explicit authorization.

## Token and network boundary

- Acquisition used only `https://graph.mapillary.com` through the official API.
  No Mapillary page was scraped.
- `MAPILLARY_ACCESS_TOKEN` was loaded only by the named, bounded backend loader.
  Its value, fragments, headers and token-bearing URLs were never printed,
  persisted, sent to the web client or committed.
- Serving the published private index does not need the Mapillary token and the
  startup procedure below deliberately avoids the repository `.env`.
- The final exact-value scan found zero runtime-token matches in tracked files;
  `.env` remains ignored and untracked.
- Real network use was limited to the authorized coverage/acquisition calls.
  MegaLoc descriptor, index, benchmark, operator and API smoke execution was
  local/offline.

Official references last checked at closure:

- [Mapillary API documentation](https://www.mapillary.com/developer/api-documentation/)
- [Accessing imagery and data through the Mapillary API](https://help.mapillary.com/hc/en-us/articles/360010234680-Accessing-imagery-and-data-through-the-Mapillary-API)
- [Mapillary CC-BY-SA and attribution guidance](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data)

## Coverage audit

The audit was metadata-only and completed before imagery acquisition. It made 20
requests and processed 20 pages against the five AOIs in
`config/phase3b3/mapillary-aoi-v1.json`; `imagery_downloaded` is false in the
audit artifact.

| AOI | Accepted | Rejected | Sequences | Contributors | Cells | Capture years | Bounded sequence-track km | Eligible |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Kayseri urban | 375 | 0 | 19 | 4 | 17 | 2019:375 | 8.923907 | yes |
| Ankara urban | 381 | 0 | 18 | 9 | 12 | 2018:11, 2019:176, 2020:63, 2023:33, 2025:98 | 5.526765 | yes; selected |
| Sivas urban | 63 | 132 incomplete geometry | 3 | 3 | 5 | 2019:1, 2021:55, 2026:7 | 1.205653 | no |
| Kayseri–Ankara corridor | 177 | 0 | 14 | 7 | 14 | 2018:37, 2019:127, 2020:13 | 10.881485 | yes |
| Kayseri–Sivas corridor | 100 | 0 | 4 | 1 | 1 | 2016:100 | 0.077480 | no |

These kilometre values are bounded sums of sequence-track segments, not unique
road-network length. The audit artifact also retains compass-bin diversity,
image/sequence density and configured upper-bound download/descriptor/index size
estimates. Kayseri had only 375 eligible images and therefore failed the explicit
1,000-image preference gate. The deterministic fallback correctly selected
`ankara-urban-v1`; no other city is described as Kayseri.

Evidence identities:

- Coverage audit file SHA-256:
  `af4149faac9a0cafcbb206ee5d25b916bab0b5dc8839f3ca4bf74f1390ee1be3`.
- Acquisition plan SHA-256:
  `c7953b22c64184f385488399936d151557ac4b4b2ca39e73806e1fab027c2bd0`.
- Current acquisition manifest SHA-256:
  `a2f143809ef09ac8e3ebff2134681ae4fc619d9518332535a0e4e0cf0bd22ee1`.

## Bounded acquisition and locked split

The plan targeted 1,500 references plus 100 holdout queries but imposed hard
limits of 2,000 total images, 2 GiB raw download, 2,500 requests and concurrency
two. The official API exposed only a much smaller usable Ankara pool:

- 81 1024-pixel renditions; 7,488,730 downloaded bytes.
- 85 requests: four metadata pages plus 81 media requests.
- 18 sequences and nine contributors.
- Capture range 2018-08-24 through 2025-12-03.
- Acquired-year distribution: 2018:3, 2019:48, 2020:7, 2023:15, 2025:8.
- Locked references: 29; reference shortfall: 1,471.
- Locked holdout: 11; holdout shortfall: 89.

The split was frozen before descriptor generation. It has no shared image ID,
exact duplicate, perceptual near duplicate or same sequence between reference and
holdout. Adjacent-frame controls apply; contributor/date separation is preferred,
and valid geographic positives remain defined transparently at 25, 100, 500 and
1,000 metres.

- Selection-lock identity:
  `8d8ad89d3c9045a6d5aa164854bee4b11d56bd0409ce51e539678b65c596d6f4`.
- Persisted split file SHA-256:
  `1e9b658ee023f3051431d5e861dc9a1bf2291024672f9059e77ec44dfaa0ebfd`.
- Phase 3B1 manifest SHA-256:
  `a77aef76e16b0997a5be61da536e2429534c97c027b2335bc98e450dfd3dbec9`.
- Phase 3B1 split-lock identity:
  `53f083d476b252069f36583fe27963b4ac75bbbac69250ff52ea175729a110e1`.

## Real MegaLoc execution and FAISS publication

The runner reused the existing Phase 3B2 artifact and refused replacement or
hash mismatch:

- Model artifact SHA-256:
  `d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8`.
- Descriptor count/dimension: 40 / 8,448, finite and L2-normalized.
- Device/batching: CUDA; batch size eight; five batch calls.
- Runtime/throughput: 18.774714 seconds / 2.130525 images per second.
- Peak CUDA memory allocated: 2,650,686,464 bytes (2.469 GiB).
- Performance-measurement network use: false.
- Measurement-only descriptors persisted: false.

The versioned `mapillary-private-demo-faiss-flatip-v1` `IndexFlatIP` publication
contains 29 reference vectors. `vectors.faiss` is 980,013 bytes; the complete
seven-file bundle is 1,044,619 bytes.

- Source-policy SHA-256:
  `72d51363f2b63de368d34d4d7bb2fc1145f93dc0469e7026100976732dfde209`.
- Index-build identity:
  `ef65227f519e75fa83e3a4a9c61a08adf4475b95dd42a5e43708dccc5d10027e`.
- Bundle metadata file SHA-256:
  `524734d07eb9f75a5d17fc426a8f38e8483b70e07877cf5875b8dd12f32b9d9f`.
- Attribution file SHA-256:
  `c97d926ab4c21d806f50df9e0086a91a729d998f386b09a6de01334fb981147d`.
- Publication marker/file SHA-256:
  `bcb538df0ed3205bc70ca0383f2c88ad88cea849fdeb82e690756b18028780a3`.
- Performance receipt internal/file SHA-256:
  `6b38d6eefeebc1f6af845349859e5f7bfe5ac86a844eb2acce3dce47cd3b9c17` /
  `612b4292673cf3cfb76aca612e8d1720697df0940cddb5ec5454963d02017335`.
- Aggregate run receipt internal/file SHA-256:
  `d0e101a71a5df76aab780c3768c9a13b7f46b0fdc02580b2a59ac0156c60b949` /
  `11afe2d9a3e9a5c963e84d2ea178d3ba96a4b01d2f56d91c57e3d2b9df2a82ab`.

The bundle opens only when model, dimension, source policy, split lock,
attribution, publication and file checksums agree. It is not distributed or
tracked by Git.

## Frozen real holdout benchmark

The benchmark used all 11 locked queries and was not used to tune sampling,
ranking or thresholds.

| Positive radius | Recall@1 | Recall@5 | Recall@10 |
|---:|---:|---:|---:|
| 25 m | 0.090909 | 0.272727 | 0.272727 |
| 100 m | 0.181818 | 0.545455 | 0.545455 |
| 500 m | 0.454545 | 0.636364 | 0.727273 |
| 1 km | 0.454545 | 0.727273 | 0.727273 |

- Median geodesic error: 7,122.3628 m; p90: 10,381.0232 m.
- Failure groups: six beyond 1 km, three within 500 m, one within 100 m and one
  within 25 m.
- Reference density within 1 km: eight queries with 1–5 references and three
  with 6–20.
- Different/same capture year: five queries with Recall@1@100 m 0.0; six with
  0.333333.
- Different/same contributor: nine queries with Recall@1@100 m 0.111111; two
  with 0.5.
- City and province accuracy are 1.0 only because this is a one-city index; they
  do not demonstrate Türkiye-wide classification.
- No calibrated similarity threshold exists. Zero queries abstained. The stored
  `abstention_coverage=1.0` means all queries produced ranked output, not that a
  calibrated abstention policy succeeded.

Benchmark result/file SHA-256:
`0c9fec5def2ac26314f626d8a9db4737c6233f8045b3fafc55581036aa067934` /
`ebf0c5f75bafc9530aa2c54b2582a3c9c77042ad67a7c7d4d03a76d09cea4acd`.

## One-time operator blind smokes

The two authorized desktop images were each queried exactly once after the
benchmark froze. They were not copied, added to the corpus, used for sampling or
threshold tuning, retained as descriptors, or stored as API analysis records.
Exact operator ground-truth coordinates were unavailable, so no exact error is
claimed. Individual coordinates and source URLs are intentionally not copied
into Git documentation; the live result carried the required source attribution.

- Kayseri-claimed image: no abstention only because no calibrated threshold
  exists; all five candidates were Ankara; claimed city absent. Top-1 raw cosine
  similarity/distance: 0.0748085 / 0.9251915.
- Ankara-claimed image: no abstention; all five candidates were Ankara; claimed
  city present. Top-1 raw cosine similarity/distance: 0.1089478 / 0.8910522.

These are retrieval observations, not proof of the photographed location.

## Local API and web startup

Prerequisites are the retained pilot bundle under
`C:\AtlasLensPilot\mapillary-demo`, the existing local MegaLoc runtime and no
listeners on ports 8794, 8000 or 5173. The worker script name reflects reused
Phase 6C infrastructure; running it does **not** set `PHASE6C_ENABLED=true`.

Terminal 1, from the repository root:

```powershell
Set-Location D:\geoSearch
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-phase6c-workers.ps1 -Device cuda
```

Terminal 2 starts only the loopback API. It removes inherited cloud/Mapillary
secrets from that terminal and changes to `C:\tmp`, so the Settings env-file
search cannot discover `D:\geoSearch\.env`:

```powershell
Remove-Item Env:MAPILLARY_ACCESS_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue

$env:PYTHONPATH = 'D:\geoSearch\services\api\src'
$env:APP_ENV = 'development'
$env:API_HOST = '127.0.0.1'
$env:DATABASE_URL = 'sqlite:///C:/tmp/atlaslens-phase3b3-demo.db'
$env:TEMP_STORAGE_DIR = 'C:\tmp\atlaslens-phase3b3-demo-storage'
$env:KEEP_UPLOADS = 'false'
$env:GLOBAL_MODEL_ENABLED = 'false'
$env:CUSTOM_MODEL_ENABLED = 'false'
$env:RETRIEVAL_ENABLED = 'false'
$env:PHASE6A_ENABLED = 'false'
$env:PHASE6B_ENABLED = 'false'
$env:PHASE6C_ENABLED = 'false'
$env:OPENAI_GEO_ENABLED = 'false'
$env:MAPILLARY_ENABLED = 'false'
$env:REFERENCE_INDEX_ENABLED = 'false'
$env:TURKIYE_REFERENCE_INDEX_ENABLED = 'false'
$env:MEGALOC_WORKER_ENABLED = 'true'
$env:MEGALOC_WORKER_HOST = '127.0.0.1'
$env:MEGALOC_WORKER_PORT = '8794'
$env:MEGALOC_DEVICE = 'cuda'
$env:MEGALOC_TIMEOUT_SECONDS = '180'
$env:ATLASLENS_TURKIYE_DEMO_ENABLED = 'true'
$env:ATLASLENS_TURKIYE_DEMO_BUNDLE_PATH = 'C:\AtlasLensPilot\mapillary-demo\derived\mapillary-faiss-index'
$env:ATLASLENS_TURKIYE_DEMO_EXPECTED_PUBLICATION_SHA256 = 'bcb538df0ed3205bc70ca0383f2c88ad88cea849fdeb82e690756b18028780a3'
$env:ATLASLENS_TURKIYE_DEMO_EXPECTED_SOURCE_POLICY_SHA256 = '72d51363f2b63de368d34d4d7bb2fc1145f93dc0469e7026100976732dfde209'
$env:ATLASLENS_TURKIYE_DEMO_EXPECTED_SELECTION_LOCK_SHA256 = '8d8ad89d3c9045a6d5aa164854bee4b11d56bd0409ce51e539678b65c596d6f4'
$env:ATLASLENS_TURKIYE_DEMO_TOP_K = '5'
$env:ATLASLENS_TURKIYE_DEMO_UNCERTAINTY_RADIUS_M = '1000'

Set-Location C:\tmp
& 'D:\geoSearch\services\api\.venv\Scripts\python.exe' -m uvicorn atlaslens_api.main:app --host 127.0.0.1 --port 8000
```

Verify status at
`http://127.0.0.1:8000/api/v1/mapillary-demo/status`. `state=active` is expected
only after worker real-inference verification and all bundle gates pass.

Terminal 3 starts Vite on loopback without loading repository web env files:

```powershell
Set-Location D:\geoSearch\apps\web
$env:ATLASLENS_IGNORE_ENV_FILE = 'true'
$env:VITE_DEV_API_TARGET = 'http://127.0.0.1:8000'
npm.cmd exec -- vite --configLoader runner --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://127.0.0.1:5173/` and select **Mapillary demo**. The page must show
`Deneysel Mapillary Referans İndeksi` and `Özel teknik demo — üretim sistemi
değildir`. A query requires an explicit authorization acknowledgment. The upload
is normalized locally, deleted after the settled request and never sent to
Mapillary or another cloud provider.

The closure smoke observed active status, an attributed five-candidate Ankara
response, `confidence=null`, `uncalibrated_unavailable`, and zero temporary files
after the query. The production frontend build served HTTP 200 on loopback only.

Stop the web and API with Ctrl+C in their terminals, then run:

```powershell
Set-Location D:\geoSearch
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-phase6c-workers.ps1
```

The validated start/stop cycle left ports 8794/8000/5173 closed, removed trusted
worker metadata and left no launcher process. The temporary SQLite/storage paths
above are not corpus or evidence storage; inspect those exact paths before any
manual deletion.

## Uncertainty, attribution and legal gate

The runner's 100 m value is a conservative operator-selected ingestion
uncertainty, not Mapillary-supplied GPS accuracy. The private API has a separate
1,000 m default presentation radius. Neither is probability, model confidence or
validated location accuracy.

All 40 split assets have stable Mapillary IDs, source pages, contributor where
available, capture date, coordinates, CC-BY-SA-4.0 identifier/link and provenance.
No signed thumbnail URL is persisted. The 29-reference index still opens against
the 40-record attribution inventory after cleanup. Mapillary imagery is not owned
by AtlasLens.

The source policy records `GO_WITH_ATTRIBUTION` only as Phase 3B1 compatibility
and private-pilot admission. It is **not** public, production, commercial or legal
clearance. `public_distribution=false`, `production_activation=false`, and
commercial use remains `LEGAL_REVIEW_REQUIRED`.

## Raw-image cleanup

Cleanup was dry-run first and bounded to
`C:\AtlasLensPilot\mapillary-demo`:

- Dry-run raw candidates: 70 / 12,103,841 bytes.
- Dry-run materialized candidates: 40 / 6,869,214 bytes.
- Executed raw deletion: 70 / 12,103,841 bytes.
- Executed Phase 3B1 materialized deletion: 40 / 6,869,214 bytes.
- Retained: 11 JPEGs / 1,754,198 bytes and 11 separate attribution sidecars.
- Remaining materialized Phase 3B1 asset files: zero.
- Current derived output: 3,952,784 bytes; whole pilot root: 5,931,113 bytes.

Metadata, attribution, hashes, descriptors and the index remain outside Git.

## Validation, disk, downloads and cloud

- Phase 3B3/worker/holdout focused tests: 51 passed.
- Phase 3B1/3B2 regressions: 75 passed, three host-specific skips.
- Full backend: 823 collected; 816 passed, seven skipped.
- Frontend: 93 passed; ESLint, TypeScript, OpenAPI drift and production build
  passed.
- Ruff and strict mypy for six changed backend source files passed.
- Credential-shape scan: zero high-confidence or unexpected literal hits; exact
  runtime token tracked hits: zero.
- Heavy changed artifact scan: zero image/model/dataset/index/database files;
  no changed file exceeds 1 MiB.
- Preflight disk: C: 52.817 GiB, D: 11.971 GiB free. Closure snapshot: C: 52.571
  GiB, D: 11.971 GiB; the 35/8 GiB gates passed.
- Downloads: exactly 81 Mapillary renditions / 7,488,730 bytes. Models: **NONE**.
  Datasets: **NONE**.
- Cloud resources, Pods, volumes, accounts, billing and spend: **NONE**.
- Git remote, push and pull request: **NONE**.

## Exact next step

Stop and review the partial private-demo evidence. Do not acquire more imagery,
expand caps, tune against the locked holdout/operator images, provision RunPod,
publish/distribute the bundle or begin commercial deployment without a new
explicitly scoped phase authorization and the required legal/rights review.
