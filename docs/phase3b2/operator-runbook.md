# Phase 3B2 offline MegaLoc and first-party capture runbook

Phase 3B2 is implemented but production activation and real capture are blocked.
The existing local MegaLoc artifact passed a bounded technical smoke; its code,
weight and training-lineage approvals remain separate and fail closed. Do not use
these commands on real media until the capture policy, contributor authority,
privacy process and model approvals have explicit review records.

No command in this workflow downloads a model, dataset or image, contacts a cloud
provider, or provisions RunPod. Phase 6C remains disabled by default.

## Capture input contract

The plan is a contained UTF-8 JSON file under `--input-root`. It must select one
of `geotagged_images`, `ordered_frames_gpx`, `ordered_frames_csv`, or `video` and
must include opaque capture-run, sequence, contributor and device identifiers.
Timestamps are timezone-aware. Sampling is restricted to 25, 50 or 100 metres
and defaults to 50 metres. GPX/CSV interpolation is linear only inside the
configured time gap; invalid WGS84 coordinates, non-increasing route timestamps,
impossible speeds and excessive route distance fail with safe reason codes.

Video extraction requires an explicit existing `ffmpeg`/`ffmpeg.exe` path in the
plan. AtlasLens never installs it. When absent, the CLI returns exactly:

```text
FFMPEG_NOT_AVAILABLE — provide an explicit existing safe executable path; AtlasLens will not download FFmpeg.
```

## Offline capture commands

Run from the repository root with the existing API environment. Replace the
paths and receipt placeholders only after the approval gate above is satisfied.

```powershell
$capture = '.\services\api\.venv\Scripts\atlaslens-corpus.exe'

& $capture inspect-capture --input-root C:\approved\capture --work-root C:\private\atlaslens-capture --plan capture-plan.json
& $capture sync-gpx --input-root C:\approved\capture --work-root C:\private\atlaslens-capture --plan capture-plan.json
& $capture sample-route --input-root C:\approved\capture --work-root C:\private\atlaslens-capture --plan capture-plan.json
& $capture import-capture --input-root C:\approved\capture --work-root C:\private\atlaslens-capture --plan capture-plan.json --dry-run
& $capture import-capture --input-root C:\approved\capture --work-root C:\private\atlaslens-capture --plan capture-plan.json
```

`--dry-run` performs synchronization and distance-sampling validation without
creating the work root or copying media. A real import writes sampled originals
to private `quarantine/` with privacy state `pending`; it never silently approves
them. Interrupted imports resume only against the same plan fingerprint and
source identities.

## Privacy and revocation

Only an explicit human review with receipt references can change `pending` to
`approved`, `rejected`, or `needs_redaction`. Approval without redaction publishes
the reviewed file to `assets/`. A redaction approval requires a contained,
manually prepared JPEG/PNG; AtlasLens validates and atomically publishes that
derived copy while retaining the original under private `originals/`. There is
no automatic face/plate detector and no redaction-accuracy claim.

```powershell
& $capture privacy-review --work-root C:\private\atlaslens-capture --asset-id <opaque-id> --state approved --reviewed-by <reviewer-id> --reviewed-at 2026-07-17T12:00:00+03:00 --decision-receipt-sha256 <64-hex>

& $capture privacy-review --work-root C:\private\atlaslens-capture --asset-id <opaque-id> --state approved --reviewed-by <reviewer-id> --reviewed-at 2026-07-17T12:00:00+03:00 --decision-receipt-sha256 <64-hex> --redaction-applied --redaction-receipt-sha256 <64-hex> --redacted-input-root C:\approved\manual-redactions --redacted-copy reviewed.png

& $capture privacy-review --work-root C:\private\atlaslens-capture --asset-id <opaque-id> --revoke-request-id <opaque-request-id> --revocation-receipt-sha256 <64-hex> --revoked-at 2026-07-17T13:00:00+03:00

& $capture build-capture-manifest --work-root C:\private\atlaslens-capture --output manifest.json
```

The manifest contains only `approved` and `ACTIVE` assets and is compatible with
the existing Phase 3B1 rights gate, ingestion, descriptor checkpointing and FAISS
publication path. Pending, rejected, needs-redaction and revoked assets remain
excluded. A revocation physically removes a published file from `assets/` and
requires the existing Phase 3B1 revocation/rebuild workflow for any previously
published descriptor/index.

## Inspected local MegaLoc artifact

Only these exact existing local files were inspected; none is tracked or copied
into Git:

- `.local/vendor/megaloc/megaloc_model.py`: 15,726 bytes, SHA-256
  `c0848dfb287ba15b519d7b54415db824e16ec2f2b5a6899507b0476cf3379767`,
  source revision `1af071c68fc3ab6c6018c5c868391763516e50f7`.
- `.local/vendor/megaloc/LICENSE`: 1,107 bytes, SHA-256
  `40c6c4894aecc5b676f0fb93697a6c1f82b08df71b25e485b662779b2c899667`;
  the inspected code license is MIT.
- `.local/models/phase6c/megaloc/model.safetensors`: 914,577,436 bytes,
  SHA-256 `d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8`,
  model revision `7cb9f7970d366fdf059963d04d372e503e8e9df9`.
- `.local/models/phase6c/megaloc/receipt.json`: 485 bytes, SHA-256
  `0acf870c846dea64c60aca7d4d5a513dbaa6ba9aaf0f87f2c157486a150a95c6`.
  The local receipt labels the weight MIT and records no dataset download.
- `.local/models/phase6c/megaloc/phase3b2-production-smoke.json`: 587 bytes,
  SHA-256 `16762c8f1a43cc30f0e61fba8a502218575fec8410be45c167eb0359ccc4efe8`.

The named training lineage (GSV-Cities, MSLS, MegaScenes, ScanNet and SF-XL) has
not received professional review here. The code label, local weight receipt and
successful technical smoke support only a bounded non-production engineering
pilot. They do not authorize production activation or commercial release.

## MegaLoc activation gate

`MEGALOC_PHASE3B2_CONFIG_PATH` and the five Türkiye reference descriptor fields
in `.env.example` are intentionally empty. The configured file must resolve under
the repository root and bind the exact local source, license, install receipt,
`model.safetensors` SHA-256/size, 8,448 dimension and preprocessing version.
Production registration additionally requires explicit source/weight/production
approval records and a matching real smoke receipt. A technical smoke cannot
promote an injected test backend or override missing approvals.

The Türkiye reference provider stays disabled by default. Even when enabled, it
reports `not_ready` unless the approved artifact identity and a checksum-verified,
source-policy-bound compatible real index both exist. It remains outside current
fusion, ranking, weights and thresholds.

## Synthetic verification only

```powershell
$env:PYTHONPATH = 'D:\geoSearch\services\api\src'
D:\geoSearch\services\api\.venv\Scripts\python.exe -m pytest D:\geoSearch\services\api\tests\test_phase3b2_megaloc_capture.py D:\geoSearch\services\api\tests\test_phase3b2_e2e.py -q
```

The fixtures are generated in temporary storage. Their one-item FAISS retrieval
result is construction proof only, not real geolocation accuracy, calibration or
commercial readiness.
