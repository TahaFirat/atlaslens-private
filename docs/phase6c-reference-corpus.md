# Phase 6C Türkiye reference-corpus runbook

## Boundary

This workflow creates a bounded inference-time retrieval gallery. It is not a
training dataset, does not use an uploaded image as a source, does not contain a
target/Kayseri preference, and never scrapes Google, Bing, search engines, or
unreviewed websites. Planning is offline. Network discovery and image download
require the two explicit execution flags shown below.

Supported remote sources are the official Mapillary Graph API and the official
KartaView photo API. Manually supplied licensed images remain supported directly
by the checked `atlaslens-megaloc-reference-input-v1` builder contract; the
acquisition command does not silently import or relabel them.

Source and license references reviewed for this foundation:

- Mapillary API/SDK: <https://mapillary.github.io/mapillary-python-sdk/>
- Mapillary image licensing and attribution: <https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data>
- KartaView photo API: <https://kartaview.org/doc/photos>
- KartaView terms, last modified 17 June 2025: <https://kartaview.org/terms>
- CC BY-SA 4.0: <https://creativecommons.org/licenses/by-sa/4.0/>

The operator must re-review current source terms before a real acquisition.
KartaView access is refused unless the command receives the exact reviewed terms
version `2025-06-17`. Mapillary creator metadata is mandatory so per-item
attribution can be retained. KartaView items retain the required
`© Grab and KartaView Contributors` credit.

## Exact backend environment

Copy `.env.example` to the ignored `services/api/.env` file if it does not already
exist. Keep every private media root outside the publishable source tree and refer
to it only as:

```text
<PRIVATE_MEDIA_ROOT>
```

Set only the backend values below. The access token has no command-line flag and
is never exposed through `VITE_*`, API output, receipts, or logs.

```text
MAPILLARY_ENABLED=false
MAPILLARY_ACCESS_TOKEN=
MAPILLARY_MAX_IMAGES=2025
MAPILLARY_IMAGE_WIDTH=1024
```

Allowed widths are `256`, `1024`, and `2048`. The default global limit is 2,025
(25 × 81 provinces). A configured Mapillary maximum is also a hard upper bound
for the command. Do not put the token in PowerShell arguments, source URLs, a
frontend environment variable, or a committed file.

## 1. Offline plan and estimate

From the repository root:

```powershell
.\services\api\.venv\Scripts\python.exe .\scripts\acquire_turkiye_references.py
```

This writes `.local\phase6c-reference\acquisition-plan.json` and prints only
safe aggregate fields: 81 planned provinces, estimated image count, estimated
storage, source enablement, and whether network access was authorized. It does
not call Mapillary or KartaView.

The checked plan derives every province uniformly from the pinned coordinate
catalogue. Each target declares urban/rural/unknown, eight heading bins, and all
four seasons plus unknown. These are selection strata, not claims that each
remote API supplies every metadata field or that imagery exists in every stratum.

Defaults are deliberately bounded:

- 2,025 total planned images;
- 25 per province;
- 4 per source sequence;
- 2 GiB hard disk limit;
- 512 KiB per-image estimate and 8 MiB actual per-image maximum;
- no more than 100 Mapillary results or 250 KartaView results per province query.

The command refuses an estimate above the configured disk limit. More than 5,000
planned images additionally requires `--confirm-large-index`; that flag never
overrides the hard disk limit.

## 2. Local administrative metadata gate

Real acquisition requires the reviewed local GeoNames reverse gazetteer. Verify
it without downloading anything:

```powershell
.\services\api\.venv\Scripts\atlas.exe gazetteer verify
```

If it is not installed, installation is a separate explicit network action:

```powershell
.\services\api\.venv\Scripts\atlas.exe gazetteer install
.\services\api\.venv\Scripts\atlas.exe gazetteer verify
```

A remote item is accepted only when the local resolver returns country `TR` and
the same province as the balanced acquisition target. City is retained only when
the resolved place is within 25 km; otherwise it is explicitly null. This local
city-based reverse lookup is not an authoritative administrative-boundary
polygon. Consequently, `covered_province_count` is corpus metadata with stated
resolver semantics, not proof of legal boundary coverage.

## 3. Explicit Mapillary acquisition

After reviewing current terms, put the client token only in the root `.env`, set
`MAPILLARY_ENABLED=true`, inspect the printed estimate, and run:

```powershell
.\services\api\.venv\Scripts\python.exe .\scripts\acquire_turkiye_references.py --execute --confirm-download
```

The Graph API token is sent in `Authorization: OAuth ...`, never in a URL.
Metadata and thumbnail hosts are HTTPS-allowlisted, redirects are revalidated,
responses are size-bounded, and 429/5xx responses use bounded exponential
backoff. The implementation keeps only an in-process metadata cache; it does not
persist signed thumbnail URLs.

## 4. Optional KartaView fallback

KartaView is disabled unless both its source flag and exact terms acceptance are
present:

```powershell
.\services\api\.venv\Scripts\python.exe .\scripts\acquire_turkiye_references.py --execute --confirm-download --enable-kartaview --accept-kartaview-terms-version 2025-06-17
```

It uses only the documented `api.openstreetcam.org/2.0/photo/` endpoint and
approved KartaView/OpenStreetCam image hosts. Coverage may be absent. Missing or
malformed source ID, sequence ID, coordinate, or authorized image URL causes an
item to be skipped; no replacement metadata is fabricated.

## Balance, provenance, and uncertainty

Discovery is oversampled by a small fixed factor, then selection is greedy and
deterministic over the least represented source, source sequence, heading bin,
capture season, and urbanicity value. A global round-robin across all 81 province
targets prevents an early or high-coverage province from filling the corpus.
Unknown heading, season, urbanicity, and city remain unknown.

Every accepted builder record contains source, source image/sequence IDs,
credential-free stable source URL, coordinates, a positive source-location
uncertainty radius, heading and capture time when supplied, country, locally
resolved province/city, exact license URL, attribution, opaque local asset key,
relative thumbnail path, SHA-256, and dHash64. The 50 m Mapillary and 75 m
KartaView radii are conservative source-location uncertainty defaults, not model
confidence or per-image accuracy claims. MegaLoc descriptor and index versions
are added and checked by the subsequent index builder.

## Resumption and outputs

Each accepted image atomically updates:

- `reference-input.json` — exact `atlaslens-megaloc-reference-input-v1` builder input;
- `acquisition-receipt.json` — reviewed source-terms versions, resumable IDs,
  relative paths, byte sizes, SHA-256, and perceptual hashes, with no coordinate,
  token, or signed URL;
- `images/<source>/<opaque-id>.<format>` — authorized JPEG/PNG/WebP bytes only.

On resume, the command requires both manifest and receipt, verifies every local
asset SHA-256, and refuses mismatched plan fingerprints or source-terms versions.
A manifest-first interrupted atomic update is recovered only from its checked
local asset. Exact SHA-256 duplicates and dHash64 near-duplicates (Hamming distance
at most four) are rejected before admission. Actual bytes are checked against the
hard disk limit before each atomic write. Original source filenames are never used.

Build and verify the separate retrieval index only after reviewing the partial or
complete acquisition report and running the leakage audit:

```powershell
.\services\api\.venv\Scripts\python.exe .\scripts\build_phase6c_leakage_descriptors.py `
  --holdout-manifest .\.local\evaluation\turkey_holdout.json `
  --holdout-id user-holdout-001 `
  --reference-input .local\phase6c-reference\reference-input.json `
  --reference-root .local\phase6c-reference `
  --worker-port 8794 `
  --device cuda `
  --output .local\evaluation\turkey-reference-megaloc-descriptors.npz
.\services\api\.venv\Scripts\python.exe .\scripts\audit_reference_leakage.py `
  --holdout-manifest .\.local\evaluation\turkey_holdout.json `
  --holdout-id user-holdout-001 `
  --reference-input .local\phase6c-reference\reference-input.json `
  --reference-root .local\phase6c-reference `
  --descriptor-npz .local\evaluation\turkey-reference-megaloc-descriptors.npz `
  --write-filtered-reference-input .local\phase6c-reference\reference-input.leakage-filtered.json `
  --output .local\evaluation\turkey-reference-leakage.initial.json
.\services\api\.venv\Scripts\python.exe .\scripts\audit_reference_leakage.py `
  --holdout-manifest .\.local\evaluation\turkey_holdout.json `
  --holdout-id user-holdout-001 `
  --reference-input .local\phase6c-reference\reference-input.leakage-filtered.json `
  --reference-root .local\phase6c-reference `
  --descriptor-npz .local\evaluation\turkey-reference-megaloc-descriptors.npz `
  --output .local\evaluation\turkey-reference-leakage.filtered.json
.\services\api\.venv\Scripts\python.exe .\scripts\build_turkey_reference_index.py --help
.\services\api\.venv\Scripts\python.exe .\scripts\verify_reference_index.py --help
```

Run the filtered-input audit only when the initial audit actually wrote a filtered
input. The initial audit remains failed and exits nonzero when it finds exclusions;
only the separate rerun may pass. Filtering is an atomic manifest copy operation:
it does not change `reference-input.json` or delete downloaded images, and it
refuses ambiguous exclusion keys or an empty result. Keep both reports as the
audit trail.

The descriptor builder is local-only and uses the prediction-side holdout
projection. It validates actual SHA-256 values and strict reference-root path
containment, then atomically writes only real descriptors from the exactly pinned
MegaLoc worker. Its console output contains counts, versions, dimension and byte
size only; it never prints paths, coordinates, truth, or vectors. The original
artifact may be reused for the filtered rerun because it is a checked superset.

The build and verify compatibility entrypoints delegate to the pinned MegaLoc
implementations; the original `build_megaloc_reference_index.py` and
`verify_megaloc_reference_index.py` names remain supported.

No real source availability or province coverage is claimed until the command
runs with authorized credentials and the resulting receipt, leakage audit, and
index verification pass. An empty or partial result is expected and must remain
visible.
