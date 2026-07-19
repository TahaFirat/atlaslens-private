# Phase 5B licensed reference runbook

## Boundary

This runbook creates a reviewed local Wikimedia Commons reference index. It does
not authorize Phase 6 scaling, generic web scraping, Google imagery, Mapillary,
KartaView, OSV-5M, or automatic reuse of user uploads.

Discovery is metadata-only. Image bytes are fetched only after a human reviews
each row, sets `approved=true`, selects `camera_raw`, `object`, or `manual` as the
coordinate kind, and accepts an explicit license allowlist.

The committed sampling cells are search centers, not image labels or measured
coverage. Their `target` values are requested discovery quotas. They are not
claims that Commons contains that many acceptable images. City-center
coordinates are approximate query anchors and must not be presented as evidence
or location truth. The operator should periodically compare them with the
installed, receipted GeoNames artifact before a production acquisition.

## Sampling configuration

Use `config/datasets/commons-phase5b-10k-cells.csv`.

- 48 deterministic cells are ordered in eight six-continent rounds.
- The planned quotas sum to 12,000 metadata candidates, providing review
  headroom for a 10,000 accepted-image target without weakening rejection rules.
- The file names 48 country codes across Africa, Asia, Europe, North America,
  South America and Oceania.
- Every query radius is 10 km, the maximum accepted by the bounded AtlasLens
  sampling schema.
- Round-robin ordering prevents an early global target from being filled only by
  the first continent.

Actual accepted counts, countries, continents, licenses and byte sizes must come
from `atlas dataset validate`; do not copy planned quotas into a results report.

## Prerequisites

1. Use a contactable Wikimedia User-Agent, for example
   `AtlasLensCommonsReview/1.0 (https://operator.example/contact)`.
2. Freeze the independent evaluation manifest and retain its operator-supplied
   fingerprint and count. Reference discovery must exclude all evaluation
   content hashes, perceptual near-duplicates and capture families.
3. Choose an explicit allowlist. The implementation permits only individually
   reviewed `CC0 1.0` or exact `CC BY` / `CC BY-SA` 2.0–4.0 records. The commands
   below intentionally show the narrower 4.0-only example.
4. Keep the output under a private local dataset root. Never commit acquired
   images, review receipts, or absolute paths.

## 1. Metadata-only discovery

```powershell
cd <repository-root>\services\api

.\.venv\Scripts\atlas.exe dataset discover commons `
  --sampling-cells ..\..\config\datasets\commons-phase5b-10k-cells.csv `
  --target 10000 `
  --output ..\..\.local\phase5b-reference\commons-review.csv `
  --user-agent "AtlasLensCommonsReview/1.0 (https://operator.example/contact)" `
  --allow-license "CC0 1.0" `
  --allow-license "CC BY 4.0" `
  --allow-license "CC BY-SA 4.0"
```

The command uses the official MediaWiki Action API, sends serial bounded
requests with `maxlag`, and writes no image. It is resumable from the review CSV.
If the target is not met, `discovery_target_not_met` is a blocker; do not lower
license, attribution, coordinate or geography requirements merely to reach 10k.

## 2. Human review

For every proposed row:

- open the canonical Commons page;
- confirm creator/credit, exact license and license URL;
- consider personality, privacy, trademark, artwork and other non-copyright
  restrictions;
- reject missing or ambiguous attribution;
- determine whether the coordinate describes the camera, depicted object, or a
  manual placement;
- set `approved=true` and the corresponding coordinate kind only after review;
- leave display denied unless the reviewed policy explicitly permits it.

Do not edit `metadata_fingerprint`. Acquisition re-queries the official metadata
and stops if it changed after review.

## 3. Explicit acquisition

Supply the locked evaluation fingerprint and count at runtime. They are
intentionally not embedded in application code or this runbook.

```powershell
.\.venv\Scripts\atlas.exe dataset acquire commons `
  --review ..\..\.local\phase5b-reference\commons-review.csv `
  --output-root ..\..\.local\phase5b-reference `
  --manifest ..\..\.local\phase5b-reference\manifest.csv `
  --user-agent "AtlasLensCommonsReview/1.0 (https://operator.example/contact)" `
  --evaluation-manifest <LOCKED_EVALUATION_MANIFEST> `
  --evaluation-root <LOCKED_EVALUATION_ASSET_ROOT> `
  --expected-evaluation-fingerprint <LOCKED_FINGERPRINT> `
  --expected-evaluation-count <LOCKED_COUNT> `
  --allow-license "CC0 1.0" `
  --allow-license "CC BY 4.0" `
  --allow-license "CC BY-SA 4.0"
```

Acquisition accepts only `upload.wikimedia.org`, writes atomic resumable
receipts, computes SHA-256 and perceptual hashes, and rejects exact/near
duplicates plus evaluation overlap. A changed source record requires a new
review.

## 4. Validation gate

```powershell
.\.venv\Scripts\atlas.exe dataset validate `
  --manifest ..\..\.local\phase5b-reference\manifest.csv `
  --asset-root ..\..\.local\phase5b-reference `
  --evaluation-manifest <LOCKED_EVALUATION_MANIFEST> `
  --evaluation-root <LOCKED_EVALUATION_ASSET_ROOT> `
  --expected-evaluation-fingerprint <LOCKED_FINGERPRINT> `
  --expected-evaluation-count <LOCKED_COUNT> `
  --minimum-count 10000 `
  --required-continents 6 `
  --minimum-countries 30 `
  --allow-license "CC0 1.0" `
  --allow-license "CC BY 4.0" `
  --allow-license "CC BY-SA 4.0"
```

The report fingerprint and source/license/continent/country/cell counts describe
the validated local corpus. A failed count or geographic-distribution gate is an
honest blocker.

## 5. Explicit model and index build

```powershell
.\.venv\Scripts\atlas.exe models install siglip2-b16-384
.\.venv\Scripts\atlas.exe models verify siglip2-b16-384

.\.venv\Scripts\atlas.exe embeddings create `
  --manifest ..\..\.local\phase5b-reference\manifest.csv `
  --input-root ..\..\.local\phase5b-reference `
  --index-dir ..\..\.local\phase5b-reference\index `
  --provider siglip2-b16-384 `
  --device cuda

.\.venv\Scripts\atlas.exe embeddings verify `
  --index-dir ..\..\.local\phase5b-reference\index

.\.venv\Scripts\atlas.exe retrieval smoke-test `
  --index-dir ..\..\.local\phase5b-reference\index `
  --image <LICENSED_QUERY_IMAGE> `
  --device cuda
```

Model installation and image acquisition are the only networked steps and are
always explicit. Runtime queries use the verified local model and index offline.
Set `RETRIEVAL_INDEX_DIR` to the verified index directory and restart the API;
`/api/v1/capabilities` must then report retrieval available before any normal
upload gate is claimed.

## Source restrictions

- Wikimedia Commons remains per-file review; there is no corpus-wide license.
- KartaView remains unavailable without a separate operator terms-approval
  receipt and implemented adapter.
- Mapillary remains unavailable without credentials and current platform/
  commercial-terms review.
- OSV-5M remains on hold because its primary sources conflict on the inherited
  CC BY-SA version and its datasheet warns against privacy-infringing OSINT use.
- User uploads never become reference or evaluation assets automatically.
