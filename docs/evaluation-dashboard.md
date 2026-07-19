# Evaluation report dashboard

The Phase 5C evaluation screen is a read-only view over benchmark reports already
created by the existing licensed evaluation workflow. It does not run a benchmark
from the browser, ingest uploads, fit calibration, compare against unpaired data,
or promote a model.

## Produce a report

Validate the exact manifest, asset root, licenses, split, and hashes before a run.
This abbreviated example intentionally contains placeholders rather than a
repository benchmark result:

```powershell
cd services\api
.\.venv\Scripts\atlas.exe benchmark validate `
  --manifest C:\licensed\evaluation\manifest.csv `
  --asset-root C:\licensed\evaluation\images `
  --allow-license CC0-1.0

.\.venv\Scripts\atlas.exe benchmark run `
  --manifest C:\licensed\evaluation\manifest.csv `
  --asset-root C:\licensed\evaluation\images `
  --allow-license CC0-1.0 `
  --provider geoclip `
  --split test `
  --output C:\private\atlaslens\reports\evaluations\geoclip-locked-test

.\.venv\Scripts\atlas.exe benchmark run `
  --manifest C:\licensed\evaluation\manifest.csv `
  --asset-root C:\licensed\evaluation\images `
  --allow-license CC0-1.0 `
  --provider atlaslens-custom-geolocation `
  --custom-model-id atlaslens-custom-geolocation --device cuda `
  --split test `
  --output C:\private\atlaslens\reports\evaluations\custom-locked-test
```

Use `atlas benchmark compare --providers geoclip,atlaslens-custom-geolocation`
with `--geoclip-report`, `--custom-report`, and the same manifest arguments to
write a paired read-only comparison. A mock/simulated provider is rejected.

The catalog expects one directory per safe report ID and a `benchmark.json`
inside it:

```text
EVALUATION_REPORT_DIR/
  geoclip-locked-test/
    benchmark.json
```

The reader validates the report schema, recomputes the summary from per-image
records, verifies counts, and rejects mismatches. Reports classified as simulated
or carrying mock/test/fixture provider identities are excluded. Local paths and
truth coordinates are not returned by the dashboard API.

## Enable local viewing

On a trusted development/test machine only:

```text
APP_ENV=development
OPERATOR_API_ENABLED=true
EVALUATION_REPORT_DIR=C:\private\atlaslens\reports\evaluations
```

Start the API and web app, then open Evaluation. The API endpoints are
`GET /api/v1/evaluations` and `GET /api/v1/evaluations/{report_id}`. A missing
catalog returns an empty list; an invalid or unsafe entry fails closed. Production
configuration rejects the unauthenticated operator API.

The screen reports explicit numerator/denominator ratios, country/region/city
metrics, distance recall, error summaries, abstention, provider failures,
latency, uncertainty coverage, geographic/scene distributions, exclusions,
calibration state, evaluation fingerprint, and limitations when present. A missing
metric remains unavailable; it is never replaced with zero accuracy.

## Interpretation

- A benchmark measurement applies only to the bound provider/model revision and
  evaluated manifest fingerprint.
- Top-K, distance recall, and oracle diagnostics are not guarantees for an
  individual upload.
- A small, landmark-heavy, geographically imbalanced, or overlapping sample does
  not establish production accuracy or calibration.
- Compare providers only on the same immutable paired sample set and count every
  failure, abstention, and exclusion.
- Shadow traffic without reviewed ground truth can measure runtime/failure/overlap,
  not accuracy.

Promotion remains an explicit CLI workflow documented in
`custom-model-handoff.md`; the dashboard cannot mutate deployment state.
