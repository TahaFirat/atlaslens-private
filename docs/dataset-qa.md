# Dataset QA

Phase 5C adds a read-only, local dataset scanner. It diagnoses operator-supplied
image, optional mask, and optional geolocation-manifest roots. It never repairs,
moves, deletes, relabels, downloads, or enrolls source data for training.

## Checks

Policy `atlaslens-dataset-qa-v1` applies bounded file/count/pixel limits and
reports safe issue codes for:

- missing, empty, unsupported, corrupt, truncated, extreme-size, near-blank,
  blurred, underexposed, and overexposed images;
- missing/corrupt masks, dimension and palette/class mismatches, all-background,
  tiny components, and suspicious foreground coverage;
- incomplete, malformed, non-finite, out-of-range, possibly swapped, or `0,0`
  coordinates and EXIF/manifest coordinate conflicts;
- exact and bounded perceptual duplicates, conflicting labels, and cross-split
  duplicate leakage;
- capture-family and sequence leakage across splits; and
- split, label, issue-code, and other safe aggregate distributions.

These heuristics are diagnostics, not a dataset-quality guarantee. Semantic or
cropped duplicates can evade compact hashes, coordinate boundary correctness is
not inferred, and warnings require human review.

## Run the scanner

Run from `services\api`. Keep the output outside every input root. The output
directory name is the report ID unless `--report-id` explicitly supplies the
same value.

```powershell
.\.venv\Scripts\atlas.exe dataset qa `
  --images C:\licensed\dataset\images `
  --masks C:\licensed\dataset\masks `
  --manifest C:\licensed\dataset\manifest.csv `
  --asset-root C:\licensed\dataset `
  --class-id 1 --class-id 2 `
  --output C:\private\atlaslens\reports\dataset-qa\training-2026-07-12

.\.venv\Scripts\atlas.exe dataset qa-report `
  --output C:\private\atlaslens\reports\dataset-qa\training-2026-07-12

.\.venv\Scripts\atlas.exe dataset review `
  --qa-report C:\private\atlaslens\reports\dataset-qa\training-2026-07-12 `
  --severity error
```

Omit `--masks`, `--manifest`, or `--asset-root` when that data is not part of the
dataset. If masks are provided, `--class-id` values define the allowed foreground
classes; background class `0` is included automatically. `--apply` is
intentionally rejected.

Each run atomically writes `report.json`, `issues.csv`, `summary.md`, and
`report.html`. `--contact-sheet` also creates a bounded RGB JPEG contact sheet
without copying source EXIF. Contact sheets still contain image pixels and must
remain access-controlled. Reports use opaque asset keys and omit absolute paths,
raw EXIF/OCR, and exact GPS values.

## Read-only dashboard

Place each report directory directly under `DATASET_QA_REPORT_DIR`, set
`APP_ENV=development` and `OPERATOR_API_ENABLED=true` only on a trusted local
machine, then open the Dataset QA screen. The API exposes only validated
`report.json` summaries/details at `/api/v1/datasets/qa` and
`/api/v1/datasets/qa/{report_id}`. It rejects unsafe report IDs, symlink escapes,
invalid schemas, and unsafe issue values.

The unauthenticated operator API is refused in production. Phase 5C does not add
authentication, remote report publishing, repair actions, or training-data
mutation.
