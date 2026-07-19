# Phase 6C completed-history validation diagnostics

This diagnostic scores predictions that already exist in the AtlasLens analysis
history. It never uploads an image, creates an analysis, reruns a provider, or
uses a cloud service. Its HTTP client accepts only a plain loopback origin and
issues bounded `GET` requests to the history and full-analysis endpoints.

The ordering is deliberate:

1. Read only `split` and `content_sha256` from the established `EvaluationRecord`
   CSV and select the validation split.
2. Fetch completed real history rows and matching full `phase6c-v1` responses.
   When more than one completed Phase 6C response has the same image hash, retain
   the newest by analysis creation time.
3. Only after every response is fetched, use `EvaluationManifestLoader` to
   validate the licensed assets and reveal truth coordinates for scoring.

The report covers raw hierarchy, MegaLoc, OSV-5M, PLONK, OCR place evidence,
all final-fusion candidates before publication filtering, public candidates,
and every Phase 6C ablation present in the responses. Each component reports
candidate coverage, nearest-candidate mean/median/p95 error, and recall within
25/100/250/750 km. Provider output is aggregate-only: observed coverage, status
counts, candidate-return coverage, candidate count, and latency distribution.

The report always sets `no_claim=true`. It is a candidate-recall diagnostic, not
a calibrated accuracy or improvement claim. Missing completed analyses count as
coverage and recall misses. Neither the JSON report nor console output contains
paths, content hashes, analysis/asset/source IDs, coordinates, OCR text, or URLs.

```powershell
$env:PYTHONPATH = "services/api/src"
services/api/.venv/Scripts/python.exe scripts/score_phase6c_validation_history.py `
  --api-base-url http://127.0.0.1:8761 `
  --manifest .local/evaluation/turkiye-development-validation.csv `
  --asset-root .local/phase6c-evaluation-acquisition `
  --allowed-license "CC BY-SA 4.0" `
  --output .local/evaluation/turkiye-phase6c-validation-diagnostic.json
```

History size, matching-analysis fetches, manifest rows/bytes, response bytes,
and request time are all bounded. A non-loopback URL, changed history snapshot,
malformed response, manifest failure, or exceeded bound returns a stable,
path-free error and does not write a partial report.
