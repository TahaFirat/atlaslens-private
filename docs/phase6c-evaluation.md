# Phase 6C leakage-safe geolocation evaluation

This foundation keeps private truth under the ignored `.local/evaluation` tree and
out of the production inference graph. No real target manifest or target image is
committed to the repository or copied into a reference index.

## Prediction/scoring boundary

`scripts/evaluate_turkey_geolocation.py` projects opaque IDs and image paths,
starts one prediction subprocess per image, collects all predictions, and only
then loads the truth-bearing manifest for scoring. The child receives this strict
JSON document on standard input:

```json
{
  "protocol_version": "atlaslens-isolated-prediction-v1",
  "request_id": "opaque-prediction-id",
  "image_path": "C:\\private\\image.jpg"
}
```

It must write exactly one JSON response to standard output:

```json
{
  "protocol_version": "atlaslens-isolated-prediction-v1",
  "request_id": "opaque-prediction-id",
  "provider_id": "real-provider-id",
  "model_revision": "verified-revision",
  "prediction": {
    "candidates": [],
    "abstained": true,
    "failure_code": null,
    "latency_ms": 10,
    "device": "cpu"
  }
}
```

Candidate objects use the existing `CandidatePrediction` evaluation contract.
The worker command is operator-supplied because this layer does not fabricate a
model result or silently select an unverified provider. Shell parsing is never
used, response size and runtime are bounded, and truth/manifest environment keys
are removed from the child environment.

```powershell
$env:PYTHONPATH = "services/api/src"
services/api/.venv/Scripts/python.exe scripts/evaluate_turkey_geolocation.py `
  --manifest .local/evaluation/turkey_holdout.json `
  --output .local/evaluation/turkey-holdout-report.json `
  --prediction-command -- <verified-worker-command> <worker-arguments>
```

Reports contain correctness/rank metrics and an opaque evaluation ID, but no
image path. A sample count below 100 cannot produce an accuracy claim.

### Phase 6C loopback HTTP worker

Run the API separately with `PHASE6C_ENABLED=true`, the reviewed local model
workers, and an admitted reference index. The provided worker submits each image
to that API in `local_only` mode, explicitly denies cloud processing and cloud
assist, polls the terminal resource, and returns the final bounded public
candidates through the isolated protocol:

```powershell
$env:PYTHONPATH = "services/api/src"
services/api/.venv/Scripts/python.exe scripts/evaluate_turkey_geolocation.py `
  --manifest .local/evaluation/turkey_holdout.json `
  --output .local/evaluation/turkey-holdout-phase6c.json `
  --timeout-seconds 300 `
  --prediction-command -- `
  services/api/.venv/Scripts/python.exe scripts/phase6c_http_prediction_worker.py `
  --api-base-url http://127.0.0.1:8000 `
  --timeout-seconds 240 `
  --max-candidates 20
```

The worker accepts only a plain loopback HTTP origin and does not read API,
manifest, truth, or proxy settings from environment variables. Its multipart
filename is a fixed format-derived name, never the original path. It requires a
real `phase6c-v1` terminal result and a passed leakage attestation whenever a
reference-index version is present. Coordinates and positive uncertainty radii
come from final public candidates; the numeric score remains the candidate's raw
uncalibrated relative rank, and country/region/city fields come only from the
public reverse-geocode object. A public abstention remains an abstention. Network,
timeout, invalid-output, and leakage-gate failures use stable path-free codes;
neither raw OCR nor exception text is written to stdout or stderr.

## Reference leakage audit

The audit consumes either the Phase 6C acquisition `reference-input.json` or the
existing licensed extended retrieval CSV, together with its asset root. It detects
target/reference SHA-256 matches, conservative 64-bit perceptual
matches across rotations/mirrors, bounded crop signatures, and prefiltered ORB
homography support. When source image or capture-family metadata exists on both
sides, identity overlap is also excluded. Every exclusion records its reason;
the input manifest is read-only.

```powershell
$env:PYTHONPATH = "services/api/src"
services/api/.venv/Scripts/python.exe scripts/build_phase6c_leakage_descriptors.py `
  --holdout-manifest .local/evaluation/turkey_holdout.json `
  --holdout-id user-holdout-001 `
  --reference-input .local/phase6c-reference/reference-input.json `
  --reference-root .local/phase6c-reference `
  --worker-port 8794 `
  --device cuda `
  --output .local/evaluation/turkey-reference-megaloc-descriptors.npz
services/api/.venv/Scripts/python.exe scripts/audit_reference_leakage.py `
  --holdout-manifest .local/evaluation/turkey_holdout.json `
  --holdout-id user-holdout-001 `
  --reference-input .local/phase6c-reference/reference-input.json `
  --reference-root .local/phase6c-reference `
  --descriptor-npz .local/evaluation/turkey-reference-megaloc-descriptors.npz `
  --output .local/evaluation/turkey-reference-leakage.json
```

For an existing extended retrieval CSV, retain the original
`--reference-manifest <csv> --reference-root <asset-root>` arguments. The Phase 6C
input's stored perceptual value is dHash64; the leakage auditor deliberately
computes its established aHash orientation/crop signatures from the real image
bytes instead of treating the two algorithms as interchangeable.

### Explicit filtered-input rerun

When the first audit excludes one or more acquisition records, an operator may
request a separate filtered builder input:

```powershell
services/api/.venv/Scripts/python.exe scripts/audit_reference_leakage.py `
  --holdout-manifest .local/evaluation/turkey_holdout.json `
  --holdout-id user-holdout-001 `
  --reference-input .local/phase6c-reference/reference-input.json `
  --reference-root .local/phase6c-reference `
  --descriptor-npz .local/evaluation/turkey-reference-megaloc-descriptors.npz `
  --write-filtered-reference-input .local/phase6c-reference/reference-input.leakage-filtered.json `
  --output .local/evaluation/turkey-reference-leakage.initial.json
```

This first command still exits nonzero and records `status=failed`; creating a
filtered file never retroactively turns that audit into a pass. It atomically
writes the same strict Phase 6C schema minus unambiguously excluded asset keys.
The original manifest and every source image remain untouched. The command
refuses a CSV source, an ambiguous key mapping, an output that would overwrite
the source/report, or a result that would remove every record. Standard output
contains aggregate counts only, not paths or reference IDs;
`filtered_reference_count` is the number of records retained in the generated
file.

Review both generated files, then audit the filtered input as a new admission
check:

```powershell
services/api/.venv/Scripts/python.exe scripts/audit_reference_leakage.py `
  --holdout-manifest .local/evaluation/turkey_holdout.json `
  --holdout-id user-holdout-001 `
  --reference-input .local/phase6c-reference/reference-input.leakage-filtered.json `
  --reference-root .local/phase6c-reference `
  --descriptor-npz .local/evaluation/turkey-reference-megaloc-descriptors.npz `
  --output .local/evaluation/turkey-reference-leakage.filtered.json
```

Only a `passed` result from this second command admits the filtered input to an
index build. Preserve the initial failed report as the exclusion audit trail.

Descriptor similarity is optional and is never synthesized. The command above
builds it from real inference through the strict loopback-only MegaLoc client.
It uses only the holdout prediction projection, so the worker receives image
bytes and device choice, never truth, coordinates, paths, or OCR. It recomputes
actual SHA-256 values, checks strict reference-root containment, exact pinned
source/model revisions and the 8,448-value L2 contract, applies count/byte/time
bounds, and atomically replaces the NPZ. Standard output is aggregate-only.

The established audit NPZ contract contains:

- scalar strings `provider` and `version`;
- one-dimensional `content_sha256` strings;
- a finite two-dimensional `vectors` array with the same row count.

The pinned builder additionally records safe `model_id`, `model_revision`,
`source_revision`, `dimension`, and `normalization` metadata; the audit loader
remains compatible with artifacts containing only the four required fields.

The artifact must contain the holdout digest and every audited reference digest.
Missing coverage makes an explicitly requested descriptor audit incomplete.
Without an artifact the report states
`not_run_no_real_descriptor_artifact`; it never substitutes pixel hashes, random
vectors, or test embeddings.

An audit is `failed` when any reference requires exclusion and `incomplete` when
a configured check could not cover its bounded candidate set. Only `passed` is a
clean index admission result. This tooling never mutates a reference index or
deletes source data.

## Licensed Türkiye development/validation suite

`scripts/build_turkiye_evaluation_manifest.py` converts a separate Phase 6C
acquisition input into the established full-coordinate evaluation CSV. It does
not download imagery. The candidate acquisition and the actual retrieval input
are both required so the builder can exclude, from real image bytes, exact
SHA-256 overlap, dHash64 near-duplicates, the same source image and every image
from a source sequence/capture family already present in retrieval.

Only Mapillary or KartaView acquisition records with an explicitly allowed
license, complete provenance, acquisition SHA/dHash receipts, positive coordinate
uncertainty and a non-null locally resolved city are admitted. Duplicate
candidate images are also removed. Source sequences stay intact during a
deterministic grouped split. The existing CSV contract calls the development
split `calibration`; validation remains `validation`. The private holdout remains
in its separate JSON process and is never emitted by this builder.

```powershell
$env:PYTHONPATH = "services/api/src"
services/api/.venv/Scripts/python.exe scripts/build_turkiye_evaluation_manifest.py `
  --candidate-reference-input .local/phase6c-evaluation-acquisition/reference-input.json `
  --candidate-root .local/phase6c-evaluation-acquisition `
  --retrieval-reference-input .local/phase6c-reference/reference-input.json `
  --retrieval-root .local/phase6c-reference `
  --allowed-license "CC BY-SA 4.0" `
  --output-manifest .local/evaluation/turkiye-development-validation.csv `
  --output-receipt .local/evaluation/turkiye-development-validation-receipt.json
```

The receipt is aggregate-only: corpus fingerprints, counts, split sizes,
exclusion reason counts and the output CSV checksum. It contains no paths,
record IDs, cities or coordinates and never authorizes an accuracy claim.
`ready` requires the configured minimum (100 by default) and non-empty
development and validation splits. A correctly built smaller suite reports
`insufficient`; no surviving records reports `empty`. Both states exit with code
1 so automation cannot mistake them for representative validation. Integrity,
schema or unsafe-path failures exit with code 2 and do not report sensitive
record details.

### Development/validation HTTP benchmark

With the Phase 6C API already running on loopback, the full-coordinate CSV can be
scored through the established benchmark runner without sending its truth fields
to the API. The adapter receives only each validated image path; the HTTP worker
still forces `local_only`, denies cloud consent and cloud assist, uses a fixed
format-derived upload filename, and accepts only a plain loopback HTTP origin.

```powershell
$env:PYTHONPATH = "services/api/src"
services/api/.venv/Scripts/python.exe scripts/benchmark_phase6c_http.py `
  --manifest .local/evaluation/turkiye-development-validation.csv `
  --asset-root .local/phase6c-evaluation-acquisition `
  --allowed-license "CC BY-SA 4.0" `
  --output-directory .local/evaluation/turkiye-phase6c-http `
  --api-base-url http://127.0.0.1:8000 `
  --timeout-seconds 240 `
  --request-timeout-seconds 30 `
  --poll-interval-seconds 0.25
```

All URL and time choices are explicit. The command rejects any `test`-split or
non-Türkiye row; this surface is only for the separate development/validation
suite and never evaluates the private final holdout. Standard output contains
aggregate completion/failure/abstention counts only. The usual `benchmark.json`,
`per-image.csv`, and `summary.md` reports contain aggregate metrics and hashed
per-image/source IDs, but no local paths. A small or unrepresentative suite still
cannot support an accuracy or improvement claim.
