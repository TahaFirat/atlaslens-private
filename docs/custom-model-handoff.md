# Custom model handoff

This runbook covers the bounded Phase 5C handoff of an already-trained model.
It does not train, export, download, calibrate, or automatically promote a
model. GeoCLIP remains a separate managed baseline and its interfaces are not
replaced.

## Accepted artifact boundary

The initial executable adapter accepts a regular, self-contained ONNX file with
the reviewed `onnx-coordinate-v1` contract. It expects one float32 RGB/NCHW input
and named coordinate and score outputs. Coordinates are latitude/longitude and
scores must be finite, bounded values with explicitly declared semantics.

The registry rejects pickle-family checkpoints, `.pt`, `.pth`, `.ckpt`,
TorchScript, remote code, symlinks, undeclared files, hash/size mismatches, and
unknown runtime adapters. A safetensors manifest is valid only for a
repository-known architecture adapter; Phase 5C does not provide a generic
safetensors executor. ONNX Runtime is an optional local dependency and is never
downloaded at API startup.

The artifact and its manifest stay outside Git in an operator-controlled private
directory. The manifest must record:

- model/provider/revision identities and the exact artifact SHA-256 and size;
- input tensor, shape, resize, color/layout/dtype, normalization, mean/std when
  applicable, and preprocessing revision;
- output tensor names, Top-K, coordinate order, score semantics, and output
  schema revision;
- training dataset fingerprint, optional capture-family fingerprint, code
  revision, and finish time;
- evaluation thresholds; and
- license name, commercial-use status, approval status, and review reference.

## Prepare and register on Windows

Run from the repository root. Replace every example value with the actual
training/export record; do not guess missing lineage or license fields.

```powershell
services\api\.venv\Scripts\python.exe scripts\prepare-custom-model-manifest.py `
  --artifact C:\private\atlas-model\model.onnx `
  --output C:\private\atlas-model\manifest.yaml `
  --model-id atlaslens-custom-geolocation `
  --provider-id custom-trained-geolocation `
  --model-version 2026-07-12.1 `
  --implementation-revision <training-code-revision> `
  --dataset-fingerprint <lowercase-sha256> `
  --training-code-revision <training-code-revision> `
  --training-completed-at 2026-07-12T00:00:00Z `
  --license-name <reviewed-license-name> `
  --license-review-reference <review-ticket-or-document> `
  --commercial-use allowed `
  --preprocessing-version <preprocessing-revision> `
  --input-tensor <input-name> `
  --coordinates-output <coordinate-output-name> `
  --scores-output <score-output-name> `
  --normalization mean_std `
  --mean <r> <g> <b> `
  --std <r> <g> <b> `
  --score-type uncalibrated_bounded_score `
  --width 384 --height 384 --top-k 5

cd services\api
.\.venv\Scripts\atlas.exe models register-local --manifest C:\private\atlas-model\manifest.yaml
.\.venv\Scripts\atlas.exe models verify atlaslens-custom-geolocation
.\.venv\Scripts\atlas.exe models info atlaslens-custom-geolocation
.\.venv\Scripts\atlas.exe models test atlaslens-custom-geolocation --image C:\licensed\smoke.jpg --device cpu
```

`register-local` copies the exact artifact and canonical manifest into the
private model cache and records immutable identities. Registration starts
disabled. Successful `verify` rehashes the copy, checks the approved license and
moves it to `shadow`. `test` is a local runtime smoke check, not an accuracy or
performance claim. If ONNX Runtime is absent or the device is unavailable, the
command fails safely.

The reviewed coordinate adapter requires each emitted raw score to be finite and
bounded from 0 through 1, while its name must begin with `uncalibrated_`. This is
a relative model value, never a probability or product confidence.

Set `CUSTOM_MODEL_ID` to the registered ID, select `CUSTOM_MODEL_DEVICE=cpu` or
`cuda`, and leave `CUSTOM_MODEL_ENABLED=true`. A missing, invalid, or unsupported
artifact remains an unavailable provider; it never falls back to simulated
coordinates.

## Evaluation and promotion

Shadow execution cannot change the user-visible ranking and cannot establish
accuracy because live uploads have no reviewed truth. Evaluate the verified custom
provider through the typed evaluation-provider adapter on the same licensed,
held-out sample set as the baseline:

```powershell
.\.venv\Scripts\atlas.exe benchmark run `
  --manifest C:\licensed\evaluation\manifest.csv `
  --asset-root C:\licensed\evaluation\images `
  --allow-license CC0-1.0 --provider geoclip --split test `
  --output C:\private\reports\geoclip

.\.venv\Scripts\atlas.exe benchmark run `
  --manifest C:\licensed\evaluation\manifest.csv `
  --asset-root C:\licensed\evaluation\images `
  --allow-license CC0-1.0 --provider atlaslens-custom-geolocation `
  --custom-model-id atlaslens-custom-geolocation --device cuda --split test `
  --output C:\private\reports\custom

.\.venv\Scripts\atlas.exe benchmark compare `
  --providers geoclip,atlaslens-custom-geolocation `
  --manifest C:\licensed\evaluation\manifest.csv `
  --asset-root C:\licensed\evaluation\images `
  --allow-license CC0-1.0 --split test `
  --geoclip-report C:\private\reports\geoclip `
  --custom-report C:\private\reports\custom `
  --output C:\private\reports\paired-comparison
```

Both runs bind the same validated split fingerprint. The comparison command
recomputes each summary from its per-image records, rejects simulated providers,
and refuses mismatched fingerprints or sample counts. It does not generate or
approve a promotion receipt. The promotion report must bind the candidate artifact identity,
candidate and baseline provider IDs, evaluation and paired-sample fingerprints,
sample count, country Top-1 and Recall@200 km deltas, median-error change, p95
latency, subgroup checks, schema compatibility, lineage, safety, license, and
explicit operator approval.

Only these transitions exist:

```powershell
.\.venv\Scripts\atlas.exe models promote atlaslens-custom-geolocation `
  --from shadow --to candidate --report C:\private\reports\candidate-promotion.yaml

.\.venv\Scripts\atlas.exe models promote atlaslens-custom-geolocation `
  --from candidate --to primary --report C:\private\reports\primary-promotion.yaml
```

Primary additionally requires runtime-isolation verification. Missing fields,
identity mismatch, insufficient samples, excessive regression/latency, failed
subgroups, unapproved license, or absent operator approval fail closed. Promotion
does not turn raw scores into probabilities; calibration remains a separate,
revision-bound artifact.

To remove a registered custom artifact deliberately:

```powershell
.\.venv\Scripts\atlas.exe models unregister atlaslens-custom-geolocation --yes
```

Do not remove a model that an active API process is using. Deployment
coordination, authenticated operator control, fleet rollout, and rollback
automation remain outside Phase 5C.
