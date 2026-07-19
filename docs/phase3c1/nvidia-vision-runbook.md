# Product Phase 3C NVIDIA vision reasoning runbook

This is a disabled-by-default, optional cloud experiment. It does not establish
location accuracy, calibrated confidence, production readiness, continued model
availability or commercial/legal clearance. Do not use query images as corpus,
training, tuning, comparison or provider-selection data. Abstention is valid and
preferred when image-only evidence is insufficient.

## Fixed service and privacy boundary

- Provider selection remains `openai` by default. NVIDIA requires explicit
  `CLOUD_VISION_PROVIDER=nvidia` and `NVIDIA_VISION_ENABLED=true` configuration.
- Every request independently requires `cloud_assisted` analysis mode and explicit
  cloud consent. A configured key never activates cloud processing by itself;
  `local_only` never calls NVIDIA.
- `NVIDIA_API_KEY` is backend-only. Never echo, inspect, log, pass on a command
  line, expose through a `VITE_*` variable or commit it. Configure it only in the
  operator's untracked server environment. Backend settings and the Vite config
  both fail closed on populated OpenAI/NVIDIA frontend credential variables.
- The recommended Phase 3C2 free-prototype model is
  `qwen/qwen3.5-122b-a10b`. Its repaired non-thinking request profile uses
  temperature `0.7`, top-p `0.8`, seed `0` and
  `chat_template_kwargs.enable_thinking=false`. This recommendation does not guarantee availability, accuracy,
  calibration or production readiness. The client accepts only
  `https://integrate.api.nvidia.com/v1`, posts to `/chat/completions`, and polls
  only the documented `/status/{requestId}` path for a bounded `202` response.
  Redirects and alternate origins are refused.
- Input preparation is in memory. It applies orientation, converts to RGB and
  emits metadata-free JPEG derivatives. Every derivative is at most 180,000 bytes
  and the set is bounded to four derivatives. Phase 3C1 does not use NVIDIA's
  separate asset-upload API, retain a derivative, or send original EXIF, a
  separate OCR transcript, candidate coordinates or ground truth.
- Image text is untrusted. The request enables no tools or browsing and asks for
  no chain of thought. Strict schema validation rejects extra/hidden-reasoning
  fields, invalid coordinates, non-positive uncertainty, unknown evidence links
  and inconsistent abstention.

## Model status and selection

- A synthetic vision probe of `qwen/qwen3.5-122b-a10b` returned HTTP 200 in
  41.499 seconds with temperature `0.6` and top-p `0.95`. Treat this only as a
  bounded connectivity/request-shape result, not a location-quality benchmark.
- `qwen/qwen3.5-397b-a17b` repeatedly ended in a read timeout during bounded live
  attempts. It is deprecated, explicit-selection-only, and scheduled for
  retirement on **2026-07-27**. It is never an automatic fallback.
- The evaluated Phi hosted route returned HTTP 410 and is unusable.
- The frozen Phase 3C2 blind checkpoint made two isolated 122B calls. Both
  reached accepted non-202 2xx handling paths, but one failed the strict output
  schema and the other returned invalid/empty model content. No candidate was
  accepted, so the private demo is not ready. Do not retry those images; diagnose
  response compatibility with synthetic/non-private input under new authorization.
- The committed response-contract repair sends no undocumented
  `response_format` or `json_schema`. It requests exactly one JSON object and
  accepts only a bare object or one complete lowercase `json` fence. Empty,
  null, list, reasoning-only, truncated, multiple-object and prose-wrapped
  content remains fail-closed; hidden `reasoning_content` is never used.
- One post-fix 32x32 metadata-free synthetic call returned HTTP 200 in 35.937
  seconds and validated as a typed abstention with zero hypotheses. This makes a
  newly authorized operator blind validation technically possible, but is not
  accuracy or private-demo readiness evidence. Do not reuse prior operator
  images without new explicit authorization.
- Selecting a model is an operator action. A timeout, error, abstention or weak
  output never authorizes automatic resend to another model. Each separately
  authorized live attempt uses transport retry zero and a total timeout no
  greater than 120 seconds. Bounded GET polling is allowed only after HTTP 202
  and remains part of the same request rather than a new inference attempt.

Historical Phase 3C1 references for the deprecated 397B target:

- [NVIDIA model and deployment page](https://build.nvidia.com/qwen/qwen3.5-397b-a17b?nim=self-hosted&section=deploy)
- [NVIDIA inference API reference](https://docs.api.nvidia.com/nim/reference/qwen-qwen3-5-397b-a17b-infer)
- [NVIDIA status-polling reference](https://docs.api.nvidia.com/nim/reference/qwen-qwen3-5-397b-a17b-statuspolling)

Do not substitute another model or endpoint silently. Any continuation or
migration requires a fresh official availability, schema, privacy, retention,
terms and license review plus a recorded follow-up decision.

## Configuration and normal API use

Use the safe template in `.env.example`; keep real values outside Git:

```dotenv
CLOUD_VISION_PROVIDER=nvidia
NVIDIA_VISION_ENABLED=true
NVIDIA_API_KEY=
NVIDIA_VISION_MODEL=qwen/qwen3.5-122b-a10b
NVIDIA_VISION_TIMEOUT_SECONDS=90
NVIDIA_VISION_MAXIMUM_IMAGE_EDGE=1280
NVIDIA_VISION_JPEG_QUALITY=82
```

The selected model identifier, endpoint and preprocessing caps are fail-closed. The
existing upload API still controls per-request mode and consent. If selection,
enablement, credential, cloud mode or consent is missing, the optional provider
is skipped without a network call. Do not add an implicit fallback, resend or
retry with another cloud provider/model.

Accepted hypotheses are unverified image-only evidence. `confidence` has the
fixed semantics `uncalibrated_model_self_assessment`, not a probability. Every
hypothesis has a positive model radius; the application adapter applies a minimum
25 km presentation radius and caps the imported self-assessment at 0.35. The
provider may instead abstain with no candidate.

## Offline verification before any live query

Run focused tests with synthetic fixtures and mock HTTP transport:

```powershell
Set-Location D:\geoSearch\services\api
.\.venv\Scripts\python.exe -m pytest tests\test_phase3c1_nvidia_vision.py tests\test_phase3c2_nvidia_profiles.py tests\test_phase3c2_nvidia_response_contract.py
.\.venv\Scripts\ruff.exe check src tests\test_phase3c1_nvidia_vision.py tests\test_phase3c2_nvidia_profiles.py tests\test_phase3c2_nvidia_response_contract.py
.\.venv\Scripts\python.exe -m mypy src

Set-Location D:\geoSearch\apps\web
$env:ATLASLENS_IGNORE_ENV_FILE='true'
npm test -- tests\vite-config-security.test.ts
```

The focused suite covers strict schema/abstention, metadata removal and the
180,000-byte cap, zero-call authorization denial, exact request shape, bounded
`202` polling, terminal authentication failure, provider-adapter mapping and
safe settings defaults. The frontend test suite covers the Vite credential
guard. Mock responses are synthetic test fixtures only; they
are not geolocation evidence or a live-service result.

Before a live query, also verify that no secret, image, model, dataset, index or
runtime artifact is staged. Do not broadly stage the worktree. Do not read or
print the real environment file/key while checking configuration.

## One-image blind query

The command intentionally accepts exactly one image and has no batch, city,
ground-truth, expected-answer or comparison argument:

```powershell
atlas nvidia-vision blind-query <PRIVATE_IMAGE> `
  --analysis-mode cloud_assisted `
  --cloud-consent `
  --execute
```

Run each authorized blind image in a separate fresh process with a timeout no
greater than 120 seconds. Do not paste the
private path into a report or shell transcript, compare one result while deciding
how to run the other, change code/config/prompt between runs, rerun a failure for
model quality, or tune from either result. The blind command sets transport
retries to zero. A documented asynchronous `202` may still produce bounded status
poll requests; `network_attempts` and `status_poll_attempts` report exact counts.

Authorization flags are checked before the key or image is read. Sanitized output
reports completion or abstention, model/prompt/schema provenance, token and
network counts, labels, country/granularity, positive uncertainty radius,
uncalibrated confidence and limitations. It deliberately omits coordinates,
local path, original filename, content hash, raw observed text, prompt, request
ID and raw provider response, and reports `retained=false`.

Record only that sanitized projection and whether the call completed, abstained
or failed. Do not infer accuracy without independently licensed ground truth and
a separately authorized, locked evaluation design. Never use the blind results
to tune prompts, thresholds, radii, confidence, provider selection or retries.

## Retention, incidents and stop conditions

AtlasLens keeps no Phase 3C derivative or raw NVIDIA response. Process memory is
released after the call, but local deletion cannot recall a request already sent
to NVIDIA. NVIDIA-side processing and retention are governed by the operator's
current account/service terms and must be reviewed before enablement.

Stop without retrying or switching endpoints when consent is absent, the key or
model is rejected, the allowlisted endpoint/model becomes unavailable, schema
validation fails, a derivative cannot meet the inline cap, the response is
truncated/oversized, or the retirement date has passed without a new decision.
Report only stable error codes and aggregate attempt counts; never record headers,
request bodies, image data, paths, prompts, raw responses or credential material.
