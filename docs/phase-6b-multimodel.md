# Phase 6B multi-model geolocation

Phase 6B is an additive, single-host pre-release extension. It preserves the
Phase 6A API, GeoCLIP provider, candidate contract, persistence lifecycle and
frontend layout. It does not add a retrieval corpus, training, calibration,
production scaling or a Phase 6C feature.

> The project uses pretrained model weights only. No training dataset is required
> for Phase 6B inference.

## Runtime design

- GeoCLIP 1.2.0 remains the operational in-process broad geographic provider.
- OSV-5M baseline and PLONK use AtlasLens-owned typed contracts in isolated
  Python 3.10 CPU workers. Both have passed real load/inference on this host;
  PLONK acceptance is limited to YFCC. PLONK OSV/iNaturalist are prepared-only.
  The main Python 3.12/CUDA environment is unchanged.
- Exactly one PLONK specialization is selected per request: strong road/street
  scenes select OSV, strong natural/low-built scenes select iNaturalist, and
  ambiguous or unavailable semantics select YFCC. A diagnostic override is
  explicit and still selects one model.
- PaddleOCR uses `PP-OCRv5_server_det` with `latin_PP-OCRv5_mobile_rec` in an
  isolated Python 3.12 CPU worker and is the verified primary OCR provider.
  RapidOCR 3.9.1/ONNX Runtime 1.27.0 has hash-bound real-inference proof and the
  existing killable-process fallback completed a full Paddle-disabled HTTP smoke.
- Workers bind only to `127.0.0.1`; the manager verifies process command,
  protocol/provider identity and pinned revisions, refuses foreign listeners,
  records PID identity privately, and stops only a matching process.
- The readiness ladder is `not_installed`, `dependencies_installed`,
  `weights_prepared`, `worker_unreachable`, `model_load_failed`,
  `inference_not_verified`, `ready`, and `disabled`. `ready` requires successful
  real load and inference. Current residency (`model_loaded`) is separate from
  retained `load_verified` and `inference_verified` proof.
- `HeavyModelScheduler` defaults to one heavy operation and one resident model.
  Timeout, cancellation and CUDA out-of-memory paths release the scheduler and
  return bounded failure states.
- SegFormer uses the reviewed 124-entry Mapillary mapping in
  `assets/mapillary/segformer_mapillary_config.json`. Relabeling changes only
  configuration/metadata, not `model.safetensors`.

## Fusion semantics

`phase6b-v1` clusters coordinates geodesically, then ranks clusters using provider
rank, sample density, independent source-family agreement, bounded same-family
support, OCR agreement/contradiction and geographic spread. Raw scores from
GeoCLIP, OSV-5M and PLONK are never averaged or compared across providers.
OSV-5M and `PLONK_OSV_5M` are both `osv5m_family`; their correlation therefore
cannot count as independent agreement. Every contribution is returned by name,
raw value, weight, contribution and reason. A public geographic candidate is
promoted only with at least two independent families; otherwise the preserved
GeoCLIP/Phase 6A result remains in force. All scores are uncalibrated relative
ranks, never correctness probabilities.

## Exact Windows preparation

Run from the repository root. Verification is offline and downloads nothing:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -VerifyOnly
.\services\api\.venv\Scripts\atlas.exe diagnostics phase6b --json
```

The optional bootstrap is an explicit network operation. It clones only pinned
source revisions, creates ignored isolated environments under `.local/workers`,
and fetches only pretrained weight repositories under `.local/models/phase6b`.
It refuses dataset names and requires at least 8 GiB free disk. Python 3.10 must
be installed separately for OSV-5M and PLONK.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -Models osv5m
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -Models plonk
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -Models paddleocr
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_phase6b_models.ps1 -Models rapidocr
```

The bootstrap is the only artifact download path. It verifies pinned sources,
packages and weights and runs a real RapidOCR process inference. A prepared
environment is not an operational claim: isolated providers become `ready` only
after the manager has run real inference and live health exposes both verification
flags. Diagnostics and `/api/v1/capabilities` remain authoritative. No download
occurs during worker/API startup or an HTTP analysis request.

The current inspected disk state is:

| Component | Private location | Current bytes | State |
|---|---|---:|---|
| GeoCLIP | managed GeoCLIP/Hugging Face cache | 1,796,094,205 | ready |
| SegFormer-B2 | `.local/models/atlaslens-segformer-b2-v4` | 109,828,507 | ready; exact 124 names |
| OSV-5M | `.local/models/phase6b/osv5m` | 1,296,336,308 | baseline ready; real CPU inference verified |
| PLONK variants | `.local/models/phase6b/plonk` | 1,542,533,898 | YFCC ready; OSV/iNaturalist prepared-only |
| PaddleOCR | `.local/models/phase6b/paddleocr` | 96,576,898 | ready; real CPU inference verified |
| RapidOCR | managed AtlasLens model cache | 31,164,880 | ready; hash-bound proof and full fallback HTTP smoke passed |

Disk values are installation diagnostics, not latency or accuracy measurements.

## Configuration and OpenAI cost control

Copy `.env.example` to the untracked repository-root `.env`. The only supported
API-key location is:

```text
<PRIVATE_MEDIA_ROOT>
OPENAI_API_KEY=
```

Never add `OPENAI_API_KEY` or an OpenAI key to a `VITE_` variable, frontend file,
command line, issue, log or test fixture. A safe Phase 6B example is:

```text
PHASE6B_ENABLED=true
OSV5M_ENABLED=true
OSV5M_WORKER_ENABLED=true
OSV5M_WORKER_HOST=127.0.0.1
OSV5M_WORKER_PORT=8791
PLONK_ENABLED=true
PLONK_WORKER_ENABLED=true
PLONK_WORKER_HOST=127.0.0.1
PLONK_WORKER_PORT=8792
PADDLEOCR_ENABLED=true
PADDLEOCR_WORKER_ENABLED=true
PADDLEOCR_WORKER_HOST=127.0.0.1
PADDLEOCR_WORKER_PORT=8793
PADDLEOCR_FALLBACK_TO_RAPIDOCR=true
GPU_MAX_HEAVY_CONCURRENCY=1
GPU_MAX_RESIDENT_MODELS=1

OPENAI_GEO_ENABLED=false
OPENAI_API_KEY=
OPENAI_GEO_MODEL=gpt-5.6-luna
OPENAI_GEO_REASONING_EFFORT=none
OPENAI_GEO_IMAGE_DETAIL=low
OPENAI_GEO_MAX_OUTPUT_TOKENS=500
OPENAI_GEO_MONTHLY_BUDGET_USD=4.50
OPENAI_GEO_DAILY_CALL_LIMIT=10
OPENAI_GEO_PER_ANALYSIS_CALL_LIMIT=1
OPENAI_GEO_ALLOW_HIGH_DETAIL_RETRY=false
OPENAI_GEO_CACHE_TTL_DAYS=30
```

OpenAI is disabled by default. A call additionally requires `cloud_assisted`
mode, request-level cloud consent and `allow_cloud_assist=true`, a backend key,
an available daily/monthly budget and a hard case. Hard cases include low local
confidence, a small leading margin, strong model disagreement, at least 750 km
model separation, strong OCR conflict, no usable local candidate, at least
1,000 km dispersion or an explicit review request. Strong local evidence skips
the call. Cache hits consume no new call. The provider uses one 768-pixel,
metadata-free, low-detail derivative; `gpt-5.6-luna`; reasoning `none`; at most
500 output tokens; no tools; no high-detail retry; and structured bounded output.
It may adjust only supplied candidate IDs within the configured bound or reject
all candidates. It cannot create coordinates.

The local SQLite ledger reserves conservatively before a call and records bounded
token/cost counters without storing images, prompts, raw OCR, raw responses or API
keys. The checked-in price receipt is an operational estimate and must be reviewed
when public pricing changes; it is not a billing guarantee.

## Run, smoke and evaluate

Start the verified isolated workers, then the existing application:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-phase6b-workers.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-api.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-web.ps1
```

The worker start command runs one real offline inference per selected worker by
default. `scripts\dev.ps1` runs the same worker lifecycle automatically. Stop
manual workers with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-phase6b-workers.ps1
```

Run a local-only smoke with a licensed local image:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\smoke_phase6b.ps1 -Image C:\licensed\smoke.jpg
```

Acceptance exercised both OCR routes. The primary PaddleOCR pipeline completed,
and a separate full HTTP run with the Paddle worker disabled completed and
persisted with provider `rapidocr-ppocrv6-local`, `fallback_used=true` and two real
detections. GeoCLIP, SegFormer, OSV-5M and PLONK also completed in that fallback
run; `phase6b-v1` returned five bounded clusters and `accuracy_claim` stayed null.
This is functional evidence, not a location-accuracy or performance measurement.

The real OpenAI smoke is never automatic and needs both explicit flags:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\smoke_phase6b.ps1 -Image C:\licensed\smoke.jpg -UseOpenAI -CloudConsent
```

Do not run that command merely to test configuration. It can make one billable
low-detail call. It first checks key and budget state and never prints the key.

The lightweight evaluator consumes a user-owned JSON manifest and records from
real runs; it does not invoke a model or download a dataset:

```powershell
.\services\api\.venv\Scripts\python.exe .\scripts\evaluate_phase6b.py `
  --manifest .\evaluation\geolocation\manifest.json `
  --results C:\private\atlaslens\phase6b-real-runs.json `
  --output C:\private\atlaslens\phase6b-evaluation.json
```

Each results record names one manifest image and one fixed profile. `predictions`
is ordered Top-K and `providers` contains the measured real provider outcomes:

```json
{
  "profile": "geoclip_only",
  "image": "images/example.jpg",
  "predictions": [
    {"latitude": 38.72, "longitude": 35.48, "country": "TR", "city": "Kayseri"}
  ],
  "providers": [
    {"provider": "geoclip", "status": "completed", "latency_ms": 42.0}
  ],
  "cloud": {"triggered": false, "cache_hit": false, "estimated_cost_usd": 0.0}
}
```

The required profile IDs are `geoclip_only`, `geoclip_osv5m`,
`geoclip_plonk`, `all_local_models`, `all_local_models_ocr`, and
`optional_cloud_assistance`. Records are measurements from actual runs; never fill
missing providers or predictions with synthetic coordinates.

It reports sample count, mean/median error, 1/25/100/750 km rates, Top-K oracle
rates, country/city accuracy, provider success/latency, descriptive fusion deltas,
OpenAI trigger/cache/cost totals and all six requested ablations. By default all
six profiles must be complete and contain at least 100 real labelled images before
`improvement_claim_allowed` can become true. Even then, dataset independence,
license, representativeness and subgroup review are separate gates.

## Windows troubleshooting

- An empty `.local/vendor/<model>` directory containing only `.git` was a previous
  `--no-checkout` bug. The current script always force-checks out the pinned commit;
  rerunning the same command safely repairs and resumes it.
- OSV-5M intentionally has no `setup.py` or `pyproject.toml`. The bootstrap installs
  its pinned checkout's official `requirements.txt` and uses the checkout as the
  inference source tree; do not run `pip install .local/vendor/osv5m`.
- Worker tooling pins `huggingface-hub==1.23.0` and calls `snapshot_download`
  through Python. Do not downgrade it to 0.33.4: that is incompatible with the
  currently resolved Transformers 5.13.1 worker stack.
- `isolated_worker_python_missing`: install CPython 3.10 through the normal
  operator process, then rerun only the requested bootstrap model. Do not replace
  the API's Python 3.12 environment.
- CUDA unavailable or out of memory: confirm the pinned API environment reports
  Torch `2.7.1+cu128`, keep both GPU limits at `1`, close competing GPU programs,
  and rerun diagnostics. Providers fail/skip safely; they do not silently emit a
  CPU result unless their configured adapter supports that fallback.
- A PaddleOCR or RapidOCR package alone is insufficient. Keep request-time model
  downloads disabled and require the pinned weight checks plus real-inference
  verification before exposing `ready`.
- A missing optional provider must not block API startup. Check both
  `atlas diagnostics phase6b` and `/api/v1/capabilities`; do not infer readiness
  from the presence of a directory.

## Domain limitations

OSV-5M is street-view oriented. PLONK specialization depends on scene routing.
GeoCLIP can return geographically broad candidates. OCR needs readable text and a
verified place resolver. OpenAI is an optional reviewer, not verified ground
truth. The restored Mapillary names describe segmentation pixels and carry no
geographic score. Only PLONK YFCC has real-inference acceptance; the prepared OSV
and iNaturalist specializations are not claimed ready. Confidence remains
uncalibrated. Phase 6C has not started.
