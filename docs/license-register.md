# AtlasLens dependency and service register

## Product Phase 3C1 NVIDIA hosted service boundary

Phase 3C1 adds no NVIDIA SDK dependency, hosted-model weight, training dataset,
evaluation corpus, reference image, index or runtime artifact. It uses the
existing pinned HTTP/Pydantic/Pillow runtime. The service is disabled by default,
requires explicit provider selection plus request-level cloud mode and consent,
and sends only metadata-free in-memory JPEG derivatives of at most 180,000 bytes.
The separate NVCF asset-upload surface is intentionally unused.

| Component | Exact service/model | Terms, provenance and host policy |
|---|---|---|
| NVIDIA hosted Qwen vision reasoning | `qwen/qwen3.5-397b-a17b`; exact origin `https://integrate.api.nvidia.com/v1`; completion `POST /chat/completions`; documented asynchronous polling `GET /status/{requestId}` | NVIDIA's official page/model card and API reference govern availability and use. They reference NVIDIA service/model terms and license notices; training-data specifics, input/output rights, retention, regional availability, acceptable use and commercial scope require current operator/legal review. Backend-only credential; no redirects, arbitrary endpoint, asset upload, startup download or committed service output. The hosted endpoint is scheduled for retirement on 2026-07-27. |

Official references last reviewed for this decision:

- [NVIDIA model and deployment page](https://build.nvidia.com/qwen/qwen3.5-397b-a17b?nim=self-hosted&section=deploy)
- [NVIDIA inference API reference](https://docs.api.nvidia.com/nim/reference/qwen-qwen3-5-397b-a17b-infer)
- [NVIDIA status-polling reference](https://docs.api.nvidia.com/nim/reference/qwen-qwen3-5-397b-a17b-statuspolling)

This register records an engineering boundary, not legal approval, model
accuracy, calibrated confidence, production readiness or continued availability
after the scheduled retirement date.

## Phase 6B pretrained-model and service boundary

Phase 6B downloads no training or retrieval dataset and commits no model weight.
`config/external-models.lock.json` permits explicit pretrained-weight preparation
only, with request-time downloads disabled. Repository/code licenses do not grant
rights in upstream training imagery, arbitrary model inputs/outputs or commercial
deployment; those remain separate operator/legal reviews.

| Component | Exact source/revision | Declared upstream license / concern | Current host policy |
|---|---|---|---|
| GeoCLIP 1.2.0 | `VicenteVivan/geo-clip` `7a1a23b49648a5872a771cfda28490a17ab17d15`; wheel SHA-256 `c29ae90b01cf177ffa50707686fc8d0bd104bc5ee8537b6c7480ed4e1a0a5a2d` | MIT code; MP-16 training-image rights and evaluation overlap are not established by that license | Existing verified in-process provider; 1,796,094,205 private cache bytes; no dataset download |
| OSV-5M baseline | source `4e6075387ecde4255410785ffb83830c9aa099f6`; weights `osv5m/baseline` revision `71548b90ac4a1aa7c37839841f411a06da82b1a6` | MIT code/model repository; full OSV-5M imagery has the already recorded license conflict and surveillance/OSINT caution | Private weights installed; real CPU inference verified in isolated Python 3.10; full dataset remains prohibited |
| PLONK | source `76d46410910c9dfec9e19ed371450ebc7051cdf3`; package `diff-plonk==0.4` | MIT code; model training-domain data rights are not a blanket image/data license | Private isolated Python 3.10 environment; exactly one specialization selected; only YFCC has real-inference acceptance |
| PLONK OSV | `nicolas-dufour/PLONK_OSV_5M` revision `e23229f4dd91d52560e8827f5bb2c68257fa162f` | Inherits OSV-oriented domain/lineage concerns | Prepared-only; not claimed ready; reviewed StreetCLIP conditioning remains absent; same source family as OSV-5M |
| PLONK YFCC | `nicolas-dufour/PLONK_YFCC` revision `4f358d09938a89ed239a847777729e95c5d187bc` | YFCC training provenance does not authorize downloading a YFCC corpus here | Private weight preparation and real CPU inference verified; no YFCC dataset |
| PLONK iNaturalist | `nicolas-dufour/PLONK_iNaturalist` revision `8da6edcbdd01ff04a61f9d06e2de23ea300d1a35` | iNaturalist-domain model; media licenses remain per-record | Prepared-only; not claimed ready; no iNaturalist dataset |
| DINOv2 PLONK auxiliary | source `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; `dinov2_vitl14_reg` weight SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` | Apache-2.0 code/weight declaration; upstream/training-data and commercial-use review remain operator responsibilities | Private pinned auxiliary only; no DINOv2 training dataset |
| PaddleOCR / PaddlePaddle | PaddleOCR source `211989f046cc1878460f9e65574690c00a127a1a`; `paddleocr==3.7.0`, `paddlepaddle==3.3.1`; detector revision `ca867c897ecbca8873081573a802ad70d499cb94`; Latin recognizer revision `ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57` | Apache-2.0 code; model-weight notices and redistribution review remain required | Private weights; real CPU inference verified in isolated Python 3.12; no implicit request-time download |
| RapidOCR / ONNX Runtime | source `7fe716f8e38bb9a43f2680159f38deb14d8b1930`; `rapidocr==3.9.1`, `onnxruntime==1.27.0`; model revision `rapidocr-3.9.1` | RapidOCR engineering code declares Apache-2.0; upstream OCR model copyright belongs to Baidu/PaddleOCR; redistribution/commercial review remains required | Private wheel-bundled ONNX weights; hash-bound real CPU inference and full Paddle-disabled HTTP fallback verified |
| OpenAI `gpt-5.6-luna` | Responses API; price receipt in `config/cloud/openai-geo-review-v1.json`, reviewed 2026-07-14 | Commercial service terms, data controls, pricing and regional availability require current operator review | Disabled by default; consented hard-case review only; backend key; metadata-free derivative; local budget/cache controls |

No `trust_remote_code=True` path is used for these providers. External source
checkouts and weight directories are ignored private artifacts. The bootstrap
verifies exact revisions but is not legal or commercial-use approval. No OSV-5M,
YFCC, iNaturalist, Mapillary or other training/retrieval dataset was downloaded.

## Phase 6A SegFormer artifact boundary

Phase 6A adds no committed model weight, checkpoint, segmentation dataset, or new
imagery corpus. The trusted operator checkpoint and both local Hugging Face
directories are ignored private artifacts. The official NVIDIA base was staged
locally only to convert and exercise the operator-owned checkpoint; it is not
bundled or downloaded at API startup.

| Component | Exact source/revision | License/provenance finding | AtlasLens policy |
|---|---|---|---|
| Trusted `last_checkpoint.pt` | Operator-owned Colab training checkpoint; SHA-256 `1b41ad916a10998722c346bd466c48bd8d40a5f4c8d032f6db6622f692f0855b`; 439,721,578 bytes | Produced by the project operator. This records artifact identity, not rights in every training input, base model, or derived output. | Trusted local input only; `weights_only=True` CPU inspection; EMA selected; never accepted through HTTP, redistributed, or committed. Training-data/model-output rights remain operator responsibilities. |
| NVIDIA SegFormer-B2 Cityscapes base | Official `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`, revision `d633b2072669ca68d8f8e309de9b52bfdbf6bf72` | The official Hugging Face model card declares license `other`; NVIDIA/upstream model-card terms, Cityscapes provenance, acceptable use, notices, and commercial deployment scope require legal review. | Explicit operator staging of `config.json`, `preprocessor_config.json`, and `pytorch_model.bin` into ignored `.local/models/base-segformer-b2-cityscapes`; no startup download and no redistribution approval implied. [Official model card/files](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024/tree/d633b2072669ca68d8f8e309de9b52bfdbf6bf72). |
| Prepared AtlasLens SegFormer-B2 | Derived locally from the two artifacts above; deployment metadata schema `atlaslens-segmentation-deployment-v1`; 124 outputs; EMA; safetensors SHA-256 `a364afc012b98cc494d1d95ac81b9db1d37e7fd09eaa021a23fcbea74fc16e8b` | Safe serialization changes the runtime format but does not change or grant upstream/training-data rights. | Ignored `.local/models/atlaslens-segformer-b2-v4`; local evaluation only until base/checkpoint/training lineage and distribution/commercial terms receive explicit approval. |
| Mapillary 124-label configuration | Repository asset `assets/mapillary/segformer_mapillary_config.json`; SHA-256 `4c5c946b1f10f49011e873968f02e9303f1cee3c0c275e7423ef5ab936b1184f` | Exact ID/name metadata only. No Mapillary image, archive or training dataset was acquired. Taxonomy provenance/terms do not authorize imagery ingestion or redistribution. | Strict inverse/count/architecture validation; configuration-only relabeling; weights unchanged; scene labels descriptive with zero direct geographic weight. |

The conversion reused the already pinned PyTorch, Transformers, Pillow, and
safetensors runtime; no broad dependency upgrade or training-only package was
added. This inventory is an engineering record, not legal approval.

## Phase 5C custom artifact and dataset-QA boundary

Phase 5C bundles no custom trained weights, training dataset, evaluation dataset,
reference corpus, or proprietary service output. The committed development mock
is authored test metadata and is never a licensed production data source.

An operator-supplied custom artifact is registrable only when its strict manifest
records the actual license name, commercial-use status, `approved` review state,
and a concrete review reference alongside dataset/code lineage and exact hashes.
Registration is an engineering control, not legal approval. The operator remains
responsible for training-data rights, model/output rights, privacy/personality
rights, export/use restrictions, transitive runtime notices, and the scope of any
commercial deployment. Unknown, pending, or rejected license status cannot pass
artifact verification or promotion.

The optional ONNX execution path relies on an operator-installed ONNX Runtime.
The dependency is not downloaded at API startup; its pinned version, MIT notice,
native/CUDA notices, vulnerability status, and target-host acceptance must be
reviewed before deployment. Dataset-QA and evaluation catalogs read local artifacts
only and do not change the underlying source licenses or authorize reuse.

## Phase 5B licensed visual retrieval decision

| Component/source | Pinned terms or revision | Phase 5B policy |
|---|---|---|
| SigLIP2 B/16 384 | Official `google/siglip2-base-patch16-384`; revision `f775b65a79762255128c981547af89addcfe0f88`; declared Apache-2.0; `model.safetensors` SHA-256 `ed72c0ace85020ae610fc817c2538b9cae5a477b012a50859c60af5b3ad30857` | Explicit install only. Query and reference images use the same pinned offline processor and 768-dimensional encoder. Target-host CUDA/CPU acceptance is required before operational claims. |
| Wikimedia Commons | Current official reuse/licensing policy plus MediaWiki `geosearch` and `imageinfo` APIs | Approved only through per-file review with an explicit `CC0 1.0` or exact `CC BY` / `CC BY-SA` 2.0–4.0 allowlist. Creator, attribution, license URL, canonical page, coordinate kind and source metadata fingerprint are mandatory. Non-copyright restrictions remain an operator responsibility. |
| KartaView | Terms page last reviewed for the 17 June 2025 version; imagery terms state CC BY-SA 4.0 and require `© Grab and KartaView Contributors` | No acquisition without a separate operator terms-approval receipt. The current adapter intentionally remains unavailable even after the gate until implementation and legal approval are complete. |
| Mapillary | Official CC BY-SA guidance plus authenticated API v4 and applicable Meta platform/commercial terms | Not an initial reference source. Disabled without credentials, current terms approval and complete per-image/user attribution. |
| OSV-5M | Hugging Face card states CC BY-SA 4.0; official CVPR datasheet cites inherited CC BY-SA 2.0 and warns against privacy-infringing surveillance/OSINT | Hold. No slice or full dataset acquisition until curators resolve the primary-source conflict and legal/ethical review approves the use. |
| RapidOCR / PP-OCRv6 | RapidOCR 3.9.1 engineering code declares Apache-2.0; upstream states OCR model copyright belongs to Baidu/PaddleOCR | Explicit dependency install and local receipt/checksum only. No runtime model download. Full model notice/redistribution review remains required. |
| ONNX Runtime | GPU or CPU 1.27.0, MIT | Explicit local dependency; native/CUDA notices and target-host acceptance are required. Not installed in this checkout. |
| GeoNames forward | Official cities500/country/admin dumps; CC BY 4.0 | Explicit `geonames-forward-v2` install with hashes and `GeoNames` attribution. Not installed on the target host. |

The committed `config/datasets/commons-phase5b-10k-cells.csv` contains only
deterministic metadata-query anchors and requested quotas. It contains no image,
retrieval result or measured coverage. Planned target totals must never be
reported as acquired or validated dataset counts. See
`docs/phase-5b-licensed-reference-runbook.md` for the explicit review,
acquisition, evaluation-overlap, validation and offline-index workflow.

## Phase 5 model decision

| Component | Revision | Source/license | Artifact policy |
|---|---|---|---|
| GeoCLIP | PyPI `1.2.0`; implementation revision `7a1a23b49648a5872a771cfda28490a17ab17d15` | Manifest source is the official PyPI wheel at `files.pythonhosted.org`; upstream code declares MIT. The code license does not license MP-16 training images or prove evaluation independence. | Wheel `geoclip-1.2.0-py3-none-any.whl`, SHA-256 `c29ae90b01cf177ffa50707686fc8d0bd104bc5ee8537b6c7480ed4e1a0a5a2d`; explicit install only; named bundled weights/gallery extracted and individually receipted; no startup download |
| CLIP ViT-L/14 | Hugging Face revision `32bd64288804d66eefd0ccbe215aa642df71cc41` | Official `openai/clip-vit-large-patch14`; upstream model-card limitations and transitive notices apply | `model.safetensors` SHA-256 `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`; exact revision staged in private cache and required for offline runtime |
| GeoNames cities15000 | Runtime receipt; schema `atlaslens-geonames-v1` | Official `https://download.geonames.org/export/dump/`; CC BY 4.0; attribution `GeoNames` | Explicit installer fetches `cities15000.zip`, `countryInfo.txt`, and `admin1CodesASCII.txt`; actual bytes, sizes and SHA-256 values are recorded in the installation receipt because no fixed hashes are declared in the application manifest |
| OSV-5M baseline | deferred to Phase 6 review | Official repo/model card MIT; dataset license conflict remains recorded below; checkpoint/runtime review incomplete | Not installed or downloaded; single-coordinate output, pickle checkpoint and research stack do not block GeoCLIP |
| SigLIP2 | Phase 5B bounded retrieval exception; production serving remains Phase 6 | Official Google/Hugging Face Apache-2.0 checkpoint pinned above | Real encoder implementation present; not a standalone GPS provider; model remains uninstalled/unmeasured on this host |

The MIT repository license does not license GeoCLIP's MP-16 training images for
redistribution and does not establish that arbitrary web images are clean held-out
evaluation samples. AtlasLens downloads neither MP-16 nor OSV-5M in Phase 5.

### Phase 5 provisional Commons evaluation recipe

The ignored local acquisition recipe contains six individually reviewed files and
requires official API metadata at acquisition time. It records the official page,
creator/credit, license URL, camera-coordinate truth kind, page ID and content
hash. Local resized derivatives are marked as changed; no image is committed.
The 2026-07-11 operational run revalidated all six official records and produced
six ignored EXIF-free local derivatives. This records a successful review run, not
a corpus-wide Commons license conclusion or permission to redistribute the files.

| Geographic coverage | Commons file | Creator | File license |
|---|---|---|---|
| Europe / France | [Eiffel Tower viewed from the Champ de Mars](https://commons.wikimedia.org/wiki/File:Eiffel_Tower_viewed_from_the_Champ_de_Mars.jpg) | APK | CC BY-SA 4.0 |
| Oceania / Australia | [Sydney Harbour including Harbour bridge and Opera House](https://commons.wikimedia.org/wiki/File:Sydney_Harbour_including_Harbour_bridge_and_Opera_House.jpg) | Gibrate1 | CC0 1.0 |
| North America / United States | [Fort Point View of Golden Gate Bridge](https://commons.wikimedia.org/wiki/File:Fort_Point_View_of_Golden_Gate_Bridge.jpg) | CaryXz5 | CC0 1.0 |
| South America / Brazil | [Rio de Janeiro from Corcovado](https://commons.wikimedia.org/wiki/File:00_1031_Brasilien,_Rio_de_Janeiro.jpg) | W. Bulach | CC BY-SA 4.0 |
| Asia / Japan | [Chureito Pagoda and Mount Fuji](https://commons.wikimedia.org/wiki/File:Chureito_Pagoda_and_Mount_Fuji_20241022.jpg) | Supanut Arunoprayote | CC BY 4.0 |
| Africa / South Africa | [Table Mountain panorama](https://commons.wikimedia.org/wiki/File:Table_mountain_1_%E2%80%93_Panorama_(Greg_Zaal_and_Rico_Cilliers_via_Poly_Haven).jpg) | Greg Zaal and Rico Cilliers | CC0 1.0 |

This landmark-heavy six-image slice has unknown overlap with MP-16 and is too small
for general accuracy, fairness, regional, or calibration claims.

Phase 5 runtime installation and benchmark acceptance are still pending final host
verification. The pins above are manifest facts, not claims that the artifacts were
successfully installed or benchmarked on the target machine.

This is an engineering inventory, not legal advice. “Manifest checked” means the
declared license field was read from the installed package metadata; it is not a
legal conclusion and does not cover every transitive dependency. Production
release requires an SBOM, notice-file generation, vulnerability scan, and legal
review in Phase 6.

## Backend runtime

Phase 4 uses the already pinned OpenCV headless package for generated-fixture local
geometry. No copyrighted photography, external dataset, or map extract was added.
The offline map fixture is synthetic test data. The Overpass-compatible provider has
no enabled network transport or default public endpoint; any future operator endpoint
requires separate terms, attribution, privacy, capacity, and acceptable-use review.

| Dependency | Pinned version | Declared license | Review status |
|---|---:|---|---|
| FastAPI | 0.116.1 | Not populated in installed metadata | Upstream source/license review required |
| Uvicorn | 0.35.0 | BSD-3-Clause | Metadata checked; notices/legal review pending |
| Pydantic / pydantic-settings | 2.11.7 / 2.10.1 | MIT | Metadata checked; notices/legal review pending |
| SQLAlchemy / Alembic | 2.0.41 / 1.16.4 | MIT | Metadata checked; notices/legal review pending |
| Pillow | 11.3.0 | MIT-CMU | Metadata checked; image codec/notices review required |
| NumPy | 2.2.6 | BSD-style; bundled components have additional notices | Metadata checked; full wheel notice review required |
| OpenCV headless | 4.12.0.88 | Apache-2.0 | Metadata checked; bundled codec/build review required |
| GeoCLIP | 1.2.0 | MIT | Official wheel pinned and hashed; weights/gallery provenance and deployment review remain required |
| PyTorch / torchvision | 2.7.1 / 0.22.1 | BSD-3-Clause / BSD-style | CUDA/native binary notices, redistribution and vulnerability review required |
| Transformers / huggingface-hub | 4.53.2 / 0.33.4 | Apache-2.0 / Apache-style | Runtime forced offline; install-time source/model-card and transitive notices remain required |
| safetensors | 0.5.3 | Installed metadata did not declare a license | Upstream source/license and binary notice review required before release |
| psycopg / psycopg-binary | 3.2.9 | LGPL-3.0 | Metadata checked; binary/linking and notices review required |
| python-multipart | 0.0.20 | Apache-2.0 | Metadata checked; notices/legal review pending |
| OpenAI Python SDK | 1.97.1 | Apache-2.0 | Metadata checked; optional service terms reviewed separately |

## Backend development

| Dependency | Pinned version | Declared license | Review status |
|---|---:|---|---|
| pytest / pytest-asyncio | 8.4.1 / 1.1.0 | MIT / Apache-2.0 | Metadata checked; development-only review pending |
| Ruff | 0.12.4 | MIT | Metadata checked; development-only review pending |
| mypy | 1.17.0 | MIT | Metadata checked; development-only review pending |
| HTTPX | 0.28.1 | BSD-3-Clause | Metadata checked; development/runtime-transitive review pending |
| piexif | 1.1.3 | MIT | Metadata checked; test-only review pending |
| Hatchling | 1.27.0 | Pending installed-metadata check | Build-only review required |

## Frontend runtime

These fields were checked from the installed top-level package manifests.

| Dependency | Pinned version | Declared license | Review status |
|---|---:|---|---|
| React / React DOM | 19.0.0 | MIT | Manifest checked; notices/legal review pending |
| TanStack React Query | 5.66.8 | MIT | Manifest checked; notices/legal review pending |
| MapLibre GL JS | 5.3.0 | BSD-3-Clause | Manifest checked; attribution/notices review pending |
| Zod | 3.24.2 | MIT | Manifest checked; notices/legal review pending |

## Frontend development

| Dependency group | Pinned versions | Declared license | Review status |
|---|---|---|---|
| Vite / React plugin / Vitest | 6.4.3 / 4.3.4 / 3.2.7 | MIT | Manifest checked; development-only review pending |
| TypeScript | 5.8.2 | Apache-2.0 | Manifest checked; development-only review pending |
| ESLint ecosystem | 9.39.4 and manifest pins | MIT | Manifest checked; development-only review pending |
| Testing Library packages | manifest pins | MIT | Manifest checked; development-only review pending |
| Playwright Test | 1.61.1 | Apache-2.0 | Manifest checked; browser binary notices review pending |
| jsdom | 26.0.0 | MIT | Manifest checked; transitive notices review pending |
| openapi-typescript | 7.6.1 | MIT | Manifest checked; development-only review pending |
| React/Node type packages | manifest pins | MIT | Manifest checked; development-only review pending |

## Containers and system packages

| Component | Pin | Status |
|---|---|---|
| Python Docker Official Image | 3.12.10-slim-bookworm | Docker/base-distribution terms and full package SBOM review required |
| Node Docker Official Image | 24.12.0-bookworm-slim | Docker/base-distribution terms and full package SBOM review required |
| nginx-unprivileged | 1.27.4-alpine3.21 | Image provenance, Nginx/Alpine notices, and vulnerability review required |
| PostGIS/PostgreSQL image | postgis/postgis:16-3.4-alpine | Database/PostGIS/image-layer notices and vulnerability review required |

Image tags are pinned for Phase 1 reproducibility but are not immutable digests.
Digest pinning and automated SBOM/signature verification are Phase 6 release work.

## Optional executables and services

| Dependency | Data boundary | Terms/license status |
|---|---|---|
| Tesseract executable/language data | Local normalized image; redacted output | Not bundled or downloaded; operator must review executable and language-data licenses |
| OpenAI API | Consent-gated normalized, downscaled, metadata-stripped image | Optional; operator must review current API/data-processing terms, account/model availability, region, and retention before enabling |
| External MapLibre style/tile provider | Browser IP and viewed map region may reach provider | None configured by default; production provider license, attribution, privacy, capacity, and SLA review required |
| GeoNames cities15000 gazetteer | Explicit operator download; prediction-time resolution is offline | Official dump declares CC BY 4.0; preserve attribution `GeoNames`; installed artifact hashes/sizes are recorded in the local receipt |

No external imagery dataset, geolocation model weights, retrieval index, Google
Street View, social-network scraper, or proprietary geolocation service is
included in Phase 1.

## Phase 3 retrieval runtime

| Dependency | Pin | Declared license / provenance | Review status |
|---|---:|---|---|
| FAISS CPU | 1.14.3 | MIT; official project and PyPI Trusted Publishing. CPython 3.12 Windows x86-64 wheel is 16.2 MB with published SHA-256 `0a5eb27184123c7ac1060c6b862978eabf0e30c1369ccf8bdb1497d35c06ad3d`. | Approved for Phase 3 engineering use; transitive/binary notices still require release review |

Sources: [official FAISS repository/license](https://github.com/facebookresearch/faiss),
[PyPI release files](https://pypi.org/project/faiss-cpu/). FAISS is an index
library; it supplies no image model or dataset.

## Phase 3 dataset research matrix

No source below is downloaded or automatically imported. Storage figures are
official published values where available, not AtlasLens measurements.

| Source | License/terms finding | Official access and metadata | Size/GPS limits | AtlasLens status |
|---|---|---|---|---|
| OSV-5M | Primary-source conflict: the official Hugging Face card says CC BY-SA 4.0, while the CVPR paper/datasheet cite inherited CC BY-SA 2.0. Both require attribution/share-alike; the datasheet also discourages surveillance/OSINT use. | Hugging Face download APIs; coordinates, hierarchy, sequence/capture/source/environment and creator fields. | 4,894,685 train + 210,122 test images; repository reports 259 GB. Coordinate accuracy is unpublished and float precision is not accuracy. | **Hold** until curator clarification and legal/ethical review; reconsider only in Phase 6. |
| Mapillary | Official help states imagery is CC BY-SA 4.0; current platform/commercial terms also apply. | API v4/official SDK; raw and computed geometry, creator, capture time, heading, camera, dimensions, sequence and thumbnails. | More than 2.4B images reported; no full-corpus byte size/bulk snapshot. GPS varies by device; moderation guidance is not an accuracy SLA. | Conditional per-record import after terms review and complete attribution. |
| Wikimedia Commons | No corpus-wide license. Individual files may use CC BY, CC BY-SA, GFDL, public domain or multiple licenses; non-copyright rights can remain. | MediaWiki geosearch + imageinfo/extmetadata for URLs, hashes, size, attribution, restrictions and coordinates. | No current full media dump; all Commons media is reported around 853 TB, geotagged subset unknown. Coordinates may be camera/object/manual/EXIF locations. | Per-file allowlist only; reject ambiguous/missing license metadata. |
| KartaView | Terms state street imagery and 3D spatial data are CC BY-SA 4.0 with required `© Grab and KartaView Contributors` credit. | Official photo/sequence APIs; raw and map-matched coordinates, heading, time, dimensions, device and sequence metadata. | “Millions” of photos; no byte total/bulk dump. GPS accuracy unpublished; raw and map-matched coordinates remain distinct. | Conditional import with exact attribution and terms-version review. |

Primary sources:

- [OSV-5M dataset card](https://huggingface.co/datasets/osv5m/osv5m),
  [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Astruc_OpenStreetView-5M_The_Many_Roads_to_Global_Visual_Geolocation_CVPR_2024_paper.pdf),
  [datasheet](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Astruc_OpenStreetView-5M_The_Many_CVPR_2024_supplemental.pdf).
- [Mapillary license](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data)
  and [official SDK](https://mapillary.github.io/mapillary-python-sdk/docs/mapillary/mapillary.interface/).
- [Commons reuse policy](https://commons.wikimedia.org/wiki/Commons%3AReusing_content_outside_Wikimedia/en),
  [licensing policy](https://commons.wikimedia.org/wiki/Commons:Licensing), and
  [metadata API](https://www.mediawiki.org/wiki/Extension%3ACommonsMetadata/en).
- [KartaView terms](https://kartaview.org/terms),
  [photo API](https://kartaview.org/doc/photos), and
  [sequence API](https://kartaview.org/doc/sequences).
