# Phase 3A cloud execution plan

Evidence and pricing snapshot: **2026-07-16**. This is a planning document, not
an authorization to acquire data or create cloud resources. Phase 3A created no
account, payment method, Pod, volume, token, model download, or dataset download.

## Decision

Use **one RunPod Secure Cloud on-demand Pod** as the primary execution platform
after a separate Phase 3B approval. Use the existing local RTX 4060 8 GB host as
the fallback for a chunked Minimal Proof only. Do not use a savings plan, Spot,
Serverless, a cluster, or a Community Cloud host.

This decision is conditional. Before any upload, AtlasLens must have an
acquisition-ready manifest, source permission, locally applied privacy controls,
a tested deletion workflow, an approved absolute stop time, and an independent
backup target. Secure Cloud is an infrastructure choice, not a legal-clearance or
data-residency conclusion.

## Platform comparison

| Platform | Current primary-source facts | Fit for this pilot | Decision |
| --- | --- | --- | --- |
| RunPod Secure Cloud | RunPod describes T3/T4 datacenters and higher reliability. Network volumes for Pods are Secure Cloud-only. The public RTX A5000 page lists 24 GB VRAM, 25 GB RAM, 9 vCPU and $0.27/GPU-hour for Secure Cloud. | A single deterministic batch Pod, persistent volume, per-second compute billing, and explicit termination controls fit the job. | **Primary**, on-demand only. |
| RunPod Community Cloud | RunPod describes peer-to-peer infrastructure and variable reliability. The A5000 page lists $0.16/hour, but network volumes for Pods are unavailable there. RunPod is no longer accepting new Community Cloud hosts, while existing resources remain. | Lower price does not offset host variability, missing network-volume workflow, and the sensitivity of first-party route imagery. | Not approved. |
| Vast.ai | Vast is a marketplace: hosts set changing compute, storage and bandwidth rates. Storage continues to bill while an instance exists; verified-host filters and on-demand rentals are available. | Technically viable, but provider-specific security, storage, bandwidth and deletion behavior add variance. No static price could be verified without a live offer. | Reconsider only by a new approval; not the fallback. |
| Google Colab | Google states that GPU type, usage limit, idle timeout and maximum VM lifetime vary and are not guaranteed, including paid plans. | Unsuitable for a resumable, governed corpus build with fixed hardware and storage. | Not approved. |
| Existing local RTX 4060 | NVIDIA specifies 8 GB GDDR6 VRAM. There is no cloud bill, but VRAM, elapsed time and current disk headroom are materially tighter. | Safe for a chunked smoke/Minimal Proof if data is streamed from separately approved encrypted storage and D: is not used for staging. | **Fallback**, Minimal Proof only. |

Primary evidence: [RunPod A5000](https://www.runpod.io/gpu-models/rtx-a5000),
[RunPod Pod selection](https://docs.runpod.io/pods/choose-a-pod),
[RunPod network volumes](https://docs.runpod.io/storage/network-volumes),
[Vast.ai pricing](https://docs.vast.ai/guides/instances/pricing),
[Vast.ai billing](https://docs.vast.ai/guides/reference/billing),
[Colab resource limits](https://research.google.com/colaboratory/faq.html), and
[NVIDIA RTX 4060 specifications](https://www.nvidia.com/en-us/geforce/news/ultimate-guide-to-4060/).

## Exact recommended RunPod shape

| Setting | Recommended MVP value | Gate |
| --- | --- | --- |
| Product | Secure Cloud Pod, on-demand | `cloudType=SECURE`; no savings plan or Spot |
| GPU | 1 x NVIDIA RTX A5000 | Exactly one GPU; no automatic substitution |
| Minimum VRAM | 24 GB | Reject a smaller or fractional GPU |
| CPU RAM / vCPU | At least 25 GB / 9 vCPU | Match or exceed the published A5000 shape |
| Container disk | 50 GB | Ephemeral OS, package and scratch data only |
| Volume disk | Not attached | A network volume replaces it at `/workspace`; do not provision duplicate persistent storage |
| Network volume | 150 GB, Standard tier | Recommended corpus needs 120.209920 GB; 29.790080 GB remains for cache and safety headroom; High-Performance is unnecessary and prohibited |
| Datacenter | Secure Cloud `EU-RO-1` candidate | Use only if the approval-day console simultaneously offers A5000 and Standard network storage there. Otherwise wait; do not silently change region or cloud type. |
| Container | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc` | Re-resolve and record the linux/amd64 manifest before launch; reject a digest mismatch |
| Runtime | CPython 3.12, PyTorch 2.7.1 + CUDA 12.8, torchvision 0.22.1 | Must match the existing project constraint and pass a paid-runtime preflight before corpus work |
| Network exposure | No public HTTP ports; key-based administrative access only when needed | No notebook or service exposed to the public Internet |
| Pod lifetime | 16 GPU-hours maximum for Recommended MVP | Absolute UTC deadline and provider `terminate-after` required before launch |
| Idle action | After 15 minutes with no active chunk and no queued work: checkpoint, sync receipts, terminate | RunPod permits stop, but container-disk data is cleared and network-volume billing continues; termination is the AtlasLens cleanup policy |

The base image digest is an exact planning pin, not an instruction to pull it in
Phase 3A. NVIDIA's current supported-tag list can retire old tags, so Phase 3B
must confirm availability, retain the digest receipt, and run a vulnerability and
license review before any paid launch. PyTorch's compatibility table records
Python 3.12 support and the CUDA 12.8 build line for PyTorch 2.7; the target host
already verified PyTorch 2.7.1 + cu128 locally. See the
[NVIDIA CUDA image source](https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/README.md),
[image digest evidence](https://hub.docker.com/layers/nvidia/cuda/12.8.1-cudnn-runtime-ubuntu24.04/images/sha256-24cb02c4b8dcaad75f6194f004874070f2bbe6040eed1002ced31cc9d717c24d),
and [PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md).

No registry account is required for this plan. After approval, the Pod may use
the pinned public base and an AtlasLens bootstrap bundle. The bundle must install
only locked Python 3.12 dependencies and must verify `python`, `torch`, CUDA,
MegaLoc artifact hash and descriptor dimension before accepting corpus data. A
future immutable custom image would require its own approval and registry plan.

## Corpus and storage alignment

The exact corpus requirements come from the Phase 3A sampling plan. Decimal GB
is used consistently; provisioned volumes are rounded upward.

| Option | References | Locked holdout | Governed assets | Required cloud GB | Provisioned volume GB | Local output GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Minimal Proof | 12,000 | 120 | 12,120 | 28.994650 | 40 | 1.168650 |
| Recommended MVP | 50,000 | 240 | 50,240 | 120.209920 | 150 | 4.757920 |
| Expansion | 180,000 | 600 | 180,600 | 432.145344 | 500 | 17.015344 |

The required cloud value already contains compressed imagery, normalized
derivatives, MegaLoc descriptors, metadata/index data, and one full
staging/rollback duplicate. The extra provisioned capacity is cache and
operational headroom, not permission to increase corpus size.

Current disk gates remain binding: C: has approximately 56.587 GiB and must
remain at or above 35 GiB; D: has approximately 10.359 GiB and must remain at or
above 8 GiB. Therefore:

- D: remains source-code only and receives no corpus, image, model-cache,
  descriptor, checkpoint or cloud-sync output.
- Recommended output may be checksum-synchronized to a proposed encrypted
  `C:\AtlasLensData\phase3\<run-id>` location outside the repository; the path is
  not created in Phase 3A.
- That C: path is a selective output/receipt destination, not a full-corpus
  backup. Before upload, a separately approved encrypted system of record must
  retain the governed source archive and rebuild inputs outside RunPod; the
  RunPod volume does not count as either required independent backup copy.
- Expansion may not perform a blind full sync. It needs a separately approved
  encrypted external/remote backup and a selective local receipt/index copy.
- Every preflight and synchronization rechecks both disk thresholds and aborts
  before writing when either reserve would be crossed.

## Data and runtime paths

RunPod mounts a Pod network volume at `/workspace`. Use immutable corpus version
paths and never mix sources with incompatible terms:

```text
/workspace/corpus/<corpus-version>/immutable/
/workspace/corpus/<corpus-version>/manifests/
/workspace/model-cache/<artifact-sha256>/
/workspace/checkpoints/<run-id>/
/workspace/outputs/<run-id>/
/workspace/revocation/<source-name>/
/tmp/atlaslens/<run-id>/                 # container disk; disposable
```

RunPod's Pod storage documentation exposes an optional **Encrypt volume** control
for the Pod volume disk only; that customer-selectable control does not apply to
the container disk or network volume. Separately, RunPod's DPA security measures
make the broader representation that encryption is used for data at rest through
AWS and Secure Cloud datacenter subprocessors, and its Terms place responsibility
for protecting customer content at rest and in transit on the customer. These are
different scopes: a product-level per-volume control versus a contractual
infrastructure statement. Neither source establishes customer-managed keys or
the exact coverage, region and subprocessor for this planned network volume.

Consequently, plaintext private route imagery is not approved for persistent
network or container storage. Keep every object on `/workspace`
application-encrypted, decrypt only one bounded shard into RAM or
`/tmp/atlaslens/<run-id>`, never checkpoint plaintext, and securely clear the
ephemeral shard before checkpoint, abort, or termination. A professional
privacy/security review must reconcile the storage documentation, DPA, Terms,
current subprocessor list, selected datacenter and international-transfer
requirements. The DPA security table currently contains an unresolved
`[INSERT]` marker in its encryption entry; do not infer coverage from it without
clarification. Approval of capture de-identification controls is also mandatory.
If any gate remains unresolved, do not upload. See the official
[storage-options documentation](https://docs.runpod.io/pods/storage/types),
[DPA](https://www.runpod.io/legal/data-processing-agreement), and
[Terms](https://www.runpod.io/legal/terms-of-service).

The model-cache path receives the already verified MegaLoc artifact only after
its hash is checked. This plan does not request or permit a new model download.
Checkpoints are append-only chunk receipts plus a temporary descriptor shard;
the final index is written to a new version and atomically promoted only after
validation.

## Upload and direct-download policy

1. Build and validate the manifest locally. Images with unresolved source rights,
   missing provenance, unapproved cloud processing, or incomplete blur state do
   not enter the upload set.
2. Encrypt the approved archive client-side and generate SHA-256 receipts. Upload
   to the RunPod network volume through its S3-compatible API, directly into an
   `incoming/<receipt-id>` prefix. Objects remain application-encrypted at rest;
   the S3 key is separate from the RunPod API key.
3. A direct source-to-volume download is allowed only when the source policy is
   `GO` or `GO_WITH_ATTRIBUTION`, the official bulk-download/API terms explicitly
   allow acquisition and caching, and the exact URL, license and hash become a
   provenance receipt. Use a short-lived signed URL where the source supports it.
4. Never scrape, relay through a notebook, place a long-lived source token on the
   Pod, or use display-only imagery rights as acquisition rights. If no official
   bulk path exists, use the locally governed upload path or abstain.
5. Verify archive hashes, decrypt only the active bounded shard into ephemeral
   RAM/container storage, and reject any asset not present in the manifest.

RunPod documents the S3-compatible transfer path and warns that recursive
operations on very large directories can be slow. Use sharded archives and
manifest-led copies rather than recursive listing. See the
[RunPod S3 API guide](https://docs.runpod.io/storage/s3-api).

## Execution and recovery

The paid job is batch-only and idempotent:

1. Preflight identity, Secure Cloud, GPU count/model/VRAM, RAM, vCPU, image
   digest, CUDA, Python, Torch, model hash, volume size, source-policy version,
   corpus version, privacy state, available credits and absolute deadline.
2. Run deterministic shards. A shard contains no adjacent holdout/reference
   sequences and is committed only after row count, descriptor dimension,
   normalized-vector check and output SHA-256 pass.
3. Write a checkpoint after at most 1,000 references or 15 minutes, whichever is
   earlier. Checkpoints record only asset IDs, versions, counts, offsets, hashes,
   timings and bounded error codes; never raw images, OCR text, coordinates,
   tokens or original filenames in logs.
4. Resume by scanning completed receipt IDs, not by trusting a mutable row count.
   A failed Pod is terminated, and a new Secure Cloud Pod in the same datacenter
   attaches the same network volume. There is no Community Cloud failover.
5. On success, build a new immutable descriptor/index version, validate it, copy
   approved outputs to the independent backup, verify both sides, and only then
   mark the run complete.

RunPod states that stopping clears container-disk data, a network volume persists
across stop or termination, and a later start may receive zero GPUs if capacity
has changed. These behaviors are the reason for network-volume checkpoints and
termination-based cleanup; a stopped Pod is not the AtlasLens retention plan.
See [Manage Pods](https://docs.runpod.io/pods/manage-pods).

## Cost model and caps

Public evidence on 2026-07-16 lists Secure Cloud RTX A5000 at **$0.27/GPU-hour**.
Standard network storage under 1 TB is **$0.07/GB/month** and running container
disk is **$0.10/GB/month**. Estimates use a 730-hour month, include one full month
of the rounded network volume, and include expected compute plus prorated 50 GB
container disk. RunPod says Pod compute/storage is billed per second, network
volumes hourly, and RunPod ingress/egress has no fee. Upstream-source transfer,
tax, VAT and exchange costs remain excluded.

| Option | Descriptor h | Evaluation h | Setup h | Expected / max GPU h | One-time compute | One-time container | Recurring network / month | Expected first month | Max first month | Hard cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Minimal Proof | 1.25 | 2.50 | 0.50 | 4.25 / 6 | $1.15 | $0.03 | $2.80 | $3.98 | $4.46 | **$10** |
| Recommended MVP | 5.25 | 5.00 | 1.00 | 11.25 / 16 | $3.04 | $0.08 | $10.50 | $13.62 | $14.93 | **$25** |
| Expansion | 18.75 | 12.50 | 2.00 | 33.25 / 48 | $8.98 | $0.23 | $35.00 | $44.21 | $48.29 | **$65** |

"One-time container" is the expected session-prorated 50 GB container-disk
charge, not a recurring retained disk. "Max first month" substitutes maximum
GPU-hours for expected hours and includes a full month of network storage. The
hard cap retains headroom for variation, but any console price, tax or other
billable item that would make the projected total exceed it blocks launch.

The source of truth for exact arithmetic is
`config/cloud/runpod-phase3-budget-v1.json`. Prices and stock vary by region and
time. The launch gate must compare the console's total hourly rate with the
snapshot; a higher rate or different resource shape requires a revised budget
and new approval. Public price evidence:
[A5000 model page](https://www.runpod.io/gpu-models/rtx-a5000),
[RunPod pricing](https://www.runpod.io/pricing), and
[Pod pricing documentation](https://docs.runpod.io/pods/pricing).

### Hard-cap enforcement

RunPod's documented default $80/hour account spend limit is a rate limit, not an
AtlasLens total-project budget. The AtlasLens hard budget is therefore enforced
by all of the following; if any control is unavailable or untested, do not
launch:

- one Pod, one GPU and one network volume only;
- auto-pay disabled and no savings/prepaid commitment;
- approval for exactly one scenario and its maximum GPU-hours;
- provider `terminate-after` plus an independently monitored absolute UTC
  deadline;
- an in-job watchdog that exits at 90% of either spend or GPU-hour cap and an
  external lifecycle watchdog that terminates at 100%;
- `notifyLowBalance=true`, plus alerts at 50%, 75% and 90% of both the hard
  budget and maximum GPU-hours;
- a console/billing audit before launch, at each alert, after Pod termination,
  and after volume deletion.

For Recommended MVP, the soft budget is **$15**, the hard budget is **$25**, the
expected GPU time is **11.25 hours**, and the cumulative maximum is **16 hours**.
Do not authorize Expansion until Recommended meets its benchmark gates.

## Shutdown, backup and deletion

Normal completion and every abort use the same order:

1. stop accepting new shards;
2. finalize the current bounded checkpoint or mark it incomplete;
3. confirm the independent governed source/derivative archive, then synchronize
   manifests, provenance, descriptor/index versions, deletion map, cost receipt,
   checkpoint receipts and validation report to the approved backups;
4. compare file counts and SHA-256 hashes;
5. terminate the Pod (do not merely leave it idle);
6. retain the network volume for at most seven days for validation and rollback;
7. after a second verified backup and a source-retention review, delete the
   network volume and retain its deletion receipt.

RunPod is not the system of record and must never be the only backup. Its docs
warn that depleted funds can cause storage deletion. Network volumes can be
increased but not decreased, so a larger scenario uses a new immutable volume
rather than mutating a smaller corpus version.

For source revocation, use the asset-level source and provenance map to build a
new corpus version that excludes every affected asset, derivative, descriptor,
index row and backup object. Quarantine and delete the old source partition;
invalidate all indexes containing it; synchronize deletion tombstones; verify
absence by asset ID, source asset ID, SHA-256 and descriptor row map; then delete
the obsolete volume/version and record receipts. Do not attempt in-place index
editing that would obscure lineage.

## Approval-day checklist

The user should do none of these actions in Phase 3A. After an explicit Phase 3B
approval, the operator must, in order:

1. reconfirm source rights, blur/privacy state, DPA/data-region requirements and
   acquisition-ready manifest status;
2. reconfirm public and console pricing, A5000 stock and the exact datacenter;
3. set the scenario hard budget and absolute UTC termination deadline;
4. validate the independent backup and local disk reserves;
5. create and fund the RunPod account, enable 2FA and low-balance notification;
6. create least-privilege lifecycle and S3 keys, store them outside Git, then
   create exactly one approved volume and Pod;
7. run a small paid preflight shard before any full corpus job.

## Evidence ledger

All links were reviewed on 2026-07-16:

- [RunPod GPU pricing](https://www.runpod.io/pricing)
- [RunPod RTX A5000 pricing and resource shape](https://www.runpod.io/gpu-models/rtx-a5000)
- [Current A5000 deployment console entry](https://console.runpod.io/deploy?gpu=RTX+A5000)
- [RunPod Pod pricing and billing units](https://docs.runpod.io/pods/pricing)
- [RunPod billing, credits, low-balance behavior and spend limit](https://docs.runpod.io/accounts-billing/billing)
- [RunPod account notification fields](https://docs.runpod.io/runpodctl/reference/runpodctl-user)
- [RunPod Secure/Community comparison](https://docs.runpod.io/pods/choose-a-pod)
- [RunPod Pod lifecycle](https://docs.runpod.io/pods/manage-pods)
- [RunPod scheduled stop/termination flags](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)
- [RunPod network-volume behavior and pricing](https://docs.runpod.io/storage/network-volumes)
- [RunPod storage types and the volume-disk encryption control](https://docs.runpod.io/pods/storage/types)
- [RunPod S3-compatible volume API](https://docs.runpod.io/storage/s3-api)
- [RunPod key management](https://docs.runpod.io/get-started/api-keys)
- [RunPod Data Processing Agreement](https://www.runpod.io/legal/data-processing-agreement)
- [RunPod Terms and shared-responsibility language](https://www.runpod.io/legal/terms-of-service)
- [Vast.ai marketplace pricing](https://docs.vast.ai/guides/instances/pricing)
- [Vast.ai billing and storage behavior](https://docs.vast.ai/guides/reference/billing)
- [Google Colab resource limits](https://research.google.com/colaboratory/faq.html)
- [NVIDIA RTX 4060 specifications](https://www.nvidia.com/en-us/geforce/news/ultimate-guide-to-4060/)
