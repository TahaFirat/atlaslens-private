# Phase 3B1 corpus job operator runbook

The checked-in job is disabled and performs no RunPod API operation. The default
entrypoint only validates configuration. The run command is a non-executing dry
run unless --execute is supplied. Production execution additionally requires an
approved configuration and a non-empty approval receipt.

## Safe local synthetic dry run

These commands use generated pytest fixtures and temporary directories. They do
not read operator media, call a provider, download an artifact, or use a network.
Run them from the repository root.

    $env:PYTHONPATH = 'D:/geoSearch/services/api/src'
    D:/geoSearch/services/api/.venv/Scripts/python.exe -m pytest services/api/tests/test_phase3b_corpus_pipeline.py services/api/tests/test_phase3b_corpus_index.py services/api/tests/test_phase3b_cloud_package.py services/api/tests/test_phase3b_api.py services/api/tests/test_phase3b_e2e.py -q
    D:/geoSearch/services/api/.venv/Scripts/python.exe scripts/phase3b/cloud_job.py validate-config --config config/cloud/runpod-phase3b-job-v1.json
    D:/geoSearch/services/api/.venv/Scripts/python.exe scripts/phase3b/cloud_job.py run --config config/cloud/runpod-phase3b-job-v1.json --run-id synthetic-local --dry-run

The first command is synthetic functional evidence only. It is not evidence of
real-world geolocation accuracy. The last command prints the explicit future
pipeline command but does not spawn it.

## Expected mounted layout

    /workspace/input/                       read-only, rights-approved inputs
      corpus/                               source bytes; never an output
        manifests/corpus.jsonl              locked manifest inside corpus root
      holdout/locked.json                   locked query projection; read-only
    /workspace/work/<run-id>/               disposable intermediate state
    /workspace/checkpoints/<run-id>/        resumable receipts and supervisor hooks
    /workspace/output/<run-id>/             atomically published outputs
      artifact-inventory.json               relative paths, byte sizes and SHA-256
      SHA256SUMS                            checksum list for regular artifacts

The input bind mount must be read-only. The container runs as UID/GID 10001 and
refuses execution when that user can write /workspace/input. Mutable roots are
run-scoped and separate. Symlinks in inventoried artifacts are refused.

## Future image build and RunPod execution

**DO NOT RUN UNTIL USER APPROVAL**

Image construction uses the digest-pinned base and existing uv.lock dependency
graph. It does not download models, datasets, or imagery, but base, OS and Python
package acquisition is still an external/network operation:

    docker build --file infra/phase3b/Dockerfile --tag atlaslens-phase3b1:approved .
    docker run --rm --network none atlaslens-phase3b1:approved validate-config --config /opt/atlaslens/config/cloud/runpod-phase3b-job-v1.json

Before execution, the operator must have all of the following:

- a user-approved RunPod resource and budget receipt;
- rights-approved immutable input and a locked manifest;
- a locked holdout JSON whose query rows include the admitted holdout asset ID,
  source, contributor, capture-run, sequence, sampling-cell, province and
  coordinates; every row must bind one-to-one to the persisted split lock;
- an approved production descriptor adapter and local weight receipt;
- a reviewed config copy with execution_enabled=true,
  approval_state=USER_APPROVED, unchanged hard limits, and no credential;
- explicit descriptor batch size, index backend/version/shard size, locked
  holdout path/hash, and positive abstention distance; the index must contain the
  paired holdout and split-lock hashes;
- verified writable work/output/checkpoint mounts and a separate backup target.

Do not edit the checked-in disabled config for an ad hoc launch. Mount the
reviewed approval-specific copy over the same container path. On an already
approved and manually provisioned Pod, the exact offline container command is:

    docker run --rm --gpus all --network none --read-only --user 10001:10001 --stop-timeout 120 --tmpfs /tmp:rw,noexec,nosuid,size=8g --mount type=bind,src=/approved/input,dst=/workspace/input,readonly --mount type=bind,src=/approved/work,dst=/workspace/work --mount type=bind,src=/approved/output,dst=/workspace/output --mount type=bind,src=/approved/checkpoints,dst=/workspace/checkpoints --mount type=bind,src=/approved/runpod-phase3b-job-v1.json,dst=/opt/atlaslens/config/cloud/runpod-phase3b-job-v1.json,readonly atlaslens-phase3b1:approved run --config /opt/atlaslens/config/cloud/runpod-phase3b-job-v1.json --run-id phase3b1-approved-001 --execute --approval-receipt phase3b1-approved-001

No public port, RunPod API key, provider token, or runtime network is needed.
The wrapper enforces a 16-hour hard deadline, a 15-minute checkpoint hook,
SIGTERM forwarding with a 120-second grace period, and final inventory creation.
The locked budget is $15 soft / $25 hard for one job and one GPU.
The wrapper validates those values and bounds runtime; it cannot observe provider
billing. The separately approved provider-side cap/watchdog remains mandatory.

## Resume after interruption

**DO NOT RUN UNTIL USER APPROVAL**

Keep the same immutable input, approved config, image digest, run ID and
checkpoint directory. Re-run the exact command above. The wrapper always passes
--resume; the pipeline must reject a changed manifest/config/source-policy hash.
It also rejects changed holdout identity or split-lock bindings. A partial index
is never a published index. Do not delete work or checkpoints until the final
inventory and benchmark receipt have been copied and verified.

## Cleanup

Preview is the safe default and redacts host paths:

    python scripts/phase3b/cloud_job.py cleanup --config config/cloud/runpod-phase3b-job-v1.json --run-id phase3b1-approved-001

**DO NOT RUN UNTIL USER APPROVAL**

Only after two independent copies and their SHA-256 values are verified:

    python scripts/phase3b/cloud_job.py cleanup --config config/cloud/runpod-phase3b-job-v1.json --run-id phase3b1-approved-001 --execute --confirm-run-id phase3b1-approved-001

Cleanup removes only the exact run-scoped work, output and checkpoint directories
from the configured roots. It refuses symlink roots or a mismatched confirmation.
RunPod volume deletion remains a separate, manually approved provider-console
action and is not implemented here.

## Rollback

Send SIGTERM, retain the immutable input and last valid checkpoints, and do not
promote the new index. Verify the previously published index metadata and
checksums, then point the later API configuration back to that version. Resume
only with matching hashes, or start a new run ID. Never overwrite or label a
partial index as complete.
