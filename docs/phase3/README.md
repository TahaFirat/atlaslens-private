# AtlasLens Product Phase 3A

Status: **planning and locally validated configuration only**.

Product Phase 3A defines a commercially cautious Türkiye reference-corpus pilot,
an independent benchmark, and a capped cloud execution plan. It does not acquire
data, create descriptors, provision infrastructure, activate Phase 6C, or change
the API, frontend, providers, ranking, confidence, or candidate behavior.

## Selected strategy

The selected initial strategy is **A — first-party/partner-owned street imagery
plus open metadata**. AtlasLens-captured imagery is the default. Partner imagery
is admissible only under a written agreement explicitly covering commercial use,
derivatives, machine-readable embeddings/indexes, retention, attribution,
privacy responsibilities, and revocation/deletion.

Research-only datasets, API-display imagery, aerial sources, and other licensed
collections remain source-specific partitions. They cannot silently enter the
commercial pilot or inherit another source's decision. This is technical due
diligence, not legal advice or legal clearance.

## Deliverables

| Area | Plan | Machine-readable control |
|---|---|---|
| Source rights | source-license-matrix.md | ../../config/corpus/source-policy-v1.json |
| Sampling and sizing | turkiye-pilot-corpus-plan.md | ../../config/corpus/turkiye-pilot-sampling-v1.json |
| Manifest governance | turkiye-pilot-corpus-plan.md | ../../config/corpus/manifest-schema-v1.json |
| Leakage | turkiye-pilot-corpus-plan.md | ../../config/corpus/leakage-policy-v1.json |
| Cloud and cost | cloud-execution-plan.md | ../../config/cloud/runpod-phase3-budget-v1.json |
| Account timing | account-action-checklist.md | budget approval gates |
| Decision and risk | go-no-go-decision.md / risk-register.md | offline validator |

## Non-negotiable boundaries

- Only GO, GO_WITH_ATTRIBUTION, or a qualifying FIRST_PARTY_ONLY record can be
  acquisition-ready.
- Missing license text means no commercial permission. Code, weights, imagery,
  metadata, and embedding/index rights are separate.
- Every asset retains immutable source identity, attribution, rights provenance,
  hashes, split role, and deletion state.
- Source-specific indexes prevent license or revocation contamination. Combining
  ranked results never erases source lineage.
- Revocation creates a successor corpus version and a tombstone covering raw
  objects, derivatives, descriptors, index shards, caches, and backups.
- Holdout queries are unavailable during reference construction and truth loads
  only after inference. Operator Kayseri/Ankara images are permanently excluded.
- Google Street View and Google Maps/Earth imagery are not index sources.
- D: remains source-code-only; corpus material is not synced into this repository.

## Offline validation

```powershell
.\services\api\.venv\Scripts\python.exe .\scripts\phase3\validate_phase3_plan.py
.\services\api\.venv\Scripts\python.exe .\scripts\phase3\validate_phase3_plan.py --manifest C:\approved\manifest.jsonl
```

Nothing here authorizes account creation, billing, acquisition, provisioning,
model execution, or Phase 6C activation.
