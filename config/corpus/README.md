# AtlasLens corpus planning controls

These Product Phase 3A files are not imported by the API, frontend, providers, or
Phase 6C runtime.

- source-policy-v1.json: source evidence and fail-closed decisions.
- manifest-schema-v1.json: one future corpus asset record.
- turkiye-pilot-sampling-v1.json: tiers, 81 provinces, sizes and storage math.
- leakage-policy-v1.json: pre-index and pre-score contamination controls.

## Validate offline

```powershell
.\services\api\.venv\Scripts\python.exe .\scripts\phase3\validate_phase3_plan.py
.\services\api\.venv\Scripts\python.exe .\scripts\phase3\validate_phase3_plan.py --manifest C:\approved\manifest.jsonl
```

The validator accepts only `.json`, `.jsonl`, `.ndjson`, and `.csv` manifests;
it rejects every other suffix before opening the path and never opens images.
Acquisition-ready rows fail on disallowed sources, missing attribution/provenance,
invalid coordinates/hashes, inactive
deletion state, duplicate IDs, or detectable sequence/capture/source/spatial
split conflicts.

Corpus versions are immutable. Revocation produces a successor version, source
tombstones, shard rebuild, cache purge, backup disposition, and deletion
attestation. No dataset, imagery, weight, descriptor, index, secret, database, or
runtime environment belongs here.
