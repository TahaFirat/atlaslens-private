# Product Phase 3A risk register

This is a technical planning control, not legal advice.

| ID | Material risk | Required control | Status |
|---|---|---|---|
| R-01 | Display/API access is mistaken for commercial index permission | Separate code, weights, imagery, API, cache, derivative, embedding and redistribution decisions; obtain written clarification | Open for platform imagery |
| R-02 | Partner lacks title or authority | Signed asset schedule granting commercial derivative/index, retention, audit, attribution, privacy and deletion rights | Open until a partner contract exists |
| R-03 | Faces, plates, homes, workers or routes create KVKK/privacy exposure | Lawful-basis review, notice, minimization, pre-admission blur, restricted quarantine, retention controls and DPIA/DPA as applicable | Professional review required |
| R-04 | Share-alike obligations contaminate unrelated data | Per-asset attribution and source-specific immutable corpus/index versions and exports | Partitioning planned; interpretation open |
| R-05 | Permission is revoked or terms change | Quarantine, tombstone, shard rebuild, cache purge, backup expiry/crypto-erasure and deletion attestation | Phase 3B procedure required |
| R-06 | Hash/crop/rotation/sequence/source/spatial/temporal/descriptor leakage | Run checked policy before indexing and scoring; require zero unresolved conflicts | Planned, not executed |
| R-07 | Dense corridor or city labels enable memorization | Keep dense tier separate and out of national holdout; use macro strata | Controlled in design |
| R-08 | Operator images influence selection | Permanent exclusion receipt; no known-answer capture or benchmark use | Required; not used here |
| R-09 | Satellite patches overlap across splits | Preserve parent scene and footprints; buffered no-overlap split | Open until an aerial source is approved |
| R-10 | Roads, seasons, cities or contributors are imbalanced | All 81 provinces, explicit strata and contributor/route/sequence caps | Planned |
| R-11 | Model/weight rights differ from corpus rights | Review each component separately and retain an alternative descriptor seam | Final professional review required |
| R-12 | GPU, idle Pod or volume exceeds budget | On-demand only, one Pod, alerts, idle stop, absolute deadline and hard cap | Configured; no resources exist |
| R-13 | Credentials or private imagery leak | Least privilege, no public ports, encrypted transfer, secret store, redacted logs and teardown | No credential created |
| R-14 | Local disks cross safety floors | Keep corpus off D:, selective backup, preserve C >=35 GiB and D >=8 GiB | Must recheck each future sync |
| R-15 | Pricing or availability changes | Refresh official console evidence before each approval | Inherent open risk |
| R-16 | Small benchmark supports a misleading claim | At least 100 locked examples; recommended 240; report denominators and subgroups | Planned |
| R-17 | Metadata licensing is treated as imagery licensing | Use OSM/GeoNames only to enrich independently authorized imagery | Controlled in policy |
| R-18 | Deletion misses replicas/backups | Checksum inventory from source asset through every derivative and two-person deletion attestation | Phase 3B implementation required |
| R-19 | RunPod exposes an optional encryption toggle for volume disk but not the selected network volume or container disk, while its DPA separately makes broader at-rest-encryption representations; Türkiye route imagery may also cross borders | Keep network-volume objects application-encrypted with customer-managed keys outside RunPod; decrypt only bounded shards into ephemeral storage and wipe each shard; reconcile product documentation, DPA, subprocessors, region, international-transfer/security and de-identification requirements before launch | Professional review required |

Stop the affected partition if terms cannot be reproduced, rights are unknown or
blocked, permission expires, privacy complaints remain, provenance is incomplete,
leakage remains, or budget/disk floors would be exceeded.
