# Product Phase 3A go/no-go decision

## Selected strategy

**A — first-party/partner-owned street imagery plus open metadata.**

AtlasLens capture is the default. Partner data requires a source-specific written
agreement granting commercial, derivative, descriptor/index, retention,
attribution, privacy, audit, and revocation rights. OSM/GeoNames may enrich
authorized imagery but do not grant imagery rights.

This is selected because it offers the clearest controller, provenance, privacy,
deletion, and commercial-rights chain. No legal clearance is claimed.

Not selected:

- Mapillary/KartaView persistent indexing without written clarification.
- Research benchmark data as a commercial corpus.
- Google Street View or Maps/Earth persistent indexing.
- A mixed physical index that erases source partitions.

## Initial pilot

Select the **Recommended MVP**: 50,000 first-party street references, zero
aerial/satellite patches in the initial strategy, 240 locked queries, all 81
provinces, a separate Kayseri–Ankara–Sivas dense tier, 120.20992 GB cloud
storage, 4.75792 GB selective local output, 5.25 descriptor GPU-hours, and 5.0
evaluation GPU-hours. The cloud budget adds bounded setup/retry capacity and is
authoritative.

Operator Kayseri/Ankara images are not acquisition seeds, route targets,
references, hard cases, or benchmark inputs.

## Phase 3B continuation gates

1. **Rights:** 100% of admitted assets are acquisition-ready with reproducible
   rights/provenance, attribution, coordinate accuracy, and active deletion state;
   zero unknown, blocked, expired, revoked, or permission-pending assets.
2. **Coverage:** all 81 province minimums and all urbanicity, road, scene and dense
   corridor cells pass; no contributor, sequence or dense tier substitutes for
   national breadth.
3. **Leakage:** every configured check completes with zero unresolved cross-split
   conflict; holdout truth loads only after predictions finish.
4. **Descriptors:** at least 99.5% produce one finite checksum-bound descriptor;
   sustained generation is at least 8,000 references per billed GPU-hour.
5. **National benchmark:** within-200-km Recall@10 is at least 0.70, province
   Recall@10 is at least 0.55, median nearest-candidate distance is at most
   150 km, and the absolute urban/rural Recall@10 gap is at most 0.20.
6. **Paired improvement:** on the same locked holdout, the paired-bootstrap 95%
   confidence interval for the change in within-200-km Recall@10 over the frozen
   six-reference baseline has a lower bound above zero.
7. **Dense tier:** non-operator Tier 2 results are reported separately and cannot
   rescue a failed national Tier 3 gate.
8. **Operations:** index/deletion dry-runs and checksummed backup pass; no secret
   appears; hard cost and C/D disk floors pass.
9. **Semantics:** similarity stays uncalibrated; provenance and positive
   uncertainty remain; no probability or production claim is made.

Any failure means stop and request a new bounded plan. It does not authorize more
data, budget, holdout reuse, relaxed gates, or Phase 6C activation.

## Provision decision

**READY FOR USER APPROVAL — exact budget and corpus are defined, but nothing has
been created.**

This is not permission to create an account, add billing, provision a Pod/volume,
acquire imagery, or run a model.

## Exactly one next phase

**Product Phase 3B — rights-approved first-party Türkiye pilot acquisition,
leakage-safe corpus assembly, and locked offline benchmark.**

Do not begin it without explicit approval, professional review of selected terms,
and a refreshed pricing check.
