# Türkiye pilot corpus and benchmark plan

## Status and boundary

This is a Phase 3A planning artifact. It creates no acquisition authority, legal
clearance, cloud resource, dataset, image, model weight, descriptor, or production
index. It does not read or use the two operator Kayseri/Ankara images and does not
change or enable Phase 6C.

The machine-readable source of sampling truth is
`config/corpus/turkiye-pilot-sampling-v1.json`; admission and split isolation are
governed by `config/corpus/manifest-schema-v1.json`,
`config/corpus/leakage-policy-v1.json`, and the independently researched
`config/corpus/source-policy-v1.json`. A source is usable only if the current
source-policy record explicitly permits the intended commercial, derivative, and
persistent embedding/index uses. This plan makes no legal-clearance claim.

## Decision-ready pilot

Build the **Recommended MVP** only after approval: 50,000 reference images,
240 independently captured locked-holdout queries, all 81 provinces, and the two
named dense corridors. Execute it as two bounded tranches:

1. Complete the 12,000-reference Minimal proof and its 120-query holdout. Stop if
   licensing, coverage, leakage, budget, or candidate-recall gates fail.
2. If every Minimal proof gate passes, extend the same immutable source partitions
   to the 50,000-reference target and 240-query holdout. Expansion to 180,000 is a
   later approval decision, not part of this pilot.

Reference images have exactly one primary tier. A duplicate that satisfies two
coverage cells is counted once. Holdout queries are governed assets but are never
reference assets.

The selected first-party/rights-transferred street strategy currently admits no
aerial or satellite source. The three option baselines therefore contain the
following explicit reference mix; zero is intentional, not missing data:

| Option | Street references | Aerial/satellite patches | Total references |
|---|---:|---:|---:|
| Minimal proof | 12,000 | 0 | 12,000 |
| Recommended MVP | 50,000 | 0 | 50,000 |
| Expansion | 180,000 | 0 | 180,000 |

A future rights-approved aerial partition requires a versioned option and new
storage/budget calculation. It is not silently added to these totals.

### Route unit semantics

A route unit is one independently receipted drive, flight, capture mission, or
partner delivery with a stable `capture_run_id`; it is not a claim that an entire
road has been imaged. Repeated travel on the same path and day remains one run.
Route units are kept whole within one split. The table counts reference-capture
routes only; the locked holdout separately requires 120/240/600 distinct capture
runs and sequences, so its queries do not hide inside these totals.

| Option | Tier 1 route units | Tier 2 route units | Total reference route units | Named road corridors |
|---|---:|---:|---:|---:|
| Minimal proof | 162 (2 per province) | 28 | 190 | 2 |
| Recommended MVP | 324 (4 per province) | 86 | 410 | 2 |
| Expansion | 810 (10 per province) | 258 | 1,068 | 2 |

Tier 2 route units are distributed as follows:

| Target | Minimal | Recommended | Expansion |
|---|---:|---:|---:|
| Kayseri city/region | 6 | 20 | 60 |
| Ankara city/region | 8 | 24 | 72 |
| Sivas city/region | 6 | 18 | 54 |
| Kayseri–Ankara corridor | 4 | 12 | 36 |
| Kayseri–Sivas corridor | 4 | 12 | 36 |

No route, frame, satellite patch, or stopping point may be selected from the
operator images, their answers, coordinates, viewpoints, or nearby locations.

## Tier 1 — national recall coverage

Tier 1 is a candidate-recall gallery, not a meter-level map. Every option covers
all 81 provinces at a uniform bounded minimum: 100, 400, or 1,500 references per
province for Minimal proof, Recommended MVP, or Expansion respectively.

The 81 planned provinces are:

| Codes | Provinces |
|---|---|
| TR-01–TR-10 | Adana; Adıyaman; Afyonkarahisar; Ağrı; Amasya; Ankara; Antalya; Artvin; Aydın; Balıkesir |
| TR-11–TR-20 | Bilecik; Bingöl; Bitlis; Bolu; Burdur; Bursa; Çanakkale; Çankırı; Çorum; Denizli |
| TR-21–TR-30 | Diyarbakır; Edirne; Elazığ; Erzincan; Erzurum; Eskişehir; Gaziantep; Giresun; Gümüşhane; Hakkari |
| TR-31–TR-40 | Hatay; Isparta; Mersin; İstanbul; İzmir; Kars; Kastamonu; Kayseri; Kırklareli; Kırşehir |
| TR-41–TR-50 | Kocaeli; Konya; Kütahya; Malatya; Manisa; Kahramanmaraş; Mardin; Muğla; Muş; Nevşehir |
| TR-51–TR-60 | Niğde; Ordu; Rize; Sakarya; Samsun; Siirt; Sinop; Sivas; Tekirdağ; Tokat |
| TR-61–TR-70 | Trabzon; Tunceli; Şanlıurfa; Uşak; Van; Yozgat; Zonguldak; Aksaray; Bayburt; Karaman |
| TR-71–TR-81 | Kırıkkale; Batman; Şırnak; Bartın; Ardahan; Iğdır; Yalova; Karabük; Kilis; Osmaniye; Düzce |

Within each province:

- urban-core/suburban assets together target 40–60 percent and rural-settlement/
  rural-road assets together target 40–60 percent;
- at least three road classes are represented and no one road class exceeds
  60 percent;
- heading coverage uses eight 45-degree bins for street imagery, with at most
  25 percent unknown headings;
- city-center, suburban, industrial, rural-road, and terrain-dominant scenes are
  represented nationally;
- at least two separated capture months and two broad seasons are sought where
  rights-bearing supply and metadata allow it; unavailable metadata stays
  unknown and the gap is reported rather than fabricated;
- a sequence/run contributes no more than 2 percent of the option or 10 percent
  of a province quota, whichever is smaller.

Sampling proceeds round-robin by province before filling additional cells. A
high-coverage metropolitan source therefore cannot exhaust the national quota.

## Tier 2 — dense investor-demo corridor

Tier 2 adds disjoint references rather than re-counting Tier 1 assets.

| Target | Minimal | Recommended | Expansion |
|---|---:|---:|---:|
| Kayseri | 750 | 3,400 | 11,500 |
| Ankara | 900 | 4,200 | 14,000 |
| Sivas | 650 | 3,000 | 10,000 |
| Kayseri–Ankara road corridor | 800 | 3,500 | 11,500 |
| Kayseri–Sivas road corridor | 800 | 3,500 | 11,500 |
| **Tier 2 total** | **3,900** | **17,600** | **58,500** |

Each dense target includes city-center, suburban, industrial, rural-road, and
terrain-dominant cells, each at a minimum 12-percent target share. No single city
or corridor exceeds 30 percent of Tier 2. These are development/demo strata and
are reported separately from the national locked holdout.

Tier 2 also records four required manifest dimensions: `road_context`,
`terrain_class`, `vegetation_state`, and `season`. Divided-highway and
service-area samples cover both named corridors; ordinary-road, rural-road, and
junction samples cover all five targets. Mountainous coverage is required in
Kayseri, Sivas, and the Kayseri–Sivas corridor, while flat-terrain coverage is
required in Ankara and the Kayseri–Ankara corridor.

Across Tier 2, every required road context has at least 40/160/480 references and
every required terrain class at least 80/320/960 references for Minimal,
Recommended, and Expansion. Each listed road-context/target pair has at least
4/16/48 references and each listed terrain/target pair at least 8/32/96. The
observed vegetation and season dimensions require at least 2/3/4 distinct values
overall by option and two per target where available. Unknown or unavailable
metadata stays unknown: a shortfall is reported, never inferred or backfilled
from blocked sources.

## Cross-view index design

The planned index is a registry of removable, immutable source partitions:

1. A street-reference partition contains authorized street images and their
   pinned MegaLoc descriptors.
2. Each authorized aerial/satellite source receives its own partition, licence
   receipt, predeclared national grid version, raster lineage, and descriptor
   build receipt. No aerial partition exists until its policy explicitly permits
   persistent storage, derivatives, embeddings, and indexing.
3. OpenStreetMap/GeoNames-like metadata, when source policy permits, remains a
   metadata catalogue and is never relabelled as imagery rights.

The option counts above include only street-reference partitions. The aerial
partition description is a gated interface design, not a claim that any aerial
patch is approved or budgeted in this plan.

Partitions are not blended on disk. A query may search admitted partitions and
merge only opaque candidate ranks, while provenance and attribution remain
attached to each candidate. Model/code permission and corpus permission are
separate gates: the presence of a MegaLoc artifact does not authorize any image
source.

Aerial grid geometry is frozen before holdout truth access. An intended
street-query-to-aerial-reference geographic match is allowed only against this
blind national grid; no patch may be centered, selected, enlarged, or retained
because of a holdout answer. Pixel/source-tile overlap across query roles remains
prohibited.

### Revocation and deletion

Every descriptor row maps one-to-one to `asset_id`, source partition, corpus
version, and provenance receipt. A revocation performs a fail-closed cascade:

1. mark source/asset rows `QUARANTINED` or `REVOKED` and remove the partition from
   search admission;
2. build a new immutable corpus version excluding the affected assets;
3. delete their originals, normalized derivatives, descriptors, ANN shards,
   caches, staging copies, and cloud backups according to the governing contract;
4. retain only non-sensitive deletion receipts and aggregate audit evidence where
   legally permitted;
5. re-run leakage, integrity, coverage, and attribution checks before admitting
   the replacement version.

This design avoids license contamination: one source can be removed without
rebuilding or redistributing unrelated source bytes, although the global routing
manifest must be regenerated.

## Tier 3 — locked independent holdout

| Option | Queries | Minimum provinces | Maximum per province | K/A/S or dense-corridor share |
|---|---:|---:|---:|---:|
| Minimal proof | 120 | 30 | 4 | 0% |
| Recommended MVP | 240 | 60 | 4 | 0% |
| Expansion | 600 | 78 | 8 | 0% |

Kayseri (TR-38), Ankara (TR-06), and Sivas (TR-58) are excluded from Tier 3.
This deliberately prevents dense-investor-demo coverage from determining the
national decision. Tier 3 uses a query source/capture programme independent of
development/validation and of every reference source name. Every query has a
distinct non-null sequence and capture run. Contributor identities are separated
from reference construction wherever they can be resolved; shared corporate
ownership is not by itself evidence of contributor independence, and an
unresolved or undocumented overlap is quarantined. Query locations are separated
by at least 5 km, with a 30-day temporal separation when source/run/spatial
context overlaps. If an independently licensed holdout source cannot be obtained,
the benchmark is blocked rather than weakened.

Urban and rural groups each target 45–55 percent. Easy, medium, and hard strata
have minimum shares of 20, 40, and 20 percent, leaving the remaining 20 percent
for measured availability while preserving those floors. Difficulty is assigned
from a frozen, source-blind rubric (text/landmark salience, scene repetition,
road/terrain ambiguity), not from model performance.

The reference builder receives no holdout bytes, paths, hashes, source IDs,
descriptors, or truth. The prediction process receives only opaque query IDs and
temporary authorized asset handles. A separate scorer loads the truth mapping
only after every prediction in the frozen run has completed. The holdout is never
used for source selection, thresholds, index density, reranking, model choice, or
stopping decisions.

## Corpus size and storage math

All units below are decimal GB. Planning assumptions are 0.75 MB compressed image
and 0.40 MB normalized derivative per governed asset. A MegaLoc reference
descriptor contains exactly 8,448 float32 values: **33,792 bytes/reference**
before index overhead. Metadata uses 6 KB/governed asset and FAISS/Parquet index
overhead is estimated at 20 percent of descriptor bytes. Staging and
rollback/version are separate, non-overlapping halves of one additional primary-
footprint reserve. Any build that cannot fit those explicit caps must stop for a
revised approval.

Local output holds two copies of the descriptors, FAISS/Parquet index, and
metadata for an atomic build and rollback, plus 0.05/0.10/0.25 GB for reports by
option. It does not include a local image mirror.

| Option | Compressed | Normalized | Descriptors | FAISS/Parquet | Metadata | Primary | Staging | Rollback/version | Total cloud | Local output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Minimal proof | 9.090 GB | 4.848 GB | 0.405504 GB | 0.081101 GB | 0.072720 GB | 14.497325 GB | 7.248662 GB | 7.248662 GB | **28.994650 GB** | 1.168650 GB |
| Recommended MVP | 37.680 GB | 20.096 GB | 1.689600 GB | 0.337920 GB | 0.301440 GB | 60.104960 GB | 30.052480 GB | 30.052480 GB | **120.209920 GB** | 4.757920 GB |
| Expansion | 135.450 GB | 72.240 GB | 6.082560 GB | 1.216512 GB | 1.083600 GB | 216.072672 GB | 108.036336 GB | 108.036336 GB | **432.145344 GB** | 17.015344 GB |

Formulae, where `R` is references and `H` is holdout queries:

```text
compressed_GB = (R + H) × 750,000 / 1,000,000,000
normalized_GB = (R + H) × 400,000 / 1,000,000,000
descriptor_GB = R × 33,792 / 1,000,000,000
faiss_parquet_index_GB = descriptor_GB × 0.20
metadata_GB = (R + H) × 6,000 / 1,000,000,000
primary_GB = compressed_GB + normalized_GB + descriptor_GB + faiss_parquet_index_GB + metadata_GB
staging_GB = primary_GB × 0.50
rollback_version_GB = primary_GB × 0.50
total_cloud_GB = primary_GB + staging_GB + rollback_version_GB
local_output_GB = 2 × (descriptor_GB + faiss_parquet_index_GB + metadata_GB) + report_allowance_GB
```

The Expansion local output is larger than the presently reported D-drive free
space once a safety reserve is retained. Expansion must not write there; no
Phase 3A output consumes that space.

## Compute estimate

The planning throughput is 12,000 reference descriptors/GPU-hour with a 25
percent retry/verification allowance. Evaluation fixes five configurations and
budgets 240 query-configuration runs/GPU-hour. These are estimates, not benchmark
measurements, and must be replaced by actual receipts before Expansion.

| Option | Descriptor GPU h | Evaluation GPU h | Total GPU h | Main limitation |
|---|---:|---:|---:|---|
| Minimal proof | 1.25 | 2.50 | 3.75 | Directional national result; sparse rural/time coverage |
| Recommended MVP | 5.25 | 5.00 | 10.25 | Aggregate national comparison, not province-level accuracy |
| Expansion | 18.75 | 12.50 | 31.25 | Not authorized; needs new storage/budget and measured throughput |

## Leakage and sequence isolation

Admission applies, in order: source-policy validation; operator-image exclusion;
source/capture grouping; exact SHA-256; like-for-like perceptual hashes; rotations
and mirrors; crop/geometric checks; full-sequence and capture-run grouping;
spatial/temporal separation; satellite tile/pixel overlap; real pinned MegaLoc
descriptor similarity; and truth-isolation verification.

An incomplete configured check is not a pass. The initial failed audit receipt is
retained; filtering requires a new clean audit and receipt. The Phase 3A offline
configuration validator never reads image bytes. Future acquisition/leakage tools
may read authorized bytes only after approval and must not log them.

## Benchmark and continuation decision

The frozen six-reference index is evaluated first on the same locked holdout.
Then each admitted pilot index is evaluated without changing queries or scoring.
Report Recall@1/5/10 within 25/50/100/200 km, province Recall@1/5/10, nearest-
candidate distance distributions, abstention/failure rate, urban/rural and
difficulty strata, source-partition ablations, and bootstrap paired confidence
intervals. No confidence calibration or accuracy claim is made from fewer than
100 completed independent queries.

Phase 3B may continue only if all governance gates pass and the national Tier 3
run meets every threshold:

- within-200-km Recall@10 at least 0.70;
- province Recall@10 at least 0.55;
- median nearest-candidate distance at most 150 km;
- absolute urban/rural Recall@10 gap at most 0.20;
- the paired bootstrap 95-percent confidence interval for improvement in
  within-200-km Recall@10 over the frozen six-reference baseline has lower bound
  above zero.

Tier 2 performance is reported separately and cannot rescue a failed Tier 3
gate. Thus a system cannot pass merely by memorizing Kayseri, Ankara, Sivas, or
the two dense corridors.

## First acquisition after a future approval

Do not acquire anything until the main go/no-go record, source-policy evidence,
account/budget approval, and any required contracts or written permissions are
complete. Then:

1. freeze empty immutable corpus/source partitions and provenance/deletion
   receipts;
2. admit first-party AtlasLens capture or contractually transferred partner
   imagery only under the selected strategy and signed rights record;
3. fill the Minimal proof Tier 1 round-robin (100 references and two route units
   per province) before adding excess metropolitan density;
4. fill the five Minimal proof Tier 2 targets without consulting operator images;
5. acquire the independent 120-query holdout through its separate owner/process;
6. consider an aerial/satellite partition only after its distinct source-policy
   decision explicitly permits persistent storage, derivatives, embeddings, and
   indexing.

Blocked, unknown, research-only, permission-pending, or legal-review-pending
sources are never used as quota backfill. Research-only benchmark data, if later
approved for research, stays outside the commercial corpus and its metrics are
labelled separately.
