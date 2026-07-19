# Phase 5B benchmark comparison

Status: **not completed; improvement gate failed by absence of candidate results**.

The evaluation manifest remains frozen at fingerprint
`9767b0ec500c0b7972aa0f26fdfb0d48560b908285c7c34cf23b18ae32b5fba9`:
30 reviewed Commons derivatives, 24 countries, six inhabited continents, unique
exact/perceptual hashes and capture families, with reference-index overlap
forbidden.

## GeoCLIP-only baseline

| Metric | Result |
|---|---:|
| Country Top-1 | 19/28 (67.8571%); 2 explicit label exclusions |
| Country Top-5 | 21/28 (75.0000%) |
| Recall@1 / 25 / 200 / 750 / 2500 km | 20.00 / 33.33 / 56.67 / 73.33 / 86.67% |
| Median / mean Top-1 error | 143.140 / 1441.540 km |
| Top-5 oracle median / mean error | 9.684 / 1024.441 km |
| Abstention | 0/30 |
| Latency | mean 393.010 ms; median 75.966 ms; p95 91.663 ms |

The first cold call is included in the latency mean. These are preliminary
fixed-slice measurements, not global superiority or an SLA.

## Ablations

| Mode | Status | Metrics |
|---|---|---|
| GeoCLIP only | available | Baseline above |
| GeoCLIP + OCR | unavailable | RapidOCR and forward GeoNames not installed |
| GeoCLIP + retrieval | unavailable | SigLIP2 and licensed index not installed |
| GeoCLIP + OCR + retrieval | unavailable | Mandatory providers unavailable |
| Full combined pipeline | unavailable | No real index/map evidence run |

`atlas benchmark compare` validates the same manifest fingerprint/count and can
combine five existing benchmark reports. Missing modes remain explicitly
`unavailable`; it never invents deltas.

## Gate calculation

No candidate country Top-1, Recall@200 km or median error exists, so degradation
and improvement cannot be calculated. Text-rich and landmark improvements are
also unmeasured. Therefore the Phase 5B measurable-improvement gate is **FAIL**
and `PROJECT_STATE.md` remains blocked. The fixed set must not be changed before
the real providers/index are installed and all five modes are rerun.

