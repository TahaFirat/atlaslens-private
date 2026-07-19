# Map providers

## Browser basemap

`VITE_MAP_PROVIDER=maplibre` is the supported Phase 5B renderer. The default
development style is `https://tiles.openfreemap.org/styles/positron`; it provides
labels, roads, borders and water and must show OpenStreetMap/OpenFreeMap
attribution. This public development dependency is not a production capacity or
availability promise. Production must set a reviewed style URL or self-hosted
equivalent with its own license, privacy, attribution, caching and SLA decision.

AtlasLens symbol overlays use OpenFreeMap's supported `Noto Sans Regular` glyph
stack. A first style or resource failure replaces the remote style with a bounded
tile-free style instead of retrying failed glyph/tile requests; candidate markers,
uncertainty geometry, selection and the textual map remain available. The offline
style omits glyph-dependent rank/count labels and relies on the textual candidate
list for those values. The Retry action is the only path that starts a fresh
remote attempt.

The result map fits candidate centers, not uncertainty polygons. Ranked markers
are the default; uncertainty is selected/toggled and low-opacity; the evidence
heatmap appears only when source diversity makes it meaningful. Antimeridian-safe
bounds, clustering, selected-marker focus, 390 px layout and a textual map
alternative are tested. Style/load failures produce a visible error without
hiding the analysis result.

`VITE_MAP_PROVIDER=google` is only a disabled integration placeholder. Any future
activation must use the official Maps JavaScript API and an operator key restricted
by HTTP referrer, API and quota. Google tiles/images may never be downloaded,
indexed, trained on or used as AtlasLens retrieval references.

## OSM evidence research

Basemap rendering and geolocation evidence are separate. `MAP_EVIDENCE_ENABLED`
controls an optional Overpass feature check for fixed public feature enums such as
roads, rail, buildings, water and airports. It never sends raw OCR text. Enabling
requires a reviewed HTTPS endpoint and a contact-bearing `OVERPASS_USER_AGENT`.

The transport pins provider-validated public IPs, verifies TLS hostname/certificate,
ignores environment proxies, refuses redirects/private addresses/unsafe headers,
and bounds request bytes, response bytes and total time. The provider adds cache,
minimum request interval, bounded retry and a circuit breaker. OSM absence is
neutral because coverage and freshness vary. Network evidence is optional; local
OCR, GeoCLIP and licensed retrieval must remain usable without it.
