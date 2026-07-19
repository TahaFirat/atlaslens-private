# AtlasLens engineering rules

- Read `PROJECT_STATE.md` before beginning work and update it after each completed phase.
- Never fabricate geolocation evidence or candidates. Test mocks stay in test code only.
- Every candidate requires provenance, a confidence value with stated semantics, and a positive uncertainty radius.
- Abstention is a valid and preferred result when evidence is insufficient.
- Uploaded images are private and temporary by default.
- Never log raw images, API keys, exact GPS values, raw OCR content, prompts, original filenames, or other sensitive payloads.
- Cloud processing requires an explicit mode choice and explicit consent. Local-only processing never invokes a cloud analysis provider.
- Tests and documentation must accompany behavioral changes.
- Preserve API compatibility unless a migration is documented in `docs/decision-log.md`.
- Large datasets and models require explicit phase approval and license review.
- Do not silently change architectural decisions; record material changes in the decision log.
- Future work must be attached to one of the numbered phases in `PROJECT_STATE.md`.
- Do not implement future-phase geolocation models, retrieval, reranking, geometric verification, or production scaling early.

