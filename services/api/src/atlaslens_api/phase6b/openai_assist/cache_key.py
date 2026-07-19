from __future__ import annotations

import hashlib
import json
import re

from atlaslens_api.phase6b.openai_assist.models import OpenAIGeoReviewEvidence

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def evidence_fingerprint(evidence: OpenAIGeoReviewEvidence) -> str:
    canonical = json.dumps(
        evidence.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_openai_geo_cache_key(
    *,
    image_sha256: str,
    model: str,
    prompt_version: str,
    candidate_evidence_fingerprint: str,
    image_detail: str,
) -> str:
    if not _SHA256.fullmatch(image_sha256) or not _SHA256.fullmatch(candidate_evidence_fingerprint):
        raise ValueError("cache fingerprints must be SHA-256 hex digests")
    payload = {
        "candidate_evidence_fingerprint": candidate_evidence_fingerprint,
        "image_detail": image_detail,
        "image_sha256": image_sha256,
        "model": model,
        "prompt_version": prompt_version,
        "version": "phase6b-openai-cache-v1",
    }
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"v1:{digest}"
