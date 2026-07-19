from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .errors import RightsValidationError
from .models import ManifestAsset, RightsAdmission, SourcePolicyDocument
from .safety import read_bounded_bytes, sha256_bytes

_READY = frozenset({"GO", "GO_WITH_ATTRIBUTION", "FIRST_PARTY_ONLY"})
_PRIVACY_READY = frozenset(
    {"NOT_DETECTED", "BLURRED_AT_SOURCE", "BLURRED_BY_ATLASLENS"}
)


class SourcePolicy:
    def __init__(self, document: SourcePolicyDocument, *, sha256: str) -> None:
        self.document = document
        self.sha256 = sha256
        self._by_name = {entry.source_name: entry for entry in document.sources}

    @classmethod
    def load(cls, path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> SourcePolicy:
        try:
            if path.is_symlink() or not path.is_file():
                raise RightsValidationError("source_policy_path_rejected")
            payload = read_bounded_bytes(
                path,
                max_bytes=max_bytes,
                error_prefix="source_policy",
            )
            raw = json.loads(payload)
            document = SourcePolicyDocument.model_validate(raw)
        except RightsValidationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise RightsValidationError("source_policy_invalid") from exc
        return cls(document, sha256=sha256_bytes(payload))

    def enforce(self, asset: ManifestAsset) -> RightsAdmission:
        entry = self._by_name.get(asset.source_name)
        if entry is None:
            raise RightsValidationError("source_not_in_policy")
        ready_decisions = set(self.document.acquisition_ready_decisions) & _READY
        if entry.current_decision not in ready_decisions:
            if entry.current_decision == "WRITTEN_PERMISSION_REQUIRED":
                raise RightsValidationError("written_permission_not_established")
            raise RightsValidationError("source_policy_not_acquisition_ready")
        if asset.deletion_revocation_state.status != "ACTIVE":
            raise RightsValidationError("asset_revoked_or_unavailable")
        if asset.personal_data_blur_state not in _PRIVACY_READY:
            raise RightsValidationError("privacy_review_incomplete")
        if not asset.acquisition_ready:
            raise RightsValidationError("asset_not_acquisition_ready")
        decisions = (
            asset.source_policy_decision,
            asset.commercial_use_decision,
            asset.derivative_index_decision,
        )
        if any(decision not in ready_decisions for decision in decisions):
            raise RightsValidationError("asset_rights_decision_not_ready")
        if any(decision != entry.current_decision for decision in decisions):
            raise RightsValidationError("asset_rights_policy_mismatch")
        receipt = asset.provenance_receipt
        if receipt.source_policy_version != self.document.policy_version:
            raise RightsValidationError("provenance_policy_version_mismatch")
        if receipt.evidence_date != entry.evidence_date:
            raise RightsValidationError("provenance_evidence_date_mismatch")
        if (
            entry.current_decision == "FIRST_PARTY_ONLY"
            and receipt.acquisition_method != "first_party_capture"
        ):
            raise RightsValidationError("first_party_provenance_required")
        return RightsAdmission(
            asset_id=asset.asset_id,
            source_id=entry.source_id,
            decision=entry.current_decision,
            policy_version=self.document.policy_version,
            policy_evidence_date=entry.evidence_date,
        )


__all__ = ["SourcePolicy"]
