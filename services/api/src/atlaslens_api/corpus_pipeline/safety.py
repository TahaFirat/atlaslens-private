from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from pydantic import BaseModel

from .errors import AssetSafetyError


def _is_link_or_junction(path: Path) -> bool:
    try:
        is_junction = bool(getattr(path, "is_junction", lambda: False)())
        return path.is_symlink() or is_junction
    except OSError as exc:
        raise AssetSafetyError("path_metadata_unavailable") from exc


def resolve_safe_root(root: Path) -> Path:
    try:
        if _is_link_or_junction(root):
            raise AssetSafetyError("approved_root_link_rejected")
        resolved = root.resolve(strict=True)
    except AssetSafetyError:
        raise
    except OSError as exc:
        raise AssetSafetyError("approved_root_unavailable") from exc
    if not resolved.is_dir():
        raise AssetSafetyError("approved_root_unavailable")
    return resolved


def resolve_contained_file(root: Path, locator: str | Path) -> Path:
    """Resolve a regular file while rejecting traversal and every in-root link component."""

    resolved_root = resolve_safe_root(root)
    supplied = Path(locator)
    try:
        if supplied.is_absolute():
            lexical = supplied
            relative = supplied.relative_to(resolved_root)
        else:
            relative = supplied
            lexical = resolved_root / supplied
    except ValueError as exc:
        raise AssetSafetyError("path_outside_approved_root") from exc
    if not relative.parts or ".." in relative.parts:
        raise AssetSafetyError("path_traversal_rejected")
    current = resolved_root
    try:
        for part in relative.parts:
            current = current / part
            if _is_link_or_junction(current):
                raise AssetSafetyError("path_link_rejected")
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except AssetSafetyError:
        raise
    except (OSError, ValueError) as exc:
        raise AssetSafetyError("path_outside_approved_root") from exc
    if not resolved.is_file() or _is_link_or_junction(resolved):
        raise AssetSafetyError("path_not_regular_file")
    return resolved


def ensure_safe_output_root(root: Path) -> Path:
    """Create a work directory and reject link/junction output destinations."""

    try:
        root.mkdir(parents=True, exist_ok=True)
        if _is_link_or_junction(root):
            raise AssetSafetyError("output_root_link_rejected")
        resolved = root.resolve(strict=True)
    except AssetSafetyError:
        raise
    except OSError as exc:
        raise AssetSafetyError("output_root_unavailable") from exc
    if not resolved.is_dir():
        raise AssetSafetyError("output_root_unavailable")
    return resolved


def resolve_output_file(root: Path, locator: str | Path) -> Path:
    resolved_root = ensure_safe_output_root(root)
    supplied = Path(locator)
    try:
        relative = supplied.relative_to(resolved_root) if supplied.is_absolute() else supplied
    except ValueError as exc:
        raise AssetSafetyError("output_path_outside_root") from exc
    if not relative.parts or ".." in relative.parts:
        raise AssetSafetyError("output_path_traversal_rejected")
    current = resolved_root
    for part in relative.parts[:-1]:
        current = current / part
        try:
            current.mkdir(exist_ok=True)
        except OSError as exc:
            raise AssetSafetyError("output_directory_unavailable") from exc
        if _is_link_or_junction(current):
            raise AssetSafetyError("output_path_link_rejected")
    destination = resolved_root.joinpath(*relative.parts)
    if destination.exists() and _is_link_or_junction(destination):
        raise AssetSafetyError("output_path_link_rejected")
    return destination


def read_bounded_bytes(path: Path, *, max_bytes: int, error_prefix: str) -> bytes:
    try:
        size = path.stat().st_size
        if size < 1 or size > max_bytes:
            raise AssetSafetyError(f"{error_prefix}_size_invalid")
        with path.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
    except AssetSafetyError:
        raise
    except OSError as exc:
        raise AssetSafetyError(f"{error_prefix}_read_failed") from exc
    if len(payload) > max_bytes:
        raise AssetSafetyError(f"{error_prefix}_size_invalid")
    return payload


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_model_bytes(model: BaseModel) -> bytes:
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (payload + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes, *, require_idempotent: bool = True) -> str:
    """Write in the destination directory, fsync, then publish with ``os.replace``."""

    if path.exists():
        if _is_link_or_junction(path) or not path.is_file():
            raise AssetSafetyError("atomic_destination_rejected")
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise AssetSafetyError("atomic_destination_unreadable") from exc
        if existing == payload:
            return sha256_bytes(payload)
        if require_idempotent:
            raise AssetSafetyError("atomic_destination_content_mismatch")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary = Path(temporary_name)
        if _is_link_or_junction(temporary):
            raise AssetSafetyError("atomic_temporary_rejected")
        os.replace(temporary, path)
        temporary_name = None
    except AssetSafetyError:
        raise
    except OSError as exc:
        raise AssetSafetyError("atomic_write_failed") from exc
    finally:
        if temporary_name is not None:
            with suppress(OSError):
                Path(temporary_name).unlink(missing_ok=True)
    return sha256_bytes(payload)


def atomic_write_model(path: Path, model: BaseModel, *, require_idempotent: bool = True) -> str:
    return atomic_write_bytes(
        path,
        canonical_model_bytes(model),
        require_idempotent=require_idempotent,
    )


__all__ = [
    "atomic_write_bytes",
    "atomic_write_model",
    "canonical_model_bytes",
    "ensure_safe_output_root",
    "read_bounded_bytes",
    "resolve_contained_file",
    "resolve_output_file",
    "resolve_safe_root",
    "sha256_bytes",
]
