"""Content-addressed local packaging for the bounded Phase 3F run."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from atlaslens_api.corpus_pipeline.errors import AssetSafetyError
from atlaslens_api.corpus_pipeline.safety import (
    atomic_write_bytes,
    read_bounded_bytes,
    resolve_contained_file,
    resolve_output_file,
    resolve_safe_root,
)

MEGALOC_SOURCE_REVISION: Final = "1af071c68fc3ab6c6018c5c868391763516e50f7"
MEGALOC_MODEL_REVISION: Final = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
MEGALOC_WEIGHT_SHA256: Final = (
    "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_FORBIDDEN_PARTS = {
    ".agents",
    ".codex",
    ".git",
    ".local",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
_ALLOWED_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
_ALLOWED_NAMES = {"Dockerfile", "Makefile"}


class Phase3FPackagingError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Phase3FPackagingError(code)


def _require_sha256(value: str, code: str) -> None:
    _require(bool(_SHA256.fullmatch(value)), code)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path, *, max_bytes: int) -> tuple[str, int]:
    try:
        size = path.stat().st_size
        _require(0 <= size <= max_bytes, "source_file_too_large")
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(chunk)
                _require(count <= max_bytes, "source_file_too_large")
                digest.update(chunk)
        _require(count == size, "source_file_changed_during_hash")
        return digest.hexdigest(), count
    except Phase3FPackagingError:
        raise
    except OSError as exc:
        raise Phase3FPackagingError("source_file_unreadable") from exc


def _relative_path(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    _require(not _WINDOWS_DRIVE.match(normalized), "source_path_invalid")
    path = PurePosixPath(normalized)
    _require(
        bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
        "source_path_invalid",
    )
    _require(str(path) not in {"", "."}, "source_path_invalid")
    return path


def _is_packageable(path: PurePosixPath) -> bool:
    lowered_parts = tuple(part.lower() for part in path.parts)
    if any(part in _FORBIDDEN_PARTS for part in lowered_parts):
        return False
    name = path.name
    lowered_name = name.lower()
    if lowered_name == ".env" or lowered_name.startswith(".env."):
        return False
    if lowered_name.startswith("asda"):
        return False
    return name in _ALLOWED_NAMES or path.suffix.lower() in _ALLOWED_SUFFIXES


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    relative_path: str
    sha256: str
    size_bytes: int


def verify_artifact(
    root: Path,
    locator: str | Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    max_bytes: int,
) -> VerifiedArtifact:
    """Verify a local artifact without copying it or exposing its absolute path."""

    _require_sha256(expected_sha256, "expected_sha256_invalid")
    _require(0 < expected_size_bytes <= max_bytes, "expected_size_invalid")
    try:
        resolved_root = resolve_safe_root(root)
        path = resolve_contained_file(resolved_root, locator)
        digest, size = _sha256_path(path, max_bytes=max_bytes)
        relative = path.relative_to(resolved_root).as_posix()
    except AssetSafetyError as exc:
        raise Phase3FPackagingError("artifact_path_invalid") from exc
    _require(size == expected_size_bytes, "artifact_size_mismatch")
    _require(digest == expected_sha256, "artifact_sha256_mismatch")
    return VerifiedArtifact(relative, digest, size)


@dataclass(frozen=True, slots=True)
class LineEndingTrust:
    canonical_lf_sha256: str
    windows_crlf_sha256: str
    max_bytes: int

    def __post_init__(self) -> None:
        _require_sha256(self.canonical_lf_sha256, "canonical_lf_sha256_invalid")
        _require_sha256(self.windows_crlf_sha256, "windows_crlf_sha256_invalid")
        _require(self.max_bytes > 0, "line_ending_max_bytes_invalid")


@dataclass(frozen=True, slots=True)
class MaterializedLFArtifact:
    source_representation: Literal["lf", "crlf"]
    source_sha256: str
    canonical_lf_sha256: str
    source_size_bytes: int
    canonical_size_bytes: int


def materialize_trusted_lf(
    source_root: Path,
    source_locator: str | Path,
    output_root: Path,
    output_locator: str | Path,
    trust: LineEndingTrust,
) -> MaterializedLFArtifact:
    """Verify LF/CRLF representations and write only the canonical LF payload."""

    try:
        source = resolve_contained_file(source_root, source_locator)
        payload = read_bounded_bytes(
            source,
            max_bytes=trust.max_bytes,
            error_prefix="source",
        )
    except AssetSafetyError as exc:
        raise Phase3FPackagingError("line_ending_source_invalid") from exc
    _require(b"\x00" not in payload, "line_ending_source_not_text")
    source_sha256 = _sha256_bytes(payload)
    if source_sha256 == trust.canonical_lf_sha256:
        representation: Literal["lf", "crlf"] = "lf"
        canonical = payload
        _require(b"\r" not in canonical, "canonical_lf_contains_cr")
    elif source_sha256 == trust.windows_crlf_sha256:
        representation = "crlf"
        _require(payload.startswith(b"\xef\xbb\xbf") is False, "windows_source_contains_bom")
        _require(b"\r\r\n" not in payload, "windows_source_mixed_newlines")
        _require(payload.replace(b"\r\n", b"").find(b"\r") < 0, "windows_source_contains_bare_cr")
        _require(payload.replace(b"\r\n", b"").find(b"\n") < 0, "windows_source_mixed_newlines")
        canonical = payload.replace(b"\r\n", b"\n")
    else:
        raise Phase3FPackagingError("source_representation_sha256_mismatch")
    _require(
        _sha256_bytes(canonical) == trust.canonical_lf_sha256,
        "canonical_lf_sha256_mismatch",
    )
    try:
        destination = resolve_output_file(output_root, output_locator)
        atomic_write_bytes(destination, canonical)
    except AssetSafetyError as exc:
        raise Phase3FPackagingError("line_ending_destination_invalid") from exc
    return MaterializedLFArtifact(
        source_representation=representation,
        source_sha256=source_sha256,
        canonical_lf_sha256=trust.canonical_lf_sha256,
        source_size_bytes=len(payload),
        canonical_size_bytes=len(canonical),
    )


@dataclass(frozen=True, slots=True)
class SourceEntry:
    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = _relative_path(self.relative_path)
        _require(str(path) == self.relative_path, "source_entry_path_not_normalized")
        _require(_is_packageable(path), "source_entry_forbidden")
        _require_sha256(self.sha256, "source_entry_sha256_invalid")
        _require(self.size_bytes >= 0, "source_entry_size_invalid")


@dataclass(frozen=True, slots=True)
class SourceManifest:
    commit_sha: str
    entries: tuple[SourceEntry, ...]
    excluded_count: int

    def __post_init__(self) -> None:
        _require(bool(_COMMIT.fullmatch(self.commit_sha)), "commit_sha_invalid")
        paths = [entry.relative_path for entry in self.entries]
        _require(paths == sorted(paths), "manifest_entries_not_sorted")
        _require(len(paths) == len(set(paths)), "manifest_entry_duplicate")
        _require(self.excluded_count >= 0, "excluded_count_invalid")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": "atlaslens-phase3f-source-manifest-v1",
            "commit_sha": self.commit_sha,
            "excluded_count": self.excluded_count,
            "entry_count": len(self.entries),
            "entries": [
                {
                    "relative_path": entry.relative_path,
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                }
                for entry in self.entries
            ],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(self.payload(), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode()

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes)


def build_source_manifest(
    workspace_root: Path,
    *,
    commit_sha: str,
    tracked_paths: Iterable[str],
    include_prefixes: Sequence[str] = (),
    max_file_bytes: int = 64 * 1024 * 1024,
    max_total_bytes: int = 512 * 1024 * 1024,
) -> SourceManifest:
    """Hash only caller-supplied tracked paths; untracked workspace files are ignored."""

    _require(bool(_COMMIT.fullmatch(commit_sha)), "commit_sha_invalid")
    _require(max_file_bytes > 0 and max_total_bytes > 0, "source_size_limit_invalid")
    try:
        root = resolve_safe_root(workspace_root)
    except AssetSafetyError as exc:
        raise Phase3FPackagingError("workspace_root_invalid") from exc
    prefixes = tuple(_relative_path(value) for value in include_prefixes)
    normalized: set[PurePosixPath] = set()
    excluded_count = 0
    for raw in tracked_paths:
        path = _relative_path(raw)
        _require(path not in normalized, "tracked_path_duplicate")
        normalized.add(path)
        if not _is_packageable(path) or (
            prefixes
            and not any(
                path == prefix or path.is_relative_to(prefix) for prefix in prefixes
            )
        ):
            excluded_count += 1
    entries: list[SourceEntry] = []
    total = 0
    for path in sorted(normalized, key=str):
        if not _is_packageable(path) or (
            prefixes
            and not any(
                path == prefix or path.is_relative_to(prefix) for prefix in prefixes
            )
        ):
            continue
        try:
            source = resolve_contained_file(root, Path(*path.parts))
        except AssetSafetyError as exc:
            raise Phase3FPackagingError("tracked_source_invalid") from exc
        digest, size = _sha256_path(source, max_bytes=max_file_bytes)
        total += size
        _require(total <= max_total_bytes, "source_total_too_large")
        entries.append(SourceEntry(str(path), digest, size))
    _require(bool(entries), "source_manifest_empty")
    return SourceManifest(commit_sha, tuple(entries), excluded_count)


@dataclass(frozen=True, slots=True)
class SourceArchive:
    relative_path: str
    sha256: str
    size_bytes: int
    manifest_sha256: str
    entry_count: int


def write_source_archive(
    workspace_root: Path,
    output_root: Path,
    output_locator: str | Path,
    manifest: SourceManifest,
) -> SourceArchive:
    """Write exactly the manifest members, rejecting any content drift."""

    try:
        root = resolve_safe_root(workspace_root)
    except AssetSafetyError as exc:
        raise Phase3FPackagingError("workspace_root_invalid") from exc
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for entry in manifest.entries:
            try:
                source = resolve_contained_file(
                    root,
                    Path(*PurePosixPath(entry.relative_path).parts),
                )
                payload = read_bounded_bytes(
                    source,
                    max_bytes=max(1, entry.size_bytes),
                    error_prefix="source",
                )
            except AssetSafetyError as exc:
                raise Phase3FPackagingError("source_archive_member_invalid") from exc
            _require(len(payload) == entry.size_bytes, "source_archive_member_size_mismatch")
            _require(
                _sha256_bytes(payload) == entry.sha256,
                "source_archive_member_sha256_mismatch",
            )
            info = tarfile.TarInfo(entry.relative_path)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
        manifest_info = tarfile.TarInfo("PHASE3F-SOURCE-MANIFEST.json")
        manifest_info.size = len(manifest.canonical_bytes)
        manifest_info.mode = 0o644
        manifest_info.mtime = 0
        manifest_info.uid = 0
        manifest_info.gid = 0
        archive.addfile(manifest_info, io.BytesIO(manifest.canonical_bytes))
    payload = stream.getvalue()
    try:
        destination = resolve_output_file(output_root, output_locator)
        digest = atomic_write_bytes(destination, payload)
        relative = destination.relative_to(resolve_safe_root(output_root)).as_posix()
    except AssetSafetyError as exc:
        raise Phase3FPackagingError("source_archive_destination_invalid") from exc
    return SourceArchive(relative, digest, len(payload), manifest.sha256, len(manifest.entries))


__all__ = [
    "LineEndingTrust",
    "MEGALOC_MODEL_REVISION",
    "MEGALOC_SOURCE_REVISION",
    "MEGALOC_WEIGHT_SHA256",
    "MaterializedLFArtifact",
    "Phase3FPackagingError",
    "SourceArchive",
    "SourceEntry",
    "SourceManifest",
    "VerifiedArtifact",
    "build_source_manifest",
    "materialize_trusted_lf",
    "verify_artifact",
    "write_source_archive",
]
