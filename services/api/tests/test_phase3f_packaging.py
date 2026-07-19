from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from atlaslens_api.phase3f.packaging import (
    LineEndingTrust,
    Phase3FPackagingError,
    build_source_manifest,
    materialize_trusted_lf,
    verify_artifact,
    write_source_archive,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_crlf_source_is_verified_and_written_as_lf_without_source_mutation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    crlf = b"line one\r\nline two\r\n"
    lf = b"line one\nline two\n"
    source = source_root / "revision.txt"
    source.write_bytes(crlf)
    trust = LineEndingTrust(
        canonical_lf_sha256=_sha256(lf),
        windows_crlf_sha256=_sha256(crlf),
        max_bytes=1024,
    )

    result = materialize_trusted_lf(
        source_root,
        "revision.txt",
        output_root,
        "megaloc/revision.txt",
        trust,
    )

    assert result.source_representation == "crlf"
    assert result.canonical_lf_sha256 == _sha256(lf)
    assert source.read_bytes() == crlf
    assert (output_root / "megaloc" / "revision.txt").read_bytes() == lf


def test_line_ending_trust_rejects_unexpected_content_before_writing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "revision.txt").write_bytes(b"unexpected\n")
    trust = LineEndingTrust(
        canonical_lf_sha256=_sha256(b"expected\n"),
        windows_crlf_sha256=_sha256(b"expected\r\n"),
        max_bytes=1024,
    )

    with pytest.raises(
        Phase3FPackagingError, match="source_representation_sha256_mismatch"
    ):
        materialize_trusted_lf(
            source_root,
            "revision.txt",
            tmp_path / "output",
            "revision.txt",
            trust,
        )

    assert not (tmp_path / "output" / "revision.txt").exists()


def test_manifest_uses_only_explicit_tracked_safe_files_and_tar_is_deterministic(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    good = workspace / "services" / "api" / "src" / "safe.py"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"print('safe')\n")
    (workspace / "services" / "api" / ".env").write_text("SECRET=unsafe", encoding="utf-8")
    (workspace / "asda.txt").write_text("excluded", encoding="utf-8")
    (workspace / "services" / "api" / "image.jpg").write_bytes(b"not-an-image")
    (workspace / "services" / "api" / "untracked.py").write_text(
        "raise RuntimeError('not packaged')", encoding="utf-8"
    )
    local_source = workspace / ".local" / "vendor" / "hidden.py"
    local_source.parent.mkdir(parents=True)
    local_source.write_text("raise RuntimeError('hidden')", encoding="utf-8")
    tracked = (
        "services/api/src/safe.py",
        "services/api/.env",
        "asda.txt",
        "services/api/image.jpg",
        ".local/vendor/hidden.py",
    )

    manifest = build_source_manifest(
        workspace,
        commit_sha="a" * 40,
        tracked_paths=tracked,
        include_prefixes=("services/api",),
    )

    assert [entry.relative_path for entry in manifest.entries] == [
        "services/api/src/safe.py"
    ]
    assert manifest.excluded_count == 4
    first = write_source_archive(workspace, tmp_path / "out", "first.tar", manifest)
    second = write_source_archive(workspace, tmp_path / "out", "second.tar", manifest)
    assert first.sha256 == second.sha256
    assert first.manifest_sha256 == manifest.sha256

    with tarfile.open(fileobj=io.BytesIO((tmp_path / "out" / "first.tar").read_bytes())) as tar:
        assert tar.getnames() == [
            "services/api/src/safe.py",
            "PHASE3F-SOURCE-MANIFEST.json",
        ]
        assert tar.extractfile("services/api/src/safe.py").read() == b"print('safe')\n"
        names = "\n".join(tar.getnames()).lower()
        assert ".env" not in names
        assert "asda" not in names
        assert "untracked" not in names
        assert ".jpg" not in names


def test_packaging_rechecks_manifest_hashes_and_rejects_tampering(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "safe.py"
    source.write_bytes(b"before\n")
    manifest = build_source_manifest(
        workspace,
        commit_sha="b" * 40,
        tracked_paths=("safe.py",),
    )
    source.write_bytes(b"after\n")

    with pytest.raises(Phase3FPackagingError, match="source_archive_member_size_mismatch"):
        write_source_archive(workspace, tmp_path / "out", "source.tar", manifest)

    assert not (tmp_path / "out" / "source.tar").exists()


def test_manifest_rejects_traversal_even_when_reported_as_tracked(tmp_path: Path) -> None:
    with pytest.raises(Phase3FPackagingError, match="source_path_invalid"):
        build_source_manifest(
            tmp_path,
            commit_sha="c" * 40,
            tracked_paths=("../outside.py",),
        )


def test_artifact_trust_checks_size_and_hash_without_modifying_file(tmp_path: Path) -> None:
    payload = b"trusted-megaloc-artifact"
    path = tmp_path / "model.bin"
    path.write_bytes(payload)

    result = verify_artifact(
        tmp_path,
        "model.bin",
        expected_sha256=_sha256(payload),
        expected_size_bytes=len(payload),
        max_bytes=1024,
    )
    assert result.sha256 == _sha256(payload)
    assert path.read_bytes() == payload

    with pytest.raises(Phase3FPackagingError, match="artifact_sha256_mismatch"):
        verify_artifact(
            tmp_path,
            "model.bin",
            expected_sha256="0" * 64,
            expected_size_bytes=len(payload),
            max_bytes=1024,
        )
