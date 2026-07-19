from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from atlaslens_api.errors import AppError

_SAFE_KEY = re.compile(r"^[a-f0-9]{32}(?:\.[a-z0-9]{1,8})?$")


class AsyncReadable(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True, slots=True)
class LocalImageHandle:
    key: str
    path: Path

    def __repr__(self) -> str:
        return "LocalImageHandle(<redacted>)"


@dataclass(frozen=True, slots=True)
class StoredUpload:
    handle: LocalImageHandle
    size: int
    sha256: str


class StorageBackend(Protocol):
    async def save_upload(self, stream: AsyncReadable, max_bytes: int) -> StoredUpload: ...

    async def create_temporary(self, suffix: str = "") -> LocalImageHandle: ...

    async def delete(self, key: str | None) -> None: ...

    async def cleanup_orphans(self, older_than: datetime) -> int: ...


class LocalTemporaryStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()

    async def initialize(self) -> None:
        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _new_handle(self, suffix: str = "") -> LocalImageHandle:
        normalized_suffix = suffix.lower()
        if normalized_suffix and not re.fullmatch(r"\.[a-z0-9]{1,8}", normalized_suffix):
            raise ValueError("unsafe storage suffix")
        key = f"{uuid4().hex}{normalized_suffix}"
        return LocalImageHandle(key=key, path=self._root / key)

    def resolve(self, key: str) -> Path:
        if not _SAFE_KEY.fullmatch(key):
            raise ValueError("invalid storage key")
        path = (self._root / key).resolve()
        if path.parent != self._root:
            raise ValueError("storage key escaped root")
        return path

    async def save_upload(self, stream: AsyncReadable, max_bytes: int) -> StoredUpload:
        handle = self._new_handle(".upload")
        digest = hashlib.sha256()
        size = 0
        try:
            with handle.path.open("xb") as destination:
                while chunk := await stream.read(64 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise AppError(
                            413,
                            "upload_too_large",
                            "error.upload_too_large",
                            "Upload too large",
                        )
                    digest.update(chunk)
                    destination.write(chunk)
        except BaseException:
            handle.path.unlink(missing_ok=True)
            raise
        if size == 0:
            handle.path.unlink(missing_ok=True)
            raise AppError(422, "empty_upload", "error.empty_upload", "Empty upload")
        return StoredUpload(handle=handle, size=size, sha256=digest.hexdigest())

    async def create_temporary(self, suffix: str = "") -> LocalImageHandle:
        handle = self._new_handle(suffix)
        await asyncio.to_thread(handle.path.touch, exist_ok=False)
        return handle

    async def clone_upload(self, key: str, max_bytes: int) -> StoredUpload:
        source = self.resolve(key)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError("retained upload is unavailable")
        handle = self._new_handle(".upload")

        def copy() -> StoredUpload:
            digest = hashlib.sha256()
            size = 0
            try:
                with source.open("rb") as reader, handle.path.open("xb") as writer:
                    while chunk := reader.read(64 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError("retained upload exceeds current limit")
                        digest.update(chunk)
                        writer.write(chunk)
            except BaseException:
                handle.path.unlink(missing_ok=True)
                raise
            if size <= 0:
                handle.path.unlink(missing_ok=True)
                raise ValueError("retained upload is empty")
            return StoredUpload(handle=handle, size=size, sha256=digest.hexdigest())

        return await asyncio.to_thread(copy)

    async def delete(self, key: str | None) -> None:
        if key is None:
            return
        try:
            path = self.resolve(key)
        except ValueError:
            return
        await asyncio.to_thread(path.unlink, missing_ok=True)

    async def cleanup_orphans(self, older_than: datetime) -> int:
        threshold = older_than.astimezone(UTC).timestamp()

        def cleanup() -> int:
            removed = 0
            if not self._root.exists():
                return removed
            for path in self._root.iterdir():
                if not path.is_file() or not _SAFE_KEY.fullmatch(path.name):
                    continue
                try:
                    if path.stat().st_mtime <= threshold:
                        path.unlink(missing_ok=True)
                        removed += 1
                except OSError:
                    continue
            return removed

        return await asyncio.to_thread(cleanup)

    async def cleanup_after(self, seconds: int) -> int:
        return await self.cleanup_orphans(datetime.now(UTC) - timedelta(seconds=seconds))
