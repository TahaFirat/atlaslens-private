from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from atlaslens_api.database import create_database_engine
from atlaslens_api.retrieval.errors import RetrievalError
from atlaslens_api.retrieval.faiss_index import FaissImageIndex
from atlaslens_api.retrieval.metadata import SQLAlchemyImageMetadataRepository
from atlaslens_api.retrieval.providers import production_embedding_provider
from atlaslens_api.retrieval.service import (
    ManifestDatasetImporter,
    open_retrieval_query_service,
    verify_metadata_consistency,
)

_DATABASE_NAME = "metadata.sqlite3"
_REMOVABLE_FILES = (
    f"{_DATABASE_NAME}-journal",
    f"{_DATABASE_NAME}-shm",
    f"{_DATABASE_NAME}-wal",
    _DATABASE_NAME,
    "vectors.faiss",
    "index.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas")
    groups = parser.add_subparsers(dest="group", required=True)
    embeddings = groups.add_parser("embeddings")
    commands = embeddings.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--input-root", type=Path, required=True)
    create.add_argument("--index-dir", type=Path, required=True)
    create.add_argument(
        "--provider",
        choices=("disabled", "siglip2", "siglip2-b16-384", "clip"),
        default="disabled",
    )
    create.add_argument("--model-cache", type=Path)
    create.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    encode = commands.add_parser("encode")
    encode.add_argument("--image", type=Path, required=True)
    encode.add_argument(
        "--provider", choices=("siglip2", "siglip2-b16-384"), default="siglip2-b16-384"
    )
    encode.add_argument("--model-cache", type=Path)
    encode.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    for command in ("verify", "info"):
        item = commands.add_parser(command)
        item.add_argument("--index-dir", type=Path, required=True)

    remove = commands.add_parser("remove")
    remove.add_argument("--index-dir", type=Path, required=True)
    remove.add_argument("--yes", action="store_true")

    retrieval = groups.add_parser("retrieval")
    retrieval_commands = retrieval.add_subparsers(dest="command", required=True)
    smoke = retrieval_commands.add_parser("smoke-test")
    smoke.add_argument("--index-dir", type=Path, required=True)
    smoke.add_argument("--image", type=Path, required=True)
    smoke.add_argument("--top-k", type=int, default=5)
    smoke.add_argument("--model-cache", type=Path)
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _provider(args: argparse.Namespace, name: str | None = None):  # type: ignore[no-untyped-def]
    provider_name = name or args.provider
    if args.model_cache is None and args.device == "auto":
        return production_embedding_provider(provider_name)
    return production_embedding_provider(
        provider_name,
        cache_root=args.model_cache,
        requested_device=args.device,
    )


def _repository(index_dir: Path, *, require_existing: bool) -> SQLAlchemyImageMetadataRepository:
    database = index_dir / _DATABASE_NAME
    if require_existing and not database.is_file():
        raise RetrievalError("retrieval metadata database is missing")
    engine = create_database_engine(f"sqlite:///{database.as_posix()}")
    return SQLAlchemyImageMetadataRepository(engine)


def _create(args: argparse.Namespace) -> dict[str, object]:
    provider = _provider(args)
    directory = args.index_dir.expanduser().resolve()
    exists = (directory / "index.json").is_file()
    records_index = (
        FaissImageIndex.open(directory)
        if exists
        else FaissImageIndex(directory, provider.spec)
    )
    repository = _repository(directory, require_existing=exists)
    try:
        summary = ManifestDatasetImporter(provider, records_index, repository).create(
            args.manifest, args.input_root
        )
        return {"status": "ready", **summary.model_dump(mode="json")}
    finally:
        repository.close()


def _encode(args: argparse.Namespace) -> dict[str, object]:
    provider = _provider(args)
    if not provider.available:
        raise RetrievalError(provider.unavailable_reason or "embedding provider is unavailable")
    embedding = provider.embed(args.image)
    return {
        "status": "ready",
        "provider": embedding.spec.model_dump(mode="json"),
        "embedding": [float(value) for value in embedding.vector.tolist()],
    }


def _smoke_test(args: argparse.Namespace) -> dict[str, object]:
    index = FaissImageIndex.open(args.index_dir.expanduser().resolve())
    provider = _provider(args, index.spec.provider)
    service = open_retrieval_query_service(args.index_dir, provider)
    try:
        hits = service.retrieve_image(args.image, args.top_k)
        return {
            "status": "ready",
            "diagnostics": service.diagnostics(),
            "hits": [
                {
                    "reference_id": hit.metadata.asset_key or str(hit.metadata.image_id),
                    "distance": hit.distance,
                    "relative_similarity": max(-1.0, min(1.0, 1.0 - hit.distance)),
                    "provider": hit.provider.model_dump(mode="json"),
                    "latitude": hit.metadata.latitude,
                    "longitude": hit.metadata.longitude,
                    "source": hit.metadata.source,
                    "license": hit.metadata.license,
                    "attribution": hit.metadata.attribution,
                    "display_allowed": hit.metadata.display_allowed,
                    "capture_family_id": hit.metadata.capture_family_id,
                }
                for hit in hits
            ],
        }
    finally:
        service.close()


def _info(args: argparse.Namespace, *, verify: bool) -> dict[str, object]:
    directory = args.index_dir.expanduser().resolve()
    index = FaissImageIndex.open(directory)
    repository: SQLAlchemyImageMetadataRepository | None = None
    try:
        if verify:
            repository = _repository(directory, require_existing=True)
            verify_metadata_consistency(index, repository)
        diagnostics = index.diagnostics()
        storage_size = diagnostics.storage_size_bytes + sum(
            path.stat().st_size
            for path in (index.directory / name for name in _REMOVABLE_FILES)
            if path.is_file() and path.name not in {"vectors.faiss", "index.json"}
        )
        return diagnostics.model_copy(update={"storage_size_bytes": storage_size}).model_dump(
            mode="json"
        )
    finally:
        if repository is not None:
            repository.close()


def _remove(args: argparse.Namespace) -> dict[str, object]:
    if not args.yes:
        raise RetrievalError("removal requires explicit --yes confirmation")
    expanded = args.index_dir.expanduser()
    if expanded.is_symlink():
        raise RetrievalError("symbolic-link index directories cannot be removed")
    directory = expanded.resolve()
    if not (directory / "index.json").is_file():
        raise RetrievalError("directory is not a recognized AtlasLens retrieval index")
    unknown = [path for path in directory.iterdir() if path.name not in _REMOVABLE_FILES]
    if unknown:
        raise RetrievalError("index directory contains unknown files and was not removed")
    for name in _REMOVABLE_FILES:
        path = directory / name
        if path.exists() or path.is_symlink():
            path.unlink()
    directory.rmdir()
    return {"status": "removed"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            result = _create(args)
        elif args.command == "encode":
            result = _encode(args)
        elif args.command == "smoke-test":
            result = _smoke_test(args)
        elif args.command == "verify":
            result = _info(args, verify=True)
        elif args.command == "info":
            result = _info(args, verify=False)
        else:
            result = _remove(args)
    except (RetrievalError, ValueError, OSError):
        print(
            json.dumps({"status": "error", "code": "retrieval_operation_failed"}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
