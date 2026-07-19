#!/usr/bin/env python3
"""Private CLI diagnostics for Phase 6B runtimes.

This command never downloads, loads, or starts a model. Isolated providers become
ready only from their live localhost worker health after a real inference.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from atlaslens_api.phase6b.runtime_verification import (
    validate_rapidocr_runtime_verification,
)
from atlaslens_api.phase6b.worker_client import (
    LocalWorkerHTTPClient,
    WorkerClientError,
    WorkerHealth,
)
from atlaslens_api.providers.rapidocr import load_verified_rapidocr_runtime

ProviderState = Literal[
    "not_installed",
    "dependencies_installed",
    "weights_prepared",
    "worker_unreachable",
    "model_load_failed",
    "inference_not_verified",
    "ready",
    "disabled",
]

PROVIDER_STATES: frozenset[str] = frozenset(
    {
        "not_installed",
        "dependencies_installed",
        "weights_prepared",
        "worker_unreachable",
        "model_load_failed",
        "inference_not_verified",
        "ready",
        "disabled",
    }
)


@dataclass(frozen=True, slots=True)
class WorkerDefinition:
    provider: str
    default_port: int
    python_version: str
    import_modules: tuple[str, ...]
    use_source_path: bool
    provider_revision: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    process_running: bool
    import_ok: bool
    weights_available: bool
    model_loaded: bool
    load_verified: bool
    inference_verified: bool
    provider_revision: str
    model_revision: str
    device: str
    last_error: str | None


HealthProbe = Callable[
    [WorkerDefinition, str, int, str], tuple[WorkerSnapshot | None, str | None]
]
ImportProbe = Callable[[Path, Sequence[str], Path | None], bool]

_WORKERS = {
    "osv5m": WorkerDefinition(
        provider="osv5m",
        default_port=8791,
        python_version="3.10",
        import_modules=("torch", "torchvision", "models.huggingface"),
        use_source_path=True,
    ),
    "plonk": WorkerDefinition(
        provider="plonk",
        default_port=8792,
        python_version="3.10",
        import_modules=("torch", "plonk"),
        use_source_path=False,
    ),
    "paddleocr": WorkerDefinition(
        provider="paddleocr",
        default_port=8793,
        python_version="3.12",
        import_modules=("paddle", "paddleocr"),
        use_source_path=False,
        provider_revision="3.7.0",
    ),
}


def _bytes(path: Path) -> int:
    if not path.exists() or path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _git_revision(path: Path) -> str | None:
    """Read a checkout revision without mutating global Git safe-directory state."""

    marker = path / ".git"
    try:
        if marker.is_file():
            line = marker.read_text(encoding="utf-8").strip()
            if not line.startswith("gitdir: "):
                return None
            git_dir = (path / line.removeprefix("gitdir: ")).resolve(strict=True)
        elif marker.is_dir():
            git_dir = marker
        else:
            return None
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if len(head) == 40 and all(character in "0123456789abcdef" for character in head):
            return head
        if not head.startswith("ref: "):
            return None
        reference = head.removeprefix("ref: ")
        loose = git_dir / reference
        if loose.is_file():
            revision = loose.read_text(encoding="ascii").strip()
            return revision if len(revision) == 40 else None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="ascii").splitlines():
                if line.startswith(("#", "^")):
                    continue
                revision, separator, name = line.partition(" ")
                if separator and name == reference and len(revision) == 40:
                    return revision
    except OSError:
        return None
    return None


def _worker_import(
    worker_python: Path,
    modules: Sequence[str],
    source_path: Path | None = None,
) -> bool:
    if not worker_python.is_file() or worker_python.is_symlink():
        return False
    source_setup = (
        f"sys.path.insert(0,{str(source_path)!r});" if source_path is not None else ""
    )
    imports = ";".join(
        f"importlib.import_module({module!r})" for module in modules
    )
    command = f"import importlib,sys;{source_setup}{imports}"
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PYTHONNOUSERSITE": "1",
        }
    )
    try:
        return subprocess.run(  # noqa: S603 -- manifest-owned interpreter.
            [str(worker_python), "-c", command],
            check=False,
            capture_output=True,
            timeout=30,
            env=environment,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _worker_health(
    definition: WorkerDefinition,
    host: str,
    port: int,
    expected_provider_revision: str,
) -> tuple[WorkerSnapshot | None, str | None]:
    async def probe() -> WorkerHealth:
        client = LocalWorkerHTTPClient(
            provider=definition.provider,
            host=host,
            port=port,
            provider_revision=expected_provider_revision,
            timeout_seconds=2.0,
            max_response_bytes=64 * 1024,
        )
        # Health opens a bounded Connection: close transport. Calling client.close()
        # would unload the model and is deliberately not appropriate for diagnostics.
        return await client.health(timeout_seconds=2.0)

    try:
        health = asyncio.run(probe())
    except WorkerClientError as exc:
        return None, exc.code
    except (OSError, RuntimeError, ValueError):
        return None, "worker_health_failed"
    return (
        WorkerSnapshot(
            process_running=health.process_running,
            import_ok=health.import_ok,
            weights_available=health.weights_available,
            model_loaded=health.model_loaded,
            load_verified=health.load_verified,
            inference_verified=health.real_inference_verified,
            provider_revision=health.provider_revision,
            model_revision=health.model_revision,
            device=health.device,
            last_error=health.last_error,
        ),
        None,
    )


def classify_isolated_state(
    *,
    enabled: bool,
    worker_python_exists: bool,
    import_ok: bool,
    weights_available: bool,
    source_revision_matches: bool,
    health: WorkerSnapshot | None,
    health_error: str | None,
) -> tuple[ProviderState, str | None]:
    if not enabled:
        return "disabled", "provider_disabled"
    if not worker_python_exists or not import_ok:
        reason = "worker_python_missing" if not worker_python_exists else "provider_import_missing"
        return "not_installed", reason
    if not weights_available:
        return "dependencies_installed", "pretrained_weights_missing"
    if not source_revision_matches:
        return "weights_prepared", "source_revision_mismatch"
    if health is None:
        return "worker_unreachable", health_error or "worker_unreachable"
    if not health.process_running:
        return "worker_unreachable", "worker_process_not_running"
    if not health.import_ok:
        return "not_installed", "worker_provider_import_failed"
    if not health.weights_available:
        return "dependencies_installed", "worker_pretrained_weights_missing"
    if not health.load_verified:
        if health.last_error is not None:
            return "model_load_failed", health.last_error
        return "weights_prepared", "model_load_not_verified"
    if not health.inference_verified:
        return "inference_not_verified", health.last_error or "real_inference_not_verified"
    return "ready", None


def _main_runtime() -> dict[str, Any]:
    torch_spec = importlib.util.find_spec("torch")
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch_importable": torch_spec is not None,
        "cuda_available": False,
        "torch_version": None,
        "cuda_runtime": None,
    }
    if torch_spec is None:
        return result
    try:
        import torch

        result.update(
            torch_version=str(torch.__version__),
            cuda_runtime=str(torch.version.cuda) if torch.version.cuda else None,
            cuda_available=bool(torch.cuda.is_available()),
        )
    except Exception:
        result["torch_importable"] = False
    return result


def _environment_bool(name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _environment_port(name: str, default: int) -> int:
    try:
        port = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return port if 1 <= port <= 65_535 else default


def _environment_path(name: str, default: Path, project_root: Path) -> Path:
    configured = Path(os.environ.get(name, str(default))).expanduser()
    return configured if configured.is_absolute() else project_root / configured


def _weights_available(name: str, root: Path, project_root: Path) -> bool:
    if root.is_symlink() or not root.is_dir():
        return False
    if name == "osv5m":
        required = (root / "config.json", root / "pytorch_model.bin")
        return all(path.is_file() and not path.is_symlink() for path in required)
    if name == "paddleocr":
        directories = (
            root / "det",
            root / "rec" / "latin_PP-OCRv5_mobile_rec",
        )
        filenames = ("inference.json", "inference.pdiparams", "inference.yml")
        return all(
            all((directory / filename).is_file() for filename in filenames)
            for directory in directories
        )
    if name == "plonk":
        dino_source = project_root / ".local" / "vendor" / "dinov2"
        dino_weights = (
            root
            / "torch"
            / "hub"
            / "checkpoints"
            / "dinov2_vitl14_reg4_pretrain.pth"
        )
        dino_ready = (
            dino_weights.is_file()
            and not dino_weights.is_symlink()
            and (dino_source / "dinov2").is_dir()
        )
        return dino_ready and any(
            (root / variant / "config.json").is_file()
            and (root / variant / "model.safetensors").is_file()
            for variant in ("yfcc", "inat")
        )
    return False


def _base_provider(
    model: Mapping[str, Any],
    *,
    weight_path: Path,
    source_path: Path | None,
) -> dict[str, Any]:
    installed_revision = _git_revision(source_path) if source_path is not None else None
    weight_bytes = _bytes(weight_path)
    return {
        "provider": model["name"],
        "source_revision": model["source_revision"],
        "installed_source_revision": installed_revision,
        "source_revision_matches": (
            installed_revision == model["source_revision"]
            if installed_revision is not None
            else None
        ),
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "model_variants": model.get("variants", []),
        "execution_mode": model["execution_mode"],
        "weight_path": str(weight_path),
        "disk_usage": weight_bytes,
        "disk_usage_bytes": weight_bytes,
        "source_disk_usage_bytes": _bytes(source_path) if source_path else 0,
        "dataset_downloaded": False,
    }


def _collect_geoclip(
    model: Mapping[str, Any], project_root: Path, local_app_data: Path
) -> dict[str, Any]:
    model_cache = _environment_path(
        "MODEL_CACHE_DIR", local_app_data / "AtlasLens" / "models", project_root
    )
    weight_path = model_cache / "geoclip"
    result = _base_provider(model, weight_path=weight_path, source_path=None)
    enabled = _environment_bool("GLOBAL_MODEL_ENABLED")
    import_ok = importlib.util.find_spec("geoclip") is not None
    required = (
        "installation.json",
        "geoclip-assets/coordinates_100K.csv",
        "geoclip-assets/image_encoder_mlp_weights.pth",
        "geoclip-assets/location_encoder_weights.pth",
        "geoclip-assets/logit_scale_weights.pth",
        "clip-snapshot/model.safetensors",
    )
    weights_available = all((weight_path / relative).is_file() for relative in required)
    if not enabled:
        state: ProviderState = "disabled"
        reason = "provider_disabled"
    elif not import_ok:
        state, reason = "not_installed", "provider_import_missing"
    elif not weights_available:
        state, reason = "dependencies_installed", "pretrained_weights_missing"
    else:
        # GeoCLIP is the preserved, separately accepted in-process baseline.
        state, reason = "ready", None
    result.update(
        state=state,
        health=state,
        reason_code=reason,
        enabled=enabled,
        dependencies_installed=import_ok,
        weights_available=weights_available,
        worker_python=None,
        worker_port=None,
        worker_reachable=None,
        model_loaded=None,
        load_verified=state == "ready",
        inference_verified=state == "ready",
        last_error=reason,
        device="auto",
    )
    return result


def _collect_rapidocr(
    model: Mapping[str, Any], project_root: Path, local_app_data: Path
) -> dict[str, Any]:
    weight_path = _environment_path(
        "RAPIDOCR_MODEL_ROOT",
        local_app_data / "AtlasLens" / "models" / "rapidocr-3.9.1",
        project_root,
    )
    result = _base_provider(model, weight_path=weight_path, source_path=None)
    enabled = _environment_bool("RAPIDOCR_ENABLED")
    import_ok = _worker_import(
        Path(sys.executable), ("rapidocr", "onnxruntime", "cv2", "numpy")
    )
    runtime = load_verified_rapidocr_runtime(weight_path, device="cpu") if import_ok else None
    weights_available = runtime is not None
    evidence = (
        validate_rapidocr_runtime_verification(
            project_root,
            weight_path,
            expected_source_revision=str(model["source_revision"]),
            expected_model_revision=str(model["model_revision"]),
        )
        if weights_available
        else None
    )
    verification = evidence.verification if evidence is not None else None
    if not enabled:
        state: ProviderState = "disabled"
        reason = "provider_disabled"
    elif not import_ok:
        state, reason = "not_installed", "provider_import_missing"
    elif not weights_available:
        state, reason = "dependencies_installed", "pretrained_weights_missing"
    elif verification is None:
        state = "inference_not_verified"
        reason = evidence.reason_code if evidence is not None else "runtime_verification_missing"
    else:
        state, reason = "ready", None
    result.update(
        state=state,
        health=state,
        reason_code=reason,
        enabled=enabled,
        dependencies_installed=import_ok,
        weights_available=weights_available,
        worker_python=sys.executable,
        worker_port=None,
        worker_reachable=verification is not None,
        model_loaded=None,
        load_verified=verification.load_verified if verification else False,
        inference_verified=verification.inference_verified if verification else False,
        last_error=reason,
        device=verification.device if verification else "cpu",
        runtime_verification_path=str(
            project_root / ".local/runtime/phase6b/rapidocr-verification.json"
        ),
    )
    return result


def _collect_isolated(
    model: Mapping[str, Any],
    project_root: Path,
    *,
    health_probe: HealthProbe,
    import_probe: ImportProbe,
) -> dict[str, Any]:
    name = str(model["name"])
    definition = _WORKERS[name]
    source_path = project_root / ".local" / "vendor" / name
    weight_path = project_root / ".local" / "models" / "phase6b" / name
    result = _base_provider(model, weight_path=weight_path, source_path=source_path)
    worker_python = _environment_path(
        f"{name.upper()}_WORKER_PYTHON",
        project_root
        / ".local"
        / "workers"
        / name
        / ".venv"
        / "Scripts"
        / "python.exe",
        project_root,
    )
    port = _environment_port(f"{name.upper()}_WORKER_PORT", definition.default_port)
    host = os.environ.get(f"{name.upper()}_WORKER_HOST", "127.0.0.1")
    enabled = _environment_bool(f"{name.upper()}_ENABLED") and _environment_bool(
        f"{name.upper()}_WORKER_ENABLED"
    )
    import_ok = import_probe(
        worker_python,
        definition.import_modules,
        source_path if definition.use_source_path else None,
    )
    weights_available = _weights_available(name, weight_path, project_root)
    installed_revision = result["installed_source_revision"]
    source_revision_matches = (
        installed_revision == model["source_revision"]
        if name in {"osv5m", "plonk"}
        else True
    )
    if name == "plonk":
        auxiliary = model.get("auxiliary")
        expected_auxiliary_revision = (
            auxiliary.get("source_revision")
            if isinstance(auxiliary, Mapping)
            else None
        )
        installed_auxiliary_revision = _git_revision(
            project_root / ".local" / "vendor" / "dinov2"
        )
        auxiliary_matches = (
            isinstance(expected_auxiliary_revision, str)
            and installed_auxiliary_revision == expected_auxiliary_revision
        )
        result.update(
            auxiliary_source_revision=expected_auxiliary_revision,
            installed_auxiliary_source_revision=installed_auxiliary_revision,
            auxiliary_source_revision_matches=auxiliary_matches,
        )
        source_revision_matches = source_revision_matches and auxiliary_matches
    health: WorkerSnapshot | None = None
    health_error: str | None = None
    if enabled and import_ok and weights_available and source_revision_matches:
        expected_provider_revision = definition.provider_revision or str(
            model["source_revision"]
        )
        if host != "127.0.0.1":
            health_error = "worker_host_not_loopback"
        else:
            health, health_error = health_probe(
                definition, host, port, expected_provider_revision
            )
    state, reason = classify_isolated_state(
        enabled=enabled,
        worker_python_exists=worker_python.is_file(),
        import_ok=import_ok,
        weights_available=weights_available,
        source_revision_matches=source_revision_matches,
        health=health,
        health_error=health_error,
    )
    result.update(
        state=state,
        health=state,
        reason_code=reason,
        enabled=enabled,
        dependencies_installed=import_ok,
        weights_available=weights_available,
        worker_python=str(worker_python),
        worker_python_version=definition.python_version,
        worker_port=port,
        worker_reachable=health is not None,
        model_loaded=health.model_loaded if health else False,
        load_verified=health.load_verified if health else False,
        inference_verified=health.inference_verified if health else False,
        last_error=(health.last_error if health else None) or reason,
        device=health.device if health else model.get("device", "auto"),
        reported_provider_revision=health.provider_revision if health else None,
        reported_model_revision=health.model_revision if health else None,
    )
    return result


def collect(
    project_root: Path,
    *,
    health_probe: HealthProbe = _worker_health,
    import_probe: ImportProbe = _worker_import,
) -> dict[str, Any]:
    lock_path = project_root / "config" / "external-models.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    local_app_data = Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    )
    providers: list[dict[str, Any]] = []
    for model in lock["models"]:
        name = model["name"]
        if name == "geoclip":
            provider = _collect_geoclip(model, project_root, local_app_data)
        elif name == "rapidocr":
            provider = _collect_rapidocr(model, project_root, local_app_data)
        elif name in _WORKERS:
            provider = _collect_isolated(
                model,
                project_root,
                health_probe=health_probe,
                import_probe=import_probe,
            )
        else:
            continue
        if provider["state"] not in PROVIDER_STATES:
            raise RuntimeError("invalid Phase 6B provider state")
        providers.append(provider)

    usage = shutil.disk_usage(project_root)
    return {
        "schema_version": "atlaslens-phase6b-diagnostics-v2",
        "visibility": "private_cli_only",
        "manifest": str(lock_path),
        "runtime": _main_runtime(),
        "disk": {"free_bytes": usage.free, "total_bytes": usage.total},
        "providers": providers,
        "policy": lock["policy"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = collect(args.project_root.resolve())
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        runtime = report["runtime"]
        print(
            f"Python {runtime['python']} | torch {runtime['torch_version'] or 'missing'} | "
            f"CUDA {'available' if runtime['cuda_available'] else 'unavailable'}"
        )
        for provider in report["providers"]:
            mib = provider["disk_usage_bytes"] / 1024 / 1024
            reason = f" ({provider['reason_code']})" if provider["reason_code"] else ""
            print(f"{provider['provider']}: {provider['state']}{reason}")
            print(
                f"  source_revision={provider['source_revision']} | "
                f"model_revision={provider['model_revision']}"
            )
            print(
                f"  worker_python={provider['worker_python'] or 'in-process'} | "
                f"worker_port={provider['worker_port'] or 'n/a'}"
            )
            print(
                f"  weight_path={provider['weight_path']} | disk={mib:.1f} MiB | "
                f"load_verified={provider['load_verified']} | "
                f"inference_verified={provider['inference_verified']} | "
                f"last_error={provider['last_error'] or 'none'}"
            )
        print("Dataset downloads: forbidden (no dataset action was performed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
