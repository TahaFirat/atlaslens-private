"""One-shot local RunPod supervisor for the authorized Phase 3F pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "services" / "api" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from atlaslens_api.phase3f.runpod import (  # noqa: E402
    PodConnection,
    RunPodAPIError,
    RunPodConfig,
    RunPodInventory,
    RunPodV1Client,
)
from atlaslens_api.phase3f.safety import (  # noqa: E402
    MAX_RUNTIME_SECONDS,
    BudgetPolicy,
    PodRequest,
    RunPodLease,
    SinglePodSession,
)
from atlaslens_api.phase3f.supervisor import (  # noqa: E402
    Phase3FSupervisorError,
    prepare_transfer_bundle,
    publish_verified_output,
    require_empty_inventory,
    require_inventory_restored,
    verify_output_archive,
)

REQUIRED_BRANCH = "feature/phase3f-multiregion-pilot"
REQUIRED_BASE = "183b7c2f410c41d3449238f156e73fed493e9dd9"
REQUIRED_ORIGIN = "https://github.com/TahaFirat/atlaslens-private.git"
IMAGE = (
    "runpod/pytorch@sha256:"
    "60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f"
)
GPU_PREFERENCES = (
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3090",
)
REMOTE_JOB_SECONDS = 5 * 60 * 60 + 30 * 60
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


class SupervisorExecutionError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SupervisorExecutionError(code)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=D:/geoSearch", *args],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    _require(completed.returncode == 0, "GIT_PREFLIGHT_FAILED")
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _preflight_repository() -> tuple[str, tuple[str, ...]]:
    _require(_git("branch", "--show-current") == REQUIRED_BRANCH, "GIT_BRANCH_MISMATCH")
    head = _git("rev-parse", "HEAD")
    _require(_git("merge-base", "main", "HEAD") == REQUIRED_BASE, "GIT_BASE_MISMATCH")
    _require(_git("remote", "get-url", "origin") == REQUIRED_ORIGIN, "GIT_ORIGIN_MISMATCH")
    status = tuple(
        line for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if line
    )
    _require(status in {(), ("?? asda.html",)}, "GIT_WORKTREE_NOT_CLEAN")
    raw = subprocess.run(
        ["git", "-c", "safe.directory=D:/geoSearch", "ls-files", "-z"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    _require(raw.returncode == 0, "GIT_TRACKED_FILES_UNAVAILABLE")
    tracked = tuple(
        value.decode("utf-8", errors="strict")
        for value in raw.stdout.split(b"\x00")
        if value
    )
    return head, tracked


def _disk_gate() -> None:
    c_free = shutil.disk_usage("C:\\").free
    d_free = shutil.disk_usage("D:\\").free
    _require(c_free >= 35 * 1024**3, "C_DISK_GATE_FAILED")
    _require(d_free >= 8 * 1024**3, "D_DISK_GATE_FAILED")


def _run_command(arguments: list[str], *, timeout_seconds: float) -> None:
    try:
        completed = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=max(1.0, timeout_seconds),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SupervisorExecutionError("TRANSFER_COMMAND_FAILED") from exc
    _require(completed.returncode == 0, "TRANSFER_COMMAND_FAILED")


def _ssh_arguments(key: Path, known_hosts: Path, connection: PodConnection) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key),
        "-p",
        str(connection.public_ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        f"root@{connection.public_ip}",
    ]


def _scp_arguments(key: Path, known_hosts: Path, connection: PodConnection) -> list[str]:
    return [
        "scp",
        "-q",
        "-i",
        str(key),
        "-P",
        str(connection.public_ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
    ]


def _wait_connection(
    client: RunPodV1Client,
    pod_id: str,
    lease: RunPodLease,
    started: float,
) -> PodConnection:
    for _ in range(120):
        lease.assert_within_limits(elapsed_seconds=int(time.monotonic() - started))
        try:
            return client.pod_connection(pod_id)
        except RunPodAPIError as exc:
            if exc.code not in {"pod_connection_pending", "pod_not_running"}:
                raise
        time.sleep(5)
    raise SupervisorExecutionError("POD_SSH_UNAVAILABLE")


def _run_watched(
    arguments: list[str],
    *,
    lease: RunPodLease,
    started: float,
) -> None:
    try:
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise SupervisorExecutionError("REMOTE_JOB_START_FAILED") from exc
    try:
        while process.poll() is None:
            lease.assert_within_limits(elapsed_seconds=int(time.monotonic() - started))
            time.sleep(15)
        _require(process.returncode == 0, "REMOTE_JOB_FAILED")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


def _operation(
    client: RunPodV1Client,
    lease: RunPodLease,
    *,
    bundle_root: Path,
    key: Path,
    known_hosts: Path,
    download_path: Path,
    run_id: str,
    started: float,
) -> dict[str, object]:
    connection = _wait_connection(client, lease.pod.pod_id, lease, started)
    ssh = _ssh_arguments(key, known_hosts, connection)
    scp = _scp_arguments(key, known_hosts, connection)
    def remaining() -> float:
        return MAX_RUNTIME_SECONDS - (time.monotonic() - started)
    remote_transfer = "/workspace/phase3f-transfer"
    _run_command(
        [*ssh, "mkdir -p /workspace/phase3f-transfer/vendor"],
        timeout_seconds=remaining(),
    )
    for local, remote in (
        (bundle_root / "atlaslens-phase3f-source.tar", f"{remote_transfer}/source.tar"),
        (bundle_root / "model.safetensors", f"{remote_transfer}/model.safetensors"),
        (bundle_root / "vendor" / "megaloc_model.py", f"{remote_transfer}/vendor/megaloc_model.py"),
        (bundle_root / "vendor" / "LICENSE", f"{remote_transfer}/vendor/LICENSE"),
    ):
        _run_command(
            [*scp, str(local), f"root@{connection.public_ip}:{remote}"],
            timeout_seconds=remaining(),
        )
        lease.assert_within_limits(elapsed_seconds=int(time.monotonic() - started))
    dependencies = (
        "httpx==0.28.1 pydantic==2.11.7 pillow==11.3.0 "
        "numpy==2.2.6 safetensors==0.5.3 faiss-cpu==1.14.3"
    )
    deadline = int(time.time() + REMOTE_JOB_SECONDS)
    remote_command = " && ".join(
        (
            "rm -rf /workspace/phase3f-repo /workspace/phase3f-output /workspace/phase3f-work",
            "mkdir -p /workspace/phase3f-repo",
            "tar -xf /workspace/phase3f-transfer/source.tar -C /workspace/phase3f-repo",
            f"python -m pip install --disable-pip-version-check --no-cache-dir {dependencies}",
            (
                "timeout --signal=TERM "
                f"{REMOTE_JOB_SECONDS} python /workspace/phase3f-repo/scripts/phase3f/cloud_job.py "
                "--repository-root /workspace/phase3f-repo "
                f"--run-id {run_id} "
                "--model /workspace/phase3f-transfer/model.safetensors "
                "--vendor-root /workspace/phase3f-transfer/vendor "
                "--work-root /workspace/phase3f-work "
                "--output-root /workspace/phase3f-output "
                f"--deadline-epoch {deadline}"
            ),
            "tar -C /workspace -cf /workspace/phase3f-transfer/output.tar phase3f-output",
        )
    )
    try:
        _run_watched(
            [*ssh, "bash", "-lc", shlex.quote(remote_command)],
            lease=lease,
            started=started,
        )
        _run_command(
            [
                *scp,
                f"root@{connection.public_ip}:{remote_transfer}/output.tar",
                str(download_path),
            ],
            timeout_seconds=remaining(),
        )
    finally:
        cleanup = (
            "rm -rf /workspace/phase3f-work /workspace/phase3f-output "
            "/workspace/phase3f-repo /workspace/phase3f-transfer"
        )
        try:
            _run_command([*ssh, cleanup], timeout_seconds=min(60.0, max(1.0, remaining())))
        except SupervisorExecutionError:
            pass
    return {
        "pod_id_sha256": hashlib.sha256(lease.pod.pod_id.encode()).hexdigest(),
        "gpu": connection.gpu_display_name,
        "hourly_price_usd": str(connection.hourly_price),
    }


def _post_inventory(
    client: RunPodV1Client,
    before: RunPodInventory,
) -> RunPodInventory:
    after = client.inventory()
    for _ in range(20):
        if after == before:
            return after
        time.sleep(3)
        after = client.inventory()
    return after


def _write_receipt(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlaslens-phase3f-supervisor")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=Path(r"C:\AtlasLensRuntime\phase3f"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        print("PHASE3F_EXECUTION_FLAG_REQUIRED")
        return 2
    run_id = os.urandom(16).hex()
    _require(bool(_RUN_ID.fullmatch(run_id)), "RUN_ID_INVALID")
    key_root = Path(r"C:\tmp")
    key = key_root / f"atlaslens-phase3f-{run_id}"
    known_hosts = key_root / f"atlaslens-phase3f-{run_id}.known-hosts"
    local_root = ROOT / ".local" / "phase3f" / "runs" / run_id
    transfer_root = local_root / "transfer"
    download_path = local_root / "output.tar"
    extracted_parent = local_root / "extracted"
    started_at = datetime.now(UTC)
    started = time.monotonic()
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        print("RUNPOD_API_KEY_NOT_VISIBLE_IN_APP")
        return 1
    try:
        _disk_gate()
        head, tracked = _preflight_repository()
        key_root.mkdir(parents=True, exist_ok=True)
        _run_command(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                f"atlaslens-phase3f-{run_id}",
                "-f",
                str(key),
            ],
            timeout_seconds=30,
        )
        public_key = key.with_suffix(".pub").read_text(encoding="ascii").strip()
        bundle = prepare_transfer_bundle(
            ROOT,
            transfer_root,
            commit_sha=head,
            tracked_paths=tracked,
            model_path=ROOT / ".local" / "models" / "phase6c" / "megaloc" / "model.safetensors",
            vendor_root=ROOT / ".local" / "vendor" / "megaloc",
        )
        config = RunPodConfig(
            image_name=IMAGE,
            gpu_type_preferences=GPU_PREFERENCES,
            container_disk_gb=40,
            min_gpu_memory_gb=16,
            ssh_public_key=public_key,
        )
        with RunPodV1Client(api_token=token, config=config) as client:
            before = client.inventory()
            require_empty_inventory(before)
            offer = client.select_gpu_offer(max_hourly_price=Decimal("0.50"))
            request = PodRequest(
                run_marker=f"atlaslens-phase3f-{run_id}",
                idempotency_key=f"atlaslens-phase3f-create-{run_id}",
                hourly_cost_usd=offer.hourly_price,
                max_runtime_seconds=MAX_RUNTIME_SECONDS,
                public_ports=(22,),
            )
            session = SinglePodSession(
                client,
                policy=BudgetPolicy(),
                cleanup_poll_attempts=20,
                cleanup_poll_seconds=3.0,
            )
            execution = session.execute(
                request,
                lambda lease: _operation(
                    client,
                    lease,
                    bundle_root=bundle.root,
                    key=key,
                    known_hosts=known_hosts,
                    download_path=download_path,
                    run_id=run_id,
                    started=started,
                ),
            )
            after = _post_inventory(client, before)
            require_inventory_restored(before, after)
        _disk_gate()
        verified = verify_output_archive(download_path, extracted_parent)
        published = publish_verified_output(verified, args.runtime_root)
        elapsed = time.monotonic() - started
        hourly = Decimal(str(execution.value["hourly_price_usd"]))
        conservative_compute = hourly * Decimal(str(elapsed)) / Decimal(3600)
        conservative_total = conservative_compute + Decimal("0.10")
        _require(conservative_total < Decimal("10"), "ABSOLUTE_BUDGET_REACHED")
        receipt = {
            "schema": "atlaslens-phase3f-local-supervisor-receipt-v1",
            "run_id": run_id,
            "outcome": verified.outcome,
            "pod_id_sha256": execution.value["pod_id_sha256"],
            "gpu": execution.value["gpu"],
            "hourly_price_usd": str(hourly),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "actual_spend_usd": None,
            "actual_spend_available": False,
            "conservative_compute_usd": str(conservative_compute),
            "conservative_storage_allowance_usd": "0.10",
            "conservative_incremental_upper_usd": str(conservative_total),
            "single_pod_create_attempts": execution.audit.create_attempts,
            "pod_termination_verified": execution.audit.termination_verified,
            "post_pods": len(after.pods),
            "post_endpoints": len(after.endpoint_ids),
            "post_network_volumes": len(after.network_volume_ids),
            "post_templates": len(after.template_ids),
            "publication_hash": verified.publication_hash,
            "output_inventory_sha256": verified.inventory_sha256,
            "output_size_bytes": verified.total_size_bytes,
            "published_path_name": published.name,
            "secret_values_included": False,
        }
        _write_receipt(
            args.runtime_root / "_receipts" / f"{published.name}.json",
            receipt,
        )
        print(
            json.dumps(
                {
                    "outcome": verified.outcome,
                    "run_id": run_id,
                    "publication_hash": verified.publication_hash,
                    "cleanup_verified": True,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    except (Phase3FSupervisorError, RunPodAPIError, SupervisorExecutionError) as exc:
        print(exc.code)
        return 1
    except BaseException:
        print("PHASE3F_SUPERVISOR_UNEXPECTED_FAILURE")
        return 1
    finally:
        for path in (key, key.with_suffix(".pub"), known_hosts):
            try:
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            except OSError:
                pass
        if local_root.exists():
            shutil.rmtree(local_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
