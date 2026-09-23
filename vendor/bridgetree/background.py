from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ACTIVE_STATES = {"starting", "running"}
WORKER_READY_FILE = ".background_worker_ready.json"


def _utc_iso(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else timestamp
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _job_paths(state_dir: str | Path, job: str) -> Dict[str, Path]:
    root = Path(state_dir).resolve()
    return {
        "root": root,
        "lock": root / f"{job}.lock",
        "spec": root / f"{job}.spec.json",
        "log": root / f"{job}.log",
        "status": root / f"{job}.status.json",
        "ready": root / f"{job}.monitor_ready.json",
        "pid": root / f"{job}.pid",
        "exit": root / f"{job}.exit",
        "run_dir": root / f"{job}.run_dir",
    }


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _not_started_status(paths: Mapping[str, Path], job: str) -> Dict[str, Any]:
    return {
        "job": job,
        "state": "not_started",
        "state_dir": str(paths["root"]),
        "log_file": str(paths["log"]),
        "status_file": str(paths["status"]),
    }


def _refresh_dead_status_unlocked(
    paths: Mapping[str, Path], status: Dict[str, Any]
) -> Dict[str, Any]:
    pid = status.get("pid")
    if status.get("state") not in ACTIVE_STATES or _pid_is_alive(
        pid if isinstance(pid, int) else None
    ):
        return status
    now = time.time()
    refreshed = {
        **status,
        "state": "interrupted",
        "finished_at": _utc_iso(now),
        "finished_at_epoch": now,
        "message": "background monitor process is no longer running",
    }
    _atomic_write_json(paths["status"], refreshed)
    return refreshed


def _archive_previous(paths: Mapping[str, Path], job: str, previous: Mapping[str, Any] | None) -> str | None:
    existing = [
        path
        for name, path in paths.items()
        if name not in {"root", "lock"} and path.exists()
    ]
    existing.extend(path for path in paths["root"].glob(f"{job}.*") if path not in existing and path != paths["lock"])
    if not existing:
        return None
    stamp_value = str((previous or {}).get("started_at", ""))
    stamp = re.sub(r"[^0-9]+", "", stamp_value)[:14] or datetime.now().strftime("%Y%m%d%H%M%S")
    history = paths["root"] / "history" / f"{job}_{stamp}"
    suffix = 1
    while history.exists():
        history = history.with_name(f"{history.name}_{suffix}")
        suffix += 1
    history.mkdir(parents=True)
    for source in sorted(set(existing)):
        if source.is_file():
            source.replace(history / source.name)
    return str(history)


def _parse_artifacts(values: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        source, separator, target = value.partition("=")
        source_path = Path(source)
        target_path = Path(target)
        if not separator or not source or not target:
            raise ValueError(f"artifact must use source=target syntax: {value}")
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError(f"artifact source must stay inside the run directory: {source}")
        if target_path.name != target or target in {".", ".."}:
            raise ValueError(f"artifact target must be one fixed filename: {target}")
        result[source] = target
    return result


def read_job_status(state_dir: str | Path, job: str, *, refresh: bool = True) -> Dict[str, Any]:
    paths = _job_paths(state_dir, job)
    status = _load_json(paths["status"])
    if status is None:
        return _not_started_status(paths, job)
    pid = status.get("pid")
    if refresh and status.get("state") in ACTIVE_STATES and not _pid_is_alive(pid if isinstance(pid, int) else None):
        # Re-read and transition under the same lock used by launch_job.  A
        # stale status refresh must never overwrite a concurrently launched
        # monitor's new live PID.
        with paths["lock"].open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            current = _load_json(paths["status"])
            if current is None:
                return _not_started_status(paths, job)
            status = _refresh_dead_status_unlocked(paths, current)
    return status


def launch_job(
    *,
    job: str,
    state_dir: str | Path,
    cwd: str | Path,
    command: Sequence[str],
    run_root: str | Path,
    run_prefix: str,
    artifacts: Mapping[str, str] | None = None,
    exact_run_dir: str | Path | None = None,
) -> Dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job):
        raise ValueError("job name may contain only letters, digits, dot, underscore, and dash")
    if not command:
        raise ValueError("background command cannot be empty")
    working_directory = Path(cwd).resolve()
    if not working_directory.is_dir():
        raise FileNotFoundError(f"background working directory does not exist: {working_directory}")
    entrypoint = Path(__file__).resolve().parents[2] / "scripts" / "background_entrypoint.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"background manager entry point does not exist: {entrypoint}")
    output_root = Path(run_root)
    if not output_root.is_absolute():
        output_root = working_directory / output_root
    output_root = output_root.resolve()
    declared_run_dir: Path | None = None
    if exact_run_dir is not None:
        declared_run_dir = Path(exact_run_dir)
        if not declared_run_dir.is_absolute():
            declared_run_dir = working_directory / declared_run_dir
        declared_run_dir = declared_run_dir.resolve()
        try:
            declared_run_dir.relative_to(output_root)
        except ValueError as exc:
            raise ValueError("exact run directory must stay inside run_root") from exc
    paths = _job_paths(state_dir, job)
    paths["root"].mkdir(parents=True, exist_ok=True)

    with paths["lock"].open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = _load_json(paths["status"])
        if previous is None:
            previous = _not_started_status(paths, job)
        else:
            previous = _refresh_dead_status_unlocked(paths, previous)
        previous_pid = previous.get("pid")
        if previous.get("state") in ACTIVE_STATES and _pid_is_alive(
            previous_pid if isinstance(previous_pid, int) else None
        ):
            return {**previous, "launch_result": "already_running"}
        history_dir = _archive_previous(paths, job, previous)
        started_epoch = time.time()
        spec = {
            "job": job,
            "cwd": str(working_directory),
            "command": list(command),
            "command_display": shlex.join(command),
            "run_root": str(output_root),
            "run_prefix": run_prefix,
            "exact_run_dir": str(declared_run_dir) if declared_run_dir is not None else None,
            "artifacts": dict(artifacts or {}),
            "started_at": _utc_iso(started_epoch),
            "started_at_epoch": started_epoch,
            "log_file": str(paths["log"]),
            "status_file": str(paths["status"]),
            "exit_file": str(paths["exit"]),
            "run_dir_file": str(paths["run_dir"]),
            "history_dir": history_dir,
        }
        _atomic_write_json(paths["spec"], spec)
        worker_command = [sys.executable, str(entrypoint), "_worker", "--spec", str(paths["spec"])]
        bootstrap_log = paths["log"].open("a", encoding="utf-8")
        try:
            worker = subprocess.Popen(
                worker_command,
                cwd=working_directory,
                stdin=subprocess.DEVNULL,
                stdout=bootstrap_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=os.environ.copy(),
            )
        except Exception as exc:
            failed = {
                **spec,
                "state": "failed",
                "pid": None,
                "exit_code": None,
                "message": f"could not start background monitor: {type(exc).__name__}: {exc}",
            }
            _atomic_write_json(paths["status"], failed)
            raise
        finally:
            bootstrap_log.close()
        _atomic_write_text(paths["pid"], f"{worker.pid}\n")
        starting = {
            **spec,
            "state": "starting",
            "pid": worker.pid,
            "launch_result": "started",
            "run_dir": str(declared_run_dir) if declared_run_dir is not None else None,
        }
        _atomic_write_json(paths["status"], starting)
        deadline = time.monotonic() + 5.0
        monitor_ready = False
        while time.monotonic() < deadline:
            ready = _load_json(paths["ready"])
            if ready is not None and ready.get("pid") == worker.pid:
                monitor_ready = True
                break
            if not _pid_is_alive(worker.pid):
                break
            time.sleep(0.01)
        starting = {**starting, "monitor_ready": monitor_ready}
        _atomic_write_json(paths["status"], starting)
        return starting


def _discover_run_dir(run_root: Path, run_prefix: str, before: set[Path]) -> Path | None:
    if not run_root.is_dir():
        return None
    candidates = [path.resolve() for path in run_root.glob(f"{run_prefix}*") if path.is_dir()]
    created = [path for path in candidates if path not in before]
    if created:
        return max(created, key=lambda path: path.stat().st_mtime_ns)
    # Resume workers intentionally write into the exact pre-existing run
    # directory.  In that case there is no newly-created directory to
    # discover, but the exact prefix remains unambiguous.  Do not guess among
    # merely prefix-matching historical runs.
    exact = (run_root / run_prefix).resolve()
    return exact if exact in candidates else None


def _copy_artifacts(
    run_dir: Path | None,
    state_dir: Path,
    artifacts: Mapping[str, str],
) -> tuple[Dict[str, str], list[str], Dict[str, str]]:
    copied: Dict[str, str] = {}
    missing: list[str] = []
    errors: Dict[str, str] = {}
    for source_name, target_name in artifacts.items():
        source = run_dir / source_name if run_dir is not None else None
        target = state_dir / target_name
        if source is None or not source.is_file():
            missing.append(source_name)
            continue
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(target)
            copied[source_name] = str(target)
        except OSError as exc:
            errors[source_name] = f"{type(exc).__name__}: {exc}"
            with suppress(OSError):
                temporary.unlink()
    return copied, missing, errors


def run_worker(spec_path: str | Path) -> int:
    spec_file = Path(spec_path).resolve()
    spec = _load_json(spec_file)
    if spec is None:
        return 125
    job = str(spec["job"])
    paths = _job_paths(spec_file.parent, job)
    pid = os.getpid()
    run_root = Path(str(spec["run_root"]))
    before = {
        path.resolve()
        for path in run_root.glob(f"{spec['run_prefix']}*")
        if path.is_dir()
    } if run_root.is_dir() else set()
    declared_value = spec.get("exact_run_dir")
    declared_run_dir = Path(str(declared_value)).resolve() if declared_value else None
    command = [str(value) for value in spec["command"]]
    exit_code = 125
    error: str | None = None
    termination_signal: int | None = None
    child: subprocess.Popen[Any] | None = None
    termination_requested_at: float | None = None
    termination_forwarded = False

    def forward_termination(signum: int, _frame: Any) -> None:
        nonlocal termination_requested_at, termination_signal
        termination_signal = signum
        if termination_requested_at is None:
            termination_requested_at = time.monotonic()

    signal.signal(signal.SIGTERM, forward_termination)
    signal.signal(signal.SIGINT, forward_termination)
    _atomic_write_json(
        paths["ready"],
        {"job": job, "pid": pid, "ready_at": _utc_iso(), "ready_at_epoch": time.time()},
    )

    with paths["lock"].open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        running = {
            **spec,
            "state": "running",
            "pid": pid,
            "launch_result": "started",
            "run_dir": str(declared_run_dir) if declared_run_dir is not None else None,
        }
        _atomic_write_json(paths["status"], running)
        if declared_run_dir is not None:
            _atomic_write_text(paths["run_dir"], f"{declared_run_dir}\n")

    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    with paths["log"].open("w", encoding="utf-8", buffering=1) as log:
        log.write("=== BridgeTree background job ===\n")
        log.write(f"job: {job}\n")
        log.write(f"pid: {pid}\n")
        log.write(f"started_at: {spec['started_at']}\n")
        log.write(f"cwd: {spec['cwd']}\n")
        log.write(f"command: {spec['command_display']}\n")
        log.write("=== command output ===\n")
        log.flush()
        try:
            child = subprocess.Popen(
                command,
                cwd=str(spec["cwd"]),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            while child.poll() is None:
                if termination_signal is not None and not termination_forwarded:
                    ready = False
                    if declared_run_dir is not None:
                        ready_value = _load_json(declared_run_dir / WORKER_READY_FILE)
                        ready = bool(
                            ready_value is not None
                            and ready_value.get("pid") == child.pid
                        )
                    grace_elapsed = (
                        termination_requested_at is not None
                        and time.monotonic() - termination_requested_at >= 5.0
                    )
                    if ready or grace_elapsed:
                        with suppress(OSError):
                            child.send_signal(termination_signal)
                        termination_forwarded = True
                time.sleep(0.02)
            exit_code = int(child.returncode)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.write(f"\nbackground worker could not execute command: {error}\n")

        run_dir = (
            declared_run_dir
            if declared_run_dir is not None and declared_run_dir.is_dir()
            else _discover_run_dir(run_root, str(spec["run_prefix"]), before)
        )
        _atomic_write_text(paths["run_dir"], f"{run_dir or ''}\n")
        copied, missing, artifact_errors = _copy_artifacts(
            run_dir,
            paths["root"],
            {str(key): str(value) for key, value in dict(spec.get("artifacts", {})).items()},
        )
        finished_epoch = time.time()
        state = (
            "interrupted"
            if termination_signal is not None
            else ("completed" if exit_code == 0 else "failed")
        )
        final = {
            **spec,
            "state": state,
            "pid": pid,
            "exit_code": exit_code,
            "finished_at": _utc_iso(finished_epoch),
            "finished_at_epoch": finished_epoch,
            "duration_seconds": finished_epoch - float(spec["started_at_epoch"]),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "copied_artifacts": copied,
            "missing_artifacts": missing,
            "artifact_errors": artifact_errors,
            "error": error,
            "termination_signal": termination_signal,
            "termination_forwarded": termination_forwarded,
        }
        _atomic_write_text(paths["exit"], f"{exit_code}\n")
        _atomic_write_json(paths["status"], final)
        log.write("\n=== background job result ===\n")
        log.write(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and inspect detached BridgeTree server jobs")
    subparsers = parser.add_subparsers(dest="action", required=True)

    start = subparsers.add_parser("start", help="start a detached job or report the existing live job")
    start.add_argument("--job", required=True)
    start.add_argument("--state-dir", required=True)
    start.add_argument("--cwd", required=True)
    start.add_argument("--run-root", required=True)
    start.add_argument("--run-prefix", required=True)
    start.add_argument("--exact-run-dir")
    start.add_argument("--artifact", action="append", default=[])
    start.add_argument("command", nargs=argparse.REMAINDER)

    status = subparsers.add_parser("status", help="print the fixed status JSON")
    status.add_argument("--job", required=True)
    status.add_argument("--state-dir", required=True)

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--spec", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.action == "_worker":
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        return run_worker(args.spec)
    if args.action == "status":
        print(json.dumps(read_job_status(args.state_dir, args.job), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    result = launch_job(
        job=args.job,
        state_dir=args.state_dir,
        cwd=args.cwd,
        command=command,
        run_root=args.run_root,
        run_prefix=args.run_prefix,
        artifacts=_parse_artifacts(args.artifact),
        exact_run_dir=args.exact_run_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the wrappers
    raise SystemExit(main())
