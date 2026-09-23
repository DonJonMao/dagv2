"""Detached, resumable diagnostic orchestration; never trains model weights.

The existing background manager owns the detached session and process log.
This module owns only diagnostic phase order, heartbeat snapshots and safe
control commands. Every model call still goes through the frozen runners.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

from .background import ACTIVE_STATES, launch_job, read_job_status
from .diagnostic_runner import atomic_json, append_jsonl, load_manifest


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _error_fields(exc: BaseException) -> dict:
    # Exception strings can contain provider payloads, credentials or URLs.
    # Full identity gates are already recorded by the frozen runners.
    return {"error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)}


def _read_optional(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class _Heartbeat:
    def __init__(self, root: Path, interval: float):
        self.root, self.interval = root, interval
        self.phase, self.state = "planning", "running"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="diagnostic-heartbeat", daemon=True)
        self.started = False

    def event(self, event: str, **fields: Any) -> None:
        row = {"at_epoch": time.time(), "at": _timestamp(), "event": event,
               "phase": self.phase, "state": self.state, **fields}
        with self.lock:
            append_jsonl(self.root / "runtime.jsonl", row)
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            with (self.root / "run.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
            print(line, flush=True)

    def snapshot(self) -> None:
        if not (self.root / "manifest.json").is_file():
            self.event("heartbeat", manifest_ready=False)
            return
        from .diagnostic_progress import write_progress
        progress = write_progress(self.root, phase=self.phase, state=self.state)
        phases = progress.get("phases", {})
        compact = {name: {k: value.get(k) for k in ("planned", "success", "failed", "pending", "unknown")}
                   for name, value in phases.items() if isinstance(value, dict)}
        self.event("heartbeat", progress=compact)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self.snapshot()
            except Exception as exc:
                # Derived progress can lag an atomic write. It never changes
                # a trial's outcome or silently drops authoritative ledgers.
                try:
                    self.event("progress_snapshot_error", **_error_fields(exc))
                except Exception:
                    print("diagnostic heartbeat persistence failed", file=sys.stderr, flush=True)

    def close(self) -> None:
        self.stop_event.set()
        if self.started:
            # Snapshot performs local reads only. Join before the final
            # snapshot so an old heartbeat cannot overwrite terminal state.
            self.thread.join()


def run_pipeline(config_path: str | Path, run_dir: str | Path, *, resume: bool = False,
                 offline: bool = False, heartbeat_seconds: float = 5.0) -> dict:
    """Run all frozen phases once, retaining per-item retry/physical budgets.

    Offline mode only plans, recovers historical scores, evaluates pending
    rows, and produces reports. It makes zero embedding/rerank/generation
    requests and is deliberately named ``offline_complete``.
    """
    from .diagnostic_config import load_diagnostic_config
    from .diagnostic_evaluation import evaluate_diagnostics, unified_diagnostic_report
    from .diagnostic_observability import ModuleEventRecorder, observation_scope, observe
    from .diagnostic_root_runner import run_root_diagnostics
    from .diagnostic_runner import (analyze_diagnostics, plan_diagnostics, run_generations,
                                    online_identity_ready, run_scores, validate_online)

    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be a positive finite number")
    root, config_path = Path(run_dir).resolve(), Path(config_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".diagnostic.pipeline.lock").open("a") as pipeline_lock:
        try:
            fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another diagnostic pipeline owns this run") from exc
        if (root / "manifest.json").exists() and not resume:
            raise ValueError("run already has a manifest; use resume or a new directory")
        if resume and not (root / "manifest.json").exists():
            raise ValueError("cannot resume without a frozen manifest")
        heartbeat = _Heartbeat(root, heartbeat_seconds)
        started = time.time()
        phases: list[dict] = []
        manifest = None
        status, error = "failed", None
        recorder = ModuleEventRecorder(root)
        ready_path = root / ".background_worker_ready.json"
        old_sigterm = signal.getsignal(signal.SIGTERM)

        def interrupt(_signum, _frame):
            raise KeyboardInterrupt("diagnostic stop requested")

        def stage(name, action):
            heartbeat.phase = name
            heartbeat.event("phase_started")
            observe("execution", "phase_started", phase=name)
            begin = time.monotonic()
            result = action()
            summary = {"phase": name, "status": "completed", "elapsed_ms": (time.monotonic() - begin) * 1000}
            if isinstance(result, dict):
                summary.update({key: result[key] for key in ("planned", "success", "failed", "subset_count",
                    "historical_score_count", "missing_score_count") if key in result})
            phases.append(summary)
            heartbeat.event("phase_completed", **{k: v for k, v in summary.items() if k != "phase"})
            observe("execution", "phase_completed", **summary)
            return result

        try:
            signal.signal(signal.SIGTERM, interrupt)
            atomic_json(ready_path, {"pid": os.getpid(), "ready_at_epoch": time.time()})
            heartbeat.thread.start()
            heartbeat.started = True
            with observation_scope(recorder):
                heartbeat.event("pipeline_started", resume=resume, offline=offline, weights_updated=False,
                                optimizer_steps=0)
                settings, _ = load_diagnostic_config(config_path)
                if resume:
                    manifest = stage("manifest", lambda: load_manifest(root))
                    # Empty role list still checks source and deployment drift,
                    # while permitting an explicitly offline unknown identity.
                    stage("resume_identity", lambda: validate_online(manifest, config_path, ()))
                else:
                    stage("planning", lambda: plan_diagnostics(config_path, root))
                    manifest = load_manifest(root)
                recorder.run_identity = manifest["manifest_id"]
                heartbeat.event("frozen_plan", manifest_id=manifest["manifest_id"],
                                counts=manifest["counts"], budgets=manifest["budgets"])
                stage("historical_analysis", lambda: analyze_diagnostics(root))
                if not offline:
                    try:
                        stage("online_identity", lambda: validate_online(
                            manifest, config_path, ("embedding", "reranker", "generator")))
                    except Exception as exc:
                        atomic_json(root / "online_preflight.json", {
                            "status": "blocked_before_network", "manifest_id": manifest["manifest_id"],
                            "error": _error_fields(exc), "missing_identity_roles":
                            online_identity_ready(manifest["deployment"], ("embedding", "reranker", "generator")),
                            "network_calls": 0})
                        raise
                    stage("score", lambda: run_scores(root, config_path, execute=True))
                    stage("fresh_analysis", lambda: analyze_diagnostics(root, score_view="fresh"))
                    stage("generation", lambda: run_generations(root, config_path, execute=True))
                stage("evaluation", lambda: evaluate_diagnostics(root, settings["questions"]))
                if not offline:
                    stage("root", lambda: run_root_diagnostics(root, config_path, execute=True))
                stage("report", lambda: unified_diagnostic_report(root))
                status = ("offline_complete" if offline else "completed_with_failures"
                          if any(p.get("failed", 0) for p in phases) else "completed")
        except KeyboardInterrupt as exc:
            status, error = "interrupted", _error_fields(exc)
        except Exception as exc:
            status, error = "failed", _error_fields(exc)
        finally:
            # Subsequent TERM cannot interrupt atomic finalization. SIGKILL
            # and host loss still recover from the durable per-trial ledger.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            heartbeat.state = status
            heartbeat.close()
            if manifest is not None:
                try:
                    unified_diagnostic_report(root)
                except Exception as exc:
                    status = "failed" if status != "interrupted" else status
                    error = {**(error or {}), "report_error_type": type(exc).__name__}
            completion = {"schema_version": 1, "status": status, "run_dir": str(root),
                          "manifest_id": manifest["manifest_id"] if manifest else None,
                          "offline": offline, "weights_updated": False, "optimizer_steps": 0,
                          "started_at_epoch": started, "finished_at_epoch": time.time(),
                          "last_phase": heartbeat.phase, "phases": phases, "error": error}
            try:
                heartbeat.state = status
                try:
                    heartbeat.event("pipeline_finished", completion_status=status, error=error)
                except Exception as exc:
                    status = "interrupted" if status == "interrupted" else "failed"
                    completion.update(status=status, error={**(error or {}),
                                      "runtime_log_error_type": type(exc).__name__})
                    heartbeat.state = status
                atomic_json(root / "completion.json", completion)
                try:
                    heartbeat.snapshot()
                except Exception as exc:
                    # progress.json is derived, not the authoritative
                    # outcome. A refresh failure must not contradict it.
                    with suppress(Exception):
                        heartbeat.event("progress_snapshot_error", **_error_fields(exc))
            finally:
                with suppress(OSError):
                    ready_path.unlink()
                signal.signal(signal.SIGTERM, old_sigterm)
        return completion


def _paths(args) -> tuple[Path, Path, Path]:
    repo = Path(__file__).resolve().parents[2]
    state = Path(args.state_dir or os.environ.get("DIAGNOSTIC_STATE_DIR", repo / "outputs/background-diagnostics")).expanduser().resolve()
    root = Path(args.run_root or os.environ.get("DIAGNOSTIC_RUN_ROOT", repo / "outputs/diagnostics")).expanduser().resolve()
    if any(p in {Path("/"), Path.home().resolve(), repo} or p in repo.parents for p in (state, root)):
        raise ValueError("state and output paths must not be filesystem, home or repository roots")
    if state == root or state in root.parents or root in state.parents:
        raise ValueError("state directory and run root must be separate, non-nested paths")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.job):
        raise ValueError("invalid diagnostic job name")
    return repo, state, root


def _previous_args(status: dict) -> list[str]:
    return [str(v) for v in status.get("command", [])]


def _arg_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _management_commands(repo: Path, state: Path, run_root: Path, job: str) -> dict:
    return {action: shlex.join(["env", "BRIDGETREE_BASE_PYTHON=" + sys.executable,
            "bash", str(repo / "scripts/run_diagnostics.sh"), action,
            "--state-dir", str(state), "--run-root", str(run_root), "--job", job])
            for action in ("status", "log", "module-log", "stop", "resume")}


def _checked_status(state: Path, job: str) -> dict:
    path = state / f"{job}.status.json"
    if path.is_symlink():
        raise ValueError("refusing symlink diagnostic status")
    if path.exists():
        value = _read_optional(path)
        if not value or value.get("job") != job or value.get("state") not in {
                "starting", "running", "completed", "failed", "interrupted"}:
            raise ValueError("cannot validate diagnostic status; refusing to treat it as a fresh job")
        if value["state"] in ACTIVE_STATES and (
                not isinstance(value.get("pid"), int) or isinstance(value["pid"], bool) or value["pid"] <= 0):
            raise ValueError("cannot validate active diagnostic monitor PID")
    return read_job_status(state, job)


def _completion_exit(result: dict) -> int:
    return (0 if result["status"] in {"completed", "completed_with_failures", "offline_complete"}
            else 130 if result["status"] == "interrupted" else 1)


def _verified_stop(status: dict, state: Path, job: str) -> dict:
    pid = status.get("pid")
    if status.get("state") not in ACTIVE_STATES or not isinstance(pid, int) or pid <= 0:
        return {"action": "stop", "result": "not_running"}
    spec = str(state / f"{job}.spec.json")
    proc = Path(f"/proc/{pid}/cmdline")
    try:
        if proc.is_file():
            command = proc.read_bytes().decode(errors="replace").split("\0")
            correct = spec in command and "_worker" in command and any(
                v.endswith("background_entrypoint.py") for v in command)
        else:
            value = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True,
                                   text=True, check=False).stdout
            correct = all(v in value for v in (spec, "_worker", "background_entrypoint.py"))
        if not correct or os.getsid(pid) != pid:
            raise RuntimeError("cannot verify the recorded detached monitor; refusing to signal this PID")
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"action": "stop", "result": "already_exited"}
    return {"action": "stop", "result": "graceful_stop_requested", "pid": pid}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="后台运行 PR1–PR4 诊断；不训练或更新模型权重")
    parser.add_argument("action", choices=["start", "resume", "status", "log", "module-log", "stop", "preflight", "_worker"])
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--state-dir")
    parser.add_argument("--run-root")
    parser.add_argument("--job", default="diagnostics")
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--module", default="events", choices=["events", "execution", "proposal", "scoring",
                        "activation", "state", "selection", "stop", "context"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", dest="execution_mode", action="store_const", const="offline")
    mode.add_argument("--online", dest="execution_mode", action="store_const", const="online")
    parser.add_argument("--resume-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.umask(0o077)
    try:
        if args.action == "_worker":
            if not args.config or not args.run_dir:
                raise ValueError("worker requires config and run directory")
            result = run_pipeline(args.config, args.run_dir, resume=args.resume_worker,
                                  offline=args.execution_mode == "offline", heartbeat_seconds=args.heartbeat_seconds)
            return _completion_exit(result)
        repo, state, run_root = _paths(args)
        status = _checked_status(state, args.job)
        if args.action == "resume" and not args.run_root and not os.environ.get("DIAGNOSTIC_RUN_ROOT"):
            recorded_root = status.get("run_root")
            if recorded_root:
                args.run_root = recorded_root
                repo, state, run_root = _paths(args)
        if args.action == "status":
            run = Path(status["run_dir"]) if status.get("run_dir") else None
            result = {**status, "progress": _read_optional(run / "progress.json") if run else None,
                      "completion": _read_optional(run / "completion.json") if run else None}
        elif args.action == "stop":
            result = _verified_stop(status, state, args.job)
        elif args.action in {"log", "module-log"}:
            if args.action == "log":
                path = state / f"{args.job}.log"
            else:
                if not status.get("run_dir"):
                    raise ValueError("no recorded diagnostic run directory")
                path = Path(status["run_dir"]) / "modules" / f"{args.module}.jsonl"
            if not path.parent.exists():
                raise ValueError("no diagnostic log directory yet")
            os.execvp("tail", ["tail", "-n", "100", "-F", str(path)])
            return 0
        else:
            if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
                raise ValueError("heartbeat seconds must be positive and finite")
            if status.get("state") in ACTIVE_STATES and args.action != "preflight":
                result = {**status, "launch_result": "already_running"}
            else:
                resume = args.action == "resume"
                previous = _previous_args(status)
                config = Path(args.config or (_arg_value(previous, "--config") if resume else None)
                              or repo / "configs/diagnostic_28.server.yaml").expanduser().resolve()
                if not config.is_file():
                    raise ValueError("configuration missing: copy diagnostic_28.yaml to a server config and pass --config")
                if args.action == "start" and status.get("state") in {"failed", "interrupted"} and not args.run_dir:
                    raise ValueError("previous run failed/interrupted; use resume, or explicitly select a new --run-dir")
                raw_run = args.run_dir or (status.get("run_dir") if resume else None)
                run = Path(raw_run).expanduser().resolve() if raw_run else run_root / ("diagnostic_" +
                    datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
                if resume and not raw_run:
                    raise ValueError("no previous run to resume")
                if run == run_root or run_root not in run.parents:
                    raise ValueError("run directory must be a child of run root")
                if not resume and run.exists() and any(run.iterdir()):
                    raise ValueError("run directory is not empty; use resume or a new run directory")
                offline = (args.action == "preflight" or args.execution_mode == "offline" or
                           (args.execution_mode is None and resume and "--offline" in previous))
                if args.action == "preflight":
                    result = run_pipeline(config, run, offline=True, heartbeat_seconds=args.heartbeat_seconds)
                else:
                    command = [sys.executable, "-u", "-m", "bridgetree.diagnostic_background", "_worker",
                               "--config", str(config), "--run-dir", str(run),
                               "--heartbeat-seconds", str(args.heartbeat_seconds)]
                    if resume:
                        command.append("--resume-worker")
                    if offline:
                        command.append("--offline")
                    result = launch_job(job=args.job, state_dir=state, cwd=repo, command=command,
                        run_root=run_root, run_prefix=run.name, exact_run_dir=run,
                        artifacts={"progress.json": f"{args.job}.progress.json",
                                   "completion.json": f"{args.job}.completion.json",
                                   "diagnostic_report.json": f"{args.job}.report.json"})
                    result["commands"] = _management_commands(repo, state, run_root, args.job)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return _completion_exit(result) if args.action == "preflight" else 0
    except Exception as exc:
        # Validation text generated here contains paths/flags, not config
        # payloads; errors raised by a model remain structured in its ledger.
        print(json.dumps({"status": "error", **_error_fields(exc),
                          "message": str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else None},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
