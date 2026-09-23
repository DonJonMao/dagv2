"""Offline subprocess fixture. This is never imported by production entrypoints."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dagbt import runner


def fixture_worker(connection, dataset, arm, config):
    connection.send({"type": "ready", "hashes": {}})
    while True:
        try:
            task = connection.recv()
        except EOFError:
            return
        if task is None:
            return
        time.sleep(config.get("experiment", {}).get("fixture_delay", 0))
        if config.get("experiment", {}).get("fixture_fail_original") and arm == "original":
            connection.send({"type": "task_error", "error": "offline deliberate failure"})
            continue
        connection.send({"type": "result", "row": {
            "unit_id": task["question"]["id"], "answer": {"status": "ok", "prediction": "offline fixture"},
            "ranking": {}, "budgets": {str(k): {"selected_doc_ids": []} for k in (5, 10, 20)},
            "diagnostics": {"fixture_only": True, "worker_pid": os.getpid()}}})


if __name__ == "__main__":
    runner.native_worker = fixture_worker
    real_popen = runner.subprocess.Popen
    def fixture_popen(command, *args, **kwargs):
        if command[1:3] == ["-m", "dagbt.runner"]:
            command = [sys.executable, str(Path(__file__).resolve()), *command[3:]]
        return real_popen(command, *args, **kwargs)
    runner.subprocess.Popen = fixture_popen
    raise SystemExit(runner.main())
