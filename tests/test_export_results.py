"""Offline, synthetic fixtures for the small read-only result export bundle."""

import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from scripts import export_results as exporter


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _row_path(run, dataset, arm, unit):
    return run / dataset / arm / "rows" / (_digest(unit) + ".json")


def _write_row(run, unit, *, dataset="hotpotqa", arm="original", status="ok", prediction="Paris", **extra):
    return _write_json(_row_path(run, dataset, arm, unit), {
        "unit_id": unit,
        "answer": {"status": status, "prediction": prediction},
        "budgets": {str(k): {"selected_doc_ids": []} for k in (5, 10, 20)},
        **extra,
    })


def _fixture(tmp_path, scopes=None, arms=("original", "fusion")):
    root = tmp_path / "repository"
    run = root / "outputs" / "fixture-run"
    scopes = scopes or {"hotpotqa": ["q1", "q2"]}
    files = []
    questions_by_dataset = {}
    for dataset, units in scopes.items():
        questions = [{"id": unit, "question": "Which city?"} for unit in units]
        if dataset == "personamem":
            questions = [{**q, "options": ["tea", "coffee"], "persona_id": "persona-1"} for q in questions]
        questions_by_dataset[dataset] = questions
        directory = root / "data" / dataset
        _write_text(directory / "questions.jsonl", "".join(json.dumps(q) + "\n" for q in questions))
        labels = [{"id": unit, "answers": ["Paris"], "gold_groups": [["doc"]]} for unit in units]
        if dataset == "personamem":
            labels = [{"id": unit, "correct_answer": "(a)"} for unit in units]
        _write_json(directory / "evaluation_only.json", labels)
        for name in ("questions.jsonl", "evaluation_only.json"):
            path = directory / name
            files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if dataset == "personamem":
            _write_json(directory / "manifest.json", {
                "schema": 1,
                "public_sha256": {"questions.jsonl": files[-2]["sha256"]},
                "evaluation_sha256": files[-1]["sha256"],
            })
    original = _write_json(root / "original_manifest.json", {"files": files})
    _write_json(run / "manifest.json", {
        "schema": 1, "datasets": list(scopes), "arms": list(arms), "question_ids": scopes,
        "questions_digest": {dataset: _digest(questions) for dataset, questions in questions_by_dataset.items()},
        "original_hash_manifest": hashlib.sha256(original.read_bytes()).hexdigest(),
        "dataset_manifests": {
            dataset: hashlib.sha256((root / "data" / dataset / "manifest.json").read_bytes()).hexdigest()
            for dataset in scopes if dataset == "personamem"
        },
        "config": {"model_profile": "fixture", "api_key": "fixture-api-key-do-not-export"},
    })
    return root, run, tmp_path / "bundle.tar.gz"


def _archive(destination):
    with tarfile.open(destination, "r:gz") as archive:
        return {member.name: archive.extractfile(member).read()
                for member in archive.getmembers() if member.isfile()}


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


def _answers(archive):
    return [json.loads(line) for line in archive["answers.jsonl"].decode("utf-8").splitlines() if line]


def test_export_current_answers_preserves_sources_and_reports_missing_and_failures(tmp_path):
    root, run, destination = _fixture(tmp_path)
    good = _write_row(run, "q1", prediction="巴黎 — café 🌲")
    failed = _write_row(run, "q2", status="timeout", prediction="")
    failed_row = json.loads(failed.read_text())
    failed_row["answer"].update(error_type="TimeoutError", error="fixture timeout")
    _write_json(failed, failed_row)
    _write_row(run, "q1", arm="fusion", prediction="Paris")
    # Historical attempts and rows outside manifest scope are never authoritative.
    _write_row(run, "unrequested", prediction="HISTORICAL-OR-OUT-OF-SCOPE")
    _write_json(run / "hotpotqa/original/attempts/old/result.json", failed_row)
    _write_json(run / "hotpotqa/original/failures/old.json", failed_row)
    _write_json(run / "requests/request.json", {"api_key": "REQUEST-CACHE-SECRET"})
    _write_json(root / "config.local.json", {"api_key": "LOCAL-CONFIG-SECRET"})
    _write_json(run / "progress.json", {"state": "generating"})
    _write_text(run / "events.jsonl", '{"event":"started"}\n')
    _write_text(run / "launcher.log", "fixture runner started\n")
    before = _snapshot(root)

    report = exporter.export_bundle(root, run, destination)

    assert _snapshot(root) == before
    archive = _archive(destination)
    records = _answers(archive)
    assert len(records) == 3
    assert {(r["arm"], r["unit_id"]) for r in records} == {
        ("original", "q1"), ("original", "q2"), ("fusion", "q1"),
    }
    result = next(r for r in records if r["arm"] == "original" and r["unit_id"] == "q1")
    assert result["answer"]["prediction"] == "巴黎 — café 🌲"
    assert result["source"]["relative_path"] == str(good.relative_to(run))
    assert result["source"]["bytes"] == good.stat().st_size
    assert result["source"]["mtime_ns"] == good.stat().st_mtime_ns
    assert len(result["source"]["answer_sha256"]) == 64
    assert int(result["source"]["answer_sha256"], 16) >= 0
    failure = next(r for r in records if r["unit_id"] == "q2")
    assert failure["answer"] == {
        "status": "timeout", "prediction": "", "error_type": "TimeoutError", "error": "fixture timeout",
    }
    assert report["counts"]["hotpotqa"]["original"] == {
        "expected": 2, "exported": 2, "status_ok": 1, "status_failed": 1, "missing": 0, "invalid": 0,
    }
    assert report["counts"]["hotpotqa"]["fusion"] == {
        "expected": 2, "exported": 1, "status_ok": 1, "status_failed": 0, "missing": 1, "invalid": 0,
    }
    assert report["errors"] == []
    assert {"answers.jsonl", "report.json", "run/manifest.json", "run/progress.json", "run/events.jsonl",
            "run/launcher.log", "original_manifest.json", "data/hotpotqa/questions.jsonl",
            "data/hotpotqa/evaluation_only.json"} <= set(archive)
    exported_content = b"\n".join(archive.values())
    for forbidden in (b"fixture-api-key-do-not-export", b"REQUEST-CACHE-SECRET", b"LOCAL-CONFIG-SECRET",
                      b"HISTORICAL-OR-OUT-OF-SCOPE"):
        assert forbidden not in exported_content
    assert not any("attempts/" in name or "requests/" in name or "failures/" in name for name in archive)
    assert not list(destination.parent.glob(destination.name + "*.partial"))


def test_read_answer_handles_unicode_escapes_and_ignores_nested_fake_answer(tmp_path):
    answer = {"status": "ok", "prediction": '北京 🌲 café: "answer": {fake}\\path\nnext line',
              "extra": {"details": ["nested", {"answer": "still inside actual answer"}]}}
    path = _write_json(tmp_path / "row.json", {
        "unit_id": "题目一",
        "metadata": {"answer": {"status": "ok", "prediction": "WRONG NESTED ANSWER"}},
        "quoted": '  "answer": { "status": "ok", "prediction": "WRONG STRING" }',
        "answer": answer,
        "diagnostics": {"full_text": "unneeded"},
    })
    assert exporter.read_answer(path) == answer


@pytest.mark.parametrize("text", [
    '{\n  "unit_id": "q1",\n  "answer": {\n    "status": "ok",\n    "prediction": "unterminated',
    '{\n  "unit_id": "q1",\n  "answer": ["ok", "Paris"]\n}\n',
    '{\n  "unit_id": "q1",\n  "metadata": {\n    "answer": {"status": "ok", "prediction": "fake"}\n  }\n}\n',
])
def test_read_answer_rejects_broken_or_nonobject_answer(tmp_path, text):
    path = _write_text(tmp_path / "row.json", text)
    with pytest.raises((ValueError, RuntimeError)):
        exporter.read_answer(path)


def test_large_diagnostics_after_answer_are_neither_read_nor_packaged(tmp_path, monkeypatch):
    root, run, destination = _fixture(tmp_path, {"hotpotqa": ["q1"]}, ("original",))
    path = _write_row(run, "q1", diagnostics={"unneeded": "DIAGNOSTIC-ONLY-" * 100_000})
    original_open = Path.open
    consumed = []

    class ReadBudget:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def _record(self, value):
            consumed.append(len(value))
            assert sum(consumed) < 64 * 1024, "Exporter read the full diagnostic payload"
            return value

        def read(self, size=-1):
            assert size >= 0, "Exporter requested the whole result file"
            return self._record(self.handle.read(size))

        def readline(self, size=-1):
            return self._record(self.handle.readline(size))

        def __iter__(self):
            return self

        def __next__(self):
            line = self.readline()
            if not line:
                raise StopIteration
            return line

        def __getattr__(self, name):
            return getattr(self.handle, name)

    def guarded_open(self, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        return ReadBudget(handle) if self == path else handle

    monkeypatch.setattr(Path, "open", guarded_open)
    exporter.export_bundle(root, run, destination)
    assert consumed and sum(consumed) < 64 * 1024
    archive = _archive(destination)
    assert _answers(archive)[0]["answer"]["prediction"] == "Paris"
    assert b"DIAGNOSTIC-ONLY" not in b"\n".join(archive.values())


def test_invalid_current_rows_are_reported_without_fabricated_answers(tmp_path):
    units = ["good", "broken", "wrong-type", "missing-answer", "pending"]
    root, run, destination = _fixture(tmp_path, {"hotpotqa": units}, ("original",))
    _write_row(run, "good")
    _write_text(_row_path(run, "hotpotqa", "original", "broken"),
                '{\n  "unit_id": "broken",\n  "answer": {\n    "status": "ok",\n')
    _write_json(_row_path(run, "hotpotqa", "original", "wrong-type"),
                {"unit_id": "wrong-type", "answer": {"status": 123, "prediction": "Paris"}})
    _write_json(_row_path(run, "hotpotqa", "original", "missing-answer"),
                {"unit_id": "missing-answer", "diagnostics": {"answer": {"status": "ok", "prediction": "fake"}}})
    before = _snapshot(root)

    report = exporter.export_bundle(root, run, destination)

    assert _snapshot(root) == before
    assert [r["unit_id"] for r in _answers(_archive(destination))] == ["good"]
    assert report["counts"]["hotpotqa"]["original"] == {
        "expected": 5, "exported": 1, "status_ok": 1, "status_failed": 0, "missing": 1, "invalid": 3,
    }
    assert {e["unit_id"] for e in report["errors"] if "unit_id" in e} == {
        "broken", "wrong-type", "missing-answer",
    }
    assert all(error["error"] for error in report["errors"])


def test_only_datasets_with_exported_answers_read_questions_and_gold(tmp_path, monkeypatch):
    root, run, destination = _fixture(tmp_path, {"hotpotqa": ["q1"], "personamem": ["p1"]}, ("original",))
    _write_row(run, "q1")
    blocked = root / "data/personamem"
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        assert blocked not in self.parents, "Unstarted dataset data must remain unread"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = exporter.export_bundle(root, run, destination)
    archive = _archive(destination)
    assert not any(name.startswith("data/personamem/") for name in archive)
    assert report["counts"]["personamem"]["original"]["missing"] == 1
    assert report["counts"]["personamem"]["original"]["exported"] == 0


def test_personamem_exports_options_labels_and_provenance_manifest(tmp_path):
    root, run, destination = _fixture(tmp_path, {"personamem": ["p1"]}, ("original",))
    _write_row(run, "p1", dataset="personamem", prediction="(a)")
    report = exporter.export_bundle(root, run, destination)
    archive = _archive(destination)
    assert report["errors"] == []
    assert {"data/personamem/questions.jsonl", "data/personamem/evaluation_only.json",
            "data/personamem/manifest.json"} <= set(archive)
    questions = json.loads(archive["data/personamem/questions.jsonl"])
    assert questions["persona_id"] == "persona-1" and questions["options"] == ["tea", "coffee"]
    assert _answers(archive)[0]["answer"]["prediction"] == "(a)"


def test_output_inside_source_run_is_rejected_including_symlink_alias(tmp_path):
    root, run, _ = _fixture(tmp_path)
    alias = tmp_path / "run-alias"
    alias.symlink_to(run, target_is_directory=True)
    before = _snapshot(root)
    for destination in (run / "bundle.tar.gz", alias / "bundle.tar.gz"):
        with pytest.raises((ValueError, RuntimeError)):
            exporter.export_bundle(root, run, destination)
    assert _snapshot(root) == before


def test_existing_destination_and_partial_are_never_overwritten(tmp_path):
    root, run, destination = _fixture(tmp_path)
    destination.write_bytes(b"existing user archive")
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        exporter.export_bundle(root, run, destination)
    assert destination.read_bytes() == b"existing user archive"
    destination.unlink()
    partial = destination.with_name(destination.name + ".partial")
    partial.write_bytes(b"existing user partial")
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        exporter.export_bundle(root, run, destination)
    assert partial.read_bytes() == b"existing user partial"
    assert not destination.exists()


@pytest.mark.parametrize("field,value", [
    ("datasets", "../outside"), ("datasets", "/tmp/outside"), ("arms", "../outside"),
    ("arms", "nested/arm"),
])
def test_unsafe_manifest_path_components_are_rejected(tmp_path, field, value):
    root, run, destination = _fixture(tmp_path)
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = [value]
    if field == "datasets":
        manifest["question_ids"] = {value: ["q1"]}
    _write_json(path, manifest)
    with pytest.raises((ValueError, RuntimeError)):
        exporter.export_bundle(root, run, destination)
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_logs_are_bounded_utf8_tails_and_bearer_credentials_are_redacted(tmp_path):
    root, run, destination = _fixture(tmp_path, {"hotpotqa": ["q1"]}, ("original",))
    _write_row(run, "q1")
    for name in ("events.jsonl", "launcher.log"):
        _write_text(run / name,
                    "SHOULD-NOT-BE-IN-TAIL\n" + "过程记录\n" * exporter.LOG_TAIL_BYTES
                    + "Authorization: Bearer fixture-bearer-secret\nLAST-EVENT\n")
    before = _snapshot(root)
    report = exporter.export_bundle(root, run, destination)
    assert _snapshot(root) == before
    archive = _archive(destination)
    for name in ("events.jsonl", "launcher.log"):
        content = archive["run/" + name]
        assert len(content) <= exporter.LOG_TAIL_BYTES
        assert b"SHOULD-NOT-BE-IN-TAIL" not in content
        assert b"fixture-bearer-secret" not in content
        assert b"[REDACTED]" in content and content.endswith(b"LAST-EVENT\n")
        assert "过程记录" in content.decode("utf-8")
        assert report["logs"][name]["tail_only"] is True


def test_result_symlink_escaping_run_is_an_invalid_row(tmp_path):
    root, run, destination = _fixture(tmp_path, {"hotpotqa": ["q1"]}, ("original",))
    outside = _write_json(tmp_path / "outside.json", {
        "unit_id": "q1", "answer": {"status": "ok", "prediction": "OUTSIDE-PRIVATE-TEXT"},
    })
    row = _row_path(run, "hotpotqa", "original", "q1")
    row.parent.mkdir(parents=True)
    row.symlink_to(outside)
    report = exporter.export_bundle(root, run, destination)
    archive = _archive(destination)
    assert _answers(archive) == []
    assert report["counts"]["hotpotqa"]["original"]["invalid"] == 1
    assert report["counts"]["hotpotqa"]["original"]["missing"] == 0
    assert report["errors"][0]["unit_id"] == "q1"
    assert b"OUTSIDE-PRIVATE-TEXT" not in b"\n".join(archive.values())
    assert row.is_symlink()


@pytest.mark.parametrize("changed", ["original_manifest.json", "data/hotpotqa/questions.jsonl",
                                     "data/hotpotqa/evaluation_only.json"])
def test_changed_data_identity_prevents_publishing_bundle(tmp_path, changed):
    root, run, destination = _fixture(tmp_path, {"hotpotqa": ["q1"]}, ("original",))
    _write_row(run, "q1")
    path = root / changed
    path.write_bytes(path.read_bytes() + b"\n")
    before = _snapshot(root)
    with pytest.raises(ValueError, match="manifest|checksum|identity"):
        exporter.export_bundle(root, run, destination)
    assert _snapshot(root) == before
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_archive_failure_removes_only_own_partial_and_never_publishes(tmp_path, monkeypatch):
    root, run, destination = _fixture(tmp_path, {"hotpotqa": ["q1"]}, ("original",))
    _write_row(run, "q1")
    before = _snapshot(root)
    observed_partial = []

    def fail_addfile(*args, **kwargs):
        observed_partial.append(destination.with_name(destination.name + ".partial").exists())
        assert not destination.exists()
        raise OSError("synthetic archive write failure")

    monkeypatch.setattr(tarfile.TarFile, "addfile", fail_addfile)
    with pytest.raises(OSError, match="synthetic archive write failure"):
        exporter.export_bundle(root, run, destination)
    assert observed_partial == [True]
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()
    assert _snapshot(root) == before


def test_concurrent_destination_creation_never_overwrites_existing_file(tmp_path, monkeypatch):
    root, run, destination = _fixture(tmp_path, {"hotpotqa": ["q1"]}, ("original",))
    _write_row(run, "q1")
    original_link = exporter.os.link

    def racing_link(source, target, *args, **kwargs):
        assert not destination.exists()
        # The temporary archive is complete before its final name can be visible.
        assert _answers(_archive(source))[0]["unit_id"] == "q1"
        destination.write_bytes(b"concurrent user archive")
        return original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(exporter.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        exporter.export_bundle(root, run, destination)
    assert destination.read_bytes() == b"concurrent user archive"
    assert not destination.with_name(destination.name + ".partial").exists()


def test_export_total_size_limit_prevents_archive_creation(tmp_path, monkeypatch):
    root, run, destination = _fixture(tmp_path, {"hotpotqa": ["q1"]}, ("original",))
    _write_row(run, "q1")
    monkeypatch.setattr(exporter, "MAX_TOTAL_BYTES", 100)
    with pytest.raises(ValueError, match="limit"):
        exporter.export_bundle(root, run, destination)
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()
