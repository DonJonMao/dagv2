#!/usr/bin/env python3
"""Private, append-free BT trace freezing and exact-input APC experiments.

No production scorer/cache/budget changes. Structural estimates never become
engine measurements. Every command creates a new output directory.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import statistics
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 1e-6, 1e-5  # dagbt.bridge.probe_reranker_protocol defaults, frozen before runs
LOCAL_ALGORITHM = 'dagbt_local_terminal_v1'
LEGACY_ALGORITHM = 'dagbt_fusion_reliability_v3'


def algorithm_identity(row):
    version = row.get('algorithm_version', LEGACY_ALGORITHM)
    if version not in (LOCAL_ALGORITHM, LEGACY_ALGORITHM):
        raise ValueError('Unknown algorithm identity')
    if version == LOCAL_ALGORITHM and not row.get('scoring_context_id'):
        raise ValueError('Local scoring context missing')
    return version, row.get('scoring_context_id')


def score_bank_key(payload, row):
    version, context = algorithm_identity(row)
    task = [payload['query'], payload['documents'], payload.get('instruction')]
    return digest([version, context, *task]) if version == LOCAL_ALGORITHM else digest(task)


def transport_record_identity(record):
    return {k:record[k] for k in ('unit_id','url','payload','algorithm_version','scoring_context_id') if k in record}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                yield number, json.loads(line)


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(path, 0o700)
    return path


def write(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def select_questions(ids, limit=50):
    # Independent of scores, success, prefix overlap, and log filesystem order.
    return sorted(set(ids), key=lambda x: (digest(x), x))[:limit]


def synthetic_requests():
    """Synthetic API cases only, explicitly not production token/benefit evidence."""
    shared = "Synthetic anchor: the launch date was in spring.\n" * 256
    a = 'Factual evidence passages:\n[Passage {"doc_id": "a", "source_id": "a"}]\n' + shared
    later = lambda tag: '\n\n[Passage {"doc_id": "' + tag + '", "source_id": "' + tag + '"}]\nSynthetic tail ' + tag
    cases = [
        ("long-prefix-different-sets", "Synthetic date question?", [a + later("b"), a + later("c")]),
        ("prior-completed-reuse", "Synthetic date question?", [a + later("d")]),
        ("query-only", "Synthetic date question?", ["Synthetic short unrelated content."]),
        ("system-only-negative", "Different synthetic geography question?", ["A desert contains stones."]),
        ("middle-document-different-context", "Synthetic date question?", ["Changed early context.\n" + a + later("b")]),
        ("canonical-early-divergence", "Synthetic date question?", ["Earlier synthetic passage z.\n" + a]),
        ("empty-set", "Synthetic date question?", ["[No evidence passages]"]),
        ("singleton", "Synthetic date question?", [a]),
        ("pair", "Synthetic date question?", [a + later("b")]),
        ("long-premises", "Synthetic date question?", [a + later("b") + later("c") + later("d")]),
        ("native-batch", "Synthetic date question?", [a, a + later("b"), a + later("c"), "[No evidence passages]"]),
        ("reversed-batch", "Synthetic date question?", ["[No evidence passages]", a + later("c"), a + later("b"), a]),
    ]
    return [{"ordinal": i, "case": name, "question_id": "synthetic:" + name,
             "payload": {"query": query, "documents": docs, "top_n": len(docs), "return_documents": False},
             "final_inputs": None, "kind": "synthetic_protocol_only"} for i, (name, query, docs) in enumerate(cases)]


def validate_payload(p):
    if not isinstance(p.get("query"), str) or not isinstance(p.get("documents"), list):
        raise ValueError("Full query/documents required; ID-only exports cannot be replayed")
    if not p["documents"] or not all(isinstance(d, str) for d in p["documents"]):
        raise ValueError("A physical rerank request requires complete serialized samples")
    if len(p["documents"]) > 4 or p.get("top_n") != len(p["documents"]):
        raise ValueError("BT physical batch/top_n contract differs")
    if set(p) - {"query", "documents", "model", "instruction", "top_n", "return_documents"}:
        raise ValueError("Unreviewed fields in request payload")


def main_collection_gate(smoke, config_path, selected, dataset_hash):
    """Require terminal ten-question source capture without selecting successes."""
    smoke = Path(smoke)
    with quiescent_source(smoke):
        manifest = json.loads((smoke / "manifest.json").read_text())
        progress = json.loads((smoke / "progress.json").read_text())
        summary = json.loads((smoke / "capture_summary.json").read_text())
        if manifest.get("kind") != "real_dagbt_workload_capture_not_accuracy_experiment" or manifest.get("selection") != "first_10_packaged_questions":
            raise ValueError("Main collection requires the fixed ten-question smoke capture")
        if progress.get("state") not in {"complete", "complete_with_failures"} or summary.get("questions") != 10 or sum(summary.get("terminal_counts", {}).values()) != 10:
            raise ValueError("All ten smoke questions must be terminal before main collection")
        if manifest.get("question_ids", {}).get("personamem") != [q["id"] for q in selected[:10]]:
            raise ValueError("Smoke question scope differs from the predetermined main prefix")
        if manifest.get("config_file_sha256") != file_hash(config_path) or manifest.get("dataset_manifest_sha256") != dataset_hash:
            raise ValueError("Smoke configuration or dataset identity changed")
        return {"smoke_manifest_sha256": file_hash(smoke / "manifest.json"),
                "smoke_summary_sha256": file_hash(smoke / "capture_summary.json"),
                "smoke_terminal_counts": summary["terminal_counts"]}


def collect_workload(config_path, output, questions=10, smoke=None):
    """Capture a bounded fixed question range through the existing fusion worker.

    The public runner CLI requires paired arms. Reuse its preflight, index and
    generate_dataset APIs instead of running an unnecessary original arm or
    changing any search/worker/budget code. No scoring labels are loaded.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from dagbt import runner
    from dagbt.config import resolve
    from dagbt.resources import ensure_index, inspect_index
    if questions not in {10, 50} or (questions == 50 and smoke is None):
        raise ValueError("Collection is limited to fixed 10-question smoke or gated fixed 50-question main")
    config = runner.validate_config(json.loads(Path(config_path).read_text()))
    method = LOCAL_ALGORITHM if config.get('fusion',{}).get('algorithm_version') == LOCAL_ALGORITHM else 'fusion'
    settings = resolve(config, method)
    budgets = {k: settings[k] for k in ("ann_calls", "set_score_calls", "llm_calls", "reader_calls")}
    if budgets != {"ann_calls": 36, "set_score_calls": 512, "llm_calls": 24, "reader_calls": 0 if method == LOCAL_ALGORITHM else 1}:
        raise ValueError("Reference algorithm budgets changed")
    selected = runner.load_questions("personamem", questions)
    dataset_hash = file_hash(ROOT / "data/personamem/manifest.json")
    gate = main_collection_gate(smoke, config_path, selected, dataset_hash) if questions == 50 else None
    out = new_output(output)
    manifest = {"schema": 1, "kind": "real_dagbt_workload_capture_not_accuracy_experiment",
                "datasets": ["personamem"], "arms": [method], "config": config,
                'algorithm_version':settings['algorithm_version'],
                "question_ids": {"personamem": [q["id"] for q in selected]},
                "questions_digest": {"personamem": digest(selected)}, "selection": f"first_{questions}_packaged_questions",
                "source_hashes": runner.frozen_sources(), "algorithm_budgets": budgets,
                "config_file_sha256": file_hash(config_path), "dataset_manifest_sha256": dataset_hash,
                "main_collection_gate": gate,
                "retry_rule": "existing transport retries; one question attempt; no explicit failed-task retry",
                "labels_read": False, "production_service_changed": False}
    write(out / "manifest.json", manifest)
    # Preserve existing caller network route except disable ambient forward
    # proxy for the exact authorized model hosts (not scoring semantics).
    hosts = [urllib.parse.urlsplit(config[k]).hostname for k in ("llm_base_url", "embedding_base_url")]
    hosts.append(urllib.parse.urlsplit(config["reranker"]["url"]).hostname)
    no_proxy = ",".join(filter(None, [os.environ.get("NO_PROXY", ""), *hosts]))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = no_proxy
    try:
        with runner.writer_lock(out):
            command = __import__("subprocess").check_output(["ps", "-p", str(os.getpid()), "-o", "command="], text=True).strip()
            runner.save(out / "pid.json", {"pid": os.getpid(), "pgid": os.getpgrp(), "output": str(out.resolve()),
                                          "started_unix": time.time(), "process_command": command})
            runner.save(out / "progress.json", {"state": "preflight", "updated_unix": time.time()})
            report = runner.preflight(config, ["personamem"], endpoints=True, output=out, arms=[method])
            runner.save(out / "preflight.json", report)
            runner.save(out / "progress.json", {"state": "preparing_index", "updated_unix": time.time()})
            ensure_index(config, "personamem", out)
            runner.save(out / "index_artifacts.json", {"personamem": inspect_index(config, "personamem")})
            counts = runner.generate_dataset(out, "personamem", selected, [method], config, retry_failed=False)
            runner.save(out / "capture_summary.json", {"terminal_counts": counts, "questions": len(selected), "labels_read": False})
            runner.save(out / "progress.json", {"state": "complete" if counts.get("ok", 0) == len(selected) else "complete_with_failures",
                                                "terminal_counts": counts, "updated_unix": time.time()})
        return {"status": "collected", "counts": counts}
    except Exception as exc:
        runner.save(out / "progress.json", {"state": "failed", "error_type": type(exc).__name__, "updated_unix": time.time()})
        return {"status": "collection_failed", "error_type": type(exc).__name__}


def scorer_metadata(events_path):
    """Join trusted cache-miss IDs and original bridge request events in order.

    Never parse role/header-looking strings from untrusted passage contents.
    IDs originate in the actual frozen scorer, not guessed from answers.
    """
    pending, active, requests, cache_counts = [], None, {}, None
    contexts = {}
    for line, event in read_rows(events_path):
        if event.get("event") == "cache_lookup" and event.get("module") == "scoring" and event.get("source") == "cache_miss":
            pending.append(event["ids"])
        elif event.get("event") == "score_batch_started" and event.get("module") == "scoring":
            n = event["batch_documents"]
            if n > len(pending):
                raise ValueError("Scorer batch lacks trusted cache-miss set identities")
            active, pending = pending[:n], pending[n:]
        elif event.get("event") == "bridge_rerank_request":
            offset = event["document_offset"]
            docs = event["documents"]
            ids = None if active is None else active[offset:offset + len(docs)]
            if ids is not None and len(ids) != len(docs):
                raise ValueError("Scorer IDs/physical batch alignment differs")
            stage = "/".join(map(str, event["stage"]))
            value = {"set_ids": ids, "documents_sha256": digest(docs), "source_event_line": line}
            if event.get('algorithm_version') == LOCAL_ALGORITHM:
                algorithm_identity(event)
                value.update(algorithm_version=LOCAL_ALGORITHM,
                    scoring_context_id=event['scoring_context_id'],node_id=event.get('node_id'),
                    scoring_query=event.get('query'))
            if stage in requests and requests[stage] != value:
                raise ValueError("Ambiguous original rerank stage identity")
            requests[stage] = value
        elif event.get("event") == "bridge_discovery_completed":
            cost = event.get("trace", {}).get("scorer_cost_cumulative")
            if cost is not None:
                trace = event.get('trace',{})
                context = trace.get('scoring_context_id', 'legacy_global')
                contexts[context] = {k:cost.get(k,0) for k in ('memory_cache_hits','persistent_cache_hits','scored_sets','logical_input_tokens_estimate')}
                cache_counts = {k:sum(c[k] for c in contexts.values()) for k in contexts[context]}
    return requests, cache_counts


def indexed_scores(body, count):
    # Reuse the frozen production validator, including explicit truncation.
    import sys
    sys.path.insert(0, str(ROOT))
    from vendor.bridgetree.dependency_scoring import _restore_indexed_scores
    return _restore_indexed_scores(body, count, "unit_interval")


@contextmanager
def quiescent_source(source):
    """Hold existing runner locks while reading a terminal capture.

    A poll timeout never makes a live writer safe to freeze. Shared locks also
    prevent a compliant runner from resuming during extraction. External logs
    without this lock protocol must be exported from a quiescent writer.
    """
    import fcntl
    source = Path(source).resolve()
    roots = {p.parent for p in source.rglob("writer.lock")}
    roots.update(p for p in (source, *source.parents) if (p / "writer.lock").is_file())
    with ExitStack() as stack:
        for root in sorted(roots):
            handle = stack.enter_context((root / "writer.lock").open("r"))
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Source has a live writer; wait for terminal collection before freezing") from exc
            progress = root / "progress.json"
            if progress.exists() and json.loads(progress.read_text()).get("state") not in {
                "complete", "complete_with_failures", "failed", "stopped"
            }:
                raise ValueError("Source progress is not terminal; incomplete collection cannot be frozen")
        yield


def freeze_transport(source, output, limit=50):
    with quiescent_source(source):
        return _freeze_transport(source, output, limit)


def _freeze_transport(source, output, limit=50):
    """Join DAG-BT call journal with complete Transport request records.

    First pass reads metadata, second streams selected question request bodies.
    Copies of bridge request/response events are not physical attempts. Retry
    attempts are retained individually, including failures with unknown reach.
    """
    source = Path(source)
    if limit <= 0:
        raise ValueError("Question limit must be positive")
    journals = sorted(source.rglob("call_events.jsonl"))
    units, paths = [], []
    for journal in journals:
        if not (journal.parent / "fusion_events.jsonl").exists():
            continue  # index builds/preflight are outside question search
        unit = None
        for _, event in read_rows(journal):
            if event.get("event") == "call_started":
                record_path = journal.parent / "requests" / (event["request_ref"] + ".json")
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    unit = str(record["unit_id"])
                    break
        # Questions without reranker calls remain in the selected denominator.
        if unit is None:
            unit = "no-rerank:" + str(journal.parent.relative_to(source))
        parts = journal.parent.parts
        idx = parts.index("attempts") if "attempts" in parts else -1
        dataset = parts[idx - 2] if idx >= 2 else "unknown"
        selection_id = dataset + ":" + unit
        units.append(selection_id)
        paths.append((unit, selection_id, dataset, journal))
    # Keep source-planned failures that never reached any model in the sample
    # denominator. task.json is authoritative, not an invented path identity.
    for task_path in sorted(source.rglob("task.json")):
        task = json.loads(task_path.read_text())
        if task.get("arm") not in ('fusion',LOCAL_ALGORITHM) or not isinstance(task.get("question"), dict):
            continue
        unit, dataset = task["question"]["id"], task["dataset"]
        units.append(dataset + ":" + unit)
    source_manifest = source / "manifest.json"
    if source_manifest.is_file():
        planned = json.loads(source_manifest.read_text()).get("question_ids", {})
        if isinstance(planned, dict):
            for dataset, question_ids in planned.items():
                if isinstance(question_ids, list):
                    units.extend(str(dataset) + ":" + str(unit) for unit in question_ids)
    chosen = select_questions(units, limit)
    out = new_output(output)
    counts = Counter()
    identities = []
    seen = set()
    physical_seen, journal_seen = set(), set()
    scorer_counts = []
    algorithms = set()

    def rows():
        spool = out / ".physical_order.sqlite"
        fd = os.open(spool, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        db = sqlite3.connect(spool)
        db.execute("CREATE TABLE attempts (started REAL, question TEXT, line INTEGER, attempt INTEGER, body TEXT)")
        for unit, selection_id, dataset, journal in paths:
            if selection_id not in chosen:
                continue
            journal_hash = file_hash(journal)
            copy_key = (selection_id, journal_hash)
            if copy_key in journal_seen:
                counts["duplicate_journal_copies"] += 1
                continue
            journal_seen.add(copy_key)
            identities.append({"journal_sha256": journal_hash, "question_id_hash": digest(unit),
                               "scorer_events_sha256": file_hash(journal.parent / "fusion_events.jsonl")})
            set_metadata, cache_counts = scorer_metadata(journal.parent / "fusion_events.jsonl")
            scorer_counts.append({"dataset": dataset, "question_id_hash": digest(unit), "cache_counts": cache_counts})
            starts = defaultdict(list)
            for _, e in read_rows(journal):
                if e.get("event") == "call_started" and e.get("kind") == "rerank_http" and not e.get("cache_hit"):
                    starts[e["request_ref"]].append(e)
            for line, event in read_rows(journal):
                if event.get("kind") != "rerank_http" or event.get("event") != "call_started":
                    continue
                counts["logical_requests"] += 1
                if event.get("cache_hit"):
                    counts["transport_response_cache_hits"] += 1
                    continue
                key = (str(journal.parent.resolve()), event["request_ref"])
                if key in seen:
                    counts["duplicate_journal_references"] += 1
                    continue
                seen.add(key)
                record_path = journal.parent / "requests" / (event["request_ref"] + ".json")
                if not record_path.exists():
                    raise ValueError("Missing complete Transport request record")
                record = json.loads(record_path.read_text())
                identity = transport_record_identity(record)
                if digest(identity) != event["request_ref"]:
                    raise ValueError("Transport request hash mismatch")
                payload = record["payload"]
                validate_payload(payload)
                scores = indexed_scores(record["response"], len(payload["documents"])) if "response" in record else None
                for attempt_index, attempt in enumerate(record.get("attempts", [])):
                    physical_key = digest([dataset, unit, event["request_ref"], attempt])
                    if physical_key in physical_seen:
                        counts["duplicate_physical_attempt_copies"] += 1
                        continue
                    physical_seen.add(physical_key)
                    # A failed call can later issue the same payload under a
                    # different stage; assign each attempt by its real time.
                    call = event
                    timed = sorted((e for e in starts[event["request_ref"]] if isinstance(e.get("unix"), (int, float))), key=lambda e: e["unix"])
                    if timed:
                        pos = bisect.bisect_right([e["unix"] for e in timed], attempt["started_unix"]) - 1
                        if pos < 0: raise ValueError("Physical attempt predates all original call events")
                        call = timed[pos]
                    metadata = set_metadata.get(call["stage"])
                    if metadata and metadata["documents_sha256"] != digest(payload["documents"]):
                        raise ValueError("Scorer event documents differ from complete request")
                    counts["physical_attempts"] += 1
                    success = attempt.get("http_status") == 200 and "finished_unix" in attempt
                    counts["successful_attempts" if success else "failed_or_unknown_attempts"] += 1
                    counts["successful_model_score_samples"] += len(payload["documents"]) if success else 0
                    row = {"dataset": dataset, "question_id": unit,
                           'algorithm_version':record.get('algorithm_version',LEGACY_ALGORITHM),
                           'scoring_context_id':record.get('scoring_context_id'),
                           'node_id':None if metadata is None else metadata.get('node_id'),
                           "request_id": event["request_ref"], "stage": call["stage"],
                           "journal_line": line, "attempt_index": attempt_index,
                           "started_unix": attempt.get("started_unix"), "payload": payload,
                           "payload_sha256": digest(payload), "success": success,
                           "model_reached": True if success else None,
                           "original_scores": scores if success else None,
                           "set_ids": None if metadata is None else metadata["set_ids"],
                           "batch_indices": list(range(len(payload["documents"]))),
                           "set_content_sha256": [digest(d) for d in payload["documents"]],
                           "transport_cache_hit": False, "final_inputs": None,
                           "request_file_sha256": file_hash(record_path)}
                    if not isinstance(row["started_unix"], (int, float)):
                        raise ValueError("Missing physical timestamp; cannot invent replay order")
                    algorithms.add(algorithm_identity(row)[0])
                    if len(algorithms) > 1:
                        raise ValueError('Cross-algorithm traces must be frozen separately')
                    db.execute("INSERT INTO attempts VALUES (?,?,?,?,?)", (row["started_unix"], row["question_id"],
                               row["journal_line"], row["attempt_index"], json.dumps(row, ensure_ascii=False)))
        # Disk sort streams selected requests; raw full-run bodies are never
        # accumulated in RAM. Original timestamps preserve question interleaving.
        db.commit()
        try:
            for ordinal, (body,) in enumerate(db.execute("SELECT body FROM attempts ORDER BY started,question,line,attempt")):
                yield {"ordinal": ordinal, **json.loads(body)}
        finally:
            db.close()
            spool.unlink()
    write_rows(out / "rerank_trace.jsonl", rows())
    manifest = {"schema": 1, "kind": "dagbt_transport_trace", "selection": "ascending SHA256(question identity)",
                'algorithm_versions':sorted(algorithms),
                "requested_questions": limit, "available_questions": len(set(units)),
                "selected_question_identities": chosen, "smoke_question_identities": chosen[:10],
                "counts": dict(counts), "source_journals": identities,
                "source_scorer_cache_counts": scorer_counts,
                "trace_sha256": file_hash(out / "rerank_trace.jsonl"),
                "order": "physical started_unix, original journal line/attempt within equal timestamps; timestamp ties explicitly deterministic",
                "retry_rule": "retain every recorded attempt; replay failures separately, never infer model reach",
                "limitations": ["final token IDs require isolated proxy observation",
                                "zero-rerank identities use attempt path when no request unit exists",
                                "set-result-cache hits require fusion/scorer logs; not inferred from absence"],
                "atol": ATOL, "rtol": RTOL}
    write(out / "manifest.json", manifest)
    return manifest


def validate_engine_row(row):
    algorithm_identity(row)
    p = row["payload"]
    prompts = p.get("prompt")
    if not isinstance(prompts, list) or not prompts or not all(
        isinstance(t, list) and t and all(type(x) is int and x >= 0 for x in t) for t in prompts
    ):
        raise ValueError("Final complete token IDs required (not strings or estimates)")
    if digest(p) != row["payload_sha256"]:
        raise ValueError("Engine payload identity mismatch")
    if not row.get("cache_namespace") or not row.get("computation_identity"):
        raise ValueError("Verified computation identity and tenant namespace required")
    if row.get("observed_at") != "proxy_post_json":
        raise ValueError("Only actual final proxy submissions qualify as engine trace")
    return prompts


def capture_requests(trace, output, proxy, smoke_only=False):
    """Submit frozen successful physical requests to the observed isolated proxy.

    Failure attempts remain in the source trace and are reported, not silently
    treated as known model work. No retry, no reordering, no result cache.
    """
    proxy = isolated_url(proxy)
    out = new_output(output)
    counts = Counter()
    selected = None
    if smoke_only:
        manifest = json.loads((Path(trace).parent / "manifest.json").read_text())
        if manifest["trace_sha256"] != file_hash(trace):
            raise ValueError("Smoke source trace changed")
        selected = set(manifest["smoke_question_identities"])
    def captured():
        for _, row in read_rows(trace):
            if selected is not None and row["dataset"] + ":" + row["question_id"] not in selected:
                continue
            if row["payload_sha256"] != digest(row["payload"]):
                raise ValueError("Frozen rerank payload changed")
            validate_payload(row["payload"])
            if not row["success"]:
                counts["historical_failed_or_unknown_attempts_excluded"] += 1
                continue
            counts["http_attempts"] += 1
            t = time.perf_counter()
            record = {"ordinal": row["ordinal"], "request_id": row["request_id"], "payload_sha256": row["payload_sha256"]}
            try:
                body = http_json(proxy + "/rerank", row["payload"])
                record.update(scores=indexed_scores(body, len(row["payload"]["documents"])), success=True,
                              usage=body.get("usage"))
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                counts["failures"] += 1
                record.update(success=False, error_type=type(exc).__name__)
            record["seconds"] = time.perf_counter() - t
            yield record
    write_rows(out / "capture_results.jsonl", captured())
    manifest = {"kind": "isolated_rerank_capture", "source_trace_sha256": file_hash(trace),
                "counts": dict(counts), "smoke_only": smoke_only, "retry_rule": "no new retries; source failures remain separate",
                "final_token_trace": "observer engine_trace.jsonl; same sequence must be cross-checked"}
    write(out / "manifest.json", manifest)
    return manifest


def prefix_analysis(rows, block_size):
    """Ideal unlimited capacity, completed prior calls only, final token recompute.

    Blocks include their entire preceding prefix. Same-batch samples cannot
    consume each other's future computation. Identity/namespace segregated.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("Actual positive KV block size required")
    blocks = defaultdict(set)
    totals = Counter()
    detail = []
    last_ordinal = -1
    completions = []
    for row in rows:
        if row["ordinal"] <= last_ordinal:
            raise ValueError("Trace order must be strictly increasing")
        last_ordinal = row["ordinal"]
        prompts = validate_engine_row(row)
        start = row.get("started_unix")
        finish = row.get("finished_unix")
        if type(start) not in (int, float) or row.get("completed") and (type(finish) not in (int, float) or finish < start):
            raise ValueError("Actual start/completion times required for causal hit estimates")
        while completions and completions[0][0] <= start:
            _, _, completed_identity, completed_blocks = heapq.heappop(completions)
            blocks[completed_identity].update(completed_blocks)
        identity = (row["computation_identity"], row["cache_namespace"])
        previous = blocks[identity]
        additions = set()
        hits = []
        for tokens in prompts:
            # At least last input position must be recomputed to yield logits.
            eligible = max(0, (len(tokens) - 1) // block_size)
            count = 0
            prefix = None
            for i in range(1, eligible + 1):
                key = digest([prefix, tokens[(i - 1) * block_size:i * block_size]])
                prefix = key
                if key in previous and count == i - 1:
                    count = i
                additions.add(key)
            hits.append(count * block_size)
            totals["full_input_tokens"] += len(tokens)
            totals["ideal_reusable_tokens"] += count * block_size
            totals["model_samples"] += 1
        if row.get("completed"):
            heapq.heappush(completions, (finish, row["ordinal"], identity, additions))
        detail.append({"ordinal": row["ordinal"], "input_lengths": list(map(len, prompts)),
                       "ideal_reusable_tokens": hits})
    return {"kind": "structural_ideal_capacity_estimate", "block_size": block_size,
            **dict(totals), "engine_hit_tokens": None, "engine_new_compute_tokens": None,
            "evictions": None, "details": detail,
            "limitations": "No capacity/eviction/scheduler inference; prior completed HTTP calls only"}


def isolated_url(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != "http" or p.hostname not in {"127.0.0.1", "localhost", "::1"} or p.username or p.password or p.query or p.fragment:
        raise ValueError("Only a loopback isolated backend is permitted (use an SSH tunnel)")
    if p.port in {8002, 18002} or not p.port:
        raise ValueError("Production ports are excluded")
    return url.rstrip("/")


def http_json(url, payload=None, timeout=180):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("BT_PREFIX_TEST_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["BT_PREFIX_TEST_API_KEY"]
    request = urllib.request.Request(url, headers=headers,
        data=None if payload is None else json.dumps(payload).encode())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def http_health(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {}
    if os.environ.get("BT_PREFIX_TEST_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["BT_PREFIX_TEST_API_KEY"]
    with opener.open(urllib.request.Request(url, headers=headers), timeout=10) as response:
        if response.status != 200:
            raise ValueError("Isolated backend is unhealthy")


def stats(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": ordered[math.ceil(.95 * len(values)) - 1]}


def metrics_snapshot(backend, definitions):
    """Only installed/observed metric names and units, never estimated counters."""
    if not definitions:
        return {"raw": None, "values": {}}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(backend + "/metrics", timeout=10) as response:
            raw = response.read().decode()
    except (urllib.error.URLError, TimeoutError):
        return {"raw": None, "values": {}}
    values = {}
    for field, definition in definitions.items():
        matches = []
        name = definition["metric"]
        for line in raw.splitlines():
            if not line or line.startswith("#"): continue
            label, *tail = line.split()
            if label.split("{", 1)[0] == name and tail:
                try: v = float(tail[0])
                except ValueError: continue
                if math.isfinite(v): matches.append(v)
        # Multiple labelled replicas cannot silently be summed across devices.
        values[field] = matches[0] if len(matches) == 1 else None
    return {"raw": raw, "values": values}


def metrics_delta(before, after, definitions):
    result = {}
    for field, definition in definitions.items():
        a, b = before["values"].get(field), after["values"].get(field)
        if a is None or b is None or definition.get("kind") != "counter" or b < a:
            result[field] = None
        else:
            result[field] = {"value": b - a, "unit": definition["unit"], "metric": definition["metric"]}
    return result


def raw_labels(response, payload):
    import sys
    sys.path.insert(0, str(ROOT))
    from vendor.bridgetree.clients import _response_declares_truncation
    if _response_declares_truncation(response):
        raise ValueError("Native backend explicitly reported input truncation")
    count = len(payload["prompt"])
    by_index = {}
    token = payload.get("allowed_token_ids")
    if not isinstance(token, list) or len(token) != 1:
        raise ValueError("Current canonical label protocol required; do not change scoring head")
    for c in response.get("choices", []):
        if _response_declares_truncation(c):
            raise ValueError("Native sample explicitly reported input truncation")
        i = c.get("index")
        lp = c.get("logprobs") or {}
        values = lp.get("token_logprobs") or []
        if type(i) is not int or i < 0 or i >= count or i in by_index:
            raise ValueError("Invalid or duplicate native index")
        if lp.get("tokens") != ["token_id:" + str(token[0])] or len(values) != 1:
            raise ValueError("Canonical label logprob absent")
        v = values[0]
        if type(v) not in {int, float} or not math.isfinite(v) or v > 1e-6:
            raise ValueError("Invalid raw logprob")
        by_index[i] = v
    if len(by_index) != count:
        raise ValueError("Incomplete native batch")
    return [by_index[i] for i in range(count)]


def replay(trace, output, backend, identity_path, condition, reset=False):
    """One arm; sequential original physical calls, no HTTP/result cache or retries.

    Run arms sequentially under deployment control. No general production
    reset endpoint: requires loopback isolation plus matching inventory.
    """
    backend = isolated_url(backend)
    identity = json.loads(Path(identity_path).read_text())
    if identity.get("isolated") is not True or identity.get("backend_url") != backend:
        raise ValueError("Inventory must attest the exact isolated backend URL")
    if identity.get("runtime_verified") is not True or identity.get("health_verified") is not True:
        raise ValueError("A launch plan is not observed runtime identity")
    evidence = identity.get("runtime_evidence_file")
    if not evidence or file_hash(evidence) != identity.get("runtime_evidence_sha256"):
        raise ValueError("Missing/tampered installed-engine runtime evidence")
    if identity.get("enable_prefix_caching") != (condition != "off"):
        raise ValueError("Observed runtime APC state does not match condition")
    if condition == "cold" and not reset:
        raise ValueError("Cold arm needs explicit isolated prefix-only reset after equal warmup")
    out = new_output(output)
    write(out / "manifest.json", {"condition": condition, "trace_sha256": file_hash(trace),
          "inventory_sha256": file_hash(identity_path), "identity": identity,
          "atol": ATOL, "rtol": RTOL, "retry_rule": "one HTTP attempt; fail visibly; no retries",
          "result_cache": "bypassed by direct final-token native requests", "concurrency": 1,
          "warmup": "operator must finish identical loading/compiler warmup before this command"})
    http_health(backend + "/health")
    if reset:
        # Prefix reset must be supported by the installed version, never inferred.
        if identity.get("prefix_reset_verified") is not True:
            raise ValueError("Installed prefix reset capability is not verified")
        http_json(backend + "/reset_prefix_cache", {})
    definitions = identity.get("metrics", {})
    before = metrics_snapshot(backend, definitions)
    started = time.perf_counter()
    counts = Counter()
    durations, question_times = [], defaultdict(float)
    scored_groups, unique_samples = set(), set()

    def results():
        for _, row in read_rows(trace):
            prompts = validate_engine_row(row)
            if row["computation_identity"] != identity["computation_identity"]:
                raise ValueError("Observed model/tokenizer/attention/position identity differs")
            if row["cache_namespace"] != identity.get("tenant_namespace_hash"):
                raise ValueError("Dedicated engine tenant differs from trace")
            record = {k: row[k] for k in ("ordinal", "request_id", "payload_sha256", "computation_identity", "cache_namespace")}
            record["label_token_id"] = row["payload"].get("allowed_token_ids", [None])[0]
            record["rerank_payload"] = row.get("rerank_payload")
            record.update(algorithm_version=row.get("algorithm_version",LEGACY_ALGORITHM),
                          scoring_context_id=row.get("scoring_context_id"))
            record["final_inputs"] = [{"index": i, "token_ids_sha256": digest(tokens), "input_length": len(tokens)}
                                      for i, tokens in enumerate(prompts)]
            if row.get("rerank_payload"):
                r = row["rerank_payload"]
                if row["request_id"] not in scored_groups:
                    counts["whole_set_score_samples"] += len(r["documents"])
                    scored_groups.add(row["request_id"])
                unique_samples.update(digest([r["query"], doc]) for doc in r["documents"])
            counts["http_attempts"] += 1
            t = time.perf_counter()
            try:
                body = http_json(backend + "/v1/completions", row["payload"])
                record.update(raw_logprobs=raw_labels(body, row["payload"]), usage=body.get("usage"), success=True)
                counts["model_samples"] += len(prompts)
                counts["full_input_tokens"] += sum(map(len, prompts))
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                # Do not persist bodies, exception strings or authorization data.
                record.update(success=False, error_type=type(exc).__name__)
                counts["failures"] += 1
            record["seconds"] = time.perf_counter() - t
            durations.append(record["seconds"])
            question_times[(row.get("dataset"), row.get("question_id", "unknown"))] += record["seconds"]
            yield record
    write_rows(out / "results.jsonl", results())
    elapsed = time.perf_counter() - started
    after = metrics_snapshot(backend, definitions)
    measured = metrics_delta(before, after, definitions)
    write(out / "metrics_snapshots.json", {"before": before, "after": after, "delta": measured})
    summary = {"counts": dict(counts), "request_seconds": stats(durations),
               "question_service_seconds": stats(list(question_times.values())),
               "sample_seconds": None, "sample_latency_note": "HTTP batch timing is not individual engine sample latency",
               "total_wall_seconds": elapsed, "samples_per_second": counts["model_samples"] / elapsed,
               "engine_kv_hit_tokens": None, "engine_new_compute_tokens": None, "device_seconds": None,
               "peak_memory": None, "evictions": None, "oom": None,
               "metrics_note": "Import installed-engine snapshots separately; usage.prompt_tokens is full input, not compute"}
    summary["measured_service_metrics"] = measured
    summary["distinct_serialized_samples"] = len(unique_samples)
    summary["http_retries"] = 0
    summary["result_cache_hits_in_core_replay"] = 0
    summary["historical_result_cache_hits"] = None
    summary["service_http_seconds_sum"] = sum(durations)
    summary["wall_time_scope"] = "measurement includes identical harness input parsing/output writing; request seconds excludes input hashing"
    write(out / "summary.json", summary)
    return summary


def compare_scores(left, right, atol=ATOL, rtol=RTOL):
    if len(left) != len(right):
        raise ValueError("Sample counts differ")
    absolute = [abs(a - b) for a, b in zip(left, right)]
    relative = [e / max(abs(a), 1e-300) for e, a in zip(absolute, left)]
    rank = lambda v: sorted(range(len(v)), key=lambda i: (-v[i], i))
    def ties(values):
        groups = defaultdict(list)
        for i, value in enumerate(values): groups[value].append(i)
        return sorted(indices for indices in groups.values() if len(indices) > 1)
    return {"equivalent": all(e <= atol + rtol * abs(a) for e, a in zip(absolute, left)),
            "max_absolute_error": max(absolute, default=0), "max_relative_error": max(relative, default=0),
            "rank_equal": rank(left) == rank(right),
            "exact_ties_equal": ties(left) == ties(right)}


def quartet(values, epsilon):
    p, pe, pg, pge = (values[k] for k in ("P", "Pe", "PG", "PGe"))
    # Preserve the source arithmetic order, not reassociated expressions.
    before, after = pe - p, pge - pg
    activation = after - before
    return {"A": activation, "M": after, "group_marginal": pge - pe,
            "decision": "retain" if activation > epsilon and after > epsilon else
                        "speculate" if activation > epsilon else "reject",
            "pivot_eligible": after < -epsilon}


def normalized_scores(results, label_ids):
    groups = {}
    for _, row in read_rows(results):
        if not row["success"]:
            raise ValueError("Failed physical attempt; normalized score unavailable")
        label = next((k for k, v in label_ids.items() if v == row.get("label_token_id")), None)
        if label not in ("yes", "no"):
            raise ValueError("Actual canonical label token identity missing")
        group = groups.setdefault(row["request_id"], {})
        if label in group:
            raise ValueError("Duplicate physical label attempt must be reported separately")
        group[label] = row
    bank, per_sample = {}, []
    for request_id, group in groups.items():
        if set(group) != {"yes", "no"}:
            raise ValueError("Incomplete yes/no computation pair")
        y, n = group["yes"], group["no"]
        if algorithm_identity(y) != algorithm_identity(n):
            raise ValueError("Label pair scoring context differs")
        p = y["rerank_payload"]
        if p is None or p != n["rerank_payload"] or len(y["raw_logprobs"]) != len(n["raw_logprobs"]):
            raise ValueError("Label pair rerank identity differs")
        if y.get("final_inputs") != n.get("final_inputs"):
            raise ValueError("Canonical yes/no label requests have different full token inputs")
        if len(y["raw_logprobs"]) != len(p["documents"]):
            raise ValueError("Whole-set score sample count differs from original batch")
        scores = []
        for i, (yes, no) in enumerate(zip(y["raw_logprobs"], n["raw_logprobs"])):
            offset = max(yes, no)
            score = math.exp(yes - offset) / (math.exp(yes - offset) + math.exp(no - offset))
            scores.append(score)
            per_sample.append({"request_id": request_id, "index": i, "score": score,
                               "algorithm_version":algorithm_identity(y)[0], "scoring_context_id":algorithm_identity(y)[1],
                               "query_sha256": digest(p["query"]), "set_content_sha256": digest(p["documents"][i])})
        key = score_bank_key(p,y)
        if key in bank and bank[key] != scores:
            raise ValueError("Repeated exact request varies; use per-physical request mapping, not a result-cache surrogate")
        bank[key] = scores
    return bank, per_sample


def compare_quartets(states, off, on):
    details = []
    for state in states:
        keys = state["sample_keys"]
        a, b = ({k: bank[keys[k]] for k in ("P", "Pe", "PG", "PGe")} for bank in (off, on))
        qa, qb = quartet(a, state["marginal_epsilon"]), quartet(b, state["marginal_epsilon"])
        details.append({"state_id": state["state_id"], "off": qa, "on": qb,
                        "raw_score_comparison": compare_scores(list(a.values()), list(b.values())),
                        "A_error": abs(qa["A"] - qb["A"]), "M_error": abs(qa["M"] - qb["M"]),
                        "action_equal": qa["decision"] == qb["decision"] and qa["pivot_eligible"] == qb["pivot_eligible"]})
    return details


def build_search_bundles(source, output):
    """Export original question-local resources; no network or labels read."""
    import sys
    import importlib
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from dagbt import runner
    from dagbt.model_runtime import load_tokenizer
    from dagbt.resources import prepare_resources, scope_resources
    from dagbt.personamem import load_questions, load_scopes
    source = Path(source).resolve()
    with quiescent_source(source):
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest.get("kind") != "real_dagbt_workload_capture_not_accuracy_experiment" or manifest.get("datasets") != ["personamem"] or manifest.get("arms") not in (["fusion"],[LOCAL_ALGORITHM]):
            raise ValueError("Bundle export requires the original scoped PersonaMem fusion capture")
        method = manifest["arms"][0]
        config = runner.validate_config(manifest["config"])
        if manifest.get("source_hashes") != runner.frozen_sources():
            raise ValueError("Original algorithm sources changed since capture")
        if manifest.get("dataset_manifest_sha256") != file_hash(ROOT / "data/personamem/manifest.json"):
            raise ValueError("Dataset identity changed since capture")
        sys.path.insert(0, str(ROOT / "dagv2"))
        inherited = os.environ.pop("DAGV2_CONFIG", None)
        try:
            pipeline = importlib.import_module("pipeline_dagv2")
        finally:
            if inherited is not None: os.environ["DAGV2_CONFIG"] = inherited
        _, resources, hashes = prepare_resources(config, "personamem", pipeline, load_tokenizer(config))
        recorded_hashes = json.loads((source / "personamem" / method / "resource_hashes.json").read_text())
        if hashes != recorded_hashes:
            raise ValueError("Original corpus/index resource identity changed since capture")
        questions = {q["id"]: q for q in load_questions(ROOT)}
        scopes = load_scopes(ROOT)
        selected = manifest["question_ids"]["personamem"]
        out = new_output(output)
        entries = []
        for unit in selected:
            q = questions[unit]
            attempts = sorted((source / "personamem" / method / "attempts" / runner.digest(unit)).glob("attempt-*"))
            if not attempts:
                entries.append({"question_id_sha256": digest(unit), "replayable": False, "reason": "no_original_attempt"})
            for attempt in attempts:
                task = json.loads((attempt / "task.json").read_text())
                if task.get("question") != q or task.get("dataset") != "personamem" or task.get("arm") != method:
                    raise ValueError("Original task differs from frozen public question/scope")
                if not (attempt / "result.json").is_file():
                    raise ValueError("Original question attempt is not terminal")
                docs, ids, vectors, _, _ = scope_resources(resources, scopes[q["scope_id"]])
                directory = new_output(out / digest([unit, attempt.name]))
                vector_path = directory / "vectors.npy"
                with os.fdopen(os.open(vector_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
                    np.save(handle, vectors, allow_pickle=False)
                bundle = {"dataset": "personamem", "method": method, "config": pipeline.e.CONFIG,
                          "algorithm_version":LOCAL_ALGORITHM if method==LOCAL_ALGORITHM else LEGACY_ALGORITHM,
                          "output_options":q.get("options"),
                          "question": {"id": unit, "question": q["user_question"]}, "reader_question": q["question"],
                          "attempt_dir": str(attempt), "ids": ids,
                          "documents": [{"doc_id": d, "title": docs[d].title, "text": docs[d].text,
                                         "metadata": docs[d].metadata, "passage": docs[d].passage} for d in ids],
                          "vectors_path": str(vector_path.resolve()), "vectors_sha256": file_hash(vector_path),
                          "scope_sha256": digest(ids), "original_resource_hashes": hashes,
                          "source_result_sha256": file_hash(attempt / "result.json")}
                write(directory / "bundle.json", bundle)
                entries.append({"question_id_sha256": digest(unit), "attempt": attempt.name, "replayable": True,
                                "bundle": str((directory / "bundle.json").relative_to(out)),
                                "bundle_sha256": file_hash(directory / "bundle.json"), "visible_documents": len(ids),
                                "source_status": json.loads((attempt / "result.json").read_text())["answer"]["status"]})
        result = {"kind": "private_scoped_search_bundles", "source_manifest_sha256": file_hash(source / "manifest.json"),
                  "planned_questions": len(selected), "entries": entries, "labels_read": False, "network_called": False,
                  "hardware_equivalence_verified": False}
        write(out / "manifest.json", result)
        return result


def search_replay(bundle_path, scores_path, label_ids, output):
    """Execute actual Engine/BridgeSession with read-only frozen observations.

    Bundle contains exact question/config/resources and original attempt dir.
    Only reranker responses come from this arm. An unseen request stops replay;
    the harness has no network fallback and no planner/Reader regeneration.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from copy import deepcopy
    from types import SimpleNamespace
    import numpy as np
    from dagbt import engine as engine_module
    from dagbt.model_runtime import load_tokenizer
    from dagbt.transport import Transport, normalize_response, prepare_request, request_identity
    import threading
    bundle = json.loads(Path(bundle_path).read_text())
    from dagbt.config import resolve
    version = resolve(bundle["config"],bundle.get("method","fusion"))["algorithm_version"]
    if bundle.get("algorithm_version",LEGACY_ALGORITHM) != version:
        raise ValueError("Bundle algorithm identity differs from method")
    if bundle["config"].get("model_profile") != "bridgetree":
        raise ValueError("Only the existing BT model profile is supported")
    if bundle["config"].get("_test_transport"):
        raise ValueError("Production replay must not enable fixture transport")
    runtime = None
    if bundle.get("dataset") == "personamem":
        import importlib
        from dagbt.runner import configure_personamem_reader, validate_config
        validate_config(bundle["config"])
        sys.path.insert(0, str(ROOT / "dagv2"))
        inherited = os.environ.pop("DAGV2_CONFIG", None)
        try:
            runtime = importlib.import_module("pipeline_dagv2")
        finally:
            if inherited is not None: os.environ["DAGV2_CONFIG"] = inherited
        runtime.e.CONFIG.update(bundle["config"])
        runtime.e.native.configure(runtime.e.CONFIG)
        runtime.e.validate_plan = runtime.repair.validate_plan
        configure_personamem_reader()
    bank, samples = normalized_scores(scores_path, label_ids)
    if any(s["algorithm_version"] != version for s in samples):
        raise ValueError("Cross-algorithm score replay is forbidden")
    source = Path(bundle["attempt_dir"]) / "requests"
    records = {}
    for path in source.glob("*.json"):
        r = json.loads(path.read_text())
        if digest(transport_record_identity(r)) != path.stem:
            raise ValueError("Original observation request identity mismatch")
        if "response" in r:
            records[path.stem] = r
    out = new_output(output)
    requested = []

    class FrozenTransport(Transport):
        def get(self, stage, url, payload, *, reserve=None, extra_reserve=0, scoring_context_id=None):
            url, payload, legacy = prepare_request(stage, url, payload, self.config)
            key = digest(request_identity(self.unit,url,payload,self.config,scoring_context_id))
            requested.append({"stage": stage, "request_ref": key})
            if key not in records:
                raise ValueError("TRAJECTORY_DIVERGENCE_UNSEEN_REQUEST:" + key)
            r = deepcopy(records[key])
            rerank = url == self.config.get("reranker", {}).get("url")
            if rerank:
                lookup = score_bank_key(payload, {"algorithm_version":version,"scoring_context_id":scoring_context_id})
                if lookup not in bank:
                    raise ValueError("TRAJECTORY_DIVERGENCE_UNSCORED_SET:" + lookup)
                r["response"] = {"results": [{"index": i, "relevance_score": s} for i, s in enumerate(bank[lookup])]}
            # Exact original physical charge (including retries), same search budgets.
            stage_text = "/".join(map(str, stage)) if isinstance(stage, (tuple, list)) else str(stage)
            kind = "rerank_http" if rerank else "embedding_http" if url.rstrip("/").endswith("/embeddings") else "reader" if stage_text.startswith("reader/") else "llm"
            for _ in range(max(1, len(r.get("attempts", [])))):
                self._reserve(kind, stage_text, reserve, extra_reserve)
            return {"response": normalize_response(r["response"], legacy), "response_ref": key, "request": payload}

        post_rerank = get
    docs = {d["doc_id"]: SimpleNamespace(**{**d, "passage": d.get("passage", "\n".join(
            d[k] for k in ("title", "text") if d.get(k)))}) for d in bundle["documents"]}
    ids = bundle["ids"]
    vectors = np.load(bundle["vectors_path"], allow_pickle=False)
    if file_hash(bundle["vectors_path"]) != bundle["vectors_sha256"]:
        raise ValueError("Frozen corpus vectors changed")
    resources = (docs, ids, vectors, SimpleNamespace(vectors=dict(zip(ids, vectors)), lock=threading.Lock()),
                 load_tokenizer(bundle["config"]))
    prior = engine_module.Transport
    engine_module.Transport = FrozenTransport
    try:
        instance = engine_module.Engine(bundle["question"], resources, SimpleNamespace(output=out),
                                       bundle["config"], bundle.get("method", "fusion"),
                                       reader_question=bundle.get("reader_question"),output_options=bundle.get("output_options"))
        result = instance.run()
    except Exception as exc:
        write(out / "replay_status.json", {"completed": False, "error_type": type(exc).__name__,
              "first_divergent_request": requested[-1] if requested else None,
              "note": "No network fallback or old-score fallback used"})
        raise
    finally:
        engine_module.Transport = prior
    write(out / "request_sequence.json", requested)
    write(out / "trajectory.json", {"bridge_traces": instance.bridge.traces, "algorithm_version":version,
          "events": instance.events, "logical_counts": dict(instance.ledger.used),
          "requests": requested})
    # Full private trace captures queue/scheduler, candidates, decisions and budgets.
    write(out / "replay_status.json", {"completed": True, "logical_counts": dict(instance.ledger.used),
          "requests_sha256": digest(requested), "scores_sha256": file_hash(scores_path),
          "bundle_sha256": file_hash(bundle_path), "result": result})
    return {"completed": True}


def compare_search(off, on, output):
    """Compare actual rerun scheduler/queue/actions and separately all quartets."""
    ignored = {"elapsed_seconds", "at_epoch", "seconds", "started_unix", "finished_unix",
               "P", "Pe", "PG", "PGe", "score_P", "score_Pe", "score_PG", "score_PGe",
               "score", "activation", "signal", "context_marginal", "priority",
               "target_marginal_before", "target_marginal_after", "reranker_elapsed_ms"}
    def structural(value):
        if isinstance(value, dict): return {k: structural(v) for k, v in value.items() if k not in ignored}
        if isinstance(value, list): return [structural(v) for v in value]
        return value
    a, b = (json.loads((Path(p) / "trajectory.json").read_text()) for p in (off, on))
    if a.get("algorithm_version",LEGACY_ALGORITHM) != b.get("algorithm_version",LEGACY_ALGORITHM):
        raise ValueError("Cross-algorithm trajectory comparison is forbidden")
    sa, sb = structural(a), structural(b)
    first = None
    for section in ("requests", "bridge_traces", "events", "logical_counts"):
        if sa[section] != sb[section]:
            if isinstance(sa[section], list):
                first_index = next((i for i, (x, y) in enumerate(zip(sa[section], sb[section])) if x != y),
                                   min(len(sa[section]), len(sb[section])))
            else: first_index = None
            first = {"section": section, "index": first_index}; break
    quartets = []
    for x, y in zip(a["bridge_traces"], b["bridge_traces"]):
        ax = x.get("search_archive", {}).get("activations", [])
        by = y.get("search_archive", {}).get("activations", [])
        if len(ax) != len(by):
            quartets.append({"measurement_count_equal": False}); continue
        for u, v in zip(ax, by):
            keys = ("P", "Pe", "PG", "PGe")
            if not all(k in u and k in v for k in keys):
                raise ValueError("Four raw scores missing from actual search archive")
            comparison = compare_scores([u[k] for k in keys], [v[k] for k in keys])
            quartets.append({"raw_scores": comparison,
                 "A_error": abs(((u[keys[3]]-u[keys[2]])-(u[keys[1]]-u[keys[0]])) -
                                ((v[keys[3]]-v[keys[2]])-(v[keys[1]]-v[keys[0]]))),
                 "M_error": abs((u[keys[3]]-u[keys[2]])-(v[keys[3]]-v[keys[2]]))})
    raw_equivalence = None if not quartets else all(q.get("raw_scores", {}).get("equivalent", False) for q in quartets)
    report = {"trajectory_equal": sa == sb, "raw_score_equivalent": raw_equivalence,
              "equivalence_proven": sa == sb and raw_equivalence is True,
              "first_divergence": first, "quartets": quartets,
              "note": "Actual source queues and pivot/speculation limits executed; numerical priorities omitted from structural equality, queue order retained"}
    out = new_output(output); write(out / "search_comparison.json", report)
    return report


def run_synthetic(output, backend=None):
    out = new_output(output)
    rows = synthetic_requests()
    for row in rows: validate_payload(row["payload"])
    write_rows(out / "synthetic_requests.jsonl", rows)
    if backend is not None:
        backend = isolated_url(backend)
        results = []
        for row in rows:
            body = http_json(backend + "/rerank", row["payload"])
            results.append({"case": row["case"], "scores": indexed_scores(body, len(row["payload"]["documents"]))})
        by_name = {r["case"]: r["scores"] for r in results}
        comparison = compare_scores(by_name["native-batch"], list(reversed(by_name["reversed-batch"])))
        write(out / "synthetic_results.json", {"results": results, "reversed_batch": comparison,
              "hardware_kv_verified": False, "note": "Protocol scores alone cannot prove KV hits"})
    write(out / "manifest.json", {"kind": "synthetic_protocol_only", "model_called": backend is not None,
          "pending_service_cases": ["actual-token block boundaries/8191+1 capacity and explicit 413",
                                   "production concurrency", "capacity-driven eviction", "isolated process restart",
                                   "different private-tenant engine instance"],
          "trace_sha256": file_hash(out / "synthetic_requests.jsonl"), "performance_claim": None})
    return {"status": "synthetic_written"}


def compare_runs(off, on, output):
    off, on = Path(off), Path(on)
    a, b = (json.loads((p / "manifest.json").read_text()) for p in (off, on))
    if a["trace_sha256"] != b["trace_sha256"] or a["atol"] != b["atol"] or a["rtol"] != b["rtol"]:
        raise ValueError("Trace/tolerance identity mismatch")
    # An arm may differ only in APC and its isolated deployment identifiers.
    ignored = {"enable_prefix_caching", "backend_url", "pid", "started_at", "deployment_id", "mode",
               "runtime_evidence_file", "runtime_evidence_sha256", "source_inventory_sha256"}
    def canonical(m):
        result = {k: v for k, v in m["identity"].items() if k not in ignored}
        if "engine_argv" in result:
            args, keep, i = result["engine_argv"], [], 0
            while i < len(args):
                if args[i] in ("--enable-prefix-caching", "--no-enable-prefix-caching"):
                    i += 1; continue
                if args[i] in ("--host", "--port"):
                    i += 2; continue
                keep.append(args[i]); i += 1
            result["engine_argv"] = keep
        return result
    if canonical(a) != canonical(b):
        raise ValueError("Other service settings changed between arms")
    ar, br = (list(read_rows(p / "results.jsonl")) for p in (off, on))
    if len(ar) != len(br):
        raise ValueError("HTTP/sample sequence lengths differ")
    differences = []
    for (_, x), (_, y) in zip(ar, br):
        if algorithm_identity(x) != algorithm_identity(y):
            raise ValueError('Cross-algorithm/scoring-context comparison is forbidden')
        if any(x[k] != y[k] for k in ("ordinal", "request_id", "payload_sha256", "computation_identity", "cache_namespace")):
            raise ValueError("Complete final input or original order differs")
        comparison = compare_scores(x["raw_logprobs"], y["raw_logprobs"]) if x["success"] and y["success"] else None
        differences.append({"ordinal": x["ordinal"], "comparison": comparison,
                            "off_seconds": x["seconds"], "on_seconds": y["seconds"]})
    sa, sb = (json.loads((p / "summary.json").read_text()) for p in (off, on))
    baseline, optimized = sa["total_wall_seconds"], sb["total_wall_seconds"]
    report = {"status": "IMPLEMENTED_NOT_HARDWARE_VERIFIED", "samples_unchanged": sa["counts"] == sb["counts"],
              "raw_logprob_comparisons": differences, "speedup": baseline / optimized,
              "reranker_time_reduction": 1 - optimized / baseline,
              "normalized_score_equivalence": None, "action_equivalence": None, "search_trajectory_equivalence": None,
              "engine_kv_hit_tokens": None, "five_pair_stability": None,
              "note": "Per-arm timing alone cannot pass final acceptance; join label scores, real KV metrics and fixed-observation search replay"}
    label_ids = a["identity"].get("label_token_ids")
    if label_ids:
        try:
            _, left = normalized_scores(off / "results.jsonl", label_ids)
            _, right = normalized_scores(on / "results.jsonl", label_ids)
            if [(x["request_id"], x["index"], x["query_sha256"], x["set_content_sha256"]) for x in left] != [
                (x["request_id"], x["index"], x["query_sha256"], x["set_content_sha256"]) for x in right]:
                raise ValueError("Normalized sample identity differs")
            comparisons = []
            for query in dict.fromkeys(x["query_sha256"] for x in left):
                comparisons.append(compare_scores([x["score"] for x in left if x["query_sha256"] == query],
                                                   [x["score"] for x in right if x["query_sha256"] == query]))
            report["normalized_score_equivalence"] = {
                "equivalent": all(c["equivalent"] for c in comparisons),
                "rank_equal": all(c["rank_equal"] for c in comparisons),
                "exact_ties_equal": all(c["exact_ties_equal"] for c in comparisons),
                "max_absolute_error": max((c["max_absolute_error"] for c in comparisons), default=0),
                "max_relative_error": max((c["max_relative_error"] for c in comparisons), default=0)}
        except ValueError:
            report["status"] = "FAILED_EQUIVALENCE_OR_REGRESSION"
    if any(d["comparison"] is None or not d["comparison"]["equivalent"] for d in differences):
        report["status"] = "FAILED_EQUIVALENCE_OR_REGRESSION"
    if report["normalized_score_equivalence"] and not all(report["normalized_score_equivalence"][k] for k in ("equivalent", "rank_equal", "exact_ties_equal")):
        report["status"] = "FAILED_EQUIVALENCE_OR_REGRESSION"
    out = new_output(output)
    write(out / "comparison.json", report)
    return report


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--source", required=True); f.add_argument("--output", required=True); f.add_argument("--questions", type=int, default=50)
    a = sub.add_parser("analyze"); a.add_argument("--trace", required=True); a.add_argument("--block-size", type=int, required=True); a.add_argument("--output", required=True)
    a = sub.add_parser("capture"); a.add_argument("--trace", required=True); a.add_argument("--proxy", required=True); a.add_argument("--output", required=True); a.add_argument("--smoke-only", action="store_true")
    r = sub.add_parser("replay"); r.add_argument("--trace", required=True); r.add_argument("--backend", required=True); r.add_argument("--identity", required=True); r.add_argument("--condition", choices=["off", "cold", "warm", "negative"], required=True); r.add_argument("--reset-isolated-prefix-cache", action="store_true"); r.add_argument("--output", required=True)
    c = sub.add_parser("compare"); c.add_argument("--off", required=True); c.add_argument("--on", required=True); c.add_argument("--output", required=True)
    s = sub.add_parser("search-replay"); s.add_argument("--bundle", required=True); s.add_argument("--scores", required=True); s.add_argument("--label-ids", required=True); s.add_argument("--output", required=True)
    s = sub.add_parser("build-search-bundles"); s.add_argument("--source", required=True); s.add_argument("--output", required=True)
    s = sub.add_parser("compare-search"); s.add_argument("--off", required=True); s.add_argument("--on", required=True); s.add_argument("--output", required=True)
    s = sub.add_parser("synthetic"); s.add_argument("--backend"); s.add_argument("--output", required=True)
    s = sub.add_parser("collect-smoke"); s.add_argument("--config", required=True); s.add_argument("--output", required=True)
    s = sub.add_parser("collect-main"); s.add_argument("--config", required=True); s.add_argument("--smoke", required=True); s.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "freeze":
        if args.questions < 1: p.error("questions must be positive")
        result = freeze_transport(args.source, args.output, args.questions)
    elif args.command == "analyze":
        result = prefix_analysis((row for _, row in read_rows(args.trace)), args.block_size)
        out = new_output(args.output); write(out / "prefix_analysis.json", result)
    elif args.command == "capture":
        result = capture_requests(args.trace, args.output, args.proxy, args.smoke_only)
    elif args.command == "replay":
        result = replay(args.trace, args.output, args.backend, args.identity, args.condition, args.reset_isolated_prefix_cache)
    elif args.command == "search-replay":
        result = search_replay(args.bundle, args.scores, json.loads(args.label_ids), args.output)
    elif args.command == "build-search-bundles":
        result = build_search_bundles(args.source, args.output)
    elif args.command == "compare-search":
        result = compare_search(args.off, args.on, args.output)
    elif args.command == "synthetic":
        result = run_synthetic(args.output, args.backend)
    elif args.command == "collect-smoke":
        result = collect_workload(args.config, args.output)
    elif args.command == "collect-main":
        result = collect_workload(args.config, args.output, questions=50, smoke=args.smoke)
    else:
        result = compare_runs(args.off, args.on, args.output)
    # Never print trace bodies, question text, tokens, or credentials.
    print(json.dumps({"command": args.command, "output": args.output, "status": result.get("status", "written")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
