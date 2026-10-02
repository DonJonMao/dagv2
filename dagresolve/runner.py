"""Independent, resumable DAG-Resolve runner over the frozen three datasets."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import importlib
import json
from pathlib import Path
import time

from dagbt.runner import (AuditedCalls, digest, file_hash, load, redacted_error,
                          request_cost, save, verify_originals)
from . import METHOD
from .flow import solve
from .runtime import DATASETS, ROOT, import_originals, load_runtime, validate_config


METRICS = ("f1", "em", "r@5", "r@10", "r@20", "all@5", "all@10", "all@20")


def code_hashes():
    paths = [*sorted((ROOT / "dagresolve").glob("*.py")),
             ROOT / "dagbt" / "runner.py", ROOT / "dagbt" / "transport.py",
             ROOT / "dagbt" / "model_runtime.py"]
    script = ROOT / "scripts" / "run_resolve.sh"
    if script.is_file():
        paths.append(script)
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def row_path(output, unit):
    return Path(output) / "rows" / (digest(unit) + ".json")


def call_directory(output, unit):
    return Path(output) / "calls" / digest(unit)


def _cost(output):
    """Keep native cost accounting, but reject partial or invalid token usage.

    Request records are never rewritten. If malformed usage would make the
    shared meter raise, reconstruct its basic counters from observed records.
    """
    output = Path(output)
    records, malformed = [], 0
    for path in (output / "requests").glob("*.json"):
        try:
            record = load(path)
            if not isinstance(record, dict):
                raise ValueError("Request record is not an object")
            records.append((path.stem, record))
        except (OSError, ValueError, TypeError):
            malformed += 1
    invalid_usage = 0
    invalid_by_kind = Counter()
    strict_totals = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    strict_by_kind = {}
    def kind_of(record):
        stage = record.get("stage", "")
        stage = "/".join(map(str, stage)) if isinstance(stage, (list, tuple)) else str(stage)
        return ("embedding_http" if str(record.get("url", "")).rstrip("/").endswith("/embeddings")
                else "reader" if stage.startswith("reader/") else "llm")
    for _, record in records:
        if "response" not in record:
            continue
        response = record["response"]
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        kind = kind_of(record)
        required = ("prompt_tokens",) if kind == "embedding_http" else tuple(strict_totals)
        valid = isinstance(usage, dict) and all(
            type(usage.get(key)) is int and usage[key] >= 0 for key in required)
        valid = valid and all(type(value) is int and value >= 0
                             for key, value in usage.items() if key in strict_totals)
        invalid_usage += int(not valid)
        invalid_by_kind[kind] += int(not valid)
        layer = strict_by_kind.setdefault(kind, {key: 0 for key in strict_totals})
        if isinstance(usage, dict):
            for key in strict_totals:
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    strict_totals[key] += value
                    layer[key] += value
    try:
        result = request_cost(output)
    except (ValueError, TypeError, OverflowError, AttributeError):
        # Invalid API fields must not conceal already incurred HTTP work.
        events, bad_events = [], 0
        path = output / "call_events.jsonl"
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Invalid call event")
                if event.get("event") == "call_started":
                    events.append(event)
            except (ValueError, TypeError):
                bad_events += 1
        result = {"logical_calls": len(events), "cache_hits": sum(bool(e.get("cache_hit")) for e in events),
                  "unique_requests": len(records), "http_attempts": 0, "http_retries": 0,
                  "successful_http_requests": 0, "incomplete_or_failed_http_attempts": 0,
                  "malformed_request_records": malformed, "incomplete_event_lines": bad_events,
                  "requests_by_stage": {}, "by_kind": {}}
        stages = Counter()
        for kind in ("embedding_http", "reader", "llm"):
            selected = [record for _, record in records if kind_of(record) == kind]
            kind_events = [event for event in events if event.get("kind") == kind]
            attempts = [attempt for record in selected for attempt in record.get("attempts", [])]
            layer = {"logical_calls": len(kind_events),
                     "cache_hits": sum(bool(e.get("cache_hit")) for e in kind_events),
                     "unique_requests": len(selected), "http_attempts": len(attempts),
                     "http_retries": sum(max(0, len(record.get("attempts", [])) - 1) for record in selected),
                     "successful_http_requests": sum("response" in record for record in selected),
                     "incomplete_or_failed_http_attempts": sum(
                         not isinstance(attempt, dict) or attempt.get("http_status") != 200 for attempt in attempts)}
            result["by_kind"][kind] = layer
            for key in ("http_attempts", "http_retries", "successful_http_requests", "incomplete_or_failed_http_attempts"):
                result[key] += layer[key]
        for _, record in records:
            stages[json.dumps(record.get("stage"), ensure_ascii=False)] += 1
        result["requests_by_stage"] = dict(stages)
        result["reasoning_logical_calls"] = result["by_kind"]["llm"]["logical_calls"]
        result["reader_logical_calls"] = result["by_kind"]["reader"]["logical_calls"]
        result["physical_llm_attempts"] = result["by_kind"]["llm"]["http_attempts"]
        result["physical_reader_attempts"] = result["by_kind"]["reader"]["http_attempts"]
    result.update(strict_totals)
    for kind, values in strict_by_kind.items():
        layer = result.setdefault("by_kind", {}).setdefault(kind, {})
        layer.update(values)
        layer["missing_usage_responses"] = invalid_by_kind[kind]
    result["missing_usage_responses"] = invalid_usage
    result["invalid_or_missing_usage_responses"] = invalid_usage
    result["token_usage_complete"] = not any(result.get(key, 0) for key in (
        "invalid_or_missing_usage_responses", "incomplete_or_failed_http_attempts",
        "malformed_request_records", "incomplete_event_lines"))
    result["token_scope"] = "observed nonnegative integer API fields only; failed or lost server work may be unreported"
    return result


def make_manifest(runtime, questions, limit):
    return {"schema": 1, "method": METHOD, "dataset": runtime.config["dataset"],
            "requested_config": runtime.requested_config, "config": runtime.config,
            "scope": "full1000" if limit is None else "explicit_subset",
            "unit_ids": [question["id"] for question in questions],
            "questions_digest": digest(questions), "asset_hashes": runtime.hashes,
            "code_hashes": code_hashes(),
            "original_manifest_sha256": file_hash(ROOT / "original_manifest.json"),
            "sampling": runtime.e.SAMPLING, "workers": 1,
            "planner_system": runtime.e.PLAN_SYSTEM, "planner_schema": runtime.e.PLAN_SCHEMA,
            "context_limit": 16384, "planner_output_tokens": 2048,
            "node_output_tokens": 512, "proposal_output_tokens": 1024,
            "judge_output_tokens": 1024, "reader_output_tokens": 1024,
            "protocol": {"max_dag_nodes": 6, "max_candidate_answers": 2,
                         "candidate_proposal": "first dependency-ordered bridge node only; at most one proposal",
                         "max_binding_repairs_per_question": 1,
                         "max_extra_embedding_queries": 2, "max_joint_judgments": 1,
                         "reader": "original v6 chain reader; explicit reader stage",
                         "generation_labels_read": False,
                         "retry_scope": "at most three persisted identical HTTP attempts"}}


def validate_saved_row(row, unit):
    if (not isinstance(row, dict) or row.get("unit_id") != unit
            or not isinstance(row.get("answer"), dict)
            or not isinstance(row["answer"].get("status"), str)
            or not isinstance(row["answer"].get("prediction"), str)
            or not isinstance(row.get("budgets"), dict)
            or any(not isinstance(row["budgets"].get(k), dict)
                   or not isinstance(row["budgets"][k].get("selected_doc_ids"), list)
                   for k in ("5", "10", "20"))):
        raise ValueError("Invalid saved result for " + str(unit))
    return row


def collect_cost(output):
    units = {path.name: _cost(path) for path in sorted((Path(output) / "calls").glob("*"))
             if path.is_dir()}
    numeric = ("logical_calls", "cache_hits", "unique_requests", "http_attempts", "http_retries",
               "successful_http_requests", "prompt_tokens", "completion_tokens", "total_tokens",
               "incomplete_or_failed_http_attempts", "missing_usage_responses",
               "malformed_request_records", "incomplete_event_lines", "reasoning_logical_calls",
               "reader_logical_calls", "physical_llm_attempts", "physical_reader_attempts")
    totals = {name: sum(cost.get(name, 0) for cost in units.values()) for name in numeric}
    totals["token_usage_complete"] = all(cost["token_usage_complete"] for cost in units.values())
    totals["token_scope"] = "observed successful responses; failed or lost server work may be unreported"
    result = {"units": units, "totals": totals}
    save(Path(output) / "cost.json", result)
    return result


def work(question, runtime, output):
    e = runtime.e
    documents, ids, vectors, index, tokenizer = runtime.resources
    calls = AuditedCalls(question["id"], call_directory(output, question["id"]), runtime.config, e)
    input_path = Path(output) / "inputs" / (digest(question["id"]) + ".json")
    started = time.monotonic()
    planner_capacity = None
    try:
        if input_path.is_file():
            row = load(input_path)
            if row.get("unit_id") != question["id"] or row.get("question") != question["question"]:
                raise ValueError("Saved input question identity changed")
            row["plan"] = e.validate_plan(row["plan"])
            if (not isinstance(row.get("candidate_doc_ids"), list)
                    or not row["candidate_doc_ids"]
                    or len(set(row["candidate_doc_ids"])) != len(row["candidate_doc_ids"])
                    or not set(row["candidate_doc_ids"]) <= set(documents)):
                raise ValueError("Saved candidate archive is invalid")
        else:
            messages = [{"role": "system", "content": e.PLAN_SYSTEM},
                        {"role": "user", "content": question["question"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            count = len(tokenizer.encode(prompt, add_special_tokens=False))
            planner_capacity = {"input_tokens": count, "output_token_reserve": 2048,
                                "safety_margin": 8, "context_limit": 16384,
                                "total_tokens_local": count + 2048 + 8,
                                "within_budget": count + 2048 + 8 <= 16384}
            if not planner_capacity["within_budget"]:
                raise AssertionError(("context_budget", count))
            plan = e.make_plan(question["question"], calls)
            pool, trace = e.archive(question["question"], plan, ids, vectors, calls)
            row = {"unit_id": question["id"], "question": question["question"],
                   "plan": plan, "candidate_doc_ids": pool, "archive_trace": trace}
            save(input_path, row)
        # The controller mutates the candidate pool. Keep the saved initial
        # archive immutable so interrupted calls replay the same execution.
        result = solve(deepcopy(row), documents, tokenizer, index, calls, runtime=runtime)
    except e.PlanError as exc:
        result = e.failure(question["id"], "planner_failed", exc)
    except AssertionError as exc:
        if not exc.args or not isinstance(exc.args[0], tuple) or exc.args[0][0] != "context_budget":
            raise
        result = e.failure(question["id"], "context_overflow", exc)
        if planner_capacity is not None and not planner_capacity["within_budget"]:
            result["ranking"]["trace"].append({"stage": "planner_capacity", **planner_capacity})
    result = validate_saved_row(result, question["id"])
    if planner_capacity is not None:
        result["planner_capacity"] = planner_capacity
    result["seconds"] = time.monotonic() - started
    result["runner_cost"] = _cost(calls.output)
    save(row_path(output, question["id"]), result)
    return result


def run(runtime, output, limit=None):
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(runtime.questions)):
        raise ValueError("limit must be an integer within the dataset size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    questions = runtime.questions if limit is None else runtime.questions[:limit]
    manifest = make_manifest(runtime, questions, limit)
    with (output / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "manifest.json"
        if path.is_file() and load(path) != manifest:
            raise ValueError("Run method, code, config, data or scope changed; use a new output directory")
        if not path.exists() and any((output / name).exists() for name in ("rows", "inputs", "calls")):
            raise ValueError("Output contains run artifacts without a manifest")
        save(path, manifest)
        completed = 0
        started = time.monotonic()
        def status(state, **extra):
            save(output / "progress.json", {"state": state, "method": METHOD,
                 "dataset": runtime.config["dataset"], "completed": completed,
                 "total": len(questions), "elapsed_seconds": time.monotonic() - started,
                 "updated_unix": time.time(), "labels_read": False, **extra})
        try:
            for question in questions:
                path = row_path(output, question["id"])
                if path.exists():
                    validate_saved_row(load(path), question["id"])
                else:
                    status("running", active_unit=question["id"])
                    work(question, runtime, output)
                completed += 1
                status("running", last_unit=question["id"])
                print(f"{completed}/{len(questions)}", flush=True)
        except BaseException as exc:
            collect_cost(output)
            status("paused", **redacted_error(exc))
            raise
        collect_cost(output)
        status("complete")
    return {"completed": completed, "total": len(questions), "output": str(output)}


def evaluate(output, *, dataset=None, config=None):
    output = Path(output)
    with (output / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, progress = load(output / "manifest.json"), load(output / "progress.json")
        if manifest.get("method") != METHOD or manifest.get("dataset") not in DATASETS:
            raise ValueError("Not a supported DAG-Resolve run")
        if dataset is not None and dataset != manifest["dataset"]:
            raise ValueError("Evaluation dataset differs from the saved run")
        if config is not None and validate_config(config) != manifest["requested_config"]:
            raise ValueError("Evaluation configuration differs from the saved run")
        if progress.get("state") != "complete" or progress.get("completed") != len(manifest["unit_ids"]):
            raise ValueError("Generation is incomplete; evaluation labels remain unopened")
        rows = [load(path) for path in (output / "rows").glob("*.json")]
        by_unit = {row.get("unit_id"): row for row in rows}
        if len(rows) != len(manifest["unit_ids"]) or set(by_unit) != set(manifest["unit_ids"]):
            raise ValueError("Generation result scope is incomplete or duplicated; labels remain unopened")
        for unit, row in by_unit.items():
            validate_saved_row(row, unit)
        if manifest["code_hashes"] != code_hashes():
            raise ValueError("Research code changed after generation")
        # The first access to label bytes happens only after the complete scope
        # and terminal result contracts have been checked above.
        verify_originals(include_labels=True)
        dataset = manifest["dataset"]
        labels_path = ROOT / "data" / dataset / "evaluation_only.json"
        labels = {row["id"]: row for row in load(labels_path)}
        if not set(by_unit) <= set(labels):
            raise ValueError("Evaluation references do not cover the run")
        import_originals(dataset)
        metrics = importlib.import_module("metrics")
        scored = []
        for unit in manifest["unit_ids"]:
            row, label = by_unit[unit], labels[unit]
            valid = row["answer"]["status"] == "ok"
            item = {"unit_id": unit, "valid": valid,
                    "f1": metrics.token_f1(row["answer"]["prediction"], label["answers"]) if valid else 0.,
                    "em": metrics.exact_match(row["answer"]["prediction"], label["answers"]) if valid else 0.}
            groups = label["gold_groups"]
            if not groups or not all(groups):
                raise ValueError("Invalid evaluation support groups")
            for k in ("5", "10", "20"):
                ids = set(row["budgets"][k]["selected_doc_ids"])
                if dataset == "musique":
                    ids = {str(doc) if str(doc).startswith("musique:") else "musique:" + str(doc) for doc in ids}
                hits = [bool(ids.intersection(group)) for group in groups]
                item["r@" + k] = sum(hits) / len(hits)
                item["all@" + k] = float(all(hits))
            scored.append(item)
        n = len(scored)
        if not n:
            raise ValueError("Empty evaluation scope")
        summary = {"method": METHOD, "dataset": dataset, "n": n, "scope": manifest["scope"],
                   "invalid_answers": sum(not item["valid"] for item in scored),
                   "answer_status_counts": dict(Counter(row["answer"]["status"] for row in rows)),
                   "metrics_percent": {name: 100 * sum(item[name] for item in scored) / n for name in METRICS},
                   "labels_sha256": file_hash(labels_path), "cost": collect_cost(output)["totals"],
                   "support_metric": "macro recall of gold title groups; any matching document satisfies a group"}
        save(output / "scores.json", scored)
        save(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "evaluate"))
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--config", default=str(ROOT / "config.example.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "dagresolve"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.limit is not None and args.action != "run":
        parser.error("--limit applies only to run; evaluate uses the saved scope")
    config = validate_config(load(args.config))
    if args.action == "evaluate":
        evaluate(args.output, dataset=args.dataset, config=config)
        return
    runtime = load_runtime(args.dataset, config)
    if args.action == "check":
        documents, _, vectors, _, tokenizer = runtime.resources
        print(json.dumps({"method": METHOD, "dataset": args.dataset,
              "questions": len(runtime.questions), "documents": len(documents),
              "embedding_dimensions": vectors.shape[1], "tokenizer_loaded": bool(tokenizer),
              "original_integrity": runtime.integrity, "asset_hashes": runtime.hashes,
              "code_hashes": code_hashes(), "network_calls": 0, "labels_read": False},
              ensure_ascii=False, indent=2))
    else:
        run(runtime, args.output, args.limit)


if __name__ == "__main__":
    main()
