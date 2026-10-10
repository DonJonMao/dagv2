"""Crash-isolated paired inference experiments; generation never opens labels.

The coordinator uses only the standard library. Native model imports and global
configuration live in persistent spawned workers, one per dataset and arm.
"""
from __future__ import annotations
from dagbt.methods import capabilities, terminal_method

import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import random
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "personamem")
METRICS = ("f1", "em", "r@5", "r@10", "r@20", "all@5", "all@10", "all@20")


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def redacted_error(exc):
    value = str(exc)
    for key, secret in os.environ.items():
        if secret and any(word in key.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "SECRET")):
            value = value.replace(secret, "[REDACTED]")
    return {"error_type": type(exc).__name__, "error": value[:2000]}


def validate_config(config):
    if "_test_transport" in config:
        raise ValueError("_test_transport is forbidden in production runner configuration")
    if config.get("model_profile", "legacy") not in ("legacy", "bridgetree"):
        raise ValueError("model_profile must be legacy or bridgetree")
    # Native caches store URLs/payloads. Credentials must stay in environment headers.
    for key, value in config.items():
        if isinstance(value, dict):
            validate_config(value)
        if any(word in key.lower() for word in ("api_key", "password", "secret", "access_token")):
            raise ValueError(f"{key}: put credentials in environment variables, never config")
        if key.endswith(("base_url", "endpoint")) or key == "url":
            if not isinstance(value, str):
                raise ValueError(f"{key}: expected a URL string")
            parsed = urllib.parse.urlsplit(value)
            if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query:
                raise ValueError(f"{key}: use an HTTP(S) URL without credentials or query parameters")
    experiment = config.get("experiment", {})
    for key in ("question_timeout_seconds", "worker_startup_timeout_seconds", "max_question_attempts"):
        if key in experiment and (not isinstance(experiment[key], (float, int)) or experiment[key] <= 0):
            raise ValueError(f"experiment.{key} must be positive")
    if "max_question_attempts" in experiment and not isinstance(experiment["max_question_attempts"], int):
        raise ValueError("experiment.max_question_attempts must be an integer")
    return config


def frozen_sources():
    paths = [ROOT / "original_manifest.json"]
    for dirname in ("dagbt", "vendor"):
        base = ROOT / dirname
        if base.exists():
            paths.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                         and p.suffix not in (".pyc", ".log"))
    paths.extend(p for p in (ROOT / "scripts").glob("*paired*") if p.is_file())
    if (ROOT / "scripts" / "run_v3.sh").is_file():
        paths.append(ROOT / "scripts" / "run_v3.sh")
    paths.extend(ROOT / 'scripts' / name for name in ('smoke_local_terminal.py','demo_local_terminal.py',
        'benchmark_bt_reranker_prefix_cache.py','smoke_residual_memory.py','demo_residual_memory.py',
        'ablate_residual_memory.py') if (ROOT / 'scripts' / name).is_file())
    paths.append(ROOT / 'serving' / 'bt_prefix_cache_probe.py')
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def verify_originals(*, include_labels=False):
    manifest = load(ROOT / "original_manifest.json")
    errors, checked, deferred = [], 0, []
    for entry in manifest["files"]:
        relative = entry["path"]
        if relative.endswith("evaluation_only.json") and not include_labels:
            deferred.append(relative)
            continue
        path = ROOT / relative
        if not path.is_file() or file_hash(path) != entry["sha256"]:
            errors.append(relative)
        checked += 1
    if errors:
        raise ValueError("Original artifact missing or modified: " + ", ".join(errors))
    return {"checked": checked, "label_hash_checks_deferred_until_scoring": deferred}


def probe_endpoint(base, model, key_env, timeout=10, *, api_key=None):
    headers = {}
    credential = os.environ.get(key_env) if api_key is None else api_key
    if credential:
        headers["Authorization"] = "Bearer " + credential
    request = urllib.request.Request(base.rstrip("/") + "/models", headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    available = [item.get("id") for item in result.get("data", [])]
    if model not in available:
        raise ValueError(f"Requested model {model!r} not advertised by /models")
    selected = [item for item in result.get("data", []) if item.get("id") == model]
    return {"base_url": base, "model": model, "models_endpoint": "ok", "advertised_model_metadata": selected,
            "advertised_metadata_digest": digest(selected),
            "weight_identity_limit": "/models metadata and model ID do not establish a weights checksum or immutable revision"}


def probe_local_tokenizer():
    from transformers import AutoTokenizer
    directory = ROOT / "package" / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    messages = [{"role": "system", "content": "You are a QA reader."},
                {"role": "user", "content": "Return one short answer from the supplied evidence."}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not isinstance(rendered, str) or not rendered.strip() or not token_ids:
        raise ValueError("Bundled tokenizer chat template produced empty output")
    return {"path": str(directory), "local_files_only": True, "chat_template_render": "ok",
            "probe_prompt_tokens": len(token_ids), "probe_render_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "files_sha256": {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()}}


def probe_reranker(config, output=None):
    settings = config.get("reranker", {})
    if not settings.get("url") or not isinstance(settings.get("model", ""), str):
        raise ValueError("BridgeTree requires reranker.url; an empty model uses the service default")
    if settings.get("score_contract", "pointwise") != "pointwise":
        raise ValueError("BridgeTree requires a pointwise reranker")
    from dagbt.bridge import probe_reranker_protocol
    from dagbt.budget import Ledger
    from dagbt.transport import Transport
    base = Path(output) if output else ROOT / "outputs" / "preflight"
    trace = base / "preflight_calls" / f"probe-{time.time_ns()}"
    maximum = 5 * int(config.get("max_identical_attempts", 3))
    ledger = Ledger({"rerank_http": maximum, "http_attempts": maximum},
                    sink=lambda event: append_jsonl(trace / "ledger_events.jsonl", event))
    transport = Transport("preflight_pointwise_protocol", trace, config, ledger, None)
    try:
        report = probe_reranker_protocol(transport, config, ledger)
    except Exception as exc:
        save(trace / "report.json", {"status": "failed", **redacted_error(exc), "ledger": ledger.public_dict(), "cost": request_cost(trace)})
        raise
    result = {"url": settings["url"], "model": settings["model"], "protocol": report,
              "trace_directory": str(trace), "ledger": ledger.public_dict(), "cost": request_cost(trace),
              "scope": "five synthetic composition/order probes; not a proof for all inputs; outside per-question budgets"}
    save(trace / "report.json", result)
    return result


def probe_fusion_planner(config, output=None, method='fusion'):
    """One real, label-free planner request; never repair or change protocols."""
    from copy import deepcopy
    from dagbt.budget import Ledger
    from dagbt.config import resolve
    from dagbt.engine import legacy_modules
    from dagbt.model_runtime import load_tokenizer
    from dagbt.reasoning import Reasoner
    from dagbt.transport import Transport
    from dagbt import prompts

    base = Path(output) if output else ROOT / "outputs" / "preflight"
    trace = base / "preflight_calls" / f"planner-{time.time_ns()}"
    settings = resolve(config, method)
    effective = deepcopy(config)
    effective["fusion"] = settings
    ledger = Ledger({}, sink=lambda event: append_jsonl(trace / "ledger_events.jsonl", event))
    tokenizer = load_tokenizer(effective)
    transport = Transport("preflight_fusion_planner_protocol", trace, effective, ledger, tokenizer)
    events = []
    def observe(event):
        events.append(event)
        append_jsonl(trace / "protocol_events.jsonl", event)
    reasoner = Reasoner(transport, tokenizer, effective, settings, ledger, observe)
    e, repair, _ = legacy_modules()
    from dagbt import local_terminal
    local = terminal_method(method)
    joint = capabilities(method).joint_reading
    from dagbt import residual_plan
    schema = local_terminal.planner_contract(e.PLAN_SCHEMA) if local else e.PLAN_SCHEMA
    system = local_terminal.PLAN_SYSTEM if local else e.PLAN_SYSTEM
    if joint:
        schema = residual_plan.planner_contract(e.PLAN_SCHEMA)
        system = residual_plan.PLAN_SYSTEM
    query = "Which city hosts the science museum visited by the fictional traveler Mira?"
    report = {"check": "fusion_planner_protocol", "version": settings['algorithm_version'], "status": "running",
              "trace_directory": str(trace), "configured_protocol": settings.get("response_format", "plain"),
              "schema_sha256": digest(schema), "synthetic_query": query,
              "logical_call_limit": 1, "repair_calls": 0, "protocol_fallback": False,
              "experiment_task": False,
              "scope": "one synthetic planner request using the actual fusion schema/validator; no labels or question budget; does not establish map/select/reader compatibility or answer quality"}
    save(trace / "report.json", report)
    started = time.monotonic()
    try:
        def validate_plan(value):
            plan = residual_plan.validate_plan(value, repair.validate_plan) if joint else local_terminal.validate_plan(value, repair.validate_plan) if local else repair.validate_plan(value)
            if len(plan["steps"]) > settings["max_initial_nodes"]:
                raise ValueError("Initial node cap exceeded")
            return plan
        plan = reasoner.request("planner", prompts.plan_system(system, personal=False), query,
                                validate_plan, schema, reserve=0)
    except Exception as exc:
        report.update(status="failed", **redacted_error(exc))
        raise
    else:
        report.update(status="passed", validated_plan=plan)
        return report
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, ledger=ledger.public_dict(),
                      cost=request_cost(trace), logical_calls=reasoner.sequence,
                      reasoning_requests=reasoner.requests, response_events=events)
        save(trace / "report.json", report)


def preflight(config, datasets, *, endpoints=True, output=None, arms=None):
    validate_config(config)
    version = config.get('fusion', {}).get('algorithm_version')
    default_method = version if terminal_method(version) else 'fusion'
    arms = list(arms or [default_method])
    from dagbt.config import resolve
    for arm in arms or ["fusion"]:
        if arm != "original":
            resolve(config, arm)
    report = {"original_integrity": verify_originals(), "datasets": {}, "endpoints": {}, "arms": arms,
              "config_digest": digest(config), "created_unix": time.time()}
    dependencies = {"numpy": "numpy", "transformers": "transformers", "yaml": "PyYAML", "jinja2": "Jinja2"}
    missing_modules = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise RuntimeError("Missing dependencies: " + ", ".join(missing_modules))
    report["dependency_versions"] = {name: importlib.metadata.version(distribution) for name, distribution in dependencies.items()}
    from dagbt.model_runtime import is_bridgetree, load_tokenizer, token_accounting, resolve_api_key
    if is_bridgetree(config):
        tokenizer = load_tokenizer(config)
        encoded = tokenizer.apply_chat_template([{"role": "user", "content": "Protocol check."}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
        report["local_tokenizer"] = {**token_accounting(config), "chat_message_adapter": "ok",
                                     "probe_prompt_tokens": len(tokenizer.encode(encoded, add_special_tokens=False))}
    else:
        report["local_tokenizer"] = probe_local_tokenizer()
    report["source_hashes"] = frozen_sources()
    report["original_manifest_sha256"] = file_hash(ROOT / "original_manifest.json")
    report["model_identity"] = {"configured_llm_model": config.get("llm_model"), "configured_embedding_model": config.get("embedding_model"),
                                "configured_reranker_model": config.get("reranker", {}).get("model"),
                                "operator_declared_deployment_identity": config.get("deployment_identity", {}),
                                "operator_declared_reranker_deployment_identity": config.get("reranker", {}).get("deployment_identity", {}),
                                "limitation": "Configured IDs and operator metadata are recorded; no model weights/revision checksum is inferred."}
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError("Unsupported dataset: " + dataset)
        if dataset == "personamem":
            if not is_bridgetree(config):
                raise ValueError("PersonaMem requires the bridgetree model profile and its derived index")
            from dagbt.personamem import validate_dataset
            validate_dataset(ROOT)
        if is_bridgetree(config):
            from dagbt.resources import inspect_index
            index_report = inspect_index(config, dataset)
            directory = ROOT / "data" / dataset
            report["datasets"][dataset] = {"questions": len(load_questions(dataset)),
                "documents": index_report["documents"], "retrieval_index": index_report,
                "current_data_sha256": {name: file_hash(directory / name) for name in ("questions.jsonl", "corpus.jsonl")}}
            if dataset == "personamem":
                report["datasets"][dataset]["current_data_sha256"].update(
                    {name: file_hash(directory / name) for name in ("scopes.json", "manifest.json")})
                report["datasets"][dataset]["task"] = "multiple_choice_personal_memory; per-question context and exclusive cutoff"
            continue
        index = ROOT / "data" / dataset / "index"
        if not (index / "passage_vectors.npy").is_file():
            raise FileNotFoundError(str(index / "passage_vectors.npy"))
        info = load(index / "manifest.json")
        if info["embedding_model"] != config["embedding_model"]:
            raise ValueError("Embedding model differs from frozen corpus index")
        questions = load_questions(dataset)
        directory = ROOT / "data" / dataset
        report["datasets"][dataset] = {"questions": len(questions), "documents": info["documents"],
            "current_data_sha256": {name: file_hash(directory / name) for name in
                                    ("questions.jsonl", "corpus.jsonl", "index/manifest.json", "index/passage_vectors.npy")}}
    if endpoints:
        for kind, env in (("llm", "DAG_LLM_API_KEY"), ("embedding", "DAG_EMBED_API_KEY")):
            report["endpoints"][kind] = probe_endpoint(config[kind + "_base_url"], config[kind + "_model"], env,
                                                      api_key=resolve_api_key(config, kind))
        if config.get("reranker") and any(arm != "original" and resolve(config, arm)["proxy_mode"] == "activation" for arm in arms):
            report["endpoints"]["reranker"] = probe_reranker(config, output)
        if is_bridgetree(config) and any(arm != "original" for arm in arms):
            method = next(arm for arm in arms if arm != 'original')
            if method == 'fusion':
                report["endpoints"]["fusion_planner_protocol"] = probe_fusion_planner(config, output)
            else:
                report["endpoints"]["fusion_planner_protocol"] = probe_fusion_planner(config, output, method)
    else:
        report["endpoints"] = {"status": "not_checked_offline_preflight"}
    report["protocol_note"] = ("BT profile uses its declared deterministic token estimator and chat adapter; missing derived indices will be built before question generation. "
                               if is_bridgetree(config) else "Bundled local chat template is rendered. ") + \
                              "/models checks advertised IDs, not weights. Online BT fusion preflight also requires one real planner request; it does not verify map/select/reader compatibility or accuracy."
    return report


def load_questions(dataset, limit=None):
    if dataset == "personamem":
        from dagbt.personamem import load_questions as load_personamem_questions
        return load_personamem_questions(ROOT, limit=limit)
    questions = [json.loads(line) for line in (ROOT / "data" / dataset / "questions.jsonl").read_text().splitlines() if line.strip()]
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    # Never expose incidental fields to generation, even if an input schema changes.
    result = [{"id": q["id"], "question": q["question"]} for q in questions]
    return result if limit is None else result[:limit]


class AuditedCalls:
    """Native transport semantics plus per-invocation accounting, including cache hits."""
    def __init__(self, unit, output, config, experiment):
        self.unit, self.output, self.config = unit, Path(output), config
        self._experiment = experiment
        self._inner = experiment.Calls(unit, output, config)
        self._adapted = None
        if config.get("model_profile") == "bridgetree":
            from dagbt.transport import Transport
            from dagbt.budget import Ledger
            self._adapted = Transport(unit, output, config, Ledger({}), None)

    def get(self, stage, url, payload):
        if self._adapted is not None:
            return self._adapted.get(stage, url, payload)
        e = self._experiment
        embedding = url.endswith("/embeddings")
        final = dict(payload) if embedding else {**payload, **e.SAMPLING}
        final["model"] = self.config["embedding_model" if embedding else "llm_model"]
        key = e.native.digest({"unit_id": self.unit, "url": url, "payload": final})
        path = self.output / "requests" / (key + ".json")
        before = load(path) if path.exists() else {}
        stage_text = "/".join(map(str, stage)) if isinstance(stage, (list, tuple)) else str(stage)
        kind = "embedding_http" if embedding else "reader" if stage_text.startswith("reader/") else "llm"
        event = {"event": "call_started", "unix": time.time(), "stage": stage, "request_ref": key,
                 "kind": kind, "cache_hit": "response" in before}
        append_jsonl(self.output / "call_events.jsonl", event)
        try:
            result = self._inner.get(stage, url, payload)
        except BaseException as exc:
            append_jsonl(self.output / "call_events.jsonl", {**event, "event": "call_failed", **redacted_error(exc)})
            raise
        append_jsonl(self.output / "call_events.jsonl", {**event, "event": "call_finished", "unix": time.time()})
        return result


def request_cost(output):
    output = Path(output)
    seed_path = output / "seeded_requests.json"
    seeded = set(load(seed_path)) if seed_path.exists() else set()
    event_path = output / "call_events.jsonl"
    events, incomplete_event_lines = [], 0
    for line in event_path.read_text().splitlines() if event_path.exists() else []:
        try:
            if line.strip():
                events.append(json.loads(line))
        except json.JSONDecodeError:
            incomplete_event_lines += 1
    logical = [e for e in events if e.get("event") == "call_started"]
    result = {"logical_calls": len(logical), "cache_hits": sum(bool(e.get("cache_hit")) for e in logical),
              "unique_requests": 0, "http_attempts": 0, "http_retries": 0, "successful_http_requests": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests_by_stage": {},
              "incomplete_or_failed_http_attempts": 0,
              "missing_usage_responses": 0, "incomplete_event_lines": incomplete_event_lines,
              "malformed_request_records": 0,
              "token_scope": "observed successful nonseeded API responses; failed/lost server work may be unreported"}
    stages = Counter()
    by_kind = {}
    def layer(kind):
        kind = "embedding_http" if kind == "embedding" else kind
        kind = kind if kind in {"llm", "reader", "embedding_http", "rerank_http"} else "unknown"
        return by_kind.setdefault(kind, {"logical_calls": 0, "cache_hits": 0, "unique_requests": 0,
            "http_attempts": 0, "http_retries": 0, "successful_http_requests": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "missing_usage_responses": 0, "incomplete_or_failed_http_attempts": 0,
            "observed_http_seconds": 0.0, "timed_http_attempts": 0})
    logical_kinds = {}
    for event in logical:
        current = layer(event.get("kind"))
        current["logical_calls"] += 1
        current["cache_hits"] += int(bool(event.get("cache_hit")))
        logical_kinds[event.get("request_ref")] = event.get("kind")
    for path in (output / "requests").glob("*.json"):
        try:
            record = load(path)
            if not isinstance(record, dict):
                raise ValueError("request record is not an object")
        except (json.JSONDecodeError, ValueError):
            result["malformed_request_records"] += 1
            continue
        stages[json.dumps(record.get("stage"), ensure_ascii=False)] += 1
        result["unique_requests"] += 1
        attempts = record.get("attempts", [])
        kind = logical_kinds.get(path.stem) or next((a.get("kind") for a in attempts if a.get("kind")), None)
        current = layer(kind)
        current["unique_requests"] += 1
        if path.stem in seeded:
            continue
        result["http_attempts"] += len(attempts)
        result["http_retries"] += max(0, len(attempts) - 1)
        result["incomplete_or_failed_http_attempts"] += sum(a.get("http_status") != 200 for a in attempts)
        current["http_attempts"] += len(attempts)
        current["http_retries"] += max(0, len(attempts) - 1)
        current["incomplete_or_failed_http_attempts"] += sum(a.get("http_status") != 200 for a in attempts)
        for attempt in attempts:
            start, end = attempt.get("started_unix"), attempt.get("finished_unix")
            if (_finite_number(start) and _finite_number(end) and end >= start):
                current["observed_http_seconds"] += end - start
                current["timed_http_attempts"] += 1
        if "response" in record:
            result["successful_http_requests"] += 1
            current["successful_http_requests"] += 1
            usage = record["response"].get("usage", {})
            if not isinstance(usage, dict) or not usage:
                result["missing_usage_responses"] += 1
                current["missing_usage_responses"] += 1
                usage = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                result[key] += int(usage.get(key, 0) or 0)
                current[key] += int(usage.get(key, 0) or 0)
    result["requests_by_stage"] = dict(stages)
    result["by_kind"] = by_kind
    result["reasoning_logical_calls"] = by_kind.get("llm", {}).get("logical_calls", 0)
    result["reader_logical_calls"] = by_kind.get("reader", {}).get("logical_calls", 0)
    result["physical_llm_attempts"] = by_kind.get("llm", {}).get("http_attempts", 0)
    result["physical_reader_attempts"] = by_kind.get("reader", {}).get("http_attempts", 0)
    result["layer_scope"] = "logical invocations include cache replay; physical attempts/usage exclude seeded old requests; reasoning and reader are separate"
    result["token_usage_complete"] = not any(result[k] for k in ("missing_usage_responses", "incomplete_or_failed_http_attempts", "malformed_request_records"))
    return result


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def original_question(q, resources, calls, e):
    """Exact e.work computation/catches, with Calls supplied for accounting.

    e.work's filesystem resume wrapper is handled by our attempt coordinator.
    All planner/archive/controller/reader algorithms remain native functions.
    """
    try:
        docs, ids, vectors, index, tokenizer = resources
        plan = e.make_plan(q["question"], calls)
        if getattr(calls, "config", {}).get("model_profile") == "bridgetree":
            from dagbt.resources import archive_for_resources
            pool, trace = archive_for_resources(q["question"], plan, ids, vectors, calls, e)
        else:
            pool, trace = e.archive(q["question"], plan, ids, vectors, calls)
        row = dict(unit_id=q["id"], question=q["question"], plan=plan, candidate_doc_ids=pool, archive_trace=trace)
        save(calls.output / "input.json", row)
        result = e.native.solve(row, docs, tokenizer, index, calls)
        # controller mutates this pool with corpus-wide node hits. Capture it only
        # after solve, without modifying any native selection or reader behavior.
        result["diagnostics"] = {**result.get("diagnostics", {}), "candidate_doc_ids": list(row["candidate_doc_ids"]),
                                 "archive_doc_ids": list(dict.fromkeys(d for t in trace for d in t.get("doc_ids", []))),
                                 "retrieval_query_count": len(trace) + result.get("ranking", {}).get("embedding_requests", 0),
                                 "semantic_validation": "legacy nonempty answer/source heuristic, not entailment verification"}
        return result
    except e.PlanError as exc:
        return e.failure(q["id"], "planner_failed", exc)
    except AssertionError as exc:
        if not exc.args or not isinstance(exc.args[0], tuple) or exc.args[0][0] != "context_budget":
            raise
        return e.failure(q["id"], "context_overflow", exc)


def configure_personamem_reader():
    """Adapt the shared reader before either algorithm counts prompt tokens."""
    reader = importlib.import_module("reader")
    original = reader.reader_messages
    if getattr(original, "_personamem_adapter", False):
        return

    def messages(**kwargs):
        result = original(**kwargs)
        marker = "Answer with the shortest exact phrase supported by the context passages."
        if marker not in result[-1]["content"]:
            raise ValueError("PersonaMem reader cannot locate the frozen QA output instructions")
        result[-1]["content"] = result[-1]["content"].rsplit(marker, 1)[0] + (
            "Choose the single best answer option for the user's question using only the supplied "
            "personal memory. These memories belong to this question's shared conversation and "
            "precede its cutoff. Respect speaker roles and changes in the user's preferences. "
            "The options are candidate answers, not evidence. The source guide only locates evidence. "
            "Return exactly one line with the selected option letter, with no explanation:\nAnswer: (a)\n"
            "Replace a with the letter of the selected option.")
        return result

    messages._personamem_adapter = True
    reader.reader_messages = messages
    # The frozen QA parser truncates at the first answer line. Preserve an
    # invalid whole response here so explanations/multiple choices cannot be
    # silently accepted just because their first line contains a valid label.
    def parse_answer(text):
        from dagbt.personamem import parse_choice
        choice = parse_choice(text, ["a", "b", "c", "d"])
        return f"({choice})" if choice is not None else str(text).strip()
    reader.parse_answer = parse_answer


def native_worker(connection, dataset, arm, config):
    # The environment prevents inherited server configuration changing native imports.
    os.environ.pop("DAGV2_CONFIG", None)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        sys.path.insert(0, str(ROOT / "dagv2"))
        pipeline = importlib.import_module("pipeline_dagv2_musique" if dataset == "musique" else "pipeline_dagv2")
        e = pipeline.e
        e.CONFIG.update(config)
        if config.get("model_profile") == "bridgetree":
            from dagbt.resources import prepare_resources
            from dagbt.model_runtime import load_tokenizer
            _, resources, hashes = prepare_resources(config, dataset, pipeline, load_tokenizer(config))
        else:
            _, resources, hashes = pipeline.prepare(dataset, ROOT / "outputs")
        memory_scopes, public_questions = None, None
        if dataset == "personamem":
            from dagbt.personamem import load_scopes
            memory_scopes = load_scopes(ROOT)
            public_questions = {q["id"]: q for q in load_questions(dataset)}
            configure_personamem_reader()
        connection.send({"type": "ready", "hashes": hashes})
    except BaseException as exc:
        connection.send({"type": "startup_error", **redacted_error(exc)})
        return
    while True:
        try:
            task = connection.recv()
        except EOFError:
            return
        if task is None:
            return
        q, output = task["question"], Path(task["output"])
        started = time.monotonic()
        try:
            question_resources = resources
            if memory_scopes is not None:
                from dagbt.resources import scope_resources
                if q != public_questions.get(q["id"]):
                    raise ValueError("PersonaMem task differs from its frozen public question/scope")
                question_resources = scope_resources(resources, memory_scopes[q["scope_id"]])
            calls = AuditedCalls(q["id"], output, e.CONFIG, e)
            generation_question = {"id": q["id"], "question": q["question"]}
            if arm == "original":
                row = original_question(generation_question, question_resources, calls, e)
            else:
                from dagbt.engine import run_question
                if dataset == "personamem":
                    # Only the current request reaches fusion planning/evidence;
                    # the final MCQ reader receives the original public options.
                    fusion_question = {"id": q["id"], "question": q["user_question"]}
                    row = run_question(fusion_question, question_resources, calls, e.CONFIG,
                                       method=arm, reader_question=q["question"], output_options=q['options'])
                else:
                    row = run_question(generation_question, question_resources, calls, e.CONFIG, method=arm)
            if memory_scopes is not None:
                from dagbt.personamem import parse_choice
                row.setdefault("diagnostics", {})["memory_scope"] = {
                    "scope_id": q["scope_id"], "shared_context_id": q["shared_context_id"],
                    "end_index_exclusive": q["end_index"], "visible_doc_ids": question_resources[1]}
                if row["answer"]["status"] == "ok" and parse_choice(row["answer"]["prediction"], q["options"]) is None:
                    row["answer"]["status"] = "invalid_choice"
                    row["answer"]["error"] = "Reader must return one unambiguous option label"
            from dagbt.model_runtime import token_accounting
            row.setdefault("diagnostics", {})["token_accounting"] = token_accounting(config)
            row["seconds"] = time.monotonic() - started
            connection.send({"type": "result", "row": row})
        except BaseException as exc:
            connection.send({"type": "task_error", **redacted_error(exc)})


class Worker:
    def __init__(self, dataset, arm, config, *, target=None, context=None):
        self.dataset, self.arm, self.config = dataset, arm, config
        self.target = target or native_worker
        self.context = context or mp.get_context("spawn")
        self.process = self.connection = None

    def start(self):
        if self.process is not None and self.process.is_alive():
            return
        self.close()
        parent, child = self.context.Pipe()
        self.connection = parent
        self.process = self.context.Process(target=self.target, args=(child, self.dataset, self.arm, self.config), daemon=True)
        self.process.start()
        child.close()
        timeout = self.config.get("experiment", {}).get("worker_startup_timeout_seconds", 300)
        message = self._receive(timeout)
        if message.get("type") != "ready":
            self.close()
            raise RuntimeError("Worker preparation failed: " + json.dumps(message))
        self.hashes = message.get("hashes", {})

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError(f"{self.dataset}/{self.arm}: worker exceeded {timeout}s")
        try:
            return self.connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"{self.dataset}/{self.arm}: worker exited (code {self.process.exitcode})") from exc

    def run(self, q, output):
        try:
            self.start()
            if self.hashes:
                resource_record = Path(output).parents[2] / "resource_hashes.json"
                if resource_record.exists() and load(resource_record) != self.hashes:
                    raise RuntimeError("Worker resource hashes changed during this run")
                save(resource_record, self.hashes)
            self.connection.send({"question": q, "output": str(output)})
            message = self._receive(self.config.get("experiment", {}).get("question_timeout_seconds", 3600))
            if message.get("type") != "result":
                raise RuntimeError("Question failed: " + json.dumps(message))
            return message["row"]
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(3)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(3)
            else:
                self.process.join(0)
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None


def validate_result(row, unit):
    if row.get("unit_id") != unit or not isinstance(row.get("answer"), dict):
        raise ValueError("Malformed result identity/answer")
    local = terminal_method(row.get('algorithm_version'))
    prediction = row['answer'].get('prediction')
    if not isinstance(row["answer"].get("status"), str) or (not isinstance(prediction, str) and not (local and prediction is None and row['answer']['status'] != 'ok')):
        raise ValueError("Malformed answer fields")
    if local:
        if row['answer'].get('answer_source') != 'dag_terminal' or row.get('budgets') != {}:
            raise ValueError('Malformed terminal result contract')
        if row['answer']['status'] == 'ok' and (not prediction or not row['answer'].get('final_node_id') or not row['answer'].get('sources')):
            raise ValueError('Successful terminal requires current supported sources')
        return row
    for k in ("5", "10", "20"):
        ids = row.get("budgets", {}).get(k, {}).get("selected_doc_ids")
        if not isinstance(ids, list) or len(ids) > int(k) or len(set(ids)) != len(ids):
            raise ValueError("Malformed retrieval selection at k=" + k)
    return row


def failure_row(q, exc, arm=None):
    row = {"unit_id": q["id"], "answer": {"status": "timeout" if isinstance(exc, TimeoutError) else "execution_failed", "prediction": "", **redacted_error(exc)},
            "ranking": {"status": "execution_failed"},
            "budgets": {str(k): {"selected_doc_ids": []} for k in (5, 10, 20)}}
    if terminal_method(arm):
        row.update(algorithm_version=arm, method=arm, budgets={})
        row['answer'].update(answer_source='dag_terminal', final_node_id=None, sources=None)
    return row


def result_path(output, dataset, arm, unit):
    return Path(output) / dataset / arm / "rows" / (digest(unit) + ".json")


def seed_cache(previous, current):
    import shutil
    seeded, skipped = [], []
    for directory in previous:
        for path in (directory / "requests").glob("*.json"):
            try:
                record = load(path)
                if not isinstance(record, dict):
                    raise ValueError("request record is not an object")
            except (json.JSONDecodeError, ValueError):
                skipped.append(str(path))
                continue
            if "response" in record:
                dest = current / "requests" / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                temporary = dest.with_name(dest.name + ".copy.tmp")
                shutil.copy2(path, temporary)
                temporary.replace(dest)
                seeded.append(path.stem)
    save(current / "seeded_requests.json", sorted(set(seeded)))
    if skipped:
        save(current / "cache_seed_skipped.json", {"invalid_records_preserved": skipped})


def generate_dataset(output, dataset, questions, arms, config, *, retry_failed=False, worker_factory=Worker):
    """One arm attempt per invocation; retries require explicit --retry-failed."""
    output = Path(output)
    workers = {arm: worker_factory(dataset, arm, config) for arm in arms}
    maximum = config.get("experiment", {}).get("max_question_attempts", 3)
    counts = Counter()
    try:
        for qi, q in enumerate(questions):
            # Deterministic rotation avoids always giving the first arm server warm-up.
            order = arms[qi % len(arms):] + arms[:qi % len(arms)]
            for arm in order:
                path = result_path(output, dataset, arm, q["id"])
                if path.exists():
                    prior = validate_result(load(path), q["id"])
                    if prior["answer"]["status"] == "ok" or not retry_failed:
                        counts["resumed_terminal"] += 1
                        continue
                unit_dir = output / dataset / arm / "attempts" / digest(q["id"])
                previous = sorted(p for p in unit_dir.glob("attempt-*") if p.is_dir())
                if len(previous) >= maximum:
                    # A killed coordinator may leave an attempt but no terminal row.
                    if not path.exists():
                        row = failure_row(q, RuntimeError("Persisted question-attempt cap reached"), arm)
                        row["attempt_directories"] = [str(p.relative_to(output)) for p in previous]
                        save(path, row)
                    counts["attempt_limit_reached"] += 1
                    continue
                attempt = unit_dir / f"attempt-{len(previous) + 1:03d}"
                attempt.mkdir(parents=True)
                seed_cache(previous, attempt)
                save(attempt / "task.json", {"dataset": dataset, "arm": arm, "question": q, "started_unix": time.time()})
                save(output / "progress.json", {"state": "generating", "dataset": dataset, "unit_id": q["id"], "arm": arm,
                                               "question_index": qi, "questions_in_dataset": len(questions), "updated_unix": time.time()})
                started = time.monotonic()
                try:
                    row = validate_result(workers[arm].run(q, attempt), q["id"])
                except Exception as exc:
                    row = failure_row(q, exc, arm)
                row["runner"] = {"dataset": dataset, "arm": arm, "attempt": len(previous) + 1,
                                 "wall_seconds": time.monotonic() - started, "cost": request_cost(attempt)}
                row["attempt_directories"] = [str(p.relative_to(output)) for p in [*previous, attempt]]
                save(attempt / "result.json", row)
                save(path, row)
                if row["answer"]["status"] != "ok":
                    save(output / dataset / arm / "failures" / (digest(q["id"]) + f"-attempt-{len(previous) + 1:03d}.json"), row)
                append_jsonl(output / "events.jsonl", {"event": "arm_terminal", "dataset": dataset, "unit_id": q["id"], "arm": arm,
                                                        "status": row["answer"]["status"], "attempt": len(previous) + 1, "unix": time.time()})
                counts[row["answer"]["status"]] += 1
    finally:
        for worker in workers.values():
            worker.close()
    return dict(counts)


def all_terminal(output, scopes, arms):
    for dataset, questions in scopes.items():
        for q in questions:
            for arm in arms:
                path = result_path(output, dataset, arm, q["id"])
                if not path.is_file():
                    return False
                validate_result(load(path), q["id"])
    return True


def total_cost(output, dataset, arm):
    numeric = Counter()
    layers = {}
    for attempt in (Path(output) / dataset / arm / "attempts").glob("*/attempt-*"):
        cost = request_cost(attempt)
        for key, value in cost.items():
            if _finite_number(value):
                numeric[key] += value
        for kind, values in cost["by_kind"].items():
            layers.setdefault(kind, Counter()).update(values)
    result = dict(numeric)
    result["by_kind"] = {kind: dict(values) for kind, values in layers.items()}
    result["token_usage_complete"] = not any(result.get(k, 0) for k in ("missing_usage_responses", "incomplete_or_failed_http_attempts", "malformed_request_records"))
    result["token_scope"] = "observed successful responses only; incomplete usage means token totals are lower bounds, not zero cost"
    return result


def current_diagnostics(row, output):
    """Recover only a failed current row's latest snapshot, never old outcomes."""
    diag = row.get("diagnostics", {})
    diag = dict(diag) if isinstance(diag, dict) else {}
    if row.get("answer", {}).get("status") != "ok" and row.get("attempt_directories"):
        partial = Path(output) / row["attempt_directories"][-1] / "fusion_partial.json"
        if partial.is_file():
            snapshot = load(partial)
            snapshot_diag = snapshot.get("diagnostics", {})
            snapshot_diag = snapshot_diag if isinstance(snapshot_diag, dict) else {}
            if not diag:
                diag = {**snapshot, **snapshot_diag, "candidate_doc_ids": snapshot.get("candidates", []),
                        "support_graph": {"nodes": snapshot.get("nodes", [])}}
            for name in ("reliability", "semantic_evidence"):
                value = snapshot_diag.get(name, snapshot.get(name))
                if name not in diag and isinstance(value, dict):
                    diag[name] = value
            for name in ("reasoning_call_count", "evidence_logical_calls", "reader_logical_calls",
                         "budgeted_llm_attempts", "budgeted_reader_attempts"):
                value = snapshot_diag.get(name, snapshot.get(name))
                if name not in diag and _finite_number(value):
                    diag[name] = value
    return diag


def normalized_semantic_evidence(diag):
    """Evidence/coverage are independent of transport protocol reliability."""
    source = diag.get("semantic_evidence")
    result = dict(source) if isinstance(source, dict) else {}
    state = result.get("evidence_state", "unknown")
    if state not in EVIDENCE_STATES:
        state = "unknown"
    selection_unknown = (state == "unknown" and "evidence_state" in result
                         or result.get("phase") == "selection_pending")
    if selection_unknown:
        state = "unknown"
    keys = ("selected_doc_ids", "selected_mapped_doc_ids", "selected_raw_only_doc_ids")
    if all(isinstance(result.get(key), list) for key in keys):
        selected, mapped, raw = (set(result[key]) for key in keys)
        if mapped.isdisjoint(raw) and selected == mapped | raw:
            if not selection_unknown:
                state = "mixed" if mapped and raw else "mapped_only" if mapped else "raw_only" if raw else "empty_context"
        else:
            state = "unknown"
            result["state_integrity"] = "selected evidence lists do not form a disjoint complete partition"
    result["evidence_state"] = state
    complete = result.get("coverage_validation_complete")
    unassessed = result.get("unassessed_requirement_ids")
    if complete is False or bool(unassessed):
        coverage = "unassessed"
    elif complete is True:
        coverage = "complete"
    else:
        coverage = "unknown"
    result["coverage_state"] = coverage
    for name in ("evidence_logical_calls", "reader_logical_calls", "budgeted_llm_attempts", "budgeted_reader_attempts"):
        # Authoritative per-run counters, never sums of copied event snapshots.
        if name not in result and _finite_number(diag.get(name)):
            result[name] = diag[name]
    if _finite_number(diag.get("reasoning_call_count")):
        # A selection-progress snapshot can predate the failed HTTP request.
        # The current engine counter includes that attempted logical call.
        result["evidence_logical_calls"] = diag["reasoning_call_count"]
    return result


def normalized_reliability(diag):
    reliability = diag.get("reliability")
    reliability = dict(reliability) if isinstance(reliability, dict) else {}
    if reliability.get("cohort") not in RELIABILITY_COHORTS:
        reliability["cohort"] = "unknown"
    return reliability


def module_metrics(row, label, dataset, output):
    """Gold-based discovery metrics and model/structural diagnostics stay distinct."""
    diag = current_diagnostics(row, output)
    reliability = normalized_reliability(diag)
    candidates = diag.get("candidate_doc_ids")
    def group_recall(ids):
        if dataset == "personamem":
            return None, None
        ids = set(map(str, ids))
        if dataset == "musique":
            ids = {d if d.startswith("musique:") else "musique:" + d for d in ids}
        hits = [bool(ids.intersection(g)) for g in label["gold_groups"]]
        return sum(hits) / len(hits), float(all(hits))
    candidate_recall, candidate_all = group_recall(candidates) if candidates is not None else (None, None)
    selected = row.get("budgets", {}).get("20", {})
    selected_recall, _ = group_recall(selected.get("selected_doc_ids", []))
    nodes = diag.get("support_graph", {}).get("nodes", row.get("ranking", {}).get("nodes", []))
    events = diag.get("events", [])
    protocol = [event for event in events if event.get("event") == "protocol_error"]
    quote_errors = [event for event in protocol if any(word in str(event.get("error", "")).lower() for word in ("quote", "span", "offset"))]
    feasibility = diag.get("reader_feasibility", {})
    ledger = diag.get("ledger", {}).get("used", {})
    metered_limits = diag.get("ledger", {}).get("limits", {})
    statuses = Counter(node.get("status", "legacy_resolved" if node.get("resolved") else "legacy_unresolved") for node in nodes)
    result = {"candidate_count": len(candidates) if candidates is not None else None,
            "gold_candidate_title_group_recall": candidate_recall,
            "gold_candidate_all_support": candidate_all,
            "gold_discovery_minus_selection_recall_at20": candidate_recall - selected_recall if candidate_recall is not None else None,
            "selected_doc_count_at20": len(selected.get("selected_doc_ids", [])),
            "model_structural_node_status_counts": dict(statuses),
            "model_unknown_nodes": statuses.get("unknown", 0), "model_ambiguous_nodes": statuses.get("ambiguous", 0),
            "structural_complete_required_at20": selected.get("complete_required"),
            "structural_necessary_covered_at20": selected.get("necessary_covered"),
            "protocol_error_events": len(protocol) if "events" in diag else None,
            "quote_or_span_protocol_error_events": len(quote_errors) if "events" in diag else None,
            "protocol_errors_by_operation": dict(Counter(str(e.get("operation")) for e in protocol)),
            "reader_prompt_tokens_local": feasibility.get("prompt_token_count"),
            "reader_context_tokens_with_output_reserve": feasibility.get("token_count", selected.get("token_count")),
            "local_token_accounting": diag.get("token_accounting", {k: feasibility[k] for k in
                                       ("token_count_is_estimate", "token_estimator_id") if k in feasibility}),
            "reader_prompt_tokens_api": (row.get("answer", {}).get("response_usage") or {}).get("prompt_tokens"),
            "retrieval_query_count": ledger.get("ann", 0 if "ann" in metered_limits else diag.get("retrieval_query_count")),
            "set_score_count": ledger.get("set_score", 0 if "set_score" in metered_limits or "retrieval_query_count" in diag else None),
            "ledger_used": ledger,
            "reliability": reliability,
            "semantic_evidence": normalized_semantic_evidence(diag),
            "interpretation": ("PersonaMem has no gold document support labels; gold_* metrics are unavailable. "
                               "Model/structural statuses do not prove semantic correctness." if dataset == "personamem" else
                               "gold_* uses held-out title-group labels after generation; model/structural statuses and complete_required do not prove semantic entailment")}
    if terminal_method(row.get('algorithm_version')):
        for key in ('gold_discovery_minus_selection_recall_at20','selected_doc_count_at20',
                    'structural_complete_required_at20','structural_necessary_covered_at20',
                    'reader_prompt_tokens_local','reader_context_tokens_with_output_reserve','reader_prompt_tokens_api'):
            result[key] = None
        source_ids = (row['answer'].get('sources') or {}).get('doc_ids', [])
        result.update(legacy_reader_metrics='not_applicable',
            semantic_evidence={'status':'not_applicable','reason':'Legacy FinalSelector evidence/coverage metrics'},
            terminal_source_doc_count=len(source_ids),
            gold_terminal_source_title_group_recall=group_recall(source_ids)[0],
            terminal_input_doc_count=len(row['answer'].get('terminal_input_doc_ids', [])),
            terminal_input_tokens_local=(diag.get('terminal_input') or {}).get('request', {}).get('input_tokens_local'),
            answer_source='dag_terminal')
    return result


def summarize_modules(rows):
    keys = ("candidate_count", "gold_candidate_title_group_recall", "gold_candidate_all_support",
            "gold_discovery_minus_selection_recall_at20", "selected_doc_count_at20", "model_unknown_nodes",
            "model_ambiguous_nodes", "structural_complete_required_at20", "structural_necessary_covered_at20",
            "protocol_error_events", "quote_or_span_protocol_error_events", "reader_prompt_tokens_local",
            "reader_context_tokens_with_output_reserve", "reader_prompt_tokens_api", "retrieval_query_count", "set_score_count")
    summary = {}
    for key in keys:
        values = [row["modules"][key] for row in rows if row["modules"][key] is not None]
        summary[key] = {"observed_tasks": len(values), "missing_tasks": len(rows) - len(values),
                        "sum": sum(values), "mean": sum(values) / len(values) if values else None}
    return summary


RELIABILITY_COHORTS = ("normal", "truncated", "partially_mapped", "truncated_and_partially_mapped", "unknown")
EVIDENCE_STATES = ("empty_context", "mapped_only", "raw_only", "mixed", "unknown")
COVERAGE_STATES = ("complete", "unassessed", "unknown")


def latest_cost_statistics(rows):
    """Top-level metrics retain their schema; named layers use dotted paths."""
    def numeric_paths(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = prefix + key
            if _finite_number(item):
                result[name] = item
            elif isinstance(item, dict) and (prefix or key == "by_kind"):
                result.update(numeric_paths(item, name + "."))
        return result
    observed = [r.get("latest_attempt_cost") for r in rows]
    values = [numeric_paths(cost) if isinstance(cost, dict) else {} for cost in observed]
    metrics = {}
    for key in sorted({key for cost in values for key in cost}):
        numbers = [cost[key] for cost in values if key in cost]
        metrics[key] = {"observed_tasks": len(numbers), "missing_tasks": len(rows) - len(numbers),
                        "sum": sum(numbers), "mean": sum(numbers) / len(numbers)}
    return {"observed_tasks": sum(isinstance(cost, dict) for cost in observed),
            "missing_tasks": sum(not isinstance(cost, dict) for cost in observed), "metrics": metrics}


def summarize_reliability(rows, answer_metric):
    """Only the currently authoritative rows enter success/failure cohorts.

    Costs here describe each row's latest attempt. Historical failed attempts
    remain in cost_all_attempts, never in a success cohort after a retry.
    """
    groups = {}
    for cohort in RELIABILITY_COHORTS:
        subset = [r for r in rows if r["valid"] and
                  r.get("modules", {}).get("reliability", {}).get("cohort", "unknown") == cohort]
        correct = sum(r[answer_metric] == 1 for r in subset)
        groups[cohort] = {"tasks": len(subset), "correct": correct,
                          answer_metric: correct / len(subset) if subset else None,
                          "cost_latest_attempt": latest_cost_statistics(subset)}
    failed = [r for r in rows if not r["valid"]]
    return {"answer_metric": answer_metric, "completion_cohorts": groups,
            "failed_tasks": len(failed), "failure_cost_latest_attempt": latest_cost_statistics(failed),
            "scope": "current authoritative task rows; cohorts contain successful answers only; costs use latest attempts; all retry costs remain in cost_all_attempts",
            "interpretation": "Cohorts contain different questions; their accuracies and costs are descriptive, not evidence of a causal benefit from truncation or partial mapping."}


def summarize_semantic_evidence(rows, answer_metric=None):
    """Current outcomes only; labels are optional for read-only live diagnostics."""
    if rows and all(r.get('modules',{}).get('semantic_evidence',{}).get('status') == 'not_applicable' for r in rows):
        return {'status':'not_applicable','reason':'Legacy FinalSelector evidence/coverage metrics; inspect terminal sources'}
    def semantic(row):
        return row.get("modules", {}).get("semantic_evidence", {})
    def group(subset):
        result = {"tasks": len(subset), "cost_latest_attempt": latest_cost_statistics(subset)}
        if answer_metric is not None:
            correct = sum(r[answer_metric] == 1 for r in subset)
            result.update(correct=correct)
            result[answer_metric] = correct / len(subset) if subset else None
        return result
    def counts(subset, key, names):
        return {name: sum(semantic(row).get(key, "unknown") == name for row in subset) for name in names}
    successful = [r for r in rows if r["valid"]]
    failed = [r for r in rows if not r["valid"]]
    evidence_groups = {state: group([r for r in successful if semantic(r).get("evidence_state", "unknown") == state])
                       for state in EVIDENCE_STATES}
    coverage_groups = {state: group([r for r in successful if semantic(r).get("coverage_state", "unknown") == state])
                       for state in COVERAGE_STATES}
    cross_counts = Counter((r.get("modules", {}).get("reliability", {}).get("cohort", "unknown"),
                            semantic(r).get("evidence_state", "unknown"),
                            semantic(r).get("coverage_state", "unknown")) for r in successful)
    numeric = {}
    for name in ("baseline_retention_ratio", "evidence_logical_calls", "reader_logical_calls",
                 "budgeted_llm_attempts", "budgeted_reader_attempts"):
        values = [semantic(r)[name] for r in rows if _finite_number(semantic(r).get(name))]
        numeric[name] = {"observed_tasks": len(values), "missing_tasks": len(rows) - len(values),
                         "sum": sum(values), "mean": sum(values) / len(values) if values else None}
    return {"answer_metric": answer_metric, "completion_cohorts": evidence_groups,
            "coverage_completion_cohorts": coverage_groups,
            "all_task_evidence_state_counts": counts(rows, "evidence_state", EVIDENCE_STATES),
            "all_task_coverage_state_counts": counts(rows, "coverage_state", COVERAGE_STATES),
            "failed_tasks": len(failed),
            "failed_evidence_state_counts": counts(failed, "evidence_state", EVIDENCE_STATES),
            "failed_coverage_state_counts": counts(failed, "coverage_state", COVERAGE_STATES),
            "failure_cost_latest_attempt": latest_cost_statistics(failed),
            "successful_reliability_evidence_coverage_counts": [
                {"reliability": cohort, "evidence_state": evidence, "coverage_state": coverage, "tasks": count}
                for (cohort, evidence, coverage), count in sorted(cross_counts.items())],
            "current_task_diagnostic_metrics": numeric,
            "scope": "current authoritative task rows only; latest failed partial snapshot may enrich missing diagnostics; live events and historical attempt results are not additional outcomes",
            "interpretation": "Protocol normality, semantic evidence composition and coverage validation are independent. Complete coverage validation does not prove that evidence is sufficient or the answer is correct. Cohort comparisons are descriptive, not causal."}


def paired_bootstrap_interval(differences, *, replicates=1000, seed=20260918):
    """Resample question-level four-arm contrasts, never arms independently."""
    n = len(next(iter(differences.values()), []))
    report = {"method": "paired_question_percentile_bootstrap", "confidence_level": .95,
              "question_n": n, "replicates": replicates, "seed": seed,
              "estimable": n >= 2, "intervals_percentage_points": None}
    if n < 2:
        return {**report, "reason": "fewer_than_two_paired_questions"}
    if any(len(values) != n for values in differences.values()):
        raise ValueError("Paired bootstrap metric vectors must have matching question counts")
    rng = random.Random(seed)
    sampled = {metric: [] for metric in differences}
    for _ in range(replicates):
        # Each selected index carries the same question's original four arm
        # scores, already reduced to its exact paired interaction contrast.
        indices = [rng.randrange(n) for _ in range(n)]
        for metric, values in differences.items():
            sampled[metric].append(100 * sum(values[i] for i in indices) / n)
    def percentile(values, quantile):
        values = sorted(values)
        location = (len(values) - 1) * quantile
        lower = int(location)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (location - lower)
    report["intervals_percentage_points"] = {metric: {"lower": percentile(values, .025), "upper": percentile(values, .975)}
                                              for metric, values in sampled.items()}
    return report


def factorial_interaction(all_scores, units, metric_names=METRICS):
    factors = ("fusion", "bt_flat", "dense_dependency", "dense_flat")
    if not set(factors) <= set(all_scores):
        return None
    shared = [unit for unit in units if all(all_scores[arm][unit]["valid"] for arm in factors)]
    def differences(subset):
        return {metric: [
            (all_scores["fusion"][unit][metric] - all_scores["bt_flat"][unit][metric])
            - (all_scores["dense_dependency"][unit][metric] - all_scores["dense_flat"][unit][metric])
            for unit in subset] for metric in metric_names}
    all_differences, shared_differences = differences(units), differences(shared)
    def point_estimates(values):
        if not next(iter(values.values()), []):
            return None
        return {metric: 100 * sum(vector) / len(vector) for metric, vector in values.items()}
    return {"formula": "(fusion - bt_flat) - (dense_dependency - dense_flat)",
            "factors": {"retrieval": ["bridge", "dense"], "selection": ["dependency", "flat"]},
            "all_task_n": len(units), "shared_four_success_n": len(shared),
            "all_task_interaction_percentage_points": point_estimates(all_differences),
            "shared_four_success_interaction_percentage_points": point_estimates(shared_differences),
            "all_task_interaction_ci95": paired_bootstrap_interval(all_differences),
            "shared_four_success_interaction_ci95": paired_bootstrap_interval(shared_differences),
            "shared_four_success_unit_ids": shared,
            "interpretation": "Descriptive paired interactions with question-level 95% percentile bootstrap intervals (1000 resamples, fixed seed). Assumes independent, exchangeable questions; shared sources can weaken independence. Intervals condition on these tasks and fixed model outputs, not model-generation randomness, and are not causal proof. Four-arm success conditioning can select a different population; original-arm failure does not exclude a four-arm success."}


def score_all(output, scopes, arms, *, label_loader=None):
    """The label loader is invoked only after *all datasets and arms* are terminal."""
    output = Path(output)
    if not all_terminal(output, scopes, arms):
        raise ValueError("Generation incomplete; evaluation labels remain unread")
    if label_loader is None:
        verify_originals(include_labels=True)
        if "personamem" in scopes:
            from dagbt.personamem import validate_dataset
            validate_dataset(ROOT, include_labels=True)
        label_loader = lambda dataset: load(ROOT / "data" / dataset / "evaluation_only.json")
    spec = importlib.util.spec_from_file_location("_dagbt_metrics", ROOT / "package" / "metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    summaries = {}
    for dataset, questions in scopes.items():
        is_personamem = dataset == "personamem"
        metric_names = ("accuracy",) if is_personamem else ('f1','em') if any(terminal_method(arm) for arm in arms) else METRICS
        answer_metric = "accuracy" if is_personamem else "em"
        labels = {r["id"]: r for r in label_loader(dataset)}
        all_scores, comparisons = {}, []
        for arm in arms:
            scored, statuses = [], Counter()
            for q in questions:
                row = load(result_path(output, dataset, arm, q["id"]))
                label, valid = labels[q["id"]], row["answer"]["status"] == "ok"
                choice = None
                answer_status = row["answer"]["status"]
                if is_personamem:
                    from dagbt.personamem import parse_choice
                    choice = parse_choice(row["answer"]["prediction"], q["options"])
                    if valid and choice is None:
                        valid, answer_status = False, "invalid_choice"
                statuses[answer_status] += 1
                item = {"unit_id": q["id"], "valid": valid, "answer_status": answer_status,
                        "prediction": row["answer"]["prediction"]}
                groups = None
                if is_personamem:
                    correct = parse_choice(label["correct_answer"], q["options"])
                    if correct is None:
                        raise ValueError("Invalid PersonaMem evaluation choice")
                    item.update(accuracy=float(valid and choice == correct), predicted_choice=choice,
                                persona_id=q["persona_id"], question_type=q.get("question_type"))
                else:
                    item.update(f1=metrics.token_f1(row["answer"]["prediction"], label["answers"]) if valid else 0.,
                                em=metrics.exact_match(row["answer"]["prediction"], label["answers"]) if valid else 0.)
                    groups = label["gold_groups"]
                    if not groups or not all(groups):
                        raise ValueError("Empty gold title group")
                item["modules"] = module_metrics(row, label, dataset, output)
                item["result_path"] = str(result_path(output, dataset, arm, q["id"]).relative_to(output))
                item["attempt_directories"] = row.get("attempt_directories", [])
                local = terminal_method(row.get('algorithm_version'))
                item["selected_doc_ids"] = {} if local else {k: row["budgets"][k]["selected_doc_ids"] for k in ("5", "10", "20")}
                if local:
                    item['legacy_selection_metrics'] = 'not_applicable'
                item["latest_attempt_cost"] = row.get("runner", {}).get("cost")
                for k in ("5", "10", "20"):
                    if is_personamem or local:
                        continue
                    ids = set(map(str, row["budgets"][k]["selected_doc_ids"]))
                    if dataset == "musique":
                        ids = {d if d.startswith("musique:") else "musique:" + d for d in ids}
                    hits = [bool(ids.intersection(g)) for g in groups]
                    item["r@" + k], item["all@" + k] = sum(hits) / len(hits), float(all(hits))
                scored.append(item)
            all_scores[arm] = {s["unit_id"]: s for s in scored}
            save(output / dataset / arm / "scores.json", scored)
        n = len(questions)
        common = [q["id"] for q in questions if all(all_scores[a][q["id"]]["valid"] for a in arms)]
        summary = {"dataset": dataset, "n": n, "common_success_n": len(common), "arms": {}, "paired": {},
                   "support_metric": (None if is_personamem else "macro recall of gold title groups; any matching document satisfies a group"),
                   "task": "personal_memory_multiple_choice" if is_personamem else "multi_hop_question_answering",
                   "failure_policy": "answer failure scores zero on all-task answer metrics; retrieval metrics use recorded selections"}
        if is_personamem:
            summary["support_metric_note"] = "Not available: PersonaMem provides no gold supporting-document labels"
            summary["choice_protocol"] = "Single option label only; ambiguous, out-of-range or explanatory responses are invalid_choice"
        for arm in arms:
            rows = list(all_scores[arm].values())
            summary["arms"][arm] = {"answer_status_counts": dict(Counter(s["answer_status"] for s in rows)),
                "failure_rate": sum(not s["valid"] for s in rows) / n,
                "all_task_metrics_percent": {k: 100 * sum(s[k] for s in rows) / n for k in metric_names},
                "common_success_metrics_percent": {k: 100 * sum(all_scores[arm][u][k] for u in common) / len(common) for k in metric_names} if common else None,
                "module_metrics": summarize_modules(rows),
                "reliability": summarize_reliability(rows, answer_metric),
                "semantic_evidence": summarize_semantic_evidence(rows, answer_metric),
                "cost_all_attempts": total_cost(output, dataset, arm)}
            if is_personamem:
                persona_scores = {}
                for item in rows:
                    persona_scores.setdefault(item["persona_id"], []).append(item["accuracy"])
                macro = sum(sum(values) / len(values) for values in persona_scores.values()) / len(persona_scores)
                summary["arms"][arm].update(persona_macro_accuracy_percent=100 * macro,
                                            personas=len(persona_scores))
        baseline = arms[0]
        for arm in arms[1:]:
            pair_ids = [q["id"] for q in questions if all_scores[baseline][q["id"]]["valid"] and all_scores[arm][q["id"]]["valid"]]
            summary["paired"][arm] = {"baseline": baseline, "pairwise_common_success_n": len(pair_ids),
                "all_task_delta_percentage_points": {k: 100 * sum(all_scores[arm][q["id"]][k] - all_scores[baseline][q["id"]][k] for q in questions) / n for k in metric_names},
                "common_success_delta_percentage_points": {k: 100 * sum(all_scores[arm][u][k] - all_scores[baseline][u][k] for u in pair_ids) / len(pair_ids) for k in metric_names} if pair_ids else None,
                f"rescue_{answer_metric}_common_success": sum(all_scores[arm][u][answer_metric] == 1 and all_scores[baseline][u][answer_metric] == 0 for u in pair_ids),
                f"harm_{answer_metric}_common_success": sum(all_scores[arm][u][answer_metric] == 0 and all_scores[baseline][u][answer_metric] == 1 for u in pair_ids)}
        interaction = factorial_interaction(all_scores, [q["id"] for q in questions], metric_names)
        if interaction is not None:
            summary["factorial_interaction"] = interaction
        for q in questions:
            comparison = {"unit_id": q["id"], "question": q["question"], "arms": {a: all_scores[a][q["id"]] for a in arms}}
            comparisons.append(comparison)
        dest = output / dataset
        (dest / "comparisons.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in comparisons))
        with (dest / "comparisons.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["unit_id", "question", *[f"{a}.{k}" for a in arms for k in ("answer_status", "prediction", *metric_names)]])
            for r in comparisons:
                writer.writerow([r["unit_id"], r["question"], *[r["arms"][a][k] for a in arms for k in ("answer_status", "prediction", *metric_names)]])
        save(dest / "summary.json", summary)
        summaries[dataset] = summary
    save(output / "summary.json", summaries)
    return summaries


@contextlib.contextmanager
def writer_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "writer.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This output directory already has a live writer") from exc
        yield


def validate_experiment_options(config, args):
    arms = args.arms
    if len(set(arms)) != len(arms) or len(arms) < 2 or arms[0] != "original":
        raise ValueError("Use at least two distinct arms, with original first")
    from dagbt.config import METHODS, resolve
    for arm in arms[1:]:
        if arm not in METHODS:
            raise ValueError("Unsupported fusion arm: " + arm)
        resolved = resolve(config, arm)
        if resolved["retrieval"] == "bridge" and resolved["proxy_mode"] == "activation" and not config.get("reranker"):
            raise ValueError("Activation-scored Bridge arms require a configured pointwise reranker")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")


def run_experiment(args):
    config = validate_config(load(args.config))
    validate_experiment_options(config, args)
    output = Path(args.output).resolve()
    datasets, arms = args.datasets, args.arms
    with writer_lock(output):
        process_command = subprocess.check_output(["ps", "-p", str(os.getpid()), "-o", "command="], text=True).strip()
        save(output / "pid.json", {"pid": os.getpid(), "pgid": os.getpgrp(), "started_unix": time.time(),
                                  "output": str(output), "process_command": process_command})
        save(output / "progress.json", {"state": "validating", "updated_unix": time.time()})
        scopes = {d: load_questions(d, args.limit) for d in datasets}
        manifest = {"schema": 1, "config": config, "datasets": datasets, "arms": arms,
                    "question_ids": {d: [q["id"] for q in qs] for d, qs in scopes.items()},
                    "questions_digest": {d: digest(qs) for d, qs in scopes.items()}, "source_hashes": frozen_sources(),
                    "original_hash_manifest": file_hash(ROOT / "original_manifest.json"),
                    "scope": "all_packaged_questions_per_dataset" if args.limit is None else f"first_{args.limit}_per_dataset",
                    "dataset_manifests": {d: file_hash(ROOT / "data" / d / "manifest.json")
                                          for d in datasets if d == "personamem"},
                    "baseline": ("native make_plan/native.solve; original archive algorithm with query-dimension guard tied to new index; shared BT model/resource adapter replaces backend protocol, token accounting and corpus vectors"
                                 if config.get("model_profile") == "bridgetree" else
                                 "untouched native make_plan/archive/native.solve, same e.work failure handling; wrapper replaces filesystem resume and call accounting only"),
                    "generation": "all configured dataset/arm tasks terminal before first label load", "kind": "inference_evaluation_not_parameter_training"}
        path = output / "manifest.json"
        if path.exists() and load(path) != manifest:
            raise ValueError("Config, source, data, arms or scope changed; use a new output directory")
        save(path, manifest)
        # Detached launch already probes the deployment. Reuse only its fresh,
        # identical configuration report; still recheck all original file hashes.
        launch_report = output / "launch_preflight.json"
        prior_probe = load(launch_report) if launch_report.is_file() else {}
        reusable = (not args.offline_preflight and prior_probe.get("config_digest") == digest(config)
                    and prior_probe.get("source_hashes") == manifest["source_hashes"]
                    and set(prior_probe.get("datasets", {})) == set(datasets)
                    and prior_probe.get("arms") == arms
                    and 0 <= time.time() - prior_probe.get("created_unix", 0) < 300
                    and "llm" in prior_probe.get("endpoints", {})
                    and (config.get("model_profile") != "bridgetree" or
                         prior_probe.get("endpoints", {}).get("fusion_planner_protocol", {}).get("status") == "passed"))
        if reusable:
            if "personamem" in datasets:
                from dagbt.personamem import validate_dataset
                validate_dataset(ROOT)
            report = {**prior_probe, "original_integrity": verify_originals(), "reused_launch_preflight": True}
        else:
            report = preflight(config, datasets, endpoints=not args.offline_preflight, output=output, arms=arms)
        save(output / "preflight.json", report)
        # SIGTERM raises through finally blocks, closing all persistent arm workers.
        old_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            if config.get("model_profile") == "bridgetree":
                from dagbt.resources import ensure_index, inspect_index
                index_artifacts = {}
                for dataset in datasets:
                    save(output / "progress.json", {"state": "preparing_index", "dataset": dataset,
                                                    "updated_unix": time.time()})
                    ensure_index(config, dataset, output)
                    index_artifacts[dataset] = inspect_index(config, dataset)
                identity_path = output / "index_artifacts.json"
                if identity_path.exists() and load(identity_path) != index_artifacts:
                    raise ValueError("Derived embedding index changed; use a new output directory")
                save(identity_path, index_artifacts)
            for dataset, questions in scopes.items():
                generate_dataset(output, dataset, questions, arms, config, retry_failed=args.retry_failed)
            save(output / "progress.json", {"state": "scoring", "updated_unix": time.time()})
            summaries = score_all(output, scopes, arms)
            failures = sum(sum(v for k, v in a["answer_status_counts"].items() if k != "ok") for d in summaries.values() for a in d["arms"].values())
            save(output / "progress.json", {"state": "complete" if not failures else "complete_with_failures", "failed_arm_tasks": failures, "updated_unix": time.time()})
        except BaseException as exc:
            save(output / "progress.json", {"state": "stopped" if isinstance(exc, KeyboardInterrupt) else "failed", **redacted_error(exc), "updated_unix": time.time()})
            raise
        finally:
            signal.signal(signal.SIGTERM, old_term)


def lock_is_held(output):
    path = Path(output) / "writer.lock"
    if not path.exists():
        return False
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def status(output):
    output = Path(output)
    return {"live_writer": lock_is_held(output),
            "pid": load(output / "pid.json") if (output / "pid.json").exists() else None,
            "progress": load(output / "progress.json") if (output / "progress.json").exists() else None,
            "log": str(output / "launcher.log")}


def diagnostics(output):
    """Read current outcomes without scoring, reading labels or changing files."""
    output = Path(output)
    report = {"output": str(output), **status(output), "datasets": {},
              "labels_read": False, "answer_accuracy_available": False,
              "scope": "manifest question IDs and current rows only; no historical result/event replay"}
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return {**report, "state": "no_manifest", "note": "No run manifest found; no experiment was started."}
    manifest = load(manifest_path)
    for dataset in manifest.get("datasets", []):
        units = manifest.get("question_ids", {}).get(dataset, [])
        if len(set(units)) != len(units):
            raise ValueError("Duplicate manifest question IDs")
        dataset_report = {}
        for arm in manifest.get("arms", []):
            rows, invalid = [], []
            for unit in units:
                path = result_path(output, dataset, arm, unit)
                if not path.is_file():
                    continue
                try:
                    row = validate_result(load(path), unit)
                except (ValueError, KeyError, TypeError) as exc:
                    invalid.append({"unit_id": unit, "path": str(path.relative_to(output)), **redacted_error(exc)})
                    continue
                diag = current_diagnostics(row, output)
                rows.append({"unit_id": unit, "valid": row["answer"]["status"] == "ok",
                             "answer_status": row["answer"]["status"],
                             "latest_attempt_cost": row.get("runner", {}).get("cost"),
                             "modules": {"reliability": normalized_reliability(diag),
                                         "semantic_evidence": normalized_semantic_evidence(diag)}})
            dataset_report[arm] = {
                "expected_tasks": len(units), "current_tasks": len(rows),
                "pending_or_invalid_tasks": len(units) - len(rows), "invalid_current_rows": invalid,
                "answer_status_counts": dict(Counter(r["answer_status"] for r in rows)),
                "successful_reliability_counts": {cohort: sum(r["valid"] and
                    r["modules"]["reliability"]["cohort"] == cohort for r in rows) for cohort in RELIABILITY_COHORTS},
                "semantic_evidence": summarize_semantic_evidence(rows),
                "cost_latest_attempt": latest_cost_statistics(rows),
                "cost_all_attempts": total_cost(output, dataset, arm)}
        report["datasets"][dataset] = dataset_report
    return report


def stop(output):
    output = Path(output).resolve()
    info = status(output)
    if not info["live_writer"]:
        return {"stopped": False, "reason": "no live OS-locked writer; stale PID file ignored"}
    record = info["pid"]
    if not record or record.get("output") != str(output):
        raise RuntimeError("No valid PID record for live writer")
    pid = record["pid"]
    command = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True).strip()
    # Comparing the command recorded after taking the writer lock handles both
    # absolute detached-launch paths and relative foreground --output paths.
    same_command = command == record["process_command"] if record.get("process_command") else str(output) in command
    if "dagbt.runner" not in command or not same_command:
        raise RuntimeError("PID identity mismatch; refusing to signal unrelated process")
    os.kill(pid, signal.SIGTERM)
    return {"stopped": True, "signal": "SIGTERM", "pid": pid}


def launch(args):
    output = Path(args.output).resolve()
    config = validate_config(load(args.config))
    validate_experiment_options(config, args)
    if lock_is_held(output):
        raise RuntimeError("Output already has a live experiment")
    report = preflight(config, args.datasets, endpoints=not args.offline_preflight, output=output, arms=args.arms)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "launch_preflight.json", report)
    command = [sys.executable, "-m", "dagbt.runner", "run", "--config", str(Path(args.config).resolve()), "--output", str(output), "--datasets", *args.datasets, "--arms", *args.arms]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if args.retry_failed:
        command += ["--retry-failed"]
    if args.offline_preflight:
        command += ["--offline-preflight"]
    with (output / "launcher.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(output / "launch.json", {"pid": process.pid, "command": command, "unix": time.time()})
    return {"launched_pid": process.pid, "output": str(output), "log": str(output / "launcher.log"), "note": "Use status to confirm the writer started; launch does not imply completion."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "launch", "preflight"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
        command.add_argument("--offline-preflight", action="store_true", help="explicitly skip all endpoint/protocol preflight calls; model calls during run still occur")
        if name != "preflight":
            command.add_argument("--output", required=True)
            command.add_argument("--arms", nargs="+", default=["original", "fusion"])
            command.add_argument("--limit", type=int)
            command.add_argument("--retry-failed", action="store_true")
    for name in ("status", "stop", "diagnostics"):
        command = sub.add_parser(name)
        command.add_argument("--output", required=True)
    command = sub.add_parser("prepare-index", help="build/resume BT-profile corpus embeddings without running question generation")
    command.add_argument("--config", required=True)
    command.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    command.add_argument("--output", default=str(ROOT / "outputs" / "index_preparation"))
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run_experiment(args)
            result = status(args.output)
        elif args.command == "launch":
            result = launch(args)
        elif args.command == "preflight":
            result = preflight(load(args.config), args.datasets, endpoints=not args.offline_preflight)
        elif args.command == "prepare-index":
            config = validate_config(load(args.config))
            if config.get("model_profile") != "bridgetree":
                raise ValueError("prepare-index requires model_profile=bridgetree; legacy packaged vectors remain unchanged")
            verify_originals()
            from dagbt.resources import ensure_index, inspect_index
            with writer_lock(args.output):
                for dataset in args.datasets:
                    ensure_index(config, dataset, Path(args.output))
                result = {d: inspect_index(config, d) for d in args.datasets}
        elif args.command == "stop":
            result = stop(args.output)
        elif args.command == "diagnostics":
            result = diagnostics(args.output)
        else:
            result = status(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if args.command == "run":
            output = Path(args.output)
            pidfile = output / "pid.json"
            if pidfile.is_file() and load(pidfile).get("pid") == os.getpid() and not lock_is_held(output):
                save(output / "progress.json", {"state": "failed", **redacted_error(exc), "updated_unix": time.time()})
        print(json.dumps(redacted_error(exc), ensure_ascii=False), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
