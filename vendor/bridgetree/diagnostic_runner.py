"""Frozen, budgeted post-hoc measurements; never changes the legacy algorithm.

Planning is offline. Online entry points require an explicit execute flag, a
matching deployment snapshot, and a predeclared physical HTTP attempt cap.
Correct labels are deliberately absent from this module's interfaces.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import random
import tarfile
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .clients import GeneratorClient, RerankerClient, build_context_plan, build_rerank_payload
from .dependency_scoring import SetReranker
from .diagnostic_config import digest, load_diagnostic_config
from .diagnostic_identity import request_hash
from .diagnostic_observability import observe
from .experiment import _visible_memory_records
from .personamem import load_shared_contexts, messages_to_memories
from .types import ContextPlan, Memory


def atomic_json(path: str | Path, value: Any) -> None:
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not an object")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid durable ledger {path.name} line {i}") from exc
            result.append(row)
    return result


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def run_lock(root: Path):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".diagnostic.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another diagnostic executor holds this run") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def source_identity() -> str:
    from .protocol import _source_package_snapshot
    return _source_package_snapshot()


def text_hash(value: str) -> str:
    """Match request-audit document hashes (raw UTF-8, not JSON strings)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deployment_snapshot(config: Any) -> dict:
    result = {}
    for role in ("embedding", "reranker", "generator"):
        value = getattr(config.models, role)
        identity = getattr(value, "deployment_identity", None)
        identity = asdict(identity) if is_dataclass(identity) else dict(identity or {})
        record = {"endpoint_sha256": text_hash(value.endpoint),
                  "model": value.model, "deployment_identity": identity,
                  "timeout_seconds": value.timeout_seconds}
        names = {"embedding": ("backend", "query_instruction", "batch_size", "local_model_path"),
                 "reranker": ("score_space", "score_contract", "task_instruction"),
                 "generator": ("temperature", "max_tokens", "context_token_budget", "provider_request_params")}
        for name in names[role]:
            field = getattr(value, name, None)
            record[name] = dict(field) if isinstance(field, Mapping) else field
        result[role] = record
    return result


def online_identity_ready(snapshot: dict, roles: tuple[str, ...]) -> list[str]:
    missing = []
    for role in roles:
        identity = snapshot[role]["deployment_identity"]
        source = identity.get("identity_source", identity.get("source", "unknown"))
        revision = any(identity.get(k) for k in ("checkpoint_revision", "weights_sha256", "deployment_revision"))
        if source in (None, "", "unknown") or not revision:
            missing.append(role)
    return missing


def _records(case: dict) -> dict[str, Memory]:
    return {m["memory_id"]: Memory(**m) for m in case["visible_memories"]}


def load_public_cases(settings: dict, config: Any) -> list[dict]:
    """Read only the public projection; never construct a gold-bearing example."""
    contexts = load_shared_contexts(settings["contexts"])
    wanted = {x["question_id"]: x for x in settings["cases"]}
    found = {}
    with Path(settings["questions"]).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            if qid not in wanted:
                continue
            if qid in found:
                raise ValueError("duplicate diagnostic question")
            # No correct_answer, answer-derived metadata or reference position
            # is projected into a scoring / generation case.
            public = {k: row[k] for k in ("question_id", "persona_id", "user_question_or_message",
                                         "all_options", "shared_context_id", "end_index_in_shared_context")}
            context_id = public["shared_context_id"]
            end = int(public["end_index_in_shared_context"])
            if context_id not in contexts or not 0 <= end <= len(contexts[context_id]):
                raise ValueError("invalid visible history prefix")
            query_time = row.get("query_time") or row.get("query_date") or row.get("cutoff") or None
            metadata = {k: row[k] for k in ("query_time", "query_date", "cutoff") if row.get(k)}
            if row.get("time"):
                value = json.loads(row["time"])
                if not isinstance(value, dict):
                    raise ValueError("time metadata must be an object")
                metadata.update(value)
            memories = messages_to_memories(contexts[context_id][:end], qid,
                                           config.data.include_system_persona, config.data.memory_granularity)
            memories = _visible_memory_records(SimpleNamespace(query_time=query_time, metadata=metadata), memories)
            definition = wanted[qid]
            ids = {m.memory_id for m in memories}
            if not set(definition["universe"]) <= ids:
                raise ValueError("diagnostic universe contains a non-visible memory")
            found[qid] = {"name": definition["name"], "question_id": qid,
                          "persona_id": public["persona_id"], "task_id": definition["task_id"],
                          "query": public["user_question_or_message"], "all_options": public["all_options"],
                          "cutoff": {"end_index": end, "query_time": query_time, "query_metadata": metadata},
                          "visible_memories": [asdict(m) for m in memories],
                          "universe": list(definition["universe"]), "controls": definition["controls"]}
    if set(found) != set(wanted):
        raise ValueError("diagnostic questions missing from data")
    return [found[c["question_id"]] for c in settings["cases"]]


def _dense_controls(settings: dict) -> dict[str, list[str]]:
    qids = {c["question_id"] for c in settings["cases"]}
    result = {}
    with tarfile.open(settings["full_archive"]["path"], "r:*") as archive:
        for member in archive.getmembers():
            if member.isfile() and "/outcomes/" in "/" + member.name and member.name.endswith(".json"):
                with archive.extractfile(member) as f:
                    value = json.load(f)
                task = value.get("task", {})
                qid = task.get("question_id")
                if task.get("method_id") == "dense" and qid in qids and value.get("status") == "success":
                    if qid in result:
                        raise ValueError("ambiguous authoritative dense outcome")
                    result[qid] = list(value["selected_ids"])
    if result.keys() != qids:
        raise ValueError("missing dense context control")
    return result


def make_scorer(case: dict, config: Any, client: Any = None, cache_dir: Path | None = None,
                session_id: str = "offline-plan") -> SetReranker:
    return SetReranker(case["query"], _records(case), client if client is not None else RerankerClient(config.models.reranker),
                       cache_dir=cache_dir, cache_identity={"diagnostic_session": session_id},
                       query_identity={"persona_id": case["persona_id"], "question_id": case["question_id"]},
                       visible_history_cutoff=case["cutoff"], batch_size=1,
                       max_input_tokens=config.dependency.reranker_max_input_tokens,
                       score_space=config.models.reranker.score_space, score_contract=config.models.reranker.score_contract)


def plan_diagnostics(config_path: str | Path, output_dir: str | Path) -> dict:
    from .diagnostic_import import load_diagnostic_archive
    settings, config = load_diagnostic_config(config_path)
    cases = load_public_cases(settings, config)
    detail = load_diagnostic_archive(settings["detail_archive"]["path"],
                                     expected_sha256=settings["detail_archive"]["sha256"])
    original = {}
    for entry in detail["artifacts"]:
        artifact = entry["artifact"]
        if artifact["task_id"] in original:
            raise ValueError("duplicate historical task identity")
        original[artifact["task_id"]] = artifact["selection"]["selected_ids"]
    for case in cases:
        ids = original.get(case["task_id"])
        if ids is None or not set(ids) <= _records(case).keys() or len(set(ids)) != len(ids):
            raise ValueError("historical selection must refer to unique visible memories")
        case["historical_selected_ids"] = list(ids)
    dense = _dense_controls(settings) if settings["include_dense_controls"] else {}
    inputs, contexts = [], []
    for case in cases:
        scorer = make_scorer(case, config)
        records = _records(case)
        universe = sorted(case["universe"])
        conditions = []
        for n in range(len(universe) + 1):
            for ids in itertools.combinations(universe, n):
                subset_id = digest([case["question_id"], list(ids)])
                prepared = scorer.prepare_set(ids)
                score_payload = build_rerank_payload(config.models.reranker, case["query"], [prepared.document], 1)
                inputs.append({"subset_id": subset_id, "question_id": case["question_id"],
                               "memory_ids": list(ids), "document": prepared.document,
                               "document_hash": text_hash(prepared.document), "query_hash": text_hash(case["query"]),
                               "wire_payload_hash": request_hash(score_payload),
                               "estimated_input_tokens": prepared.estimated_input_tokens,
                               "token_count_is_estimate": True, "server_input_tokens": None,
                               "feasible": scorer.feasible(ids)})
                conditions.append((subset_id, "subset", ids))
        if dense:
            conditions.append((digest([case["question_id"], "dense_control"]), "dense_control", dense[case["question_id"]]))
        for control in case["controls"]:
            ids = [v if v.startswith(case["question_id"] + ":") else case["question_id"] + ":" + v
                   for v in control["memory_ids"]]
            conditions.append((digest([case["question_id"], "control", control["name"]]), control["name"], ids))
        for condition_id, kind, ids in conditions:
            if not set(ids) <= records.keys() or len(set(ids)) != len(ids):
                raise ValueError("control memory IDs must be unique and visible")
            plan = build_context_plan(case["query"], [records[i] for i in ids], case["all_options"],
                                      generator_config=config.models.generator, selected_ids=ids, strict=False)
            contexts.append({"condition_id": condition_id, "question_id": case["question_id"], "kind": kind,
                             "context_plan": plan.public_dict(), "wire_payload_hash": request_hash(plan.request_dict()),
                             "feasible": plan.within_budget})
    counts = {"score_logical_inputs": len(inputs), "generation_trials": len(contexts) * settings["repeats"],
              "root_runs": len(cases) * (1 + len(settings["root_seeds"]))}
    for name, count in counts.items():
        if count > settings["budgets"][name]:
            raise ValueError(f"predeclared {name} budget too small ({count})")
    if not all(x["feasible"] for x in inputs + contexts):
        raise ValueError("at least one frozen scoring/reader context is over its estimated capacity")
    frozen = {"schema_version": 1, "purpose": "posthoc_diagnostic", "eligible_for_benchmark": False,
              "source_snapshot_hash": source_identity(), "deployment": deployment_snapshot(config),
              "public_data_hash": digest(cases), "cases": cases, "score_inputs": inputs, "contexts": contexts,
              "archives": {k: settings[k] for k in ("detail_archive", "full_archive")},
              "dependency": asdict(config.dependency), "budgets": settings["budgets"], "counts": counts,
              "repeats": settings["repeats"], "order_seed": settings["order_seed"],
              "root_seeds": settings["root_seeds"], "task_max_attempts": settings["task_max_attempts"],
              "score_mode": "fresh_all", "generation_response_cache": "disabled",
              "control_note": "Explicit controls only; no automatically invented length-matched text."}
    manifest_id = digest(frozen)
    rng = random.Random(settings["order_seed"])
    trials = []
    for repeat in range(settings["repeats"]):
        block = list(contexts)
        rng.shuffle(block)
        for c in block:
            trials.append({"trial_id": digest([manifest_id, c["condition_id"], repeat]),
                           "condition_id": c["condition_id"], "question_id": c["question_id"],
                           "repeat_index": repeat})
    frozen["manifest_id"] = manifest_id
    frozen["trials"] = trials
    frozen["trials_hash"] = digest(trials)
    root = Path(output_dir).resolve()
    with run_lock(root):
        if (root / "manifest.json").exists():
            existing = load_manifest(root)
            if existing != frozen:
                raise ValueError("refusing to replace a different frozen diagnostic run")
        else:
            atomic_json(root / "manifest.json", frozen)
        atomic_json(root / "plan_summary.json", plan_summary(frozen))
    return plan_summary(frozen)


def load_manifest(root: str | Path) -> dict:
    path = Path(root)
    value = json.loads((path if path.name == "manifest.json" else path / "manifest.json").read_text())
    frozen = {k: v for k, v in value.items() if k not in {"manifest_id", "trials", "trials_hash"}}
    if digest(frozen) != value["manifest_id"] or digest(value["trials"]) != value["trials_hash"]:
        raise ValueError("diagnostic manifest identity mismatch")
    expected = {(c["condition_id"], r) for c in value["contexts"] for r in range(value["repeats"])}
    by_condition = {c["condition_id"]: c for c in value["contexts"]}
    if len(by_condition) != len(value["contexts"]):
        raise ValueError("duplicate frozen condition identity")
    seen = set()
    for trial in value["trials"]:
        key = (trial["condition_id"], trial["repeat_index"])
        if key in seen or trial["trial_id"] != digest([value["manifest_id"], *key]):
            raise ValueError("invalid frozen trial identity")
        if (trial["condition_id"] not in by_condition
                or trial["question_id"] != by_condition[trial["condition_id"]]["question_id"]):
            raise ValueError("trial question identity differs from frozen context")
        seen.add(key)
    if seen != expected:
        raise ValueError("frozen trials do not cover predeclared conditions")
    rng = random.Random(value["order_seed"])
    expected_order = []
    for repeat in range(value["repeats"]):
        block = list(value["contexts"])
        rng.shuffle(block)
        expected_order.extend((c["condition_id"], repeat) for c in block)
    if [(t["condition_id"], t["repeat_index"]) for t in value["trials"]] != expected_order:
        raise ValueError("trial order differs from predeclared blocked randomization")
    for c in value["contexts"]:
        plan = ContextPlan.from_public_dict(c["context_plan"])
        if request_hash(plan.request_dict()) != c["wire_payload_hash"]:
            raise ValueError("frozen wire payload mismatch")
    return value


def plan_summary(manifest: dict) -> dict:
    return {"manifest_id": manifest["manifest_id"], "purpose": manifest["purpose"],
            "counts": manifest["counts"], "budgets": manifest["budgets"],
            "unknown_deployments": online_identity_ready(manifest["deployment"], ("embedding", "reranker", "generator")),
            "network_calls": 0, "generation_cache": "disabled", "score_mode": "fresh_all",
            "token_count_is_estimate": True, "eligible_for_benchmark": False}


def validate_online(manifest: dict, config_path: str | Path, roles: tuple[str, ...]) -> Any:
    _, config = load_diagnostic_config(config_path)
    current = deployment_snapshot(config)
    if current != manifest["deployment"]:
        raise ValueError("deployment/request settings differ from frozen manifest; make a new run")
    if asdict(config.dependency) != manifest["dependency"]:
        raise ValueError("dependency settings differ from frozen manifest")
    if source_identity() != manifest["source_snapshot_hash"]:
        raise ValueError("source snapshot differs from frozen manifest; make a new run")
    missing = online_identity_ready(current, roles)
    if missing:
        raise ValueError("unfrozen deployment identity: " + ", ".join(missing))
    return config


def _phase_budget(root: Path, manifest: dict, phase: str):
    from .request_audit import TransportBudget
    events = read_jsonl(root / "requests.jsonl")
    used = sum(e.get("event") == "http_attempt_started" and e.get("phase") == phase for e in events)
    return TransportBudget(manifest["budgets"][phase + "_transport_attempts"], used=used)


def _task_attempt(root: Path, phase: str, item_id: str, manifest: dict) -> tuple[int, dict | None]:
    path = root / phase / (item_id + ".json")
    if path.exists():
        value = json.loads(path.read_text())
        if (value.get("manifest_id") != manifest["manifest_id"] or value.get("item_id") != item_id
                or value.get("phase") != phase):
            raise ValueError("result belongs to another frozen manifest")
        # Terminal results, including errors, must never silently become a
        # fresh trial when --resume is used repeatedly.
        if value["status"] in {"success", "error", "budget_exhausted"}:
            return 0, value
    events = read_jsonl(root / "attempts.jsonl")
    matched = [e for e in events if e.get("phase") == phase and e.get("item_id") == item_id]
    if any(e.get("manifest_id") != manifest["manifest_id"] for e in matched):
        raise ValueError("attempt ledger belongs to a different frozen manifest")
    terminal_events = [e for e in matched if e.get("event") == "task_attempt_completed"
                       or (e.get("event") == "task_attempt_failed"
                           and (not e.get("retryable", False)
                                or e["task_attempt"] >= manifest["task_max_attempts"]))]
    if terminal_events:
        # Recover any durable terminal outcome after a crash between event
        # fsync and the atomic result write. A known failure must retain its
        # original error and must not gain a retry merely because of a crash.
        # Only retryable failures with task attempts left can fall through.
        restored = {k: v for k, v in terminal_events[-1].items() if k not in {"event", "at_epoch"}}
        atomic_json(path, restored)
        return 0, restored
    attempts = [e["task_attempt"] for e in matched]
    return max(attempts, default=0) + 1, None


def _execute_item(root: Path, manifest: dict, phase: str, item_id: str, budget: Any,
                  action: Any, metadata: dict) -> dict:
    from .clients import HTTPTransportError
    from .request_audit import AuditWriteError, JsonlAuditSink, TransportBudgetExceeded, request_audit_scope
    attempt, terminal = _task_attempt(root, phase, item_id, manifest)
    if terminal is not None:
        observe("execution", "task_reused", **{**metadata, "phase": phase,
                "task_id": item_id, "item_id": item_id, "status": terminal["status"],
                "task_attempt": terminal.get("task_attempt")})
        return terminal
    limit = manifest["task_max_attempts"]
    result = {"manifest_id": manifest["manifest_id"], "item_id": item_id, "phase": phase, **metadata}
    if attempt > limit:
        result.update(status="error", error_type="InterruptedAttempt", outcome_unknown=True,
                      task_attempt=attempt - 1)
    else:
        for index in range(attempt, limit + 1):
            started = {**result, "event": "task_attempt_started", "task_attempt": index, "at_epoch": time.time()}
            append_jsonl(root / "attempts.jsonl", started)
            observe("execution", started, task_id=item_id)
            try:
                with request_audit_scope({**metadata, "phase": phase, "task_id": item_id, "item_id": item_id,
                                          "task_attempt": index, "stage": phase, "run_identity": manifest["manifest_id"]},
                                         sink=JsonlAuditSink(root / "requests.jsonl"), budget=budget):
                    response = action()
                result.update(response)
                result.update(status="success", task_attempt=index, outcome_unknown=False)
            except AuditWriteError:
                # Losing the audit trail is a pipeline failure, not a model
                # trial eligible for retries or silent continuation.
                raise
            except Exception as exc:
                retryable = isinstance(exc, HTTPTransportError) and bool(getattr(exc, "retryable", False))
                result.update(status="budget_exhausted" if isinstance(exc, TransportBudgetExceeded) else "error",
                              error_type=type(exc).__name__, task_attempt=index,
                              retryable=retryable, outcome_unknown=isinstance(exc, (HTTPTransportError, TimeoutError)),
                              http_status=getattr(exc, "status_code", None))
                if hasattr(exc, "diagnostic_partial_artifacts"):
                    result["partial_artifacts"] = exc.diagnostic_partial_artifacts
                append_jsonl(root / "attempts.jsonl", {**result, "event": "task_attempt_failed", "at_epoch": time.time()})
                observe("execution", "task_attempt_failed", **{**metadata, "phase": phase,
                        "task_id": item_id, "item_id": item_id, "task_attempt": index,
                        "status": result["status"], "cause_type": type(exc).__name__,
                        "http_status": getattr(exc, "status_code", None)})
                if retryable and index < limit:
                    continue
            else:
                append_jsonl(root / "attempts.jsonl", {**result, "event": "task_attempt_completed", "at_epoch": time.time()})
                observe("execution", "task_attempt_completed", **{**metadata, "phase": phase,
                        "task_id": item_id, "item_id": item_id, "task_attempt": index, "status": "success"})
            break
    atomic_json(root / phase / (item_id + ".json"), result)
    return result


def run_scores(root: str | Path, config_path: str | Path, *, execute: bool = False,
               client: Any = None) -> dict:
    root = Path(root)
    manifest = load_manifest(root)
    if not execute:
        return {**plan_summary(manifest), "phase": "score", "execute": False}
    try:
        config = validate_online(manifest, config_path, ("reranker",))
    except Exception as exc:
        atomic_json(root / "score_gate.json", {"status": "blocked_before_network", "error_type": type(exc).__name__,
                    "reason": str(exc), "network_calls": 0, "manifest_id": manifest["manifest_id"]})
        raise
    with run_lock(root):
        budget = _phase_budget(root, manifest, "score")
        scorers = {c["question_id"]: make_scorer(c, config, client, root / "cache" / "scores", manifest["manifest_id"])
                   for c in manifest["cases"]}
        outcomes = []
        for item in manifest["score_inputs"]:
            scorer = scorers[item["question_id"]]
            if text_hash(scorer.serialize_set(item["memory_ids"])) != item["document_hash"]:
                raise ValueError("score serialization drift")
            payload = build_rerank_payload(config.models.reranker, scorer.query, [scorer.serialize_set(item["memory_ids"])], 1)
            if request_hash(payload) != item["wire_payload_hash"]:
                raise ValueError("score wire payload drift")
            def action(item=item, scorer=scorer):
                score = scorer.score_sets([item["memory_ids"]], reason="diagnostic_fresh_all")[0]
                return {"score": score, "score_space": scorer.score_space,
                        "score_contract": scorer.score_contract, "origin": "current_session_measurement",
                        "scorer_events": list(scorer.events)}
            outcomes.append(_execute_item(root, manifest, "score", item["subset_id"], budget, action,
                                          {k: item[k] for k in ("subset_id", "question_id", "memory_ids", "document_hash")}))
        result = _phase_summary(manifest, "score", outcomes, budget)
        atomic_json(root / "score_summary.json", result)
        return result


def run_generations(root: str | Path, config_path: str | Path, *, execute: bool = False,
                    client: Any = None) -> dict:
    root = Path(root)
    manifest = load_manifest(root)
    if not execute:
        return {**plan_summary(manifest), "phase": "generation", "execute": False}
    try:
        config = validate_online(manifest, config_path, ("generator",))
    except Exception as exc:
        atomic_json(root / "generation_gate.json", {"status": "blocked_before_network", "error_type": type(exc).__name__,
                    "reason": str(exc), "network_calls": 0, "manifest_id": manifest["manifest_id"]})
        raise
    with run_lock(root):
        budget = _phase_budget(root, manifest, "generation")
        generator = client if client is not None else GeneratorClient(config.models.generator)
        contexts = {c["condition_id"]: c for c in manifest["contexts"]}
        outcomes = []
        for trial in manifest["trials"]:
            context = contexts[trial["condition_id"]]
            plan = ContextPlan.from_public_dict(context["context_plan"])
            def action(plan=plan):
                # Deliberately bypass GenerationCache. TrialStore provides
                # idempotent resume, not cross-repeat answer reuse.
                observe("context", "generation_context", selected_ids=list(plan.selected_ids),
                        final_memory_count=len(plan.selected_ids), context_hash=plan.context_hash,
                        request_hash=request_hash(plan.request_dict()), context_within_budget=plan.within_budget,
                        generator_input_tokens_estimate=plan.token_count,
                        token_count_is_estimate=plan.token_count_is_estimate,
                        context_budget=plan.budget, context_budget_status=plan.budget_status)
                response = generator.answer_plan(plan)
                if not isinstance(response, str) or not response.strip():
                    raise ValueError("empty generator response")
                return {"response": response, "generation_cache_hit": False}
            outcomes.append(_execute_item(root, manifest, "generation", trial["trial_id"], budget, action,
                                          {**trial, "wire_payload_hash": context["wire_payload_hash"]}))
        result = _phase_summary(manifest, "generation", outcomes, budget)
        atomic_json(root / "generation_summary.json", result)
        return result


def _phase_summary(manifest: dict, phase: str, results: list[dict], budget: Any) -> dict:
    return {"manifest_id": manifest["manifest_id"], "phase": phase,
            "planned": len(results), "success": sum(r["status"] == "success" for r in results),
            "failed": sum(r["status"] != "success" for r in results),
            "transport_attempts_used": budget.used, "transport_attempts_cap": manifest["budgets"][phase + "_transport_attempts"],
            "response_cache_enabled": False, "eligible_for_benchmark": False}


def analyze_diagnostics(root: str | Path, *, score_view: str = "historical") -> dict:
    """No service construction or calls. Historical and fresh score views never mix."""
    from .dependency_diagnostics import analyze_reachability, audit_diagnostic_cases, replay_selector
    from .diagnostic_import import import_historical_scores, load_diagnostic_archive
    root = Path(root)
    manifest = load_manifest(root)
    spec = manifest["archives"]["detail_archive"]
    cases = [{"case_id": c["name"], "question_id": c["question_id"], "task_id": c["task_id"],
              "universe_ids": c["universe"]} for c in manifest["cases"]]
    if score_view == "historical":
        result = audit_diagnostic_cases(spec["path"], expected_sha256=spec["sha256"], cases=cases,
                                        max_selection_sets=manifest["dependency"]["max_selection_sets"])
        result["manifest_id"] = manifest["manifest_id"]
        atomic_json(root / "offline_analysis.json", result)
        return result
    if score_view != "fresh":
        raise ValueError("score_view must be historical or fresh")
    loaded = load_diagnostic_archive(spec["path"], expected_sha256=spec["sha256"])
    artifacts = {e["artifact"]["task_id"]: e for e in loaded["artifacts"]}
    reader_feasible = {c["condition_id"]: c["feasible"] for c in manifest["contexts"] if c["kind"] == "subset"}
    results = []
    for case in cases:
        inputs = [i for i in manifest["score_inputs"] if i["question_id"] == case["question_id"]]
        records, missing, feasible = [], [], {}
        for item in inputs:
            key = tuple(sorted(item["memory_ids"]))
            feasible[key] = item["feasible"] and reader_feasible[item["subset_id"]]
            path = root / "score" / (item["subset_id"] + ".json")
            row = json.loads(path.read_text()) if path.exists() else {}
            if row and (row.get("manifest_id") != manifest["manifest_id"] or row.get("subset_id") != item["subset_id"]):
                raise ValueError("current score result identity mismatch")
            if row.get("status") == "success":
                records.append({"ids": list(key), "score": row["score"], "source_item_id": item["subset_id"]})
            else:
                missing.append(list(key))
        entry = artifacts[case["task_id"]]
        historical = import_historical_scores(entry["artifact"], provenance=entry["provenance"])
        view = {"records": records, "origin": "fresh_all_current_session",
                "search_seen_ids": historical["search_seen_ids"]}
        observed = [r for r in records if feasible[tuple(r["ids"])]]
        best = max((r["score"] for r in observed), default=None)
        results.append({"case_id": case["case_id"], "question_id": case["question_id"], "missing_ids": missing,
                        "score_origin": view["origin"], "score_records": records,
                        "optimum_scope": "complete_feasible_subsets_of_U" if not missing else "observed_subsets_only",
                        "best_R": best, "best_R_ids": [r["ids"] for r in observed if r["score"] == best],
                        "reachability": analyze_reachability(case, entry["artifact"]["search"]["bundles"], view,
                                                             feasibility=feasible),
                        "restricted_simulation": replay_selector(entry["artifact"], view, scope="restricted_universe",
                            case=case, feasibility=feasible, max_selection_sets=manifest["dependency"]["max_selection_sets"]),
                        "budget_note": "Frozen historical search charges precede counterfactual selection; not fresh measurement cost."})
    result = {"manifest_id": manifest["manifest_id"], "score_view": "fresh", "network_calls": 0,
              "cases": results, "historical_scores_merged": False, "eligible_for_benchmark": False}
    atomic_json(root / "fresh_analysis.json", result)
    return result
