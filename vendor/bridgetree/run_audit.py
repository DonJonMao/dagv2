from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .config import RetrievalConfig
from .information import InformationObjective
from .module_metrics import MODULE_NAMES
from .personamem import PERSONAMEM_REVISION, PERSONAMEM_SOURCE_SHA256
from .training import (
    DEFAULT_MAIN_TABLE_METHODS,
    FORMAL_32K_PARTITION_QUERIES,
    FORMAL_32K_SEARCH_SPACE,
    FORMAL_32K_SEED,
    FORMAL_32K_SPLIT,
)
from .types import context_plan_hash


def _question_hash(question_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(question_ids).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(root: Path, name: str, errors: list[str], default: Any) -> Any:
    path = root / name
    if not path.is_file():
        errors.append(f"missing required artifact: {name}")
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON artifact {name}: {exc}")
        return default


def _load_jsonl(root: Path, name: str, errors: list[str]) -> list[Dict[str, Any]]:
    path = root / name
    if not path.is_file():
        errors.append(f"missing required artifact: {name}")
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSONL artifact {name}:{line_number}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"non-object JSONL record {name}:{line_number}")
            continue
        records.append(value)
    return records


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _summary_complete(
    summary: Mapping[str, Any],
    expected_queries: int,
    expected_hash: str,
    label: str,
    errors: list[str],
) -> None:
    _check(summary.get("queries") == expected_queries, f"{label}: aggregate query count mismatch", errors)
    _check(summary.get("attempted_queries") == expected_queries, f"{label}: attempted query count mismatch", errors)
    _check(summary.get("successful_queries") == expected_queries, f"{label}: successful query count mismatch", errors)
    _check(summary.get("failed_queries") == 0, f"{label}: failed_queries is not zero", errors)
    _check(summary.get("failure_rate") == 0.0, f"{label}: failure_rate is not zero", errors)
    _check(
        summary.get("attempted_question_id_sha256") == expected_hash,
        f"{label}: attempted question hash mismatch",
        errors,
    )
    _check(
        summary.get("successful_question_id_sha256") == expected_hash,
        f"{label}: successful question hash mismatch",
        errors,
    )


def audit_tuning_run(
    run_dir: str | Path,
    *,
    require_full_32k: bool = False,
    max_parse_failure_rate: float = 0.05,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Independently re-read and validate a completed tuning run's persisted artifacts."""
    if not 0.0 <= max_parse_failure_rate <= 1.0:
        raise ValueError("max_parse_failure_rate must be in [0, 1]")
    root = Path(run_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        raise FileNotFoundError(f"tuning run directory does not exist: {root}")

    required_plain = ("metrics.csv", "pareto_frontier.json")
    for name in required_plain:
        _check((root / name).is_file(), f"missing required artifact: {name}", errors)
    resolved = _load_json(root, "resolved_config.json", errors, {})
    training_config = _load_json(root, "training_config.json", errors, {})
    run_manifest = _load_json(root, "run_manifest.json", errors, {})
    split_manifest = _load_json(root, "split_manifest.json", errors, {})
    trials = _load_json(root, "trials.json", errors, [])
    final = _load_json(root, "final_summary.json", errors, {})
    best_config = _load_json(root, "best_config.json", errors, {})
    run_status = _load_json(root, "run_status.json", errors, {})
    progress = _load_json(root, "progress.json", errors, {})
    failures = _load_jsonl(root, "failures.jsonl", errors)
    events = _load_jsonl(root, "events.jsonl", errors)
    examples = _load_jsonl(root, "example_metrics.jsonl", errors)

    combined = resolved.get("config", {}) if isinstance(resolved, dict) else {}
    app = combined.get("app", {}) if isinstance(combined, dict) else {}
    tuning = combined.get("tuning", {}) if isinstance(combined, dict) else {}
    data = app.get("data", {}) if isinstance(app, dict) else {}
    models = app.get("models", {}) if isinstance(app, dict) else {}
    generator_config = models.get("generator", {}) if isinstance(models, dict) else {}
    context_token_budget = int(generator_config.get("context_token_budget", 0) or 0)
    split_data = {name: split_manifest.get(name, {}) for name in ("train", "validation", "test")}
    validation_queries = int(split_data["validation"].get("queries", -1))
    test_queries = int(split_data["test"].get("queries", -1))
    validation_hash = str(split_data["validation"].get("question_id_sha256", ""))
    test_hash = str(split_data["test"].get("question_id_sha256", ""))
    main_methods = tuple(tuning.get("main_table_methods", ()))
    diagnostic_methods = tuple(tuning.get("diagnostic_methods", ()))

    _check(run_status.get("status") == "completed", "run_status is not completed", errors)
    _check(run_status.get("failure_count") == 0, "run_status failure_count is not zero", errors)
    _check(not failures, f"failures.jsonl contains {len(failures)} records", errors)
    _check(final.get("selection_status") == "selected_on_external_validation_outcome", "selection failed", errors)
    _check(final.get("best_trial") is not None, "best_trial is missing", errors)
    _check(final.get("best_retrieval_config") == best_config.get("retrieval"), "best config mismatch", errors)
    configured_objective = tuning.get("objective_metric")
    _check(
        configured_objective == "auto" or final.get("objective_metric") == configured_objective,
        "final objective metric differs from tuning config",
        errors,
    )
    _check(progress.get("status") == "completed", "progress.json is not completed", errors)
    _check(progress.get("phase") == "completed", "progress.json terminal phase mismatch", errors)
    _check(progress.get("best_trial") == final.get("best_trial"), "progress best trial mismatch", errors)
    _check(
        progress.get("selection_status") == final.get("selection_status"),
        "progress selection status mismatch",
        errors,
    )
    _check(run_status.get("best_trial") == final.get("best_trial"), "run status best trial mismatch", errors)
    _check(
        run_status.get("selection_status") == final.get("selection_status"),
        "run status selection mismatch",
        errors,
    )
    _check(run_manifest.get("command") == "tune", "run manifest command is not tune", errors)
    _check(
        run_manifest.get("optimization_kind") == "training_free_configuration_tuning",
        "run manifest optimization kind mismatch",
        errors,
    )
    _check(run_manifest.get("data_revision") == PERSONAMEM_REVISION, "run manifest data revision mismatch", errors)
    _check(run_manifest.get("data_split") == data.get("split"), "run manifest data split mismatch", errors)
    _check(run_manifest.get("validation_queries") == validation_queries, "manifest validation count mismatch", errors)
    _check(run_manifest.get("test_queries") == test_queries, "manifest test count mismatch", errors)
    _check(run_manifest.get("trial_count") == len(trials), "manifest trial count mismatch", errors)
    _check(final.get("trial_count") == len(trials), "final trial count mismatch", errors)
    _check(
        final.get("evaluated_validation_queries") == validation_queries,
        "final validation query count mismatch",
        errors,
    )
    _check(final.get("evaluated_test_queries") == test_queries, "final test query count mismatch", errors)

    combined_payload = json.dumps(combined, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected_config_hash = hashlib.sha256(combined_payload.encode("utf-8")).hexdigest()
    _check(resolved.get("config_hash") == expected_config_hash, "resolved config hash mismatch", errors)

    personas = [set(split_data[name].get("personas", ())) for name in ("train", "validation", "test")]
    _check(not personas[0] & personas[1], "train/validation persona leakage", errors)
    _check(not personas[0] & personas[2], "train/test persona leakage", errors)
    _check(not personas[1] & personas[2], "validation/test persona leakage", errors)
    partition_queries = sum(int(split_data[name].get("queries", -1)) for name in ("train", "validation", "test"))
    _check(run_manifest.get("dataset_queries") == partition_queries, "manifest dataset query count mismatch", errors)
    _check(
        run_manifest.get("train_queries_reserved") == split_data["train"].get("queries"),
        "manifest train query count mismatch",
        errors,
    )

    if require_full_32k:
        _check(data.get("split") == "32k", "formal audit requires data.split=32k", errors)
        _check(data.get("include_system_persona") is True, "formal audit requires the system persona", errors)
        _check(
            data.get("memory_granularity") == "user_assistant_pair",
            "formal audit requires user-assistant-pair memories",
            errors,
        )
        _check(
            run_manifest.get("source_sha256") == PERSONAMEM_SOURCE_SHA256["32k"],
            "formal audit requires the pinned official source checksums",
            errors,
        )
        _check(app.get("seed") == FORMAL_32K_SEED, "formal audit requires app seed=42", errors)
        _check(tuning.get("seed") == FORMAL_32K_SEED, "formal audit requires tuning seed=42", errors)
        _check(
            tuning.get("split")
            == {
                "train_ratio": FORMAL_32K_SPLIT.train_ratio,
                "validation_ratio": FORMAL_32K_SPLIT.validation_ratio,
                "test_ratio": FORMAL_32K_SPLIT.test_ratio,
            },
            "formal audit requires the pinned persona split",
            errors,
        )
        _check(len(trials) == 16, "formal audit requires exactly 16 trials", errors)
        _check(
            validation_queries == FORMAL_32K_PARTITION_QUERIES["validation"],
            "formal audit requires 84 validation queries",
            errors,
        )
        _check(
            test_queries == FORMAL_32K_PARTITION_QUERIES["test"],
            "formal audit requires 73 test queries",
            errors,
        )
        _check(main_methods == DEFAULT_MAIN_TABLE_METHODS, "formal audit requires all seven ordered methods", errors)
        _check(tuning.get("diagnostic_methods") == ["bridgetree"], "formal audit requires one search method", errors)
        _check(tuning.get("keep_example_metrics") is True, "formal audit requires example metrics", errors)
        _check(tuning.get("validation_generate") is True, "formal validation generation is disabled", errors)
        _check(tuning.get("final_generate") is True, "formal final generation is disabled", errors)
        _check(tuning.get("fail_on_evaluation_error") is True, "formal strict failure handling is disabled", errors)
        _check(final.get("objective_metric") == "outcome.answer_accuracy", "unexpected objective metric", errors)
        formal_retrieval = asdict(RetrievalConfig())
        actual_retrieval = dict(app.get("retrieval", {}))
        for tuned_name in ("initial_width", "branch_width", "search_budget"):
            formal_retrieval.pop(tuned_name)
            actual_retrieval.pop(tuned_name, None)
        _check(
            actual_retrieval == formal_retrieval,
            "formal audit requires the pinned retrieval protocol outside the search axes",
            errors,
        )
        schedule = tuning.get("schedule", {})
        _check(schedule.get("max_validation_queries") is None, "formal validation is limited", errors)
        _check(schedule.get("max_test_queries") is None, "formal test is limited", errors)
        _check(final.get("evaluated_validation_queries") == 84, "final validation count mismatch", errors)
        _check(final.get("evaluated_test_queries") == 73, "final test count mismatch", errors)

    trial_map: Dict[int, Mapping[str, Any]] = {}
    observed_space = set()
    for trial in trials if isinstance(trials, list) else []:
        trial_index = int(trial.get("trial", -1))
        retrieval = trial.get("retrieval", {})
        _check(trial_index not in trial_map, f"duplicate trial index: {trial_index}", errors)
        trial_map[trial_index] = retrieval
        observed_space.add(
            (retrieval.get("initial_width"), retrieval.get("branch_width"), retrieval.get("search_budget"))
        )
        validation = trial.get("validation", {})
        _check(
            set(validation) == set(diagnostic_methods),
            f"validation method set mismatch: trial {trial_index}",
            errors,
        )
        for method in diagnostic_methods:
            summary = validation.get(method, {})
            label = f"validation trial {trial_index} method {method}"
            _summary_complete(summary, validation_queries, validation_hash, label, errors)
            if tuning.get("validation_generate") is True:
                parse_rate = summary.get("modules", {}).get("outcome", {}).get("parse_failure_rate")
                if parse_rate is None or float(parse_rate) > max_parse_failure_rate:
                    errors.append(f"{label}: parse failure rate exceeds threshold")

    search_space = tuning.get("search_space", {})
    expected_space = set(
        product(
            search_space.get("initial_width", ()),
            search_space.get("branch_width", ()),
            search_space.get("search_budget", ()),
        )
    )
    _check(observed_space == expected_space, "observed trial grid does not match configured search space", errors)
    _check(set(trial_map) == set(range(1, len(trials) + 1)), "trial indices are not contiguous", errors)
    if require_full_32k:
        formal_space = {
            (initial_width, branch_width, search_budget)
            for initial_width, branch_width, search_budget in product(
                FORMAL_32K_SEARCH_SPACE.initial_width,
                FORMAL_32K_SEARCH_SPACE.branch_width,
                FORMAL_32K_SEARCH_SPACE.search_budget,
            )
        }
        _check(expected_space == formal_space, "formal audit requires the pinned 2x2x4 search space", errors)

    best_trial = int(final.get("best_trial", -1))
    _check(trial_map.get(best_trial) == best_config.get("retrieval"), "best trial retrieval config mismatch", errors)
    test_metrics = final.get("test_metrics", {})
    _check(set(test_metrics) == set(main_methods), "final method set mismatch", errors)
    _check(final.get("common_test_question_id_sha256") == test_hash, "final common test hash mismatch", errors)
    for method in main_methods:
        summary = test_metrics.get(method, {})
        _summary_complete(summary, test_queries, test_hash, f"test method {method}", errors)
        if tuning.get("final_generate") is True:
            parse_rate = summary.get("modules", {}).get("outcome", {}).get("parse_failure_rate")
            if parse_rate is None or float(parse_rate) > max_parse_failure_rate:
                errors.append(f"test method {method}: parse failure rate exceeds threshold")

    expected_event_groups = {
        *(('validation', trial_index, method) for trial_index in trial_map for method in diagnostic_methods),
        *(("test", best_trial, method) for method in main_methods),
    }
    event_counts = Counter(
        (str(event.get("phase", "")), int(event.get("trial", -1)), str(event.get("method", "")))
        for event in events
    )
    _check(set(event_counts) == expected_event_groups, "event phase/trial/method set mismatch", errors)
    _check(all(count == 1 for count in event_counts.values()), "duplicate aggregate events are present", errors)
    _check(len(events) == len(expected_event_groups), "aggregate event count mismatch", errors)
    _check(
        [event.get("event_id") for event in events] == list(range(1, len(events) + 1)),
        "event IDs are not contiguous",
        errors,
    )
    _check(
        not any(event.get("phase") in {"train", "train_progress", "validation_probe"} for event in events),
        "inert train/probe events are present",
        errors,
    )

    expected_example_count = (
        len(trials) * len(diagnostic_methods) * validation_queries + len(main_methods) * test_queries
    )
    _check(len(examples) == expected_example_count, "example_metrics row count mismatch", errors)
    sequence_by_group: Dict[tuple[str, int, str], list[str]] = defaultdict(list)
    example_keys = set()
    max_unique_budget_ratio = 0.0
    max_parse_rate = 0.0
    for row_number, record in enumerate(examples, start=1):
        phase = str(record.get("phase", ""))
        trial_index = int(record.get("trial", -1))
        method = str(record.get("method", ""))
        question_id = str(record.get("question_id", ""))
        key = (phase, trial_index, method, question_id)
        _check(key not in example_keys, f"duplicate example metric key at row {row_number}", errors)
        example_keys.add(key)
        sequence_by_group[(phase, trial_index, method)].append(question_id)
        retrieval = trial_map.get(trial_index, {})
        modules = record.get("modules", {})
        _check(set(modules) == set(MODULE_NAMES), f"module set mismatch at row {row_number}", errors)
        cost = modules.get("cost", {})
        outcome = modules.get("outcome", {})
        budget = int(retrieval.get("search_budget", 0) or 0)
        visited = float(cost.get("unique_visited_nodes", -1))
        if budget > 0:
            max_unique_budget_ratio = max(max_unique_budget_ratio, visited / budget)
        _check(budget > 0 and 0 <= visited <= budget, f"unique-node budget violation at row {row_number}", errors)
        context_size = int(retrieval.get("context_size", 0) or 0)
        selected_ids = record.get("selected_memory_ids", ())
        final_count = float(cost.get("final_context_count", -1))
        final_tokens = float(cost.get("final_context_tokens", -1))
        _check(
            context_size > 0 and len(selected_ids) <= context_size and 0 <= final_count <= context_size,
            f"final context-size violation at row {row_number}",
            errors,
        )
        _check(
            context_token_budget > 0 and 0 <= final_tokens <= context_token_budget,
            f"final context-token budget violation at row {row_number}",
            errors,
        )
        _check(len(selected_ids) == len(set(selected_ids)), f"duplicate selected memory ID at row {row_number}", errors)
        _check(final_count == len(selected_ids), f"final context count mismatch at row {row_number}", errors)
        if "parse_failure_rate" in outcome:
            parse_failure = float(outcome["parse_failure_rate"])
            max_parse_rate = max(max_parse_rate, parse_failure)

    for trial_index in trial_map:
        for method in diagnostic_methods:
            sequence = sequence_by_group.get(("validation", trial_index, method), [])
            label = f"validation trial {trial_index} method {method}"
            _check(len(sequence) == validation_queries, f"{label}: example count mismatch", errors)
            _check(_question_hash(sequence) == validation_hash, f"{label}: example hash mismatch", errors)
    for method in main_methods:
        sequence = sequence_by_group.get(("test", best_trial, method), [])
        _check(len(sequence) == test_queries, f"test example count mismatch: {method}", errors)
        _check(_question_hash(sequence) == test_hash, f"test example hash mismatch: {method}", errors)

    for module in MODULE_NAMES:
        module_records = _load_jsonl(root, f"modules/{module}.jsonl", errors)
        _check(len(module_records) == len(events), f"module event count mismatch: {module}", errors)
        for event, module_record in zip(events, module_records):
            identity = ("event_id", "phase", "trial", "step", "method")
            _check(
                all(module_record.get(name) == event.get(name) for name in identity),
                f"module event identity mismatch: {module} event {event.get('event_id')}",
                errors,
            )
    if (root / "metrics.csv").is_file():
        with (root / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            metric_rows = sum(1 for _row in csv.reader(handle)) - 1
        _check(metric_rows > 0, "metrics.csv has no metric rows", errors)
    temporary_files = sorted(str(path.relative_to(root)) for path in root.rglob("*.tmp"))
    _check(not temporary_files, f"temporary artifacts remain: {temporary_files}", errors)

    if max_parse_rate > 0.0:
        warnings.append(f"at least one individual answer parse failed; max indicator={max_parse_rate}")
    report = {
        "status": "passed" if not errors else "failed",
        "audited_at": time.time(),
        "run_dir": str(root),
        "require_full_32k": require_full_32k,
        "max_parse_failure_rate": max_parse_failure_rate,
        "errors": errors,
        "warnings": warnings,
        "evidence": {
            "trial_count": len(trials),
            "event_count": len(events),
            "example_count": len(examples),
            "validation_queries_per_trial": validation_queries,
            "test_queries_per_method": test_queries,
            "final_method_count": len(main_methods),
            "failure_record_count": len(failures),
            "max_unique_budget_ratio": max_unique_budget_ratio,
            "max_individual_parse_failure": max_parse_rate,
            "common_test_question_id_sha256": test_hash,
        },
        "training_config_matches_resolved": training_config == tuning,
    }
    _check(report["training_config_matches_resolved"], "training_config.json differs from resolved tuning", errors)
    report["status"] = "passed" if not errors else "failed"
    report["errors"] = errors
    _atomic_json(root / "completion_audit.json", report)
    if errors and raise_on_error:
        audit_path = root / "completion_audit.json"
        raise RuntimeError(f"tuning completion audit failed with {len(errors)} errors: {audit_path}")
    return report


def audit_semantic_run(
    run_dir: str | Path,
    *,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Independently audit a persisted L0/L1/S0--S2-shuffle matrix.

    The audit is deliberately structural: it verifies shared frozen graph and
    quality identities, whitelist selection, exact context-plan hashes, and
    explicit failure records.  It does not turn the PSD surrogate into an
    answer-accuracy claim.
    """
    requested = Path(run_dir).resolve()
    if not requested.is_dir():
        raise FileNotFoundError(f"semantic run directory does not exist: {requested}")
    # ``run-semantic-matrix`` writes a timestamped ``semantic_*`` directory
    # below the user supplied output directory, just like the TMIC runner.
    # Accepting the parent here is important for the CLI contract and avoids
    # making callers discover an implementation-specific timestamp.  Only
    # children with a manifest are candidates; an unrelated/partial directory
    # must not be selected merely because its name matches the prefix.
    root = requested
    if not (root / "run_manifest.json").is_file():
        candidates = sorted(
            (
                item
                for item in root.glob("semantic_*")
                if item.is_dir() and (item / "run_manifest.json").is_file()
            ),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            root = candidates[0]
    errors: list[str] = []
    warnings: list[str] = []
    manifest = _load_json(root, "run_manifest.json", errors, {})
    summary = _load_json(root, "summary.json", errors, {})
    failures = _load_jsonl(root, "failures.jsonl", errors)
    default_expected = ("L0", "L1", "S0", "S1", "S2", "S3", "S2-shuffle")

    if not isinstance(manifest, Mapping):
        manifest = {}
    if not isinstance(summary, Mapping):
        summary = {}
    architectures = tuple(str(value) for value in manifest.get("architectures", ()))
    expected = architectures or default_expected
    _check(
        bool(architectures),
        "semantic manifest has no architecture order",
        errors,
    )
    _check(
        summary.get("shared_candidate_pool") is True,
        "semantic run does not declare a shared candidate pool",
        errors,
    )

    # The manifest question list is the denominator.  A hash check here
    # catches accidental reordering as well as an omitted/duplicated query.
    raw_question_ids = manifest.get("question_ids", ())
    if isinstance(raw_question_ids, (str, bytes)) or not isinstance(raw_question_ids, Sequence):
        errors.append("manifest question_ids must be a sequence")
        manifest_question_ids: list[str] = []
    else:
        manifest_question_ids = [str(value) for value in raw_question_ids]
    if len(manifest_question_ids) != len(set(manifest_question_ids)):
        errors.append("manifest question_ids contain duplicates")
    expected_manifest_hash = _question_hash(manifest_question_ids)
    if manifest.get("question_id_sha256") not in (None, expected_manifest_hash):
        errors.append("manifest question_id_sha256 mismatch")
    try:
        expected_queries = int(manifest.get("queries", len(manifest_question_ids)))
    except (TypeError, ValueError):
        expected_queries = -1
        errors.append("manifest queries is invalid")
    _check(expected_queries == len(manifest_question_ids), "manifest query count does not match question_ids", errors)
    _check(expected_queries > 0, "semantic manifest contains no queries", errors)

    if failures:
        # A failed architecture/query is not a successful zero-information
        # observation.  It remains in the artifact for diagnosis, but an
        # audit must fail so callers cannot compare changing denominators.
        errors.append(f"failures.jsonl contains {len(failures)} records")
    if isinstance(manifest.get("failures"), list) and manifest.get("failures") != failures:
        errors.append("manifest failures differ from failures.jsonl")
    if isinstance(summary.get("failures"), list) and summary.get("failures") != failures:
        errors.append("summary failures differ from failures.jsonl")
    failure_keys: set[tuple[str, str]] = set()
    for position, failure in enumerate(failures, start=1):
        if not isinstance(failure, Mapping):
            errors.append(f"failure record {position} is not an object")
            continue
        question_id = str(failure.get("question_id", ""))
        architecture = str(failure.get("architecture", ""))
        key = (question_id, architecture)
        if not question_id or question_id not in set(manifest_question_ids):
            errors.append(f"failure record {position} references an unknown question")
        if architecture not in expected:
            errors.append(f"failure record {position} references an unknown architecture")
        if key in failure_keys:
            errors.append(f"duplicate failure record: {question_id}/{architecture}")
        failure_keys.add(key)

    records_by_label: dict[str, list[dict[str, Any]]] = {}
    record_keys: set[tuple[str, str]] = set()
    quality_families = {
        label: ("rho2" if label in {"L0", "L1"} else "pointwise") for label in expected
    }
    prediction_failure_keys: set[tuple[str, str]] = set()

    def _quality_digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()

    def _audit_plan(label: str, position: int, record: Mapping[str, Any], plan: Any) -> None:
        prefix = f"{label}:{position}"
        context_hash = record.get("context_hash")
        if not isinstance(plan, Mapping):
            errors.append(f"{prefix}: missing ContextPlan")
            if context_hash not in (None, ""):
                errors.append(f"{prefix}: context hash is present without ContextPlan")
            return
        required = (
            "selected_ids",
            "chronological_ids",
            "serialized_context",
            "messages",
            "token_count",
            "token_count_is_estimate",
            "budget",
            "budget_status",
            "prompt_hash",
            "context_hash",
            "request",
        )
        missing = [key for key in required if key not in plan]
        if missing:
            errors.append(f"{prefix}: ContextPlan missing fields {missing}")
            return
        selected = plan.get("selected_ids")
        chronological = plan.get("chronological_ids")
        messages = plan.get("messages")
        request = plan.get("request")
        if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence):
            errors.append(f"{prefix}: ContextPlan selected_ids is invalid")
            selected = []
        if isinstance(chronological, (str, bytes)) or not isinstance(chronological, Sequence):
            errors.append(f"{prefix}: ContextPlan chronological_ids is invalid")
            chronological = []
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            errors.append(f"{prefix}: ContextPlan messages is invalid")
            messages = []
        if not isinstance(request, Mapping):
            errors.append(f"{prefix}: ContextPlan request is invalid")
            request = {}
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping) or "role" not in message or "content" not in message:
                errors.append(f"{prefix}: ContextPlan message is malformed")
                continue
            # Preserve the raw values for the strict canonical hash.  Coercing
            # an integer role/content (or a custom object) to text would let a
            # tampered persisted plan pass audit with a different wire
            # representation than the one that was actually recorded.
            if not isinstance(message["role"], str) or not isinstance(message["content"], str):
                errors.append(f"{prefix}: ContextPlan message role/content must be strings")
                continue
            normalized_messages.append({"role": message["role"], "content": message["content"]})
        try:
            expected_hash = context_plan_hash(
                selected_ids=selected,
                chronological_ids=chronological,
                serialized_context=plan.get("serialized_context"),
                messages=normalized_messages,
                token_count=plan.get("token_count"),
                token_count_is_estimate=plan.get("token_count_is_estimate"),
                budget=plan.get("budget"),
                budget_status=plan.get("budget_status"),
                prompt_hash=plan.get("prompt_hash"),
                request=request,
            )
            if str(plan.get("context_hash")) != expected_hash:
                errors.append(f"{prefix}: ContextPlan context hash does not match content/request")
        except (TypeError, ValueError, KeyError):
            errors.append(f"{prefix}: ContextPlan hash inputs are malformed")
        if context_hash != plan.get("context_hash"):
            errors.append(f"{prefix}: context hash differs from ContextPlan")
        record_selected = record.get("selected_memory_ids", ())
        record_greedy = record.get("selected_in_greedy_order", ())
        if (
            isinstance(record_selected, Sequence)
            and not isinstance(record_selected, (str, bytes))
            and [str(value) for value in chronological] != [str(value) for value in record_selected]
        ):
            errors.append(f"{prefix}: ContextPlan chronology differs from selected context")
        if (
            isinstance(record_greedy, Sequence)
            and not isinstance(record_greedy, (str, bytes))
            and [str(value) for value in selected] != [str(value) for value in record_greedy]
        ):
            errors.append(f"{prefix}: ContextPlan greedy IDs differ from selection")
        request_messages = request.get("messages") if isinstance(request, Mapping) else None
        if request_messages != normalized_messages:
            errors.append(f"{prefix}: request messages differ from ContextPlan messages")

    for label in expected:
        records = _load_jsonl(root, f"predictions_{label}.jsonl", errors)
        records_by_label[label] = records
        if len(records) != expected_queries:
            errors.append(f"{label}: prediction count does not match manifest queries")
        seen_questions: set[str] = set()
        for position, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                errors.append(f"{label}:{position}: prediction is not an object")
                continue
            question_id = str(record.get("question_id", ""))
            if not question_id:
                errors.append(f"{label}:{position}: missing question_id")
            elif question_id in seen_questions:
                errors.append(f"{label}:{position}: duplicate question_id {question_id}")
            seen_questions.add(question_id)
            record_keys.add((question_id, label))
            if question_id not in set(manifest_question_ids):
                errors.append(f"{label}:{position}: question is outside manifest")

            if str(record.get("status", "success")) != "success":
                prediction_failure_keys.add((question_id, label))
                continue
            provenance = record.get("provenance", {})
            if not isinstance(provenance, Mapping):
                errors.append(f"{label}:{position}: provenance is not an object")
                provenance = {}
            graph = provenance.get("frozen_graph")
            graph_hash = record.get("shared_graph_hash")
            if not isinstance(graph, Mapping):
                errors.append(f"{label}:{position}: missing frozen graph provenance")
                graph = {}
            graph_provenance_hash = graph.get("graph_hash")
            if not isinstance(graph_hash, str) or not graph_hash:
                errors.append(f"{label}:{position}: missing graph hash")
            if graph_provenance_hash != graph_hash:
                errors.append(f"{label}:{position}: graph hash differs from frozen graph provenance")

            domain = provenance.get("proposal_domain", ())
            selected = record.get("selected_memory_ids", ())
            if isinstance(domain, Sequence) and not isinstance(domain, (str, bytes)):
                domain_ids = {str(value) for value in domain}
                if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
                    selected_ids = [str(value) for value in selected]
                    if len(selected_ids) != len(set(selected_ids)):
                        errors.append(f"{label}:{position}: duplicate selected memory ID")
                    if not set(selected_ids).issubset(domain_ids):
                        errors.append(f"{label}:{position}: selected ID is outside proposal domain")
            else:
                errors.append(f"{label}:{position}: proposal domain is missing")

            _audit_plan(label, position, record, provenance.get("context_plan"))
            quality = record.get("shared_quality")
            if not isinstance(quality, Mapping) or not quality:
                errors.append(f"{label}:{position}: shared quality table is missing")
            declared_family = record.get("quality_source_declared")
            if declared_family != quality_families.get(label):
                errors.append(
                    f"{label}:{position}: quality_source_declared must be {quality_families.get(label)!r}"
                )
            runtime_source = record.get("quality_source")
            if not isinstance(runtime_source, str) or not runtime_source:
                errors.append(f"{label}:{position}: quality_source is missing")
            elif label in {"L0", "L1"} and runtime_source != "rho2":
                errors.append(f"{label}:{position}: legacy rows must use rho2 quality")
            elif label not in {"L0", "L1"} and runtime_source == "rho2":
                errors.append(f"{label}:{position}: semantic rows must use pointwise quality")
            if isinstance(quality, Mapping) and quality:
                declared_digest = record.get("quality_digest")
                actual_digest = _quality_digest(quality)
                if declared_digest not in (None, actual_digest):
                    errors.append(f"{label}:{position}: quality_digest mismatch")

    expected_set = set(manifest_question_ids)
    by_question: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for label, records in records_by_label.items():
        for record in records:
            if not isinstance(record, Mapping):
                continue
            question_id = str(record.get("question_id", ""))
            if question_id:
                by_question[question_id][label] = record
    for question_id in sorted(expected_set | set(by_question)):
        rows = by_question.get(question_id, {})
        missing_labels = [label for label in expected if label not in rows]
        if missing_labels:
            errors.append(f"{question_id}: missing architecture rows {missing_labels}")
        graph_hashes = {str(row.get("shared_graph_hash", "")) for row in rows.values()}
        graph_hashes.discard("")
        if len(graph_hashes) > 1:
            errors.append(f"{question_id}: architectures use different frozen graph hashes")
        # L0/L1 and the S rows are intentionally different quality families.
        # Require identity only within each family; otherwise a legitimate
        # rho²-vs-reranker comparison would be rejected as a protocol error.
        for family, family_labels in (
            ("rho2", ("L0", "L1")),
            ("pointwise", ("S0", "S1", "S2", "S3", "S2-shuffle")),
        ):
            quality_hashes = {
                _quality_digest(rows[label].get("shared_quality", {}))
                for label in family_labels
                if label in rows
            }
            if len(quality_hashes) > 1:
                errors.append(f"{question_id}: {family} architectures use different frozen quality tables")
            source_values = {
                str(rows[label].get("quality_source", ""))
                for label in family_labels
                if label in rows
            }
            if len(source_values) > 1:
                errors.append(f"{question_id}: {family} architectures use different quality sources")

    manifest_families = manifest.get("quality_source_families")
    if not isinstance(manifest_families, Mapping):
        errors.append("semantic manifest quality_source_families is missing")
    else:
        for label, family in quality_families.items():
            if str(manifest_families.get(label, "")) != family:
                errors.append(f"manifest quality family mismatch for {label}")
    manifest_sources = manifest.get("quality_sources")
    if not isinstance(manifest_sources, Mapping):
        errors.append("semantic manifest quality_sources is missing")
    else:
        for label in expected:
            observed_sources = {
                str(row.get("quality_source", ""))
                for row in records_by_label.get(label, ())
                if isinstance(row, Mapping)
            }
            declared_source = str(manifest_sources.get(label, ""))
            if declared_source and observed_sources and observed_sources != {declared_source}:
                errors.append(f"{label}: record quality source differs from manifest")

    # Failure rows must be exactly the complement of successful prediction
    # rows.  This catches a stale failures file left over from a rerun.
    expected_keys = {(question_id, label) for question_id in manifest_question_ids for label in expected}
    if (record_keys | failure_keys) != expected_keys:
        errors.append("prediction/failure records do not cover the manifest query-by-architecture grid")
    unexpected_overlap = (record_keys & failure_keys) - prediction_failure_keys
    if unexpected_overlap:
        errors.append("a query/architecture appears in both predictions and failures")

    summary_architectures = summary.get("architectures", {})
    if not isinstance(summary_architectures, Mapping):
        errors.append("summary architectures is not an object")
        summary_architectures = {}
    for label in expected:
        row = summary_architectures.get(label)
        if not isinstance(row, Mapping):
            errors.append(f"summary is missing architecture {label}")
            continue
        expected_success = sum(
            1
            for record in records_by_label[label]
            if isinstance(record, Mapping) and record.get("status", "success") == "success"
        )
        expected_failed = expected_queries - expected_success
        for key, value in (
            ("queries", expected_success),
            ("attempted_queries", expected_queries),
            ("successful_queries", expected_success),
            ("failed_queries", expected_failed),
        ):
            if row.get(key) != value:
                errors.append(f"{label}: summary {key} mismatch")
        expected_rate = expected_failed / expected_queries if expected_queries > 0 else 1.0
        try:
            if not np.isclose(float(row.get("failure_rate")), expected_rate, atol=1e-12, rtol=0.0):
                errors.append(f"{label}: summary failure_rate mismatch")
        except (TypeError, ValueError):
            errors.append(f"{label}: summary failure_rate is invalid")

    # Persist a machine-readable audit beside the run, as the TMIC auditor
    # does.  The report itself includes the selected child when a parent was
    # supplied, which makes shell automation unambiguous.
    report = {
        "status": "passed" if not errors else "failed",
        "requested_dir": str(requested),
        "run_dir": str(root),
        "errors": errors,
        "warnings": warnings,
        "evidence": {
            "architectures": list(expected),
            "queries": expected_queries,
            "records_by_architecture": {label: len(records_by_label[label]) for label in expected},
            "failure_records": len(failures),
            "shared_candidate_pool": bool(summary.get("shared_candidate_pool")),
            "manifest_question_id_sha256": expected_manifest_hash,
            "quality_families": quality_families,
        },
    }
    output_path = root / "semantic_audit.json"
    _atomic_json(output_path, report)
    report["output_path"] = str(output_path)
    if errors and raise_on_error:
        raise RuntimeError("semantic run audit failed: " + "; ".join(errors))
    return report


def audit_tmic_run(
    input_dir: str | Path,
    *,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Audit a persisted ``run-tmic-matrix`` directory.

    The checker is intentionally independent of retrieval code: it validates
    the serialized A0--A4 provenance, posterior/path normalization, PSD atoms,
    exact-partition domains, and certificate diagnostics.  If a parent
    directory is supplied, the newest ``tmic_*`` child is selected.
    """
    requested = Path(input_dir).resolve()
    root = requested
    if not (root / "run_manifest.json").is_file():
        candidates = sorted(
            (item for item in root.glob("tmic_*") if (item / "run_manifest.json").is_file()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            root = candidates[0]
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        raise FileNotFoundError(f"TMIC run directory does not exist: {root}")
    manifest_path = root / "run_manifest.json"
    manifest = _load_json(root, "run_manifest.json", errors, {})
    summary = _load_json(root, "summary.json", errors, {})
    architectures = [str(value) for value in manifest.get("architectures", ())]
    required_architectures = {"A0", "A1", "A2", "A3", "A4"}
    _check(required_architectures.issubset(set(architectures)), "A0-A4 matrix is incomplete", errors)
    _check(len(architectures) == len(set(architectures)), "duplicate architecture labels", errors)

    manifest_question_ids = [str(value) for value in manifest.get("question_ids", ())]
    if manifest_question_ids:
        _check(
            len(manifest_question_ids) == len(set(manifest_question_ids)),
            "manifest question_ids contain duplicates",
            errors,
        )
        expected_manifest_hash = hashlib.sha256("\n".join(manifest_question_ids).encode("utf-8")).hexdigest()
        _check(
            manifest.get("question_id_sha256") == expected_manifest_hash,
            "manifest question_id_sha256 mismatch",
            errors,
        )
    if manifest.get("protocol") is not None:
        _check(manifest.get("protocol") == "confirmatory_v1", "TMIC protocol name mismatch", errors)
    if manifest.get("data_revision") is not None:
        _check(manifest.get("data_revision") == PERSONAMEM_REVISION, "TMIC data revision mismatch", errors)

    prediction_files: dict[str, list[Dict[str, Any]]] = {}
    question_sets: dict[str, set[str]] = {}
    certificate_count = 0
    exhaustive_checks = 0
    exhaustive_matches = 0

    def finite_number(value: Any) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    for label in architectures:  # auxiliary rows are allowed
        records = _load_jsonl(root, f"predictions_{label}.jsonl", errors)
        prediction_files[label] = records
        seen_questions: set[str] = set()
        for row_index, record in enumerate(records, start=1):
            question_id = str(record.get("question_id", ""))
            if not question_id:
                errors.append(f"{label}: prediction {row_index} has no question_id")
            if question_id in seen_questions:
                errors.append(f"{label}: duplicate question_id {question_id}")
            seen_questions.add(question_id)
            provenance = record.get("provenance", {})
            if not isinstance(provenance, Mapping):
                errors.append(f"{label}/{question_id}: provenance is not an object")
                continue
            required = (
                "selected_in_greedy_order",
                "transition",
                "paths",
                "information_atoms",
                "domains",
                "bounds",
                "selection_steps",
            )
            for key in required:
                if key not in provenance:
                    errors.append(f"{label}/{question_id}: missing provenance field {key}")

            greedy = provenance.get("selected_in_greedy_order", ())
            if not isinstance(greedy, list):
                errors.append(f"{label}/{question_id}: greedy IDs are not a list")
                greedy = []
            _check(
                len(greedy) == len(set(str(value) for value in greedy)),
                f"{label}/{question_id}: duplicate greedy IDs",
                errors,
            )
            selected = record.get("selected_memory_ids", provenance.get("chronological_ids", ()))
            if isinstance(selected, list):
                _check(
                    len(selected) == len(set(str(value) for value in selected)),
                    f"{label}/{question_id}: duplicate selected IDs",
                    errors,
                )

            # Transition is P_q, so every non-empty row must be stochastic;
            # zero-row uniformization is represented explicitly in the
            # temporal diagnostics and is still a valid stochastic row.
            transition_raw = provenance.get("transition")
            transition = None
            if transition_raw is not None:
                try:
                    transition = np.asarray(transition_raw, dtype=np.float64)
                    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
                        raise ValueError("transition is not square")
                    if np.any(~np.isfinite(transition)) or np.any(transition < -1e-10):
                        raise ValueError("transition contains invalid values")
                    row_sums = transition.sum(axis=1)
                    if len(row_sums) and not np.allclose(row_sums, 1.0, atol=1e-7):
                        errors.append(f"{label}/{question_id}: transition rows are not normalized")
                except (TypeError, ValueError):
                    errors.append(f"{label}/{question_id}: invalid transition matrix")
                    transition = None

            diagnostics = provenance.get("tmic_diagnostics", {})
            if not isinstance(diagnostics, Mapping):
                diagnostics = {}
            memory_ids = [str(value) for value in diagnostics.get("memory_ids", ())]
            if not memory_ids:
                temporal_diag = diagnostics.get("temporal", {})
                if isinstance(temporal_diag, Mapping):
                    memory_ids = [str(value) for value in temporal_diag.get("ids", ())]
            if not memory_ids:
                memory_ids = [
                    str(item.get("memory_id"))
                    for item in provenance.get("nodes", ())
                    if isinstance(item, Mapping)
                ]
            excluded = {str(value) for value in diagnostics.get("excluded_candidate_ids", ())}

            # Node-level provenance is the authoritative ID universe for
            # discovered parent/path records.  Keep the full-bank IDs from
            # the temporal diagnostics as an additional (superset) check.
            node_records = provenance.get("nodes", ())
            discovered_ids: set[str] = set()
            if isinstance(node_records, list):
                for node_index, node in enumerate(node_records, start=1):
                    if not isinstance(node, Mapping):
                        errors.append(f"{label}/{question_id}: invalid node record {node_index}")
                        continue
                    node_id = str(node.get("memory_id", ""))
                    if not node_id:
                        errors.append(f"{label}/{question_id}: node {node_index} has no memory_id")
                    else:
                        if node_id in discovered_ids:
                            errors.append(f"{label}/{question_id}: duplicate node ID {node_id}")
                        discovered_ids.add(node_id)
                    parent_map = node.get("parent_posterior", {})
                    if not isinstance(parent_map, Mapping):
                        errors.append(f"{label}/{question_id}/{node_id}: parent_posterior is not an object")
                        continue
                    parent_values: list[float] = []
                    for _parent_id, value in parent_map.items():
                        if not finite_number(value) or float(value) < -1e-10:
                            errors.append(f"{label}/{question_id}/{node_id}: invalid parent posterior")
                        else:
                            parent_values.append(max(0.0, float(value)))
                    if parent_values and not np.isclose(sum(parent_values), 1.0, atol=1e-8):
                        errors.append(f"{label}/{question_id}/{node_id}: parent posterior does not sum to one")
                    parent_id = node.get("parent_id")
                    if parent_id is not None and str(parent_id) == node_id:
                        errors.append(f"{label}/{question_id}/{node_id}: self parent")

            known_ids = set(memory_ids) | discovered_ids

            paths = provenance.get("paths", {})
            if isinstance(paths, Mapping):
                for candidate, values in paths.items():
                    candidate = str(candidate)
                    if not isinstance(values, list):
                        errors.append(f"{label}/{question_id}/{candidate}: paths is not a list")
                        continue
                    posterior_values: list[float] = []
                    for path in values:
                        if not isinstance(path, Mapping):
                            errors.append(f"{label}/{question_id}/{candidate}: invalid path record")
                            continue
                        posterior = path.get("posterior", 0.0)
                        support = path.get("support", 0.0)
                        if not finite_number(posterior) or float(posterior) < -1e-10:
                            errors.append(f"{label}/{question_id}/{candidate}: invalid path posterior")
                        else:
                            posterior_values.append(max(0.0, float(posterior)))
                        if not finite_number(support) or float(support) < -1e-10:
                            errors.append(f"{label}/{question_id}/{candidate}: invalid path support")
                        path_ids = [str(value) for value in path.get("path_ids", ())]
                        if not path_ids or path_ids[-1] != candidate or len(path_ids) != len(set(path_ids)):
                            errors.append(f"{label}/{question_id}/{candidate}: malformed/cyclic path")
                        if known_ids and any(value not in known_ids for value in path_ids):
                            errors.append(f"{label}/{question_id}/{candidate}: path references an unknown memory ID")
                        parent_map = path.get("parent_posterior", {})
                        if isinstance(parent_map, Mapping):
                            parent_values = []
                            for parent_id, value in parent_map.items():
                                if not finite_number(value) or float(value) < -1e-10:
                                    errors.append(f"{label}/{question_id}/{candidate}: invalid parent posterior")
                                else:
                                    parent_values.append(max(0.0, float(value)))
                                if known_ids and str(parent_id) not in known_ids:
                                    errors.append(
                                        f"{label}/{question_id}/{candidate}: parent posterior references unknown ID"
                                    )
                            if parent_values and not np.isclose(sum(parent_values), 1.0, atol=1e-8):
                                errors.append(
                                    f"{label}/{question_id}/{candidate}: parent posterior does not sum to one"
                                )
                            # For a non-root path, at least its immediate
                            # predecessor must be represented in the stored
                            # immediate-parent posterior map.  Alternative
                            # parents may legitimately appear in the map and
                            # need not lie on this particular path.
                            if len(path_ids) > 1 and parent_values and path_ids[-2] not in {
                                str(key) for key in parent_map
                            }:
                                errors.append(
                                    f"{label}/{question_id}/{candidate}: path predecessor missing from parent posterior"
                                )
                        else:
                            errors.append(f"{label}/{question_id}/{candidate}: parent_posterior is not an object")
                    if values and not np.isclose(sum(posterior_values), 1.0, atol=1e-8):
                        errors.append(f"{label}/{question_id}/{candidate}: path posterior does not sum to one")

            atoms = provenance.get("information_atoms", {})
            if isinstance(atoms, Mapping):
                for candidate, atom in atoms.items():
                    try:
                        if not isinstance(atom, Mapping):
                            raise ValueError("atom is not an object")
                        matrix = np.asarray(atom.get("matrix"), dtype=np.float64)
                        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
                            raise ValueError("not a finite square matrix")
                        symmetric = (matrix + matrix.T) * 0.5
                        minimum = float(np.min(np.linalg.eigvalsh(symmetric))) if matrix.size else 0.0
                        if minimum < -1e-8:
                            errors.append(f"{label}/{question_id}/{candidate}: information atom is not PSD")
                        trace = atom.get("trace")
                        if trace is not None and (
                            not finite_number(trace)
                            or abs(float(trace) - float(np.trace(symmetric))) > 1e-7
                        ):
                            errors.append(f"{label}/{question_id}/{candidate}: atom trace mismatch")
                    except (TypeError, ValueError, np.linalg.LinAlgError):
                        errors.append(f"{label}/{question_id}/{candidate}: invalid information atom")

            domains = provenance.get("domains", {})
            domain_values: list[tuple[str, ...]] = []
            if isinstance(domains, Mapping):
                for branch_id, values in domains.items():
                    if not isinstance(values, list):
                        errors.append(f"{label}/{question_id}/{branch_id}: domain is not a list")
                        continue
                    domain_values.append(tuple(str(value) for value in values))
            flattened = [value for values in domain_values for value in values]
            if len(flattened) != len(set(flattened)):
                errors.append(f"{label}/{question_id}: certificate domains overlap")
            known_eligible = set(memory_ids) - excluded
            cert_status = str(provenance.get("certificate_status", "not_requested"))
            if known_eligible and domain_values and (
                label == "A4" or cert_status in {"available", "certificate_unavailable"}
            ):
                domain_union = set(flattened)
                if domain_union != known_eligible:
                    errors.append(
                        f"{label}/{question_id}: certificate domain union mismatch "
                        f"(expected {len(known_eligible)}, got {len(domain_union)})"
                    )

            bounds = provenance.get("bounds", {})
            if isinstance(bounds, Mapping):
                for branch_id, bound in bounds.items():
                    if not isinstance(bound, Mapping):
                        errors.append(f"{label}/{question_id}/{branch_id}: bound is not an object")
                        continue
                    if bound.get("valid") is True:
                        for field_name in ("U", "smax", "rho_upper"):
                            if field_name in bound and (
                                not finite_number(bound[field_name]) or float(bound[field_name]) < -1e-10
                            ):
                                errors.append(f"{label}/{question_id}/{branch_id}: invalid bound {field_name}")

            selection_bounds = diagnostics.get("selection_bounds", ())
            if isinstance(selection_bounds, list):
                for bound_row in selection_bounds:
                    if not isinstance(bound_row, Mapping):
                        errors.append(f"{label}/{question_id}: invalid selection bound row")
                        continue
                    best = bound_row.get("best_discovered_margin", 0.0)
                    upper = bound_row.get("frontier_upper_bound", 0.0)
                    gap = bound_row.get("gap", 0.0)
                    if not all(finite_number(value) for value in (best, upper, gap)) or float(gap) < -1e-8:
                        errors.append(f"{label}/{question_id}: invalid certificate gap")
                    expected_gap = (
                        max(0.0, float(upper) - float(best))
                        if finite_number(best) and finite_number(upper)
                        else 0.0
                    )
                    if finite_number(gap) and abs(float(gap) - expected_gap) > 1e-7:
                        errors.append(f"{label}/{question_id}: certificate gap mismatch")
                    if bound_row.get("certified") and not bound_row.get("bound_valid", False):
                        errors.append(f"{label}/{question_id}: certified step has invalid bound")

            if cert_status not in {"not_requested", "available", "certificate_unavailable"}:
                errors.append(f"{label}/{question_id}: unknown certificate status {cert_status}")
            if cert_status == "available":
                certificate_count += 1
                if not domains or not isinstance(diagnostics.get("domains_valid", True), bool):
                    errors.append(f"{label}/{question_id}: available certificate has no valid domain evidence")
                if diagnostics.get("domains_valid") is False:
                    errors.append(f"{label}/{question_id}: certificate status available but domains_valid=false")

                # When every eligible item has an atom, independently replay
                # the exact finite-domain greedy sequence.  If unseen items
                # remain, the certificate may still be valid, but an
                # exhaustive replay is intentionally reported as unavailable.
                atom_map = {}
                if isinstance(atoms, Mapping):
                    for candidate, atom in atoms.items():
                        if isinstance(atom, Mapping) and atom.get("matrix") is not None:
                            with contextlib.suppress(TypeError, ValueError):
                                atom_map[str(candidate)] = np.asarray(atom["matrix"], dtype=np.float64)
                if atom_map and (not known_eligible or known_eligible.issubset(set(atom_map))):
                    try:
                        objective = InformationObjective(atoms=atom_map)
                        expected_ids, _margins = objective.exhaustive_greedy(
                            list(atom_map), k=len(provenance.get("selection_steps", ()))
                        )
                        actual_ids = [
                            str(step.get("memory_id"))
                            for step in provenance.get("selection_steps", ())
                            if isinstance(step, Mapping)
                        ]
                        exhaustive_checks += 1
                        if actual_ids[: len(expected_ids)] != expected_ids:
                            errors.append(
                                f"{label}/{question_id}: certified greedy sequence differs from exhaustive replay"
                            )
                        else:
                            exhaustive_matches += 1
                    except (KeyError, TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
                        warnings.append(f"{label}/{question_id}: exhaustive replay unavailable")

        question_sets[label] = seen_questions

    # Every architecture must be evaluated on the same persisted question
    # set.  A per-architecture failure is therefore evidence, not a silent
    # denominator change.
    expected_questions = set(manifest_question_ids)
    if not expected_questions:
        expected_questions = set().union(*question_sets.values()) if question_sets else set()
    for label, values in question_sets.items():
        if values != expected_questions:
            errors.append(
                f"{label}: common question set mismatch (expected {len(expected_questions)}, got {len(values)})"
            )
    if summary and isinstance(summary.get("architectures"), Mapping):
        for label in architectures:
            row = summary["architectures"].get(label, {})
            if isinstance(row, Mapping):
                _check(
                    row.get("queries") == len(question_sets.get(label, set())),
                    f"{label}: summary query count mismatch",
                    errors,
                )
                _check(
                    row.get("failed_queries", 0) == 0,
                    f"{label}: summary reports failed queries",
                    errors,
                )

    # C changes only the stopping rule, so A3/A4 selection equality is an
    # explicit comparison metric rather than an invariant.  A valid early
    # certificate may expose fewer nodes and legitimately select a different
    # context.  Whenever the context *is* identical, however, the shared
    # generation cache/protocol requires an identical response.
    a3 = {str(item.get("question_id")): item for item in prediction_files.get("A3", ())}
    a4 = {str(item.get("question_id")): item for item in prediction_files.get("A4", ())}
    common = set(a3) & set(a4)
    equal = 0
    for question_id in sorted(common):
        left_hash, right_hash = a3[question_id].get("context_hash"), a4[question_id].get("context_hash")
        equal += int(left_hash == right_hash)
        shared_response_differs = a3[question_id].get("response") != a4[question_id].get("response")
        if manifest.get("generation") and left_hash == right_hash and shared_response_differs:
            errors.append(f"A3/A4 shared context has different generation responses: {question_id}")

    evidence = {
        "run_dir": str(root),
        "manifest": str(manifest_path),
        "architectures": architectures,
        "prediction_counts": {label: len(records) for label, records in prediction_files.items()},
        "common_question_count": len(expected_questions),
        "a3_a4_common_questions": len(common),
        "a3_a4_context_hash_equal": equal,
        "certificate_available_records": certificate_count,
        "exhaustive_replay_checks": exhaustive_checks,
        "exhaustive_replay_matches": exhaustive_matches,
        "summary_present": bool(summary),
    }
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "evidence": evidence,
    }
    output_path = root / "tmic_audit.json"
    _atomic_json(output_path, report)
    if errors and raise_on_error:
        raise RuntimeError("TMIC audit failed: " + "; ".join(errors))
    return {"output_path": str(output_path), **report}
