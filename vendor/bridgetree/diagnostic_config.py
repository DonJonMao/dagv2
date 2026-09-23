"""Strict, separate configuration for post-hoc dependency diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .dependency_config import load_dependency_config


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_fields(value: Any, allowed: set[str], required: set[str], name: str) -> dict:
    if not isinstance(value, Mapping) or set(value) - allowed or required - set(value):
        raise ValueError(f"invalid {name} fields")
    return dict(value)


def exact_int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def load_diagnostic_config(path: str | Path) -> tuple[dict, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    fields = {"schema_version", "deployment_config", "deployment_override", "detail_archive",
              "full_archive", "questions", "contexts", "cases", "repeats", "order_seed",
              "task_max_attempts", "budgets", "root_seeds", "include_dense_controls"}
    result = strict_fields(raw, fields, fields - {"deployment_override"}, "diagnostic config")
    if exact_int(result["schema_version"], "schema_version", 1) != 1:
        raise ValueError("unsupported diagnostic schema")
    for name in ("deployment_config", "deployment_override", "questions", "contexts"):
        if result.get(name) is not None:
            p = Path(result[name]).expanduser()
            result[name] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    for name in ("detail_archive", "full_archive"):
        spec = strict_fields(result[name], {"path", "sha256"}, {"path", "sha256"}, name)
        p = Path(spec["path"]).expanduser()
        spec["path"] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
        if file_digest(spec["path"]) != spec["sha256"]:
            raise ValueError(f"{name} SHA256 mismatch")
        result[name] = spec
    for name in ("repeats", "task_max_attempts"):
        exact_int(result[name], name, 1)
    exact_int(result["order_seed"], "order_seed")
    if type(result["include_dense_controls"]) is not bool:
        raise ValueError("include_dense_controls must be boolean")
    budgets = {"score_logical_inputs", "generation_trials", "score_transport_attempts",
               "generation_transport_attempts", "root_runs", "root_transport_attempts"}
    result["budgets"] = strict_fields(result["budgets"], budgets, budgets, "budgets")
    for name, value in result["budgets"].items():
        exact_int(value, name)
    if not isinstance(result["root_seeds"], list):
        raise ValueError("root_seeds must be a list")
    for value in result["root_seeds"]:
        exact_int(value, "root seed")
    if len(set(result["root_seeds"])) != len(result["root_seeds"]):
        raise ValueError("root seeds must be unique")
    if not isinstance(result["cases"], list) or not result["cases"]:
        raise ValueError("cases must be a nonempty list")
    seen = set()
    for i, case in enumerate(result["cases"]):
        case = strict_fields(case, {"name", "question_id", "task_id", "universe", "controls"},
                             {"name", "question_id", "task_id", "universe"}, "case")
        qid = case["question_id"]
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("case question IDs must be unique")
        seen.add(qid)
        universe = case["universe"]
        if not isinstance(universe, list) or not 1 <= len(universe) <= 10:
            raise ValueError("diagnostic universe requires 1..10 memories")
        if any(not isinstance(v, str) or not v for v in universe):
            raise ValueError("universe memory IDs must be nonempty strings")
        case["universe"] = [v if v.startswith(qid + ":") else qid + ":" + v for v in universe]
        if len(set(case["universe"])) != len(universe):
            raise ValueError("duplicate universe memory ID")
        controls = case.setdefault("controls", [])
        if not isinstance(controls, list):
            raise ValueError("controls must be a predeclared list")
        names = set()
        for c in controls:
            strict_fields(c, {"name", "memory_ids", "rationale"}, {"name", "memory_ids", "rationale"}, "control")
            if not isinstance(c["name"], str) or not c["name"] or c["name"] in {"subset", "dense_control"}:
                raise ValueError("control name must be nonempty and not reserved")
            if not isinstance(c["rationale"], str) or not c["rationale"].strip():
                raise ValueError("control rationale must be a nonempty string")
            if not isinstance(c["memory_ids"], list) or any(not isinstance(v, str) or not v for v in c["memory_ids"]):
                raise ValueError("control memory_ids must be a list of nonempty strings")
            if c["name"] in names:
                raise ValueError("duplicate control name")
            names.add(c["name"])
        result["cases"][i] = case
    app = load_dependency_config(result["deployment_config"], result.get("deployment_override"))
    if app.data.split != "32k" or app.data.memory_granularity != "user_assistant_pair":
        raise ValueError("this diagnostic protocol requires PersonaMem-v1 32k user_assistant_pair")
    return result, app
