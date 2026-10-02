"""Load frozen DAG v2 resources for the independent DAG-Resolve entry point.

The original import loader applies only its existing v6 adaptations. Research
functions are called explicitly and never installed into the native modules.
"""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.parse


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
DEFAULT_CONFIG = {
    "llm_base_url": "http://127.0.0.1:8020/v1",
    "embedding_base_url": "http://127.0.0.1:8019/v1",
    "llm_model": "qwen3.8-27b",
    "embedding_model": "nvidia/NV-Embed-v2",
    "request_timeout_seconds": 600,
    "max_identical_attempts": 3,
}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    from dagbt.runner import validate_config as validate_shared
    validate_shared(config)
    if config.get("model_profile", "legacy") != "legacy":
        raise ValueError("DAG-Resolve uses the frozen NV index and legacy model profile")
    normalized = {**DEFAULT_CONFIG, **config}
    for name in ("llm_model", "embedding_model"):
        if not isinstance(normalized[name], str) or not normalized[name].strip():
            raise ValueError(name + " must be a nonempty model identifier")
    for name in ("llm_base_url", "embedding_base_url"):
        parsed = urllib.parse.urlsplit(normalized[name])
        if parsed.fragment:
            raise ValueError(name + " cannot contain a URL fragment")
        normalized[name] = normalized[name].rstrip("/")
    attempts = normalized["max_identical_attempts"]
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("max_identical_attempts must be an integer from 1 to 3")
    timeout = normalized["request_timeout_seconds"]
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("request_timeout_seconds must be positive")
    return normalized


def import_originals(dataset):
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset: " + str(dataset))
    for path in (ROOT / "package", ROOT / "dagv2"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # The explicitly supplied config is authoritative, even if the calling
    # shell also has a legacy pipeline override set.
    previous = os.environ.pop("DAGV2_CONFIG", None)
    try:
        pipeline = importlib.import_module(
            "pipeline_dagv2_musique" if dataset == "musique" else "pipeline_dagv2")
    finally:
        if previous is not None:
            os.environ["DAGV2_CONFIG"] = previous
    e = pipeline.e
    return SimpleNamespace(e=e, v6=importlib.import_module("frozen_flow_v6"),
                           flow=importlib.import_module("frozen_flow"),
                           core=importlib.import_module("core"), pipeline=pipeline)


def load_runtime(dataset, config):
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset: " + str(dataset))
    config = validate_config(config)
    index_manifest = json.loads((ROOT / "data" / dataset / "index" / "manifest.json").read_text())
    if config["embedding_model"] != index_manifest["embedding_model"]:
        raise ValueError("Embedding model differs from the frozen NV corpus index; no index fallback")
    from dagbt.runner import verify_originals
    integrity = verify_originals()  # Evaluation labels remain unopened.
    runtime = import_originals(dataset)
    # Reuse the original question-only planner and data preparation, but never
    # redirect native.solve or flow.controller to DAG-Resolve implementations.
    runtime.e.CONFIG = {**runtime.e.CONFIG, **config}
    questions, resources, hashes = runtime.pipeline.prepare(dataset, ROOT / "outputs")
    runtime.questions = questions
    runtime.resources = resources
    runtime.config = dict(runtime.e.CONFIG)
    runtime.requested_config = config
    runtime.integrity = integrity
    runtime.hashes = {str(Path(path).relative_to(ROOT)): value for path, value in hashes.items()}
    return runtime
