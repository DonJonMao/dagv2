from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .clients import GeneratorClient, RerankerClient, RerankItem
from .config import load_config
from .experiment import _public_app_config
from .run_audit import audit_tuning_run
from .training import load_tuning_config, preflight_tuning, run_tuning_experiment


class OfflineDeterministicEmbedder:
    """Small deterministic encoder used only to exercise the complete scheduler offline."""

    dimension = 16

    @classmethod
    def _vector(cls, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.asarray([digest[index] / 127.5 - 1.0 for index in range(cls.dimension)], dtype=np.float64)
        norm = float(np.linalg.norm(values))
        return values / norm if norm > 0.0 else np.eye(1, cls.dimension, dtype=np.float64)[0]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float64)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector("offline-query:" + text)


def _offline_answer(
    _client: GeneratorClient,
    _query: str,
    _memories,
    _answer_options: str = "",
) -> str:
    return "(a) offline scheduler validation"


def _offline_rerank(
    _client: RerankerClient,
    _query: str,
    documents: Sequence[str],
    top_n: int,
) -> list[RerankItem]:
    return [RerankItem(index=index, score=1.0 - index * 1e-6) for index in range(min(top_n, len(documents)))]


def validate_full_32k_offline(
    config_path: str | Path = "configs/default.yaml",
    tuning_config_path: str | Path = "configs/train.yaml",
    output_dir: str | Path = "outputs/offline-full-32k-validation",
) -> Dict[str, Any]:
    """Execute the formal full-size scheduler with conspicuously fake local clients."""
    output_root = Path(output_dir)
    cache_dir = output_root / "cache"
    run_parent = output_root / "runs"
    app_config = load_config(config_path)
    app_config = replace(
        app_config,
        models=replace(
            app_config.models,
            embedding=replace(
                app_config.models.embedding,
                backend="remote",
                endpoint="offline://deterministic-embedding",
                model="offline-deterministic-embedding-16d",
            ),
            reranker=replace(
                app_config.models.reranker,
                endpoint="offline://deterministic-reranker",
                model="offline-deterministic-reranker",
            ),
            generator=replace(
                app_config.models.generator,
                endpoint="offline://constant-generator",
                model="offline-constant-option-a",
                api_key="offline-no-secret",
            ),
        ),
        runtime=replace(app_config.runtime, device="cpu", cache_dir=str(cache_dir), output_dir=str(run_parent)),
    )
    tuning_config = load_tuning_config(tuning_config_path)
    tuning_config = replace(tuning_config, output_dir=str(run_parent), keep_example_metrics=True)
    embedder = OfflineDeterministicEmbedder()
    preflight = preflight_tuning(
        app_config,
        tuning_config,
        embedder=embedder,
        check_services=False,
        require_full_32k=True,
    )
    started = time.perf_counter()
    original_answer = GeneratorClient.answer
    original_rerank = RerankerClient.rerank
    GeneratorClient.answer = _offline_answer
    RerankerClient.rerank = _offline_rerank
    try:
        result = run_tuning_experiment(app_config, tuning_config, embedder, output_dir=run_parent)
    finally:
        GeneratorClient.answer = original_answer
        RerankerClient.rerank = original_rerank
    run_dir = Path(result["run_dir"])
    completion_audit = audit_tuning_run(run_dir, require_full_32k=True, raise_on_error=True)
    report = {
        "status": "passed",
        "validation_kind": "offline_full_size_scheduler_only_not_experiment_results",
        "elapsed_seconds": time.perf_counter() - started,
        "run_dir": str(run_dir),
        "resolved_app_config": _public_app_config(app_config),
        "tuning_config": asdict(tuning_config),
        "preflight": preflight,
        "assertions": completion_audit["evidence"],
        "completion_audit": completion_audit,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "OFFLINE_VALIDATION_REPORT.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report
