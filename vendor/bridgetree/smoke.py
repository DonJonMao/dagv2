from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .clients import context_token_count
from .config import RetrievalConfig
from .retriever import BridgeTreeRetriever
from .types import Memory


def run_synthetic_smoke(output_dir: str | Path = "outputs/smoke") -> Dict[str, Any]:
    """Exercise all named runtime combinations without model services."""
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    vectors = np.asarray(
        [
            [0.8, 0.6],
            [0.0, 1.0],
            [0.7, -0.714],
            [-0.3, 0.954],
            [-0.8, 0.6],
        ],
        dtype=np.float32,
    )
    memories = [
        Memory(f"m{index + 1}", f"synthetic memory {index + 1}", float(index), f"s{index}") for index in range(5)
    ]
    core = RetrievalConfig(
        initial_width=1,
        branch_width=1,
        context_size=1,
        search_budget=5,
        max_depth=2,
        cluster_mode="fixed",
        cluster_count=1,
        feature_mode="rho",
        selection_mode="rho_logdet",
        stop_mode="budget",
        diagnostic_level="light",
    )
    configurations = {
        "core": core,
        "path": replace(core, feature_mode="path_conditioned", selection_mode="path_logdet"),
        "certificate": replace(
            core,
            feature_mode="path_conditioned",
            selection_mode="path_logdet",
            stop_mode="certificate_or_budget",
        ),
        "full_current": replace(
            core,
            cluster_mode="effective_rank",
            feature_mode="path_conditioned",
            selection_mode="path_logdet",
            stop_mode="certificate_or_budget",
            max_depth=3,
        ),
    }
    runs = {}
    for label, config in configurations.items():
        result = BridgeTreeRetriever(config).retrieve("synthetic q->m1->m2", query, memories, vectors)
        result.cost_tracker.final_context_count = len(result.selected)
        result.cost_tracker.final_context_tokens = context_token_count(result.selected)
        runs[label] = {
            "config": asdict(config),
            "selected": result.selected_in_greedy_order,
            "discovered": list(result.nodes),
            "cost": result.cost.to_dict(),
            "stop_reason": result.stop_reason,
            "tree_semantics": result.first_arrival_semantics,
        }
    if "m2" not in runs["core"]["discovered"]:
        raise AssertionError("depth=2 synthetic smoke did not discover the bridge candidate m2")
    payload = json.dumps(runs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # ``time.time_ns`` is normally sufficient for run IDs, but some filesystems
    # (and concurrent CLI invocations) can expose the same clock value.  Make
    # creation collision-safe so two smoke checks never fail merely because
    # they started together.
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    root = output_root / f"synthetic_{stamp}"
    suffix = 1
    while True:
        try:
            root.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            root = output_root / f"synthetic_{stamp}_{suffix}"
            suffix += 1
    summary = {
        "status": "passed",
        "generator_calls": 0,
        "embedding_service_calls": 0,
        "configuration_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "runs": runs,
    }
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return {"run_dir": str(root), **summary}
