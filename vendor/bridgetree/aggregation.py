from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .metrics import (
    exact_sign_flip_test,
    gain_damage_net,
    paired_bootstrap_interval,
    persona_cluster_bootstrap_interval,
    persona_macro_accuracy,
    question_micro_accuracy,
)


def _base_label(run_label: str) -> str:
    return re.sub(r"_seed\d+$", "", run_label)


def aggregate_runs(
    input_dir: str | Path,
    reference_label: str = "core",
    bootstrap_seed: int = 42,
    bootstrap_resamples: int = 2000,
) -> Dict[str, Any]:
    root = Path(input_dir)
    runs = []
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dir = manifest_path.parent
        summary_path = run_dir / "summary.json"
        predictions_path = run_dir / "predictions.jsonl"
        if not summary_path.exists() or not predictions_path.exists():
            continue
        predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line]
        runs.append(
            {
                "run_dir": str(run_dir),
                "label": _base_label(str(manifest.get("run_label", manifest.get("method", "unknown")))),
                "seed": int(manifest.get("seed", 0)),
                "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                "predictions": predictions,
            }
        )
    if not runs:
        raise ValueError(f"no completed experiment runs found in {root}")

    by_label: Dict[str, list[Dict[str, Any]]] = {}
    for run in runs:
        by_label.setdefault(run["label"], []).append(run)
    if reference_label not in by_label:
        # TMIC matrix artifacts use A0 as the fixed R1 reference, while the
        # historical multi-run scripts use ``core``.  Preserve an explicit
        # caller choice, but make the default useful for either artifact type.
        if reference_label == "core" and "A0" in by_label:
            reference_label = "A0"
        else:
            raise ValueError(f"reference label is unavailable: {reference_label}")

    metric = None
    for candidate in ("answer_accuracy", "recall_at_k", "bridge_recall_at_k"):
        if any(
            prediction.get("outcome", {}).get(candidate) is not None
            for run in runs
            for prediction in run["predictions"]
        ):
            metric = candidate
            break

    reference_values: Dict[tuple[int, str], float] = {}
    if metric:
        for run in by_label[reference_label]:
            for prediction in run["predictions"]:
                value = prediction.get("outcome", {}).get(metric)
                if value is not None:
                    reference_values[(run["seed"], prediction["question_id"])] = float(value)

    def persona_values(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        grouped: Dict[str, list[float]] = {}
        for prediction in predictions:
            value = prediction.get("outcome", {}).get(metric) if metric else None
            persona = prediction.get("persona_id")
            if value is None or persona is None:
                continue
            grouped.setdefault(str(persona), []).append(float(value))
        return {persona: sum(values) / len(values) for persona, values in grouped.items() if values}

    labels: Dict[str, Any] = {}
    for label, label_runs in sorted(by_label.items()):
        seeds = sorted({run["seed"] for run in label_runs})
        costs = [run["summary"].get("cost", {}).get("mean", {}) for run in label_runs]
        cost_names = sorted({name for record in costs for name in record})
        mean_cost = {name: sum(float(record.get(name, 0.0)) for record in costs) / len(costs) for name in cost_names}
        values_by_key = {}
        if metric:
            for run in label_runs:
                for prediction in run["predictions"]:
                    value = prediction.get("outcome", {}).get(metric)
                    if value is not None:
                        values_by_key[(run["seed"], prediction["question_id"])] = float(value)
        common = sorted(set(values_by_key) & set(reference_values))
        paired = None
        if metric and common:
            paired = paired_bootstrap_interval(
                [values_by_key[key] for key in common],
                [reference_values[key] for key in common],
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
        # Primary protocol statistics are persona macro averages.  Keep the
        # question-level micro mean as a clearly labeled auxiliary value.
        pooled_predictions = [prediction for run in label_runs for prediction in run["predictions"]]
        persona_macro = persona_macro_accuracy(pooled_predictions, metric or "answer_accuracy")
        micro_values = [
            question_micro_accuracy(run["predictions"], metric or "answer_accuracy") for run in label_runs
        ]
        # Aggregate paired correction counts on the first common seed when
        # multiple seeds are present; seed-paired records remain available in
        # the detailed run list.
        gain_damage = None
        sign_flip = None
        cluster_bootstrap = None
        if label != reference_label:
            common_seeds = sorted(
                {run["seed"] for run in label_runs}
                & {run["seed"] for run in by_label[reference_label]}
            )
            if common_seeds:
                seed_value = common_seeds[0]
                treatment_run = next(run for run in label_runs if run["seed"] == seed_value)
                baseline_run = next(run for run in by_label[reference_label] if run["seed"] == seed_value)
                gain_damage = gain_damage_net(
                    baseline_run["predictions"],
                    treatment_run["predictions"],
                    metric or "answer_accuracy",
                )
                treatment_persona = persona_values(treatment_run["predictions"])
                baseline_persona = persona_values(baseline_run["predictions"])
                common_personas = sorted(set(treatment_persona) & set(baseline_persona))
                if common_personas:
                    tmap = {persona: treatment_persona[persona] for persona in common_personas}
                    bmap = {persona: baseline_persona[persona] for persona in common_personas}
                    if len(common_personas) <= 20:
                        sign_flip = exact_sign_flip_test(tmap, bmap)
                    cluster_bootstrap = persona_cluster_bootstrap_interval(
                        {persona: tmap[persona] - bmap[persona] for persona in common_personas},
                        seed=bootstrap_seed,
                        resamples=bootstrap_resamples,
                    )
        # Preserve stratified summaries from every seed instead of letting a
        # dict comprehension overwrite earlier runs.  Each field/group is
        # pooled at the question level while its persona macro remains the
        # primary within-group statistic.
        stratified: Dict[str, Any] = {}
        for field in ("question_type", "topic", "memory_scale"):
            grouped: Dict[str, list[Mapping[str, Any]]] = {}
            for run in label_runs:
                for prediction in run["predictions"]:
                    grouped.setdefault(str(prediction.get(field, "unknown")), []).append(prediction)
            stratified[field] = {
                group: {
                    "queries": len(records),
                    "question_micro": question_micro_accuracy(records, metric or "answer_accuracy"),
                    "persona_macro": persona_macro_accuracy(records, metric or "answer_accuracy"),
                }
                for group, records in sorted(grouped.items())
            }

        labels[label] = {
            "runs": len(label_runs),
            "seeds": seeds,
            "seed_requirement_met": len(seeds) >= 3,
            "queries_with_outcome": len(values_by_key),
            "mean_outcome": (sum(values_by_key.values()) / len(values_by_key) if values_by_key else None),
            "question_micro_accuracy": (
                sum(value for value in micro_values if value is not None)
                / len([value for value in micro_values if value is not None])
                if any(value is not None for value in micro_values)
                else None
            ),
            "persona_macro_accuracy": persona_macro,
            "gain_damage_net": gain_damage,
            "sign_flip": sign_flip,
            "persona_cluster_bootstrap": cluster_bootstrap,
            "stratified": stratified,
            "paired_vs_reference": paired,
            "mean_cost": mean_cost,
            "run_dirs": [run["run_dir"] for run in label_runs],
        }
    result = {
        "input_dir": str(root),
        "reference_label": reference_label,
        "outcome_metric": metric,
        "bootstrap": {
            "unit": "query_seed_pair",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
            "confidence": 0.95,
        },
        "labels": labels,
    }
    output_path = root / "aggregate_summary.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output_path": str(output_path), **result}
