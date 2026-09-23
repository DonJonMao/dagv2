from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

from .aggregation import aggregate_runs
from .chain_experiment import build_full_plan, write_plan
from .clients import RerankerClient, build_embedder
from .config import apply_runtime_overrides, load_config
from .experiment import METHODS, run_personamem_experiment, run_semantic_matrix, run_tmic_matrix
from .offline_validation import validate_full_32k_offline
from .personamem import PERSONAMEM_REPO, PERSONAMEM_REVISION, prepare_split
from .protocol import audit_protocol, freeze_protocol, init_protocol
from .run_audit import audit_semantic_run, audit_tmic_run, audit_tuning_run
from .server_bundle import build_server_bundle, verify_bundle_offline_launcher
from .smoke import run_synthetic_smoke
from .training import (
    EFFECT_FIRST_VALIDATION_METHODS,
    load_tuning_config,
    preflight_tuning,
    run_effect_first_validation,
    run_tuning_experiment,
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "BridgeTree/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def download_personamem(raw_dir: str | Path, splits: list[str]) -> None:
    root = Path(raw_dir)
    for split in splits:
        for filename in (f"questions_{split}.csv", f"shared_contexts_{split}.jsonl"):
            destination = root / filename
            if destination.exists() and destination.stat().st_size > 0:
                print(f"exists: {destination}")
                continue
            url = (
                f"https://huggingface.co/datasets/{PERSONAMEM_REPO}/resolve/"
                f"{PERSONAMEM_REVISION}/{filename}?download=true"
            )
            print(f"downloading {filename} -> {destination}")
            _download(url, destination)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-width", type=int)
    parser.add_argument("--branch-width", type=int)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--search-budget", type=int)
    parser.add_argument("--max-ann-calls", type=int)
    parser.add_argument("--max-candidate-exposure", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--cluster-mode", choices=("none", "fixed", "effective_rank"))
    parser.add_argument("--cluster-count", type=int)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--min-cluster-size", type=int)
    parser.add_argument("--search-order", choices=("best_first", "bfs"))
    parser.add_argument(
        "--profile",
        choices=("legacy_core", "legacy_path", "semantic_path_v1", "semantic", "legacy"),
    )
    parser.add_argument(
        "--feature-mode",
        choices=("rho", "path_conditioned", "cached_memory", "query_conditioned"),
    )
    parser.add_argument(
        "--selection-mode",
        choices=(
            "rho_topk",
            "mmr",
            "rho_logdet",
            "path_logdet",
            "semantic_path_logdet",
            "pure_rerank",
            "frozen_listwise",
        ),
    )
    parser.add_argument("--stop-mode", choices=("budget", "certificate_or_budget"))
    parser.add_argument("--diagnostic-level", choices=("off", "light", "full"))
    parser.add_argument("--root-anchor-weight", type=float)
    parser.add_argument("--dense-pool-width", type=int)
    parser.add_argument("--anchor-width", type=int)
    parser.add_argument("--expand-branch-count", type=int)
    parser.add_argument("--branch-overfetch-width", type=int)
    parser.add_argument("--branch-keep-width", type=int)
    parser.add_argument("--probe-mode", choices=("centroid", "query_anchor"))
    parser.add_argument("--path-filter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--rerank-use-options",
        dest="use_answer_options",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--rerank-include-time",
        dest="include_time_metadata",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--bridge-query-instruction")
    parser.add_argument("--final-rerank-instruction")
    parser.add_argument("--path-filter-instruction")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--index-backend", choices=("exact", "faiss"))
    parser.add_argument(
        "--temporal-measure",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the angular-rank plus time-measure transition operator",
    )
    parser.add_argument(
        "--measure-propagation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Propagate branch mass over real member transitions",
    )
    parser.add_argument(
        "--state-information",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the query-conditioned state information PSD atoms",
    )
    parser.add_argument(
        "--information-certificate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable finite-domain information bounds for stopping",
    )
    parser.add_argument("--state-basis-mode", choices=("option_contrast", "identity"))
    parser.add_argument("--certificate-domain", choices=("exact_partition", "frozen_pool"))
    parser.add_argument(
        "--proposal-mode",
        choices=(
            "legacy_first_arrival",
            "dense",
            "centroid",
            "real_member_vector",
            "real_member_query_anchor",
            "round_robin",
        ),
    )
    parser.add_argument("--relation-mode", choices=("cosine", "angular", "legacy_temporal_overlap"))
    parser.add_argument("--quality-mode", choices=("direct_cosine", "rho", "frozen_reranker", "mapping", "constant"))
    parser.add_argument(
        "--path-mode",
        choices=("legacy", "none", "single_path", "posterior_expected_scatter", "shuffle"),
    )
    parser.add_argument("--certificate-mode", choices=("off", "lazy", "certificate_or_budget"))
    parser.add_argument("--context-unit", choices=("memory", "token"))
    parser.add_argument("--quality-score-space", choices=("unit_interval", "logit_difference"))
    parser.add_argument("--scorer-fingerprint")
    parser.add_argument("--representation-mode", choices=("cached_memory", "query_conditioned"))
    parser.add_argument("--memory-granularity", choices=("user_only", "user_assistant_pair"))
    parser.add_argument(
        "--include-system-persona",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def _resolved_config(args: argparse.Namespace):
    return apply_runtime_overrides(load_config(args.config, args.override_config), vars(args))


def _resolved_tuning_config(args: argparse.Namespace):
    tuning = load_tuning_config(args.tuning_config)
    search_changes = {}
    if args.initial_width is not None:
        search_changes["initial_width"] = (args.initial_width,)
    if args.branch_width is not None:
        search_changes["branch_width"] = (args.branch_width,)
    if args.search_budget is not None:
        search_changes["search_budget"] = (args.search_budget,)
    if search_changes:
        tuning = replace(tuning, search_space=replace(tuning.search_space, **search_changes))
    if args.seed is not None:
        tuning = replace(tuning, seed=args.seed)
    if getattr(args, "phase", None) is not None:
        tuning = replace(tuning, phase=args.phase)
    if getattr(args, "protocol_manifest", None) is not None:
        tuning = replace(tuning, protocol_manifest=args.protocol_manifest)
    tuning.validate()
    return tuning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bridgetree")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-personamem", help="Download the pinned official PersonaMem-v1 files")
    download.add_argument("--raw-dir", default="data/raw/personamem-v1")
    download.add_argument("--split", action="append", choices=("32k", "128k", "1M"), default=[])

    prepare = subparsers.add_parser("prepare-personamem", help="Validate and normalize a PersonaMem split")
    prepare.add_argument("--raw-dir", default="data/raw/personamem-v1")
    prepare.add_argument("--processed-dir", default="data/processed/personamem-v1")
    prepare.add_argument("--split", choices=("32k", "128k", "1M"), default="32k")

    configured_data = subparsers.add_parser(
        "prepare-configured-data",
        help="Download, pinned-checksum, and prepare formal PersonaMem 32K data from app config",
    )
    configured_data.add_argument("--config", default="configs/default.yaml")
    configured_data.add_argument("--override-config")
    configured_data.add_argument("--download-missing", action=argparse.BooleanOptionalAction, default=True)

    run = subparsers.add_parser("run", help="Run a PersonaMem retrieval/e2e experiment")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--override-config")
    run.add_argument("--method", choices=METHODS, default="bridgetree")
    run.add_argument("--limit", type=int)
    run.add_argument("--generate", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--bridge-gold", help="Independent JSONL gold memory annotations")
    run.add_argument("--output-dir")
    run.add_argument("--run-label")
    _add_runtime_arguments(run)

    chain_plan = subparsers.add_parser(
        "chain-plan", help="Freeze the full train-free BridgeTree-Chain task plan"
    )
    chain_plan.add_argument("--queries", default="data/processed/personamem-v1/32k/queries.jsonl")
    chain_plan.add_argument("--output", default="outputs/chain/planned_tasks.jsonl")
    chain_plan.add_argument("--dataset-revision", default="personamem-v1-32k")
    chain_plan.add_argument("--config-hash", default="chain_full")
    chain_plan.add_argument("--method", action="append", dest="methods")

    chain_run = subparsers.add_parser(
        "chain-run",
        help="Run or preflight the train-free conditional-activation Chain experiment",
    )
    chain_run.add_argument("--config", default="configs/chain_full.yaml")
    chain_run.add_argument("--override-config")
    chain_run.add_argument("--output-dir")
    chain_run.add_argument("--protocol-manifest")
    chain_mode = chain_run.add_mutually_exclusive_group()
    chain_mode.add_argument("--resume", action="store_true")
    chain_mode.add_argument("--preflight-only", action="store_true")

    sweep = subparsers.add_parser("sweep", help="Run required methods over one or more search budgets")
    sweep.add_argument("--config", default="configs/default.yaml")
    sweep.add_argument("--override-config")
    sweep.add_argument("--method", action="append", choices=METHODS, default=[])
    sweep.add_argument("--budget", action="append", type=int, default=[])
    sweep.add_argument(
        "--budget-protocol",
        choices=("matched_ann_calls", "matched_candidate_exposure"),
        default="matched_candidate_exposure",
    )
    sweep.add_argument("--limit", type=int)
    sweep.add_argument("--generate", action=argparse.BooleanOptionalAction, default=False)
    sweep.add_argument("--bridge-gold")
    sweep.add_argument("--output-dir")
    _add_runtime_arguments(sweep)

    tune = subparsers.add_parser(
        "tune",
        aliases=["train"],
        help="Tune retrieval configuration with external validation outcomes",
    )
    tune.add_argument("--config", default="configs/default.yaml")
    tune.add_argument("--override-config")
    tune.add_argument("--tuning-config", "--training-config", dest="tuning_config", default="configs/train.yaml")
    tune.add_argument("--output-dir")
    tune.add_argument(
        "--audit-full-32k",
        action="store_true",
        help="Require the persisted run to pass the independent formal 32K completion audit",
    )
    tune.add_argument("--max-parse-failure-rate", type=float, default=0.05)
    tune.add_argument(
        "--phase",
        choices=("development", "development-seen", "confirmatory", "confirmatory-test", "full", "full-benchmark"),
    )
    tune.add_argument("--protocol-manifest", "--manifest", dest="protocol_manifest")
    _add_runtime_arguments(tune)

    preflight = subparsers.add_parser(
        "preflight-tuning",
        help="Validate tuning data, protocol, search space, and model services",
    )
    preflight.add_argument("--config", default="configs/default.yaml")
    preflight.add_argument("--override-config")
    preflight.add_argument("--tuning-config", "--training-config", dest="tuning_config", default="configs/train.yaml")
    preflight.add_argument("--check-services", action=argparse.BooleanOptionalAction, default=False)
    preflight.add_argument("--require-full-32k", action="store_true")
    preflight.add_argument(
        "--phase",
        choices=("development", "development-seen", "confirmatory", "confirmatory-test", "full", "full-benchmark"),
    )
    preflight.add_argument("--protocol-manifest", "--manifest", dest="protocol_manifest")
    _add_runtime_arguments(preflight)

    check = subparsers.add_parser("check-ascend", help="Report Ascend/PyTorch runtime availability")
    check.add_argument("--strict", action="store_true")

    synthetic = subparsers.add_parser("smoke-synthetic", help="Run an offline q-to-m1-to-m2 retrieval smoke test")
    synthetic.add_argument("--output-dir", default="outputs/smoke")

    protocol = subparsers.add_parser("protocol", help="Initialize, audit, or freeze the leakage-safe data protocol")
    protocol_sub = protocol.add_subparsers(dest="protocol_command", required=True)
    protocol_init_parser = protocol_sub.add_parser("init", help="Persist confirmatory_v1 role/question IDs")
    protocol_init_parser.add_argument(
        "--output",
        "--output-path",
        dest="output_path",
        default="outputs/protocol/confirmatory_v1/protocol_manifest.json",
    )
    protocol_init_parser.add_argument("--raw-dir", default="data/raw/personamem-v1")
    protocol_init_parser.add_argument("--split", choices=("32k", "128k", "1M"), default="32k")
    protocol_init_parser.add_argument("--seed", type=int, default=42)
    protocol_init_parser.add_argument("--force", action="store_true")
    protocol_init_parser.add_argument(
        "--from-legacy-manifest",
        dest="from_legacy_manifest",
        help="Convert an existing legacy split manifest instead of re-splitting the source",
    )
    protocol_audit_parser = protocol_sub.add_parser("audit", help="Audit a persisted protocol manifest")
    protocol_audit_parser.add_argument("--protocol", default="confirmatory_v1")
    protocol_audit_parser.add_argument(
        "--manifest",
        "--protocol-manifest",
        dest="manifest",
        default="outputs/protocol/confirmatory_v1/protocol_manifest.json",
    )
    protocol_audit_parser.add_argument("--raw-dir", default="data/raw/personamem-v1")
    protocol_audit_parser.add_argument("--split", choices=("32k", "128k", "1M"), default="32k")
    protocol_audit_parser.add_argument("--require-frozen", action="store_true")
    protocol_freeze_parser = protocol_sub.add_parser("freeze", help="Freeze a protocol manifest")
    protocol_freeze_parser.add_argument("--protocol", default="confirmatory_v1")
    protocol_freeze_parser.add_argument(
        "--manifest",
        "--protocol-manifest",
        dest="manifest",
        default="outputs/protocol/confirmatory_v1/protocol_manifest.json",
    )
    protocol_freeze_parser.add_argument("--config-hash")
    protocol_freeze_parser.add_argument("--raw-dir", default="data/raw/personamem-v1")
    protocol_freeze_parser.add_argument("--split", choices=("32k", "128k", "1M"), default="32k")

    # A top-level alias is retained for shell scripts which predate the
    # nested ``protocol freeze`` spelling.
    freeze = subparsers.add_parser("freeze", help="Alias for protocol freeze")
    freeze.add_argument("--protocol", default="confirmatory_v1")
    freeze.add_argument(
        "--manifest",
        "--protocol-manifest",
        dest="manifest",
        default="outputs/protocol/confirmatory_v1/protocol_manifest.json",
    )
    freeze.add_argument("--config-hash")
    freeze.add_argument("--raw-dir", default="data/raw/personamem-v1")
    freeze.add_argument("--split", choices=("32k", "128k", "1M"), default="32k")

    matrix = subparsers.add_parser("run-tmic-matrix", help="Run the fixed A0--A4 TMIC matrix")
    matrix.add_argument("--config", default="configs/default.yaml")
    matrix.add_argument("--override-config")
    matrix.add_argument(
        "--phase",
        choices=("development", "development-seen", "confirmatory", "confirmatory-test", "full", "full-benchmark"),
        default="development",
    )
    matrix.add_argument("--protocol-manifest", "--manifest")
    matrix.add_argument("--limit", type=int)
    matrix.add_argument("--generate", action=argparse.BooleanOptionalAction, default=False)
    matrix.add_argument("--output-dir")
    matrix.add_argument("--include-auxiliary", action="store_true")
    matrix.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the deterministic local embedder (no embedding service calls)",
    )
    _add_runtime_arguments(matrix)

    semantic_matrix = subparsers.add_parser(
        "run-semantic-matrix",
        help="Run the fixed L0/L1/S0/S1/S2/S3/S2-shuffle matrix on shared frozen candidate pools",
    )
    semantic_matrix.add_argument("--config", default="configs/default.yaml")
    semantic_matrix.add_argument("--override-config")
    semantic_matrix.add_argument(
        "--phase",
        choices=("development", "development-seen", "confirmatory", "confirmatory-test", "full", "full-benchmark"),
        default="development",
    )
    semantic_matrix.add_argument("--protocol-manifest", "--manifest")
    semantic_matrix.add_argument("--limit", type=int)
    semantic_matrix.add_argument("--generate", action=argparse.BooleanOptionalAction, default=False)
    semantic_matrix.add_argument("--output-dir")
    semantic_matrix.add_argument(
        "--rows",
        default="S0,S1,S2",
        help="Comma-separated semantic rows (default: S0,S1,S2; S3/shuffle/L0/L1 are opt-in)",
    )
    semantic_matrix.add_argument(
        "--baseline",
        choices=("dense_rerank", "none"),
        default="dense_rerank",
        help="Optional independent baseline candidate source",
    )
    semantic_matrix.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deterministic local embeddings and direct-cosine quality (no model service calls)",
    )
    semantic_matrix.add_argument(
        "--no-reranker",
        action="store_true",
        help="Use the explicit offline/direct quality adapter even when a reranker endpoint is configured",
    )
    _add_runtime_arguments(semantic_matrix)

    semantic_audit = subparsers.add_parser(
        "audit-semantic",
        help="Audit persisted semantic matrix provenance, shared pools, context hashes, and failures",
    )
    semantic_audit.add_argument("--input-dir", required=True)
    semantic_audit.add_argument("--raise-on-error", action="store_true")

    tmic_audit = subparsers.add_parser("audit-tmic", help="Audit persisted TMIC matrix provenance and certificates")
    tmic_audit.add_argument("--input-dir", required=True)
    tmic_audit.add_argument("--raise-on-error", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate multi-seed runs with paired bootstrap CIs")
    aggregate.add_argument("--input-dir", required=True)
    aggregate.add_argument("--reference-label", default="core")
    aggregate.add_argument("--bootstrap-seed", type=int, default=42)
    aggregate.add_argument("--bootstrap-resamples", type=int, default=2000)

    package = subparsers.add_parser(
        "package-server",
        help="Build and verify a portable server bundle containing formal 32K data",
    )
    package.add_argument("--output-dir", default="dist/server")
    package.add_argument(
        "--verify-offline-launcher",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    offline_full = subparsers.add_parser(
        "validate-full-32k-offline",
        help="Run the complete 32K scheduler with explicit fake local clients",
    )
    offline_full.add_argument("--config", default="configs/default.yaml")
    offline_full.add_argument("--tuning-config", default="configs/train.yaml")
    offline_full.add_argument("--output-dir", default="outputs/offline-full-32k-validation")

    effect = subparsers.add_parser(
        "validate-effect-first",
        help="Run the predefined reranker-guided method matrix on validation personas only",
    )
    effect.add_argument("--config", default="configs/default.yaml")
    effect.add_argument("--override-config", default="configs/personamem32k_effect_first.yaml")
    effect.add_argument("--method", action="append", choices=EFFECT_FIRST_VALIDATION_METHODS, default=[])
    effect.add_argument("--output-dir", default="outputs/effect-first-validation")
    effect.add_argument("--limit", type=int)
    effect.add_argument("--generate", action=argparse.BooleanOptionalAction, default=True)
    _add_runtime_arguments(effect)

    audit = subparsers.add_parser(
        "audit-tuning-run",
        help="Independently validate a completed tuning run from its persisted artifacts",
    )
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--require-full-32k", action="store_true")
    audit.add_argument("--max-parse-failure-rate", type=float, default=0.05)
    diagnostic_plan = subparsers.add_parser("diagnostic-plan", help="Freeze an offline, budgeted post-hoc diagnostic plan")
    diagnostic_plan.add_argument("--config", default="configs/diagnostic_28.yaml")
    diagnostic_plan.add_argument("--output-dir", required=True)
    for command in ("diagnostic-analyze", "diagnostic-score", "diagnostic-generate", "diagnostic-roots", "diagnostic-evaluate", "diagnostic-report"):
        entry = subparsers.add_parser(command, help="PR1–PR4 frozen dependency diagnostics (not a benchmark method)")
        entry.add_argument("--run-dir", required=True)
        if command in {"diagnostic-score", "diagnostic-generate", "diagnostic-roots"}:
            entry.add_argument("--config", default="configs/diagnostic_28.yaml")
            entry.add_argument("--execute", action="store_true", help="Actually call only frozen deployments under hard budgets")
            entry.add_argument("--resume", action="store_true", help="Resume the same predeclared trial identities (terminal outcomes are retained)")
        elif command == "diagnostic-analyze":
            entry.add_argument("--score-view", choices=("historical", "fresh"), default="historical")
        elif command == "diagnostic-evaluate":
            entry.add_argument("--gold-source", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command.startswith("diagnostic-"):
        from .diagnostic_runner import analyze_diagnostics, plan_diagnostics, run_generations, run_scores
        from .diagnostic_evaluation import evaluate_diagnostics, unified_diagnostic_report
        if args.command == "diagnostic-plan":
            result = plan_diagnostics(args.config, args.output_dir)
        elif args.command == "diagnostic-analyze":
            result = analyze_diagnostics(args.run_dir, score_view=args.score_view)
        elif args.command == "diagnostic-score":
            result = run_scores(args.run_dir, args.config, execute=args.execute)
        elif args.command == "diagnostic-generate":
            result = run_generations(args.run_dir, args.config, execute=args.execute)
        elif args.command == "diagnostic-roots":
            from .diagnostic_root_runner import run_root_diagnostics
            result = run_root_diagnostics(args.run_dir, args.config, execute=args.execute)
        elif args.command == "diagnostic-evaluate":
            result = evaluate_diagnostics(args.run_dir, args.gold_source)
        else:
            result = unified_diagnostic_report(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "download-personamem":
        download_personamem(args.raw_dir, args.split or ["32k"])
        return 0
    if args.command == "prepare-personamem":
        print(json.dumps(prepare_split(args.raw_dir, args.processed_dir, args.split), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-configured-data":
        config = load_config(args.config, args.override_config)
        split = config.data.split
        if split != "32k":
            raise ValueError("prepare-configured-data is the pinned formal 32K data gate")
        raw_root = Path(config.data.raw_dir)
        source_paths = (
            raw_root / f"questions_{split}.csv",
            raw_root / f"shared_contexts_{split}.jsonl",
        )
        if any(not path.is_file() or path.stat().st_size <= 0 for path in source_paths):
            if not args.download_missing:
                raise FileNotFoundError(f"PersonaMem {split} source is missing and download is disabled")
            download_personamem(raw_root, [split])
        result = prepare_split(
            raw_root,
            config.data.processed_dir,
            split,
            verify_pinned_source=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        config = _resolved_config(args)
        embedder = build_embedder(config.models.embedding, device=config.runtime.device)
        result = run_personamem_experiment(
            config,
            args.method,
            embedder,
            limit=args.limit,
            generate=args.generate,
            bridge_gold_path=args.bridge_gold,
            output_dir=args.output_dir,
            run_label=args.run_label,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "chain-run":
        from .dependency_config import load_dependency_config
        from .dependency_experiment import run_dependency_experiment

        config = load_dependency_config(args.config, args.override_config)
        if args.resume and args.output_dir is None:
            raise ValueError("chain-run --resume requires --output-dir for the original run")
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = (
                Path(config.runtime.output_dir)
                / f"dependency_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
            )
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def interrupt_for_sigterm(signum, _frame):
            raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

        signal.signal(signal.SIGTERM, interrupt_for_sigterm)
        try:
            result = run_dependency_experiment(
                config,
                output_dir,
                resume=args.resume,
                preflight_only=args.preflight_only,
                protocol_manifest=args.protocol_manifest,
            )
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in {
            "completed", "completed_with_failures", "preflight_complete"
        } else 1
    if args.command == "chain-plan":
        query_path = Path(args.queries)
        if not query_path.is_file():
            raise FileNotFoundError(f"query manifest does not exist: {query_path}")
        with query_path.open(encoding="utf-8") as handle:
            queries = [json.loads(line) for line in handle if line.strip()]
        methods = tuple(args.methods) if args.methods else (
            "dense", "dense_rerank", "rfmem", "semantic_s2", "chain_h1_no_closure",
            "chain_h2_no_closure", "chain_full", "chain_no_join", "chain_dense_pool",
        )
        tasks = build_full_plan(
            queries,
            dataset_revision=args.dataset_revision,
            config_hash=args.config_hash,
            methods=methods,
        )
        write_plan(args.output, tasks)
        print(json.dumps({"output": str(args.output), "expected_tasks": len(tasks)}, ensure_ascii=False))
        return 0
    if args.command == "sweep":
        config = _resolved_config(args)
        embedder = build_embedder(config.models.embedding, device=config.runtime.device)
        methods = args.method or list(METHODS)
        if args.budget:
            budgets = args.budget
        elif args.budget_protocol == "matched_ann_calls":
            budgets = [config.retrieval.max_ann_calls or 1]
        else:
            budgets = [config.retrieval.max_candidate_exposure or config.retrieval.search_budget]
        sweep_root = Path(args.output_dir or config.runtime.output_dir) / f"sweep_{time.time_ns()}"
        sweep_root.mkdir(parents=True, exist_ok=False)
        runs = []
        static_methods = {
            "dense",
            "dense_rerank",
            "dense_rerank_20",
            "dense_rerank_28",
            "full_pool_rerank",
            "rfmem_familiarity",
        }
        for method in methods:
            method_budgets = [None] if method in static_methods else budgets
            for budget in method_budgets:
                current = config
                if budget is not None and args.budget_protocol == "matched_ann_calls":
                    if budget <= 0:
                        raise ValueError("matched ANN-call budgets must be positive")
                    current = replace(config, retrieval=replace(config.retrieval, max_ann_calls=budget))
                elif budget is not None:
                    if budget < max(config.retrieval.initial_width, config.retrieval.context_size):
                        raise ValueError("candidate-exposure budgets must cover initial_width and context_size")
                    current = replace(
                        config,
                        retrieval=replace(config.retrieval, max_candidate_exposure=budget),
                    )
                result = run_personamem_experiment(
                    current,
                    method,
                    embedder,
                    limit=args.limit,
                    generate=args.generate,
                    bridge_gold_path=args.bridge_gold,
                    output_dir=sweep_root,
                )
                actual_cost = result["summary"].get("cost", {}).get("mean", {})
                runs.append(
                    {
                        "budget": budget,
                        "budget_protocol": "actual_cost" if budget is None else args.budget_protocol,
                        "actual_cost": actual_cost,
                        **result,
                    }
                )
        manifest = {
            "methods": methods,
            "budget_protocol": args.budget_protocol,
            "budgets": budgets,
            "runs": runs,
        }
        with (sweep_root / "sweep_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({"sweep_dir": str(sweep_root), **manifest}, ensure_ascii=False, indent=2))
        return 0
    if args.command in {"tune", "train"}:
        if args.command == "train":
            print("warning: `bridgetree train` is deprecated; use `bridgetree tune`", file=sys.stderr)
        config = _resolved_config(args)
        training_config = _resolved_tuning_config(args)
        embedder = build_embedder(config.models.embedding, device=config.runtime.device)
        result = run_tuning_experiment(
            config,
            training_config,
            embedder,
            output_dir=args.output_dir,
        )
        if args.audit_full_32k:
            completion_audit = audit_tuning_run(
                result["run_dir"],
                require_full_32k=True,
                max_parse_failure_rate=args.max_parse_failure_rate,
                raise_on_error=True,
            )
            result = {**result, "completion_audit": completion_audit}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight-tuning":
        config = _resolved_config(args)
        training_config = _resolved_tuning_config(args)
        embedder = (
            build_embedder(config.models.embedding, device=config.runtime.device) if args.check_services else None
        )
        result = preflight_tuning(
            config,
            training_config,
            embedder=embedder,
            check_services=args.check_services,
            require_full_32k=args.require_full_32k,
            phase=args.phase,
            protocol_manifest=args.protocol_manifest,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "check-ascend":
        status = {"torch": False, "torch_npu": False, "npu_available": False}
        try:
            import torch

            status["torch"] = True
            try:
                import torch_npu  # noqa: F401

                status["torch_npu"] = True
                status["npu_available"] = bool(torch.npu.is_available())
            except (ImportError, AttributeError):
                pass
        except ImportError:
            pass
        print(json.dumps(status, indent=2))
        return int(args.strict and not status["npu_available"])
    if args.command == "smoke-synthetic":
        print(json.dumps(run_synthetic_smoke(args.output_dir), ensure_ascii=False, indent=2))
        return 0
    if args.command == "protocol":
        if args.protocol_command == "init":
            result = init_protocol(
                output_path=args.output_path,
                raw_dir=args.raw_dir,
                split=args.split,
                seed=args.seed,
                force=args.force,
                from_legacy_manifest=args.from_legacy_manifest,
            )
        elif args.protocol_command == "audit":
            if args.protocol != "confirmatory_v1":
                raise ValueError("only confirmatory_v1 is supported")
            result = audit_protocol(
                args.manifest,
                raw_dir=args.raw_dir,
                split=args.split,
                require_frozen=args.require_frozen,
                raise_on_error=True,
            )
        else:
            if args.protocol != "confirmatory_v1":
                raise ValueError("only confirmatory_v1 is supported")
            result = freeze_protocol(
                args.manifest,
                config_hash=args.config_hash,
                raw_dir=args.raw_dir,
                split=args.split,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "freeze":
        if args.protocol != "confirmatory_v1":
            raise ValueError("only confirmatory_v1 is supported")
        result = freeze_protocol(
            args.manifest,
            config_hash=args.config_hash,
            raw_dir=args.raw_dir,
            split=args.split,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-tmic-matrix":
        config = _resolved_config(args)
        if args.offline:
            from .offline_validation import OfflineDeterministicEmbedder

            embedder = OfflineDeterministicEmbedder()
        else:
            embedder = build_embedder(config.models.embedding, device=config.runtime.device)
        result = run_tmic_matrix(
            config,
            embedder,
            phase=args.phase,
            limit=args.limit,
            generate=args.generate,
            protocol_manifest=args.protocol_manifest,
            output_dir=args.output_dir,
            include_auxiliary=args.include_auxiliary,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-semantic-matrix":
        config = _resolved_config(args)
        if args.offline:
            from .offline_validation import OfflineDeterministicEmbedder

            embedder = OfflineDeterministicEmbedder()
            reranker = None
        else:
            embedder = build_embedder(config.models.embedding, device=config.runtime.device)
            reranker = None if args.no_reranker else RerankerClient(config.models.reranker)
        result = run_semantic_matrix(
            config,
            embedder,
            phase=args.phase,
            limit=args.limit,
            generate=args.generate,
            protocol_manifest=args.protocol_manifest,
            output_dir=args.output_dir,
            reranker=reranker,
            rows=args.rows,
            baseline=args.baseline,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-semantic":
        result = audit_semantic_run(args.input_dir, raise_on_error=args.raise_on_error)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-tmic":
        result = audit_tmic_run(args.input_dir, raise_on_error=args.raise_on_error)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "aggregate":
        result = aggregate_runs(
            args.input_dir,
            reference_label=args.reference_label,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "package-server":
        repository_root = Path(__file__).resolve().parents[2]
        result = build_server_bundle(repository_root, args.output_dir)
        if args.verify_offline_launcher:
            result.update(verify_bundle_offline_launcher(result["archive"], sys.executable))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-full-32k-offline":
        result = validate_full_32k_offline(
            config_path=args.config,
            tuning_config_path=args.tuning_config,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-effect-first":
        config = _resolved_config(args)
        embedder = build_embedder(config.models.embedding, device=config.runtime.device)
        result = run_effect_first_validation(
            config,
            embedder,
            methods=args.method or EFFECT_FIRST_VALIDATION_METHODS,
            output_dir=args.output_dir,
            limit=args.limit,
            generate=args.generate,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-tuning-run":
        result = audit_tuning_run(
            args.run_dir,
            require_full_32k=args.require_full_32k,
            max_parse_failure_rate=args.max_parse_failure_rate,
            raise_on_error=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
