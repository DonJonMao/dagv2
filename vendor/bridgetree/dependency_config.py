"""Strict configuration for conditional-dependency retrieval experiments.

The dependency entry point deliberately has its own small configuration
surface.  The legacy ``retrieval``, ``bridge_rerank`` and ``chain`` sections
describe different algorithms and are rejected when they occur in a
dependency overlay instead of being silently ignored.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .config import AppConfig, load_config
from .evidence_config import EvidenceBridgeConfig, EvidenceSearchConfig, EvidenceSelectionConfig
from .root_tie_diagnostics import validate_root_tie_settings

DEFAULT_DEPENDENCY_METHODS = (
    "dense",
    "dense_rerank",
    "activation",
    "context_marginal",
    "activation_fixed_pool",
)
OPTIONAL_DEPENDENCY_METHODS = (
    "activation_no_pairs",
    "activation_singleton_selection",
    "evidence_bridge",
)
ALL_DEPENDENCY_METHODS = frozenset(DEFAULT_DEPENDENCY_METHODS + OPTIONAL_DEPENDENCY_METHODS)


def _strict_int(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    """Return an exact integer while rejecting booleans and truncation."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric != float(int(numeric)):
            raise ValueError(f"{name} must be an integer")
        result = int(numeric)
    elif isinstance(value, str):
        text = value.strip()
        try:
            numeric = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not text or not math.isfinite(numeric) or numeric != float(int(numeric)):
            raise ValueError(f"{name} must be an integer")
        result = int(numeric)
    else:
        raise ValueError(f"{name} must be an integer")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _strict_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class DependencyConfig:
    """Search/scoring resource limits for one dependency method and question."""

    initial_width: int = 12
    initial_expansion_width: int = 4
    proposal_width: int = 4
    max_ann_calls: int = 36
    max_scored_sets: int = 512
    max_selection_sets: int = 512
    pair_rescue_width: int = 4
    reranker_batch_size: int = 32
    reranker_max_input_tokens: int = 8192
    root_tie_break: str = "legacy_lexical"
    root_tie_seed: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "initial_width",
            "initial_expansion_width",
            "proposal_width",
            "max_ann_calls",
            "max_scored_sets",
            "max_selection_sets",
            "reranker_batch_size",
            "reranker_max_input_tokens",
        ):
            object.__setattr__(self, name, _strict_int(getattr(self, name), f"dependency.{name}", positive=True))
        object.__setattr__(
            self,
            "pair_rescue_width",
            _strict_int(self.pair_rescue_width, "dependency.pair_rescue_width", nonnegative=True),
        )
        seed = self.root_tie_seed
        if seed is not None:
            seed = _strict_int(seed, "dependency.root_tie_seed", nonnegative=True)
        mode, seed = validate_root_tie_settings(self.root_tie_break, seed)
        object.__setattr__(self, "root_tie_break", mode)
        object.__setattr__(self, "root_tie_seed", seed)


@dataclass(frozen=True)
class DependencyExecutionConfig:
    """Whole-run method list and observable progress frequencies."""

    methods: tuple[str, ...] = DEFAULT_DEPENDENCY_METHODS
    log_every_questions: int = 10
    evaluate_every_questions: int = 25
    heartbeat_seconds: float = 30.0
    infrastructure_task_max_attempts: int = 3
    infrastructure_retry_initial_seconds: float = 15.0
    infrastructure_retry_multiplier: float = 4.0
    infrastructure_retry_max_seconds: float = 300.0

    def __post_init__(self) -> None:
        methods_value = self.methods
        if isinstance(methods_value, (str, bytes)) or not isinstance(methods_value, Sequence):
            raise ValueError("execution.methods must be a sequence")
        methods: list[str] = []
        for raw_method in methods_value:
            if not isinstance(raw_method, str) or not raw_method.strip():
                raise ValueError("execution.methods must contain non-empty strings")
            method = raw_method.strip()
            if method not in ALL_DEPENDENCY_METHODS:
                raise ValueError(f"unsupported dependency method: {method}")
            if method in methods:
                raise ValueError(f"duplicate dependency method: {method}")
            methods.append(method)
        if not methods:
            raise ValueError("execution.methods cannot be empty")
        object.__setattr__(self, "methods", tuple(methods))
        object.__setattr__(
            self,
            "log_every_questions",
            _strict_int(self.log_every_questions, "execution.log_every_questions", positive=True),
        )
        object.__setattr__(
            self,
            "evaluate_every_questions",
            _strict_int(self.evaluate_every_questions, "execution.evaluate_every_questions", positive=True),
        )
        object.__setattr__(
            self,
            "heartbeat_seconds",
            _strict_positive_float(self.heartbeat_seconds, "execution.heartbeat_seconds"),
        )
        object.__setattr__(
            self,
            "infrastructure_task_max_attempts",
            _strict_int(
                self.infrastructure_task_max_attempts,
                "execution.infrastructure_task_max_attempts",
                positive=True,
            ),
        )
        initial_retry = _strict_positive_float(
            self.infrastructure_retry_initial_seconds,
            "execution.infrastructure_retry_initial_seconds",
        )
        retry_multiplier = _strict_positive_float(
            self.infrastructure_retry_multiplier,
            "execution.infrastructure_retry_multiplier",
        )
        maximum_retry = _strict_positive_float(
            self.infrastructure_retry_max_seconds,
            "execution.infrastructure_retry_max_seconds",
        )
        if retry_multiplier < 1.0:
            raise ValueError(
                "execution.infrastructure_retry_multiplier must be at least 1"
            )
        if maximum_retry < initial_retry:
            raise ValueError(
                "execution.infrastructure_retry_max_seconds must be at least "
                "execution.infrastructure_retry_initial_seconds"
            )
        object.__setattr__(
            self, "infrastructure_retry_initial_seconds", initial_retry
        )
        object.__setattr__(
            self, "infrastructure_retry_multiplier", retry_multiplier
        )
        object.__setattr__(
            self, "infrastructure_retry_max_seconds", maximum_retry
        )


@dataclass(frozen=True)
class DependencyRunConfig:
    """Resolved deployment plus the configuration used by the new entry point."""

    app: AppConfig
    dependency: DependencyConfig = field(default_factory=DependencyConfig)
    execution: DependencyExecutionConfig = field(default_factory=DependencyExecutionConfig)
    evidence_bridge: EvidenceBridgeConfig = field(default_factory=EvidenceBridgeConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.app, AppConfig):
            raise ValueError("app must be an AppConfig")
        self.app.validate()
        if self.app.models.reranker.score_contract != "pointwise":
            raise ValueError("dependency scoring requires models.reranker.score_contract=pointwise")
        if not isinstance(self.evidence_bridge, EvidenceBridgeConfig):
            raise ValueError("evidence_bridge must be EvidenceBridgeConfig")
        if "evidence_bridge" in self.methods and self.evidence_bridge.gap_ann_calls >= self.dependency.max_ann_calls:
            raise ValueError("gap_ann_calls must leave an ANN budget for initial retrieval and search")

    @property
    def app_config(self) -> AppConfig:
        """Compatibility spelling for callers that make the wrapper explicit."""

        return self.app

    @property
    def seed(self) -> int:
        return self.app.seed

    @property
    def models(self):
        return self.app.models

    @property
    def data(self):
        return self.app.data

    @property
    def runtime(self):
        return self.app.runtime

    @property
    def methods(self) -> tuple[str, ...]:
        return self.execution.methods

    def resolved_dict(self) -> dict[str, Any]:
        """Return only fields that affect the dependency run.

        Legacy retrieval configuration is intentionally absent because it is
        not consumed by this algorithm.  Credentials are also absent from a
        persisted resolved configuration and from its protocol identity.
        """

        models = asdict(self.app.models)
        models.get("generator", {}).pop("api_key", None)
        dependency = asdict(self.dependency)
        # Adding an opt-in diagnostic must not change existing run/config
        # identities when it is disabled. Enabled settings remain explicit.
        if self.dependency.root_tie_break == "legacy_lexical":
            dependency.pop("root_tie_break")
            dependency.pop("root_tie_seed")
        result = {
            "seed": self.app.seed,
            "models": models,
            "data": asdict(self.app.data),
            "runtime": asdict(self.app.runtime),
            "dependency": dependency,
            "execution": asdict(self.execution),
        }
        if "evidence_bridge" in self.methods:
            result["evidence_bridge"] = asdict(self.evidence_bridge)
            result["evidence_method_version"] = "evidence_bridge_v1"
        return result

    def config_hash(self) -> str:
        payload = json.dumps(self.resolved_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Friendly aliases used by callers that prefer the resource/run distinction in
# the type name.  The canonical public names above remain concise.
DependencyResourceConfig = DependencyConfig
ExecutionConfig = DependencyExecutionConfig


_TOP_LEVEL_FIELDS = {"base_config", "seed", "models", "data", "runtime", "dependency", "execution", "methods",
                     "evidence_bridge"}
_LEGACY_OVERLAY_FIELDS = {"chain", "retrieval", "bridge_rerank"}
_EXECUTION_FIELDS = set(DependencyExecutionConfig.__dataclass_fields__)
_RUNTIME_FIELDS = {"device", "cache_dir", "output_dir"}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        raise
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration: {path}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return {str(key): value for key, value in loaded.items()}


def _replace_section(current: Any, raw: Any, name: str) -> Any:
    values = _strict_mapping(raw, name)
    allowed = set(current.__dataclass_fields__)
    _reject_unknown(values, allowed, name)
    try:
        return replace(current, **values)
    except TypeError as exc:
        raise ValueError(f"invalid {name} configuration") from exc


def _apply_overlay(base: DependencyRunConfig, raw_overlay: Mapping[str, Any], *, source: Path) -> DependencyRunConfig:
    raw = {str(key): value for key, value in raw_overlay.items()}
    forbidden = sorted(set(raw) & _LEGACY_OVERLAY_FIELDS)
    if forbidden:
        raise ValueError(
            "dependency overlay cannot set legacy section(s): " + ", ".join(forbidden)
        )
    _reject_unknown(raw, _TOP_LEVEL_FIELDS - {"base_config"}, "top-level")

    app = base.app
    if "seed" in raw:
        app = replace(app, seed=_strict_int(raw["seed"], "seed", nonnegative=True))

    if "models" in raw:
        model_values = _strict_mapping(raw["models"], "models")
        _reject_unknown(model_values, {"embedding", "reranker", "generator"}, "models")
        models = app.models
        for name, section in model_values.items():
            models = replace(models, **{name: _replace_section(getattr(models, name), section, f"models.{name}")})
        app = replace(app, models=models)

    if "data" in raw:
        app = replace(app, data=_replace_section(app.data, raw["data"], "data"))

    execution_updates: dict[str, Any] = {}
    if "runtime" in raw:
        runtime_values = _strict_mapping(raw["runtime"], "runtime")
        _reject_unknown(runtime_values, _RUNTIME_FIELDS | (_EXECUTION_FIELDS - {"methods"}), "runtime")
        app_values = {key: value for key, value in runtime_values.items() if key in _RUNTIME_FIELDS}
        if app_values:
            app = replace(app, runtime=_replace_section(app.runtime, app_values, "runtime"))
        execution_updates.update(
            {key: value for key, value in runtime_values.items() if key in _EXECUTION_FIELDS}
        )

    dependency = base.dependency
    if "dependency" in raw:
        values = _strict_mapping(raw["dependency"], "dependency")
        _reject_unknown(values, set(DependencyConfig.__dataclass_fields__), "dependency")
        dependency = replace(base.dependency, **values)

    if "execution" in raw:
        values = _strict_mapping(raw["execution"], "execution")
        _reject_unknown(values, _EXECUTION_FIELDS, "execution")
        execution_updates.update(values)
    if "methods" in raw:
        if "methods" in execution_updates:
            raise ValueError("methods cannot be set both top-level and in execution")
        execution_updates["methods"] = raw["methods"]
    execution = replace(base.execution, **execution_updates) if execution_updates else base.execution
    evidence_bridge = base.evidence_bridge
    if "evidence_bridge" in raw:
        values = _strict_mapping(raw["evidence_bridge"], "evidence_bridge")
        _reject_unknown(values, set(EvidenceBridgeConfig.__dataclass_fields__), "evidence_bridge")
        for name, cls in (("search", EvidenceSearchConfig), ("selection", EvidenceSelectionConfig)):
            if name in values:
                nested = _strict_mapping(values[name], f"evidence_bridge.{name}")
                _reject_unknown(nested, set(cls.__dataclass_fields__), f"evidence_bridge.{name}")
                values[name] = replace(getattr(evidence_bridge, name), **nested)
        evidence_bridge = replace(evidence_bridge, **values)
    result = DependencyRunConfig(app=app, dependency=dependency, execution=execution,
                                 evidence_bridge=evidence_bridge)
    return result


def _default_app_path(path: Path) -> Path:
    candidate = path.resolve().parent / "default.yaml"
    if not candidate.is_file():
        raise ValueError(
            f"dependency overlay {path} has no base_config and no sibling default.yaml"
        )
    return candidate


def _load_dependency_path(path: Path, seen: tuple[Path, ...]) -> DependencyRunConfig:
    resolved = path.expanduser().resolve()
    if resolved in seen:
        chain = " -> ".join(str(item) for item in (*seen, resolved))
        raise ValueError(f"cyclic base_config chain: {chain}")
    raw = _read_yaml(resolved)

    if "base_config" in raw:
        base_value = raw.pop("base_config")
        if not isinstance(base_value, str) or not base_value.strip():
            raise ValueError("base_config must be a non-empty path string")
        base_path = Path(base_value).expanduser()
        if not base_path.is_absolute():
            base_path = resolved.parent / base_path
        # A dependency overlay may inherit another dependency overlay or a
        # plain deployment AppConfig.
        base_raw = _read_yaml(base_path.resolve())
        if set(base_raw) & ({"dependency", "execution", "methods", "base_config", "chain"}):
            base = _load_dependency_path(base_path, (*seen, resolved))
        else:
            base = DependencyRunConfig(load_config(base_path))
        return _apply_overlay(base, raw, source=resolved)

    # A regular application configuration is a valid deployment base when
    # loaded directly.  Its legacy sections are not an overlay and are simply
    # omitted from the dependency protocol identity.
    looks_like_app_config = "models" in raw and ("retrieval" in raw or "bridge_rerank" in raw)
    if looks_like_app_config:
        if "chain" in raw:
            raise ValueError("dependency configuration cannot contain the legacy chain section")
        return DependencyRunConfig(load_config(resolved))

    base = DependencyRunConfig(load_config(_default_app_path(resolved)))
    return _apply_overlay(base, raw, source=resolved)


def load_dependency_config(
    path: str | Path,
    overlay_path: str | Path | None = None,
) -> DependencyRunConfig:
    """Load a dependency run without accepting ineffective legacy knobs.

    ``path`` may be a dependency YAML with ``base_config``, a plain deployment
    ``AppConfig`` file, or (when ``overlay_path`` is supplied) the deployment
    base.  Relative ``base_config`` paths are resolved beside the declaring
    file.
    """

    if overlay_path is None:
        return _load_dependency_path(Path(path), ())
    # Command-line overlays may be applied to either a plain deployment
    # config or an already resolved dependency config such as chain_full.yaml.
    # In both cases preserve all inherited dependency/execution values and
    # apply only the explicitly supplied fields.
    base = _load_dependency_path(Path(path), ())
    overlay = _read_yaml(Path(overlay_path).expanduser().resolve())
    if "base_config" in overlay:
        raise ValueError("overlay_path cannot also declare base_config")
    return _apply_overlay(base, overlay, source=Path(overlay_path))


def parse_dependency_config(
    raw: Mapping[str, Any],
    *,
    app: AppConfig,
) -> DependencyRunConfig:
    """Parse an already-loaded overlay against an explicit deployment."""

    if not isinstance(raw, Mapping):
        raise ValueError("dependency configuration must be a mapping")
    values = dict(raw)
    if "base_config" in values:
        raise ValueError("parse_dependency_config does not resolve base_config")
    return _apply_overlay(DependencyRunConfig(app), values, source=Path("<mapping>"))


__all__ = [
    "ALL_DEPENDENCY_METHODS",
    "DEFAULT_DEPENDENCY_METHODS",
    "OPTIONAL_DEPENDENCY_METHODS",
    "DependencyConfig",
    "DependencyExecutionConfig",
    "DependencyResourceConfig",
    "DependencyRunConfig",
    "ExecutionConfig",
    "load_dependency_config",
    "parse_dependency_config",
]
