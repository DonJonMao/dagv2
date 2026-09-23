"""BridgeTree Preference-RAG reference implementation."""

from .chain_experiment import ChainTask, Outcome, build_full_plan, summarize_outcomes
from .chain_judge import Claim, JointScore, PublicQuery, Verification
from .chain_search import ChainSearcher, EvidenceState, SearchArchive, Terminal, choose_terminal
from .chain_support import ClosureResult, close_support
from .clients import ContextPlanError
from .config import AppConfig, RetrievalConfig, load_config
from .information import (
    InformationObjective,
    PureRerankSelector,
    SemanticFeatureProvider,
    SemanticPathLogDetSelector,
    StateBasisProvider,
)
from .measure import (
    BranchMeasure,
    FrozenGraphMeasure,
    angular_navigation_affinity,
    branch_measure,
    parent_posterior,
    propagate_frozen_graph,
    propagate_mass,
    shuffle_frozen_graph,
)
from .retriever import BridgeTreeRetriever
from .semantic import (
    build_query_conditioned_representation_text,
    discover_frozen_graph,
    legacy_trace_from_result,
    rho_squared_quality_records,
    semantic_retrieve,
)
from .temporal import TimeMark, build_transition_matrix
from .types import (
    ContextPlan,
    FrozenProposalGraph,
    InformationAtom,
    Memory,
    PathHypothesis,
    QualityRecord,
    RetrievalResult,
    SemanticAtom,
    TemporalMark,
    context_plan_hash,
)

__all__ = [
    "AppConfig",
    "BridgeTreeRetriever",
    "Memory",
    "RetrievalConfig",
    "RetrievalResult",
    "PathHypothesis",
    "InformationAtom",
    "TimeMark",
    "build_transition_matrix",
    "BranchMeasure",
    "branch_measure",
    "propagate_mass",
    "parent_posterior",
    "StateBasisProvider",
    "InformationObjective",
    "TemporalMark",
    "QualityRecord",
    "FrozenProposalGraph",
    "SemanticAtom",
    "ContextPlan",
    "ContextPlanError",
    "PublicQuery",
    "JointScore",
    "Claim",
    "Verification",
    "EvidenceState",
    "Terminal",
    "SearchArchive",
    "ChainSearcher",
    "ClosureResult",
    "close_support",
    "choose_terminal",
    "ChainTask",
    "Outcome",
    "build_full_plan",
    "summarize_outcomes",
    "context_plan_hash",
    "FrozenGraphMeasure",
    "angular_navigation_affinity",
    "propagate_frozen_graph",
    "shuffle_frozen_graph",
    "SemanticFeatureProvider",
    "SemanticPathLogDetSelector",
    "PureRerankSelector",
    "discover_frozen_graph",
    "semantic_retrieve",
    "rho_squared_quality_records",
    "legacy_trace_from_result",
    "build_query_conditioned_representation_text",
    "load_config",
]
