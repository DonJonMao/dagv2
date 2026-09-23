"""Leakage-safe confirmatory/development data protocol.

``confirmatory_v1`` deliberately reverses the historical tuning roles: the
old validation+test personas are development-seen, while the old train
personas are the one-shot confirmatory-test set.  The first persisted manifest
is authoritative; later commands refuse silent re-splitting or configuration
changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .personamem import PERSONAMEM_REVISION, PersonaMemExample, iter_examples

PROTOCOL_NAME = "confirmatory_v1"
PROTOCOL_VERSION = 1
ROLE_DEVELOPMENT = "development-seen"
ROLE_CONFIRMATORY = "confirmatory-test"
ROLE_FULL = "full-benchmark"
ROLE_NAMES = (ROLE_DEVELOPMENT, ROLE_CONFIRMATORY, ROLE_FULL)


def _strict_protocol_int(value: Any, name: str, *, nonnegative: bool = False) -> int:
    """Parse an integer metadata field without accepting bool/fractional values."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    try:
        if str(value).strip() != str(result):
            raise ValueError(f"{name} must be an integer")
    except AttributeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result

_PHASE_ALIASES = {
    "development": ROLE_DEVELOPMENT,
    ROLE_DEVELOPMENT: ROLE_DEVELOPMENT,
    "confirmatory": ROLE_CONFIRMATORY,
    ROLE_CONFIRMATORY: ROLE_CONFIRMATORY,
    "full": ROLE_FULL,
    ROLE_FULL: ROLE_FULL,
}


def canonical_phase(phase: str) -> str:
    """Return the persisted role name for a user-facing phase alias."""
    try:
        return _PHASE_ALIASES[str(phase)]
    except KeyError as exc:
        raise ValueError(f"unknown protocol phase: {phase}") from exc


def stable_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def persona_hash(values: Sequence[str]) -> str:
    return stable_hash(sorted(set(str(value) for value in values)))


@dataclass(frozen=True)
class ProtocolRole:
    name: str
    personas: tuple[str, ...]
    question_ids: tuple[str, ...]
    question_hash: str
    persona_hash: str
    queries: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "personas": list(self.personas),
            "personas_count": len(self.personas),
            "persona_hash": self.persona_hash,
            "question_ids": list(self.question_ids),
            "question_hash": self.question_hash,
            "question_id_sha256": self.question_hash,
            "queries": self.queries,
        }


@dataclass(frozen=True)
class ProtocolManifest:
    protocol: str
    version: int
    seed: int
    dataset_revision: str
    split: str
    roles: Mapping[str, ProtocolRole]
    frozen: bool = False
    config_hash: str | None = None
    created_at: float = 0.0
    source_revision: str = ""
    question_cutoff: Any = None
    memory_policy: Mapping[str, Any] = field(default_factory=dict)
    source_sha256: Mapping[str, str] = field(default_factory=dict)
    worktree_clean: bool | None = None
    uncommitted_diff_hash: str | None = None
    source_package_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        role_dict = {name: role.to_dict() for name, role in self.roles.items()}
        return {
            "protocol": self.protocol,
            "protocol_name": self.protocol,
            "version": self.version,
        "seed": self.seed,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "roles": role_dict,
            # Flat aliases make the manifest convenient for shell/audit tools.
            "development_seen": role_dict.get(ROLE_DEVELOPMENT),
            "confirmatory_test": role_dict.get(ROLE_CONFIRMATORY),
            "full_benchmark": role_dict.get(ROLE_FULL),
            "frozen": self.frozen,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "source_revision": self.source_revision or self.dataset_revision,
            "question_cutoff": self.question_cutoff,
            "memory_policy": dict(self.memory_policy),
            "source_sha256": dict(self.source_sha256),
            "worktree_clean": self.worktree_clean,
            "uncommitted_diff_hash": self.uncommitted_diff_hash,
            "source_package_hash": self.source_package_hash,
        }


def _load_all_examples(
    examples: Sequence[PersonaMemExample] | None = None,
    *,
    raw_dir: str | Path = "data/raw/personamem-v1",
    split: str = "32k",
) -> list[PersonaMemExample]:
    if examples is not None:
        return list(examples)
    root = Path(raw_dir)
    return list(iter_examples(root / f"questions_{split}.csv", root / f"shared_contexts_{split}.jsonl"))


def _legacy_split(examples: Sequence[PersonaMemExample], seed: int):
    # Import lazily to avoid protocol <-> training import cycles.
    from .training import SplitProtocol, split_examples_by_persona

    return split_examples_by_persona(examples, SplitProtocol(), seed)


def build_protocol_manifest(
    examples: Sequence[PersonaMemExample],
    *,
    seed: int = 42,
    split: str = "32k",
    dataset_revision: str = PERSONAMEM_REVISION,
    question_cutoff: Any = None,
    memory_policy: Mapping[str, Any] | None = None,
    source_sha256: Mapping[str, str] | None = None,
    worktree_clean: bool | None = None,
    uncommitted_diff_hash: str | None = None,
    source_package_hash: str | None = None,
) -> ProtocolManifest:
    if not examples:
        raise ValueError("cannot initialize a protocol from an empty example set")
    if isinstance(seed, bool):
        raise ValueError("protocol seed must be an integer")
    try:
        numeric_seed = int(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("protocol seed must be an integer") from exc
    if str(seed).strip() != str(numeric_seed):
        raise ValueError("protocol seed must be an integer")
    seed = numeric_seed
    if source_package_hash is None:
        source_package_hash = _source_package_snapshot()
    splits = _legacy_split(examples, seed)
    development = tuple(
        sorted(
            tuple(splits.validation) + tuple(splits.test),
            key=lambda item: (str(item.question_id), str(item.persona_id)),
        )
    )
    confirmatory = tuple(sorted(splits.train, key=lambda item: (str(item.question_id), str(item.persona_id))))
    full = tuple(sorted(examples, key=lambda item: (str(item.question_id), str(item.persona_id))))

    def role(name: str, items: Sequence[PersonaMemExample]) -> ProtocolRole:
        qids = tuple(str(item.question_id) for item in items)
        personas = tuple(sorted({str(item.persona_id) for item in items}))
        return ProtocolRole(name, personas, qids, stable_hash(qids), persona_hash(personas), len(qids))

    return ProtocolManifest(
        protocol=PROTOCOL_NAME,
        version=PROTOCOL_VERSION,
        seed=seed,
        dataset_revision=str(dataset_revision),
        split=str(split),
        roles={
            ROLE_DEVELOPMENT: role(ROLE_DEVELOPMENT, development),
            ROLE_CONFIRMATORY: role(ROLE_CONFIRMATORY, confirmatory),
            ROLE_FULL: role(ROLE_FULL, full),
        },
        created_at=time.time(),
        source_revision=str(dataset_revision),
        question_cutoff=question_cutoff,
        memory_policy=dict(memory_policy or {}),
        source_sha256=dict(source_sha256 or {}),
        worktree_clean=worktree_clean,
        uncommitted_diff_hash=uncommitted_diff_hash,
        source_package_hash=source_package_hash,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_snapshot(raw_dir: str | Path, split: str) -> tuple[dict[str, str], str | None]:
    """Hash the exact source files used by a protocol manifest."""
    root = Path(raw_dir)
    hashes: dict[str, str] = {}
    for name in (f"questions_{split}.csv", f"shared_contexts_{split}.jsonl"):
        path = root / name
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = digest
    if not hashes:
        return hashes, None
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashes, hashlib.sha256(payload).hexdigest()


def _source_identity_path_is_excluded(relative: str, path: Path) -> bool:
    """Exclude generated state and private runtime credentials from code identity."""

    if any(
        part in {
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".cache",
            ".git",
        }
        or part.endswith(".egg-info")
        for part in path.parts
    ):
        return True
    name = path.name.lower()
    if name == ".ds_store" or name == ".env" or name.startswith(".env."):
        return True
    if relative == "configs/credentials.local.yaml":
        return True
    if "credential" in name and path.suffix.lower() in {".yaml", ".yml", ".json"}:
        return True
    return path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}


def _source_package_snapshot() -> str | None:
    """Hash the runnable source package, including untracked source files.

    A data-file checksum is not a code identity.  Confirmatory manifests need
    to record the implementation that produced them, and ``git diff`` alone
    misses newly created (untracked) modules.  Prefer Git's tracked plus
    untracked file listing, then fall back to a deterministic filesystem walk
    when the package is copied without a Git checkout.
    """
    root = Path(__file__).resolve().parents[2]
    include_roots = ("src/bridgetree", "reference", "configs", "scripts", "pyproject.toml")
    paths: list[Path] = []
    try:
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "--"] + list(include_roots),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            paths = [root / line for line in result.stdout.splitlines() if line.strip()]
    except OSError:
        paths = []
    if not paths:
        for relative in include_roots:
            candidate = root / relative
            if candidate.is_file():
                paths.append(candidate)
            elif candidate.is_dir():
                paths.extend(path for path in candidate.rglob("*") if path.is_file())
    normalized: list[tuple[str, Path]] = []
    for path in paths:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if _source_identity_path_is_excluded(relative, path):
            continue
        if path.is_file():
            normalized.append((relative, path))
    if not normalized:
        return None
    digest = hashlib.sha256()
    for relative, path in sorted(normalized):
        try:
            content = path.read_bytes()
        except OSError:
            return None
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _worktree_snapshot() -> tuple[bool | None, str | None]:
    """Return clean-state and uncommitted diff identity without credentials."""
    try:
        root = Path(__file__).resolve().parents[2]
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=root,
            check=False,
            capture_output=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--no-ext-diff"],
            cwd=root,
            check=False,
            capture_output=True,
        ).stdout
        # ``git diff`` does not contain untracked files.  Include each
        # untracked path and its bytes in the identity while keeping the
        # status boolean sensitive to every worktree change.
        status_text = status.decode("utf-8", errors="surrogateescape")
        digest = hashlib.sha256()
        digest.update(diff)
        entries = [entry for entry in status_text.split("\0") if entry]
        for entry in sorted(entries):
            digest.update(entry.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            # Porcelain v1 uses the path after the two-character status code.
            raw_path = entry[3:] if len(entry) >= 3 else ""
            if raw_path and " -> " in raw_path:
                raw_path = raw_path.split(" -> ", 1)[1]
            candidate = root / raw_path
            if candidate.is_file():
                with suppress(OSError):
                    digest.update(candidate.read_bytes())
            digest.update(b"\0")
        return not bool(status.strip()), digest.hexdigest()
    except OSError:
        return None, None


def _legacy_manifest_roles(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extract legacy train/validation/test (or modern role) records."""
    roles = value.get("roles") if isinstance(value.get("roles"), Mapping) else {}

    def pick(*names: str) -> Mapping[str, Any]:
        for name in names:
            item = roles.get(name, value.get(name))
            if isinstance(item, Mapping):
                return item
        return {}

    development = pick(ROLE_DEVELOPMENT, "development_seen", "development")
    confirmatory = pick(ROLE_CONFIRMATORY, "confirmatory_test", "confirmatory")
    full = pick(ROLE_FULL, "full_benchmark", "full")
    if not development:
        validation = pick("validation", "valid")
        test = pick("test")
        validation_ids = validation.get("question_ids", validation.get("ids", ()))
        test_ids = test.get("question_ids", test.get("ids", ()))
        development = {
            "question_ids": list(validation_ids or ()) + list(test_ids or ()),
            "personas": sorted(
                {str(item) for item in (validation.get("personas", ()) or ())}
                | {str(item) for item in (test.get("personas", ()) or ())}
            ),
        }
    if not confirmatory:
        train = pick("train", "training")
        confirmatory = {
            "question_ids": list(train.get("question_ids", train.get("ids", ())) or ()),
            "personas": list(train.get("personas", ()) or ()),
        }
    if not full:
        full = {
            "question_ids": list(development.get("question_ids", ())) + list(confirmatory.get("question_ids", ())),
            "personas": sorted(
                {str(item) for item in (development.get("personas", ()) or ())}
                | {str(item) for item in (confirmatory.get("personas", ()) or ())}
            ),
        }
    return dict(development), dict(confirmatory), dict(full)


def _build_manifest_from_legacy(
    legacy_path: str | Path,
    *,
    examples: Sequence[PersonaMemExample] | None,
    raw_dir: str | Path,
    split: str,
    seed: int,
) -> dict[str, Any]:
    path = Path(legacy_path)
    if not path.is_file():
        raise FileNotFoundError(f"legacy manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("legacy manifest must be a JSON object")
    raw_roles = _legacy_manifest_roles(value)
    loaded = _load_all_examples(examples, raw_dir=raw_dir, split=split)
    by_id = {str(item.question_id): item for item in loaded}

    # The historical split manifest persisted persona lists, counts and a
    # question-ID hash, but often omitted the IDs themselves.  Reconstructing
    # from the current source is safe only when the persisted persona/count/
    # hash triple identifies exactly one ordered sequence.  Never emit an
    # empty role (or silently deduplicate stale IDs) when that cannot be
    # established.
    def make_role(name: str, raw: Mapping[str, Any], phase_hint: str) -> dict[str, Any]:
        raw_ids_value = raw.get("question_ids", raw.get("ids", ()))
        if isinstance(raw_ids_value, (str, bytes)):
            raise ValueError(f"legacy {name} question_ids must be a sequence")
        if raw_ids_value is not None and not isinstance(raw_ids_value, Sequence):
            raise ValueError(f"legacy {name} question_ids must be a sequence")
        explicit_ids = [str(item) for item in (raw_ids_value or ())]
        if len(explicit_ids) != len(set(explicit_ids)):
            raise ValueError(f"legacy {name} question IDs contain duplicates")
        personas = [str(item) for item in raw.get("personas", ()) or ()]
        if len(personas) != len(set(personas)):
            raise ValueError(f"legacy {name} personas contain duplicates")
        expected_count = raw.get("queries", raw.get("question_count"))
        expected_hash = raw.get("question_id_sha256", raw.get("question_hash"))

        if explicit_ids:
            ids = explicit_ids
        else:
            if not personas:
                raise ValueError(
                    f"legacy {name} omits question_ids and personas; cannot reconstruct its question set"
                )
            matching = [item for item in loaded if str(item.persona_id) in set(personas)]
            if expected_count is not None:
                try:
                    if isinstance(expected_count, bool) or int(expected_count) != len(matching):
                        raise ValueError(
                            f"legacy {name} query count does not match current examples for persisted personas"
                        )
                except (TypeError, ValueError) as exc:
                    if isinstance(exc, ValueError) and str(exc).startswith("legacy "):
                        raise
                    raise ValueError(f"legacy {name} query count is invalid") from exc
            if not matching:
                raise ValueError(f"legacy {name} has no current examples for persisted personas")
            # Try every order that has been used by the legacy split code.
            # A persisted hash makes the choice unambiguous.  If a legacy
            # manifest omitted the combined role's hash, prefer the historical
            # split order (when available) instead of silently imposing a new
            # source/lexicographic ordering.
            candidate_orders: list[list[str]] = [
                [str(item.question_id) for item in matching],
                [str(item.question_id) for item in sorted(matching, key=lambda item: str(item.question_id))],
            ]
            historical_order: list[str] | None = None
            try:
                legacy_splits = _legacy_split(loaded, int(value.get("seed", seed)))
                phase_items = {
                    "development": tuple(legacy_splits.validation) + tuple(legacy_splits.test),
                    "confirmatory": tuple(legacy_splits.train),
                    "full": tuple(loaded),
                }.get(phase_hint)
                if phase_items is not None:
                    historical_order = [
                        str(item.question_id) for item in phase_items if str(item.persona_id) in set(personas)
                    ]
                    candidate_orders.append(historical_order)
            except (TypeError, ValueError):
                pass
            matching_orders = []
            for order in candidate_orders:
                if len(order) != len(set(order)) or len(order) != len(matching):
                    continue
                if expected_hash is None or str(expected_hash) == stable_hash(order):
                    matching_orders.append(order)
            # Remove duplicate candidate orders while preserving deterministic
            # order.  If a hash was supplied, require exactly one match; if it
            # was absent, use the historical phase order, falling back to
            # source order only for a manifest that carries no split ordering.
            unique_orders = list(dict.fromkeys(tuple(order) for order in matching_orders))
            if expected_hash is not None:
                if len(unique_orders) != 1:
                    raise ValueError(
                        f"legacy {name} question hash cannot uniquely reconstruct its ordered question IDs"
                    )
                ids = list(unique_orders[0])
            else:
                # The phase order is appended after source and sorted above.
                # ``full`` intentionally resolves to source order, while the
                # development/confirmatory legacy roles retain their original
                # split scheduler order.
                ids = list(historical_order or (unique_orders[-1] if unique_orders else candidate_orders[0]))

        unknown_ids = [identifier for identifier in ids if identifier not in by_id]
        if unknown_ids:
            raise ValueError(f"legacy {name} references questions absent from current source: {unknown_ids[:5]}")
        if expected_count is not None:
            try:
                if isinstance(expected_count, bool) or int(expected_count) != len(ids):
                    raise ValueError(f"legacy {name} persisted query count mismatch")
            except (TypeError, ValueError) as exc:
                if isinstance(exc, ValueError) and str(exc).startswith("legacy "):
                    raise
                raise ValueError(f"legacy {name} query count is invalid") from exc
        if expected_hash is not None and str(expected_hash) != stable_hash(ids):
            raise ValueError(f"legacy {name} question hash mismatch")
        inferred_personas = sorted({str(by_id[item].persona_id) for item in ids})
        if personas and set(personas) != set(inferred_personas):
            raise ValueError(f"legacy {name} persona list does not match its question IDs")
        personas = personas or inferred_personas
        return {
            "name": name,
            "personas": sorted(set(personas)),
            "personas_count": len(set(personas)),
            "persona_hash": persona_hash(sorted(set(personas))),
            "question_ids": ids,
            "question_hash": stable_hash(ids),
            "question_id_sha256": stable_hash(ids),
            "queries": len(ids),
        }

    development, confirmatory, full = raw_roles
    source_hashes, _data_package_hash = _source_snapshot(raw_dir, split)
    source_package_hash = _source_package_snapshot()
    clean, diff_hash = _worktree_snapshot()
    manifest = {
        "protocol": PROTOCOL_NAME,
        "protocol_name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "seed": int(value.get("seed", seed)),
        "dataset_revision": str(value.get("dataset_revision", value.get("data_revision", PERSONAMEM_REVISION))),
        "split": str(value.get("split", split)),
        "roles": {
            ROLE_DEVELOPMENT: make_role(ROLE_DEVELOPMENT, development, "development"),
            ROLE_CONFIRMATORY: make_role(ROLE_CONFIRMATORY, confirmatory, "confirmatory"),
            ROLE_FULL: make_role(ROLE_FULL, full, "full"),
        },
        "frozen": False,
        "config_hash": None,
        "created_at": time.time(),
        "source_revision": str(value.get("dataset_revision", value.get("data_revision", PERSONAMEM_REVISION))),
        "question_cutoff": value.get("question_cutoff", value.get("cutoff")),
        "memory_policy": value.get("memory_policy", value.get("data_policy", {})) or {},
        "source_sha256": source_hashes,
        "source_package_hash": source_package_hash,
        "worktree_clean": clean,
        "uncommitted_diff_hash": diff_hash,
        "initialized_from": str(path),
        "initialized_from_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_confirmation_required": True,
    }
    for role_name, alias in (
        (ROLE_DEVELOPMENT, "development_seen"),
        (ROLE_CONFIRMATORY, "confirmatory_test"),
        (ROLE_FULL, "full_benchmark"),
    ):
        manifest[alias] = manifest["roles"][role_name]
    return manifest


def init_protocol(
    *,
    output_path: str | Path = "outputs/protocol/confirmatory_v1/protocol_manifest.json",
    examples: Sequence[PersonaMemExample] | None = None,
    raw_dir: str | Path = "data/raw/personamem-v1",
    split: str = "32k",
    seed: int = 42,
    force: bool = False,
    from_legacy_manifest: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol") != PROTOCOL_NAME:
            raise ValueError("existing protocol manifest has a different protocol name")
        # Never silently accept a stale/corrupt persisted split.  The source
        # is cheap to re-read and this check is what makes the first manifest
        # authoritative across later invocations.
        audit_protocol(existing, examples=examples, raw_dir=raw_dir, split=split, raise_on_error=True)
        return existing
    if path.exists() and force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("frozen", False):
            raise PermissionError("a frozen protocol manifest cannot be rewritten")
    if from_legacy_manifest is not None:
        value = _build_manifest_from_legacy(
            from_legacy_manifest,
            examples=examples,
            raw_dir=raw_dir,
            split=split,
            seed=seed,
        )
    else:
        source_hashes, _data_package_hash = _source_snapshot(raw_dir, split)
        source_package_hash = _source_package_snapshot()
        clean, diff_hash = _worktree_snapshot()
        manifest = build_protocol_manifest(
            _load_all_examples(examples, raw_dir=raw_dir, split=split),
            seed=seed,
            split=split,
            source_sha256=source_hashes,
            source_package_hash=source_package_hash,
            worktree_clean=clean,
            uncommitted_diff_hash=diff_hash,
            memory_policy={"memory_granularity": "user_assistant_pair", "include_system_persona": True},
        )
        value = manifest.to_dict()
        value["initialized_from"] = "legacy_persona_split"
    value["source_confirmation_required"] = True
    _atomic_json(path, value)
    return value


def load_protocol_manifest(path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    if not destination.is_file():
        raise FileNotFoundError(f"protocol manifest is missing: {destination}")
    value = json.loads(destination.read_text(encoding="utf-8"))
    if value.get("protocol", value.get("protocol_name")) != PROTOCOL_NAME:
        raise ValueError(f"expected protocol {PROTOCOL_NAME}")
    return value


def _role_value(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    roles = manifest.get("roles", {})
    if isinstance(roles, Mapping) and name in roles:
        value = roles[name]
        return value if isinstance(value, Mapping) else {}
    aliases = {
        ROLE_DEVELOPMENT: "development_seen",
        ROLE_CONFIRMATORY: "confirmatory_test",
        ROLE_FULL: "full_benchmark",
    }
    value = manifest.get(aliases[name], {})
    if not isinstance(value, Mapping):
        return {}
    return value


def _sequence_field(value: Any) -> tuple[Any, ...] | None:
    """Return a manifest list/tuple without accepting scalar strings."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    return tuple(value)


def _manifest_structure_errors(
    manifest: Mapping[str, Any],
    *,
    expected_split: str | None = None,
) -> list[str]:
    """Validate the non-source-dependent protocol manifest structure.

    This check is intentionally shared by ``audit_protocol`` and
    ``protocol_gate``.  A gate must not accept a manifest which merely has the
    right protocol/version/hash fields while omitting the role/question
    provenance that makes the split reproducible.
    """
    errors: list[str] = []
    protocol = manifest.get("protocol")
    protocol_name = manifest.get("protocol_name")
    if protocol is None and protocol_name is None:
        errors.append("protocol name is missing")
    if protocol is not None and protocol != PROTOCOL_NAME:
        errors.append("protocol name mismatch")
    if protocol_name is not None and protocol_name != PROTOCOL_NAME:
        errors.append("protocol_name mismatch")
    if protocol is not None and protocol_name is not None and protocol != protocol_name:
        errors.append("protocol and protocol_name disagree")

    version = manifest.get("version")
    try:
        if _strict_protocol_int(version, "protocol version") != PROTOCOL_VERSION:
            errors.append("protocol version mismatch")
    except ValueError:
        errors.append("protocol version is missing or invalid")

    revision = manifest.get("dataset_revision")
    if not isinstance(revision, str) or not revision.strip():
        errors.append("dataset revision is missing")
    elif revision != PERSONAMEM_REVISION:
        errors.append("dataset revision mismatch")
    split = manifest.get("split")
    if not isinstance(split, str) or not split.strip():
        errors.append("dataset split is missing")
    elif expected_split is not None and split != str(expected_split):
        errors.append("dataset split mismatch")

    seed = manifest.get("seed")
    try:
        _strict_protocol_int(seed, "protocol seed", nonnegative=True)
    except ValueError:
        errors.append("protocol seed is missing or invalid")

    frozen = manifest.get("frozen")
    if not isinstance(frozen, bool):
        errors.append("frozen flag is missing or invalid")
    elif frozen and not isinstance(manifest.get("config_hash"), str):
        errors.append("frozen protocol requires a config hash")
    if (
        "config_hash" in manifest
        and manifest.get("config_hash") is not None
        and (
            not isinstance(manifest.get("config_hash"), str)
            or not manifest.get("config_hash").strip()
            or len(manifest.get("config_hash")) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in manifest.get("config_hash"))
        )
    ):
        errors.append("config hash is invalid")
    source_revision = manifest.get("source_revision")
    if source_revision is not None and (not isinstance(source_revision, str) or not source_revision.strip()):
        errors.append("source revision is invalid")
    memory_policy = manifest.get("memory_policy")
    if memory_policy is not None and not isinstance(memory_policy, Mapping):
        errors.append("memory policy is not an object")
    source_hashes = manifest.get("source_sha256")
    if source_hashes is not None:
        if not isinstance(source_hashes, Mapping):
            errors.append("source_sha256 is not an object")
        else:
            for name, digest in source_hashes.items():
                if not isinstance(digest, str) or len(digest) != 64 or any(
                    character not in "0123456789abcdefABCDEF" for character in digest
                ):
                    errors.append(f"source_sha256 has invalid digest for {name}")
    for name in ("worktree_clean",):
        if name in manifest and manifest[name] is not None and not isinstance(manifest[name], bool):
            errors.append(f"{name} must be boolean or null")
    for name in ("uncommitted_diff_hash", "source_package_hash"):
        if name in manifest and manifest[name] is not None:
            digest = manifest[name]
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in digest
            ):
                errors.append(f"{name} is not a sha256 digest")

    roles = manifest.get("roles")
    if roles is not None and not isinstance(roles, Mapping):
        errors.append("roles is not an object")
    elif isinstance(roles, Mapping):
        missing_roles = [name for name in ROLE_NAMES if name not in roles]
        if missing_roles:
            errors.append("roles is missing: " + ", ".join(missing_roles))
    for role_name in ROLE_NAMES:
        role = _role_value(manifest, role_name)
        if not role:
            errors.append(f"missing role {role_name}")
            continue
        if role.get("name") != role_name:
            errors.append(f"{role_name}: role name is missing or invalid")
        personas = _sequence_field(role.get("personas"))
        question_ids = _sequence_field(role.get("question_ids"))
        if personas is None:
            errors.append(f"{role_name}: personas must be a list")
            personas = ()
        if question_ids is None:
            errors.append(f"{role_name}: question_ids must be a list")
            question_ids = ()
        if not personas:
            errors.append(f"{role_name}: persona list is empty")
        if not question_ids:
            errors.append(f"{role_name}: question_ids list is empty")
        normalized_personas = [str(value) for value in personas]
        normalized_questions = [str(value) for value in question_ids]
        if len(normalized_personas) != len(set(normalized_personas)):
            errors.append(f"{role_name}: duplicate personas")
        if len(normalized_questions) != len(set(normalized_questions)):
            errors.append(f"{role_name}: duplicate question IDs")
        count = role.get("personas_count")
        try:
            if _strict_protocol_int(count, f"{role_name}.personas_count", nonnegative=True) != len(normalized_personas):
                errors.append(f"{role_name}: personas_count mismatch")
        except ValueError:
            errors.append(f"{role_name}: personas_count is missing or invalid")
        queries = role.get("queries")
        try:
            if _strict_protocol_int(queries, f"{role_name}.queries", nonnegative=True) != len(normalized_questions):
                errors.append(f"{role_name}: queries count mismatch")
        except ValueError:
            errors.append(f"{role_name}: queries count is missing or invalid")
        question_hash = role.get("question_hash")
        question_alias = role.get("question_id_sha256")
        if not isinstance(question_hash, str) or not question_hash:
            question_hash = question_alias
        if not isinstance(question_hash, str) or question_hash != stable_hash(normalized_questions):
            errors.append(f"{role_name}: question hash is missing or invalid")
        if question_alias is not None and question_alias != stable_hash(normalized_questions):
            errors.append(f"{role_name}: question_id_sha256 mismatch")
        declared_persona_hash = role.get("persona_hash")
        if not isinstance(declared_persona_hash, str) or declared_persona_hash != persona_hash(normalized_personas):
            errors.append(f"{role_name}: persona hash is missing or invalid")
    return errors


def audit_protocol(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    examples: Sequence[PersonaMemExample] | None = None,
    raw_dir: str | Path = "data/raw/personamem-v1",
    split: str = "32k",
    require_frozen: bool = False,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    manifest = (
        load_protocol_manifest(manifest_or_path)
        if isinstance(manifest_or_path, (str, Path))
        else dict(manifest_or_path)
    )
    errors: list[str] = _manifest_structure_errors(manifest, expected_split=split)
    warnings: list[str] = []
    if require_frozen and not manifest.get("frozen", False):
        errors.append("protocol is not frozen")
    try:
        current = _load_all_examples(examples, raw_dir=raw_dir, split=split)
    except Exception as exc:
        current = []
        errors.append(f"unable to load current source: {type(exc).__name__}: {exc}")
    persisted_source_hashes = manifest.get("source_sha256")
    if isinstance(persisted_source_hashes, Mapping):
        current_hashes, _current_data_package_hash = _source_snapshot(raw_dir, split)
        for name, digest in persisted_source_hashes.items():
            if name not in current_hashes:
                errors.append(f"source file is missing: {name}")
            elif current_hashes[name] != digest:
                errors.append(f"source file hash mismatch: {name}")
    persisted_package = manifest.get("source_package_hash")
    if persisted_package:
        current_package = _source_package_snapshot()
        if current_package is None:
            errors.append("source package snapshot is unavailable")
        elif persisted_package != current_package:
            errors.append("source package hash mismatch")
    by_id: dict[str, PersonaMemExample] = {}
    for item in current:
        key = str(item.question_id)
        if key in by_id:
            errors.append(f"duplicate question_id in current source: {key}")
        by_id[key] = item
    seen: dict[str, str] = {}
    # The full role is a deliberately materialized union, not a third
    # partition.  Duplicate-ID checks therefore apply only to the two
    # disjoint roles below; checking it against ``full-benchmark`` would flag
    # every legitimate question twice.
    partition_roles = (ROLE_DEVELOPMENT, ROLE_CONFIRMATORY)
    for role_name in ROLE_NAMES:
        role = _role_value(manifest, role_name)
        raw_ids = _sequence_field(role.get("question_ids", ()))
        raw_personas = _sequence_field(role.get("personas", ()))
        ids = [str(value) for value in (raw_ids or ())]
        personas = {str(value) for value in (raw_personas or ())}
        if len(ids) != len(set(ids)):
            errors.append(f"{role_name}: duplicate question IDs")
        try:
            declared_personas_count = _strict_protocol_int(
                role.get("personas_count", len(personas)),
                f"{role_name}.personas_count",
                nonnegative=True,
            )
        except ValueError:
            declared_personas_count = -1
        if len(personas) != declared_personas_count:
            errors.append(f"{role_name}: persona count metadata mismatch")
        if role.get("name") not in (None, role_name):
            errors.append(f"{role_name}: role name metadata mismatch")
        try:
            declared_queries = _strict_protocol_int(
                role.get("queries", len(ids)), f"{role_name}.queries", nonnegative=True
            )
        except ValueError:
            declared_queries = -1
        if declared_queries != len(ids):
            errors.append(f"{role_name}: query count/hash metadata mismatch")
        expected_hash = role.get("question_hash", role.get("question_id_sha256", ""))
        if expected_hash != stable_hash(ids):
            errors.append(f"{role_name}: question hash mismatch")
        if role.get("persona_hash") and role.get("persona_hash") != persona_hash(sorted(personas)):
            errors.append(f"{role_name}: persona hash mismatch")
        for question_id in ids:
            if role_name in partition_roles and question_id in seen:
                errors.append(f"question appears in multiple roles: {question_id}")
            if role_name in partition_roles:
                seen[question_id] = role_name
            if question_id not in by_id:
                errors.append(f"question is missing from current source: {question_id}")
            elif str(by_id[question_id].persona_id) not in personas:
                errors.append(f"persona/question mismatch in {role_name}: {question_id}")
        actual_personas = {
            str(by_id[question_id].persona_id)
            for question_id in ids
            if question_id in by_id
        }
        if actual_personas != personas:
            errors.append(f"{role_name}: persona list does not match persisted question IDs")
    full = _role_value(manifest, ROLE_FULL)
    full_ids = {str(value) for value in full.get("question_ids", ())}
    if set(seen) != full_ids:
        errors.append("role union does not equal full-benchmark IDs")
    if full_ids and set(by_id) != full_ids:
        errors.append("current source question set cannot confirm persisted full-benchmark IDs")
    # ``full-benchmark`` is intentionally the union of the two disjoint
    # evaluation roles, so it must overlap both of them.  Only the two
    # *partition* roles are required to be persona-disjoint.  The previous
    # implementation compared the full role as well and therefore rejected
    # every valid manifest produced by ``build_protocol_manifest``.
    persona_sets = [
        set(str(value) for value in _role_value(manifest, name).get("personas", ()))
        for name in ROLE_NAMES
    ]
    if persona_sets[0] & persona_sets[1]:
        errors.append("development and confirmatory persona roles overlap")
    if persona_sets[0] | persona_sets[1] != persona_sets[2]:
        errors.append("full-benchmark personas do not equal the partition persona union")
    if len(current) and full.get("queries") not in (None, len(current)):
        errors.append("full-benchmark query count differs from source")
    role_counts: dict[str, int] = {}
    for name in ROLE_NAMES:
        raw_count = _role_value(manifest, name).get("queries", 0)
        try:
            role_counts[name] = _strict_protocol_int(raw_count, f"{name}.queries", nonnegative=True)
        except ValueError:
            role_counts[name] = 0
    evidence = {
        "protocol": manifest.get("protocol"),
        "frozen": bool(manifest.get("frozen", False)),
        "role_counts": role_counts,
        "persona_counts": {name: len(_role_value(manifest, name).get("personas", ())) for name in ROLE_NAMES},
        "source_queries": len(current),
        "source_revision": PERSONAMEM_REVISION,
    }
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "evidence": evidence,
    }
    if errors and raise_on_error:
        raise RuntimeError("protocol audit failed: " + "; ".join(errors))
    return report


def freeze_protocol(
    manifest_path: str | Path,
    *,
    config_hash: str | None = None,
    output_path: str | Path | None = None,
    examples: Sequence[PersonaMemExample] | None = None,
    raw_dir: str | Path = "data/raw/personamem-v1",
    split: str = "32k",
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = load_protocol_manifest(path)
    audit_protocol(manifest, examples=examples, raw_dir=raw_dir, split=split, raise_on_error=True)
    effective_config_hash = config_hash or manifest.get("config_hash")
    if not isinstance(effective_config_hash, str) or not effective_config_hash.strip():
        raise ValueError("freezing a protocol requires --config-hash")
    if len(effective_config_hash) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in effective_config_hash
    ):
        raise ValueError("config_hash must be a SHA-256 hex digest")
    if manifest.get("frozen") and str(manifest.get("config_hash", "")).lower() != effective_config_hash.lower():
        raise ValueError("frozen protocol config hash cannot be changed")
    updated = {
        **manifest,
        "frozen": True,
        "config_hash": effective_config_hash.lower(),
        "frozen_at": time.time(),
    }
    destination = Path(output_path) if output_path is not None else path
    _atomic_json(destination, updated)
    return updated


def protocol_gate(
    phase: str,
    *,
    manifest: Mapping[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    config_hash: str | None = None,
    action: str = "run",
) -> dict[str, Any]:
    """Enforce phase-specific mutation/tuning rules."""
    if manifest is not None and manifest_path is not None:
        raise ValueError("protocol gate accepts either manifest or manifest_path, not both")
    if manifest is None:
        if manifest_path is None:
            raise ValueError("protocol gate requires a manifest")
        manifest = load_protocol_manifest(manifest_path)
    if not isinstance(manifest, Mapping):
        raise PermissionError("protocol gate requires a manifest object")
    structure_errors = _manifest_structure_errors(manifest)
    if structure_errors:
        raise PermissionError("protocol manifest structure is invalid: " + "; ".join(structure_errors))
    normalized = canonical_phase(phase)
    if normalized == ROLE_CONFIRMATORY:
        if not manifest.get("frozen", False):
            raise PermissionError("confirmatory-test requires a frozen protocol manifest")
        if action.lower() in {"tune", "train", "modify_config", "rewrite_manifest"}:
            raise PermissionError("confirmatory-test forbids tune/train/config modification")
        frozen_hash = manifest.get("config_hash")
        # A frozen split without a configuration identity is not a
        # confirmatory protocol: callers could silently change retrieval
        # settings between freeze and execution.  Require both sides of the
        # comparison, and compare canonical string spellings.
        if not isinstance(frozen_hash, str) or not frozen_hash.strip():
            raise PermissionError("confirmatory-test requires the frozen protocol config hash")
        if not isinstance(config_hash, str) or not config_hash.strip():
            raise PermissionError("confirmatory-test requires the frozen protocol config hash")
        if (
            len(frozen_hash) != 64
            or len(config_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in frozen_hash)
            or any(character not in "0123456789abcdefABCDEF" for character in config_hash)
        ):
            raise PermissionError("confirmatory-test config hashes must be SHA-256 digests")
        if frozen_hash.lower() != config_hash.lower():
            raise PermissionError("confirmatory-test config hash differs from frozen protocol")
    return {"status": "allowed", "phase": normalized, "action": action, "config_hash": config_hash}


def protocol_examples(
    manifest: Mapping[str, Any] | str | Path,
    examples: Sequence[PersonaMemExample],
    phase: str,
) -> tuple[PersonaMemExample, ...]:
    value = load_protocol_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    if not isinstance(value, Mapping):
        raise ValueError("protocol manifest must be a mapping")
    role_name = canonical_phase(phase)
    role = _role_value(value, role_name)
    if not role:
        raise ValueError(f"protocol manifest has no role {role_name}")
    raw_ids = role.get("question_ids", ())
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        raise ValueError(f"protocol role {role_name} has invalid question_ids")
    ids = [str(item) for item in raw_ids]
    if len(ids) != len(set(ids)):
        raise ValueError(f"protocol role {role_name} has duplicate question_ids")
    by_id: dict[str, PersonaMemExample] = {}
    for item in examples:
        identifier = str(item.question_id)
        if identifier in by_id:
            raise ValueError(f"examples contain duplicate question_id: {identifier}")
        by_id[identifier] = item
    missing = [identifier for identifier in ids if identifier not in by_id]
    if missing:
        raise ValueError(f"cannot confirm persisted {role_name} question set: missing {missing[:5]}")
    # Preserve the persisted order.  The order is part of the manifest hash
    # and is useful for reproducible batching; sorting here would silently
    # change a legacy split while leaving its declared hash untouched.
    selected = tuple(by_id[identifier] for identifier in ids)
    if len(selected) != len(ids):
        raise ValueError(f"cannot confirm persisted {role_name} question set")
    return selected


# Verbose aliases used by CLI integrations and notebooks.
protocol_init = init_protocol
protocol_audit = audit_protocol
protocol_freeze = freeze_protocol


__all__ = [
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "ROLE_DEVELOPMENT",
    "ROLE_CONFIRMATORY",
    "ROLE_FULL",
    "ProtocolRole",
    "ProtocolManifest",
    "stable_hash",
    "persona_hash",
    "canonical_phase",
    "build_protocol_manifest",
    "init_protocol",
    "load_protocol_manifest",
    "audit_protocol",
    "freeze_protocol",
    "protocol_gate",
    "protocol_examples",
    "protocol_init",
    "protocol_audit",
    "protocol_freeze",
]
