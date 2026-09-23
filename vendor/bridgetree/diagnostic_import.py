"""Read-only imports of literal dependency scores, never a scoring backend.

An imported observation is historical evidence, not a cache entry for a
currently deployed model.  Source locations are retained for every literal.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import math
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


class DiagnosticImportError(ValueError):
    pass


class HistoricalScoreConflict(DiagnosticImportError):
    pass


def canonical_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DiagnosticImportError("memory IDs must be a sequence")
    if any(not isinstance(value, str) or not value for value in values):
        raise DiagnosticImportError("memory IDs must be nonempty strings")
    return tuple(sorted(set(values)))


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise DiagnosticImportError(f"{name} must be a finite number")
    return float(value)


def _json_object(data: bytes, source: str) -> dict[str, Any]:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DiagnosticImportError(f"duplicate JSON field {key!r}: {source}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=no_duplicates,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               DiagnosticImportError(f"nonfinite JSON number: {source}")))
    except (ValueError, UnicodeDecodeError) as exc:
        raise DiagnosticImportError(f"invalid or truncated JSON: {source}") from exc
    if not isinstance(value, dict):
        raise DiagnosticImportError(f"expected JSON object: {source}")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_diagnostic_archive(
    path: str | Path, *, expected_sha256: str | None = None,
    max_member_bytes: int = 64 * 1024 * 1024,
    max_total_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Read candidate_pool JSON from a tar, directory, or single JSON file.

    Nothing is extracted or written. Directory identity is the SHA-256 of a
    canonical list of relative member names and content digests. Tar/file
    identity is the hash of the original file. Links and unsafe names fail.
    """
    root = Path(path).expanduser()
    if root.is_symlink():
        raise DiagnosticImportError("diagnostic source must not be a symlink")
    if not root.exists():
        raise FileNotFoundError(root)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (max_member_bytes, max_total_bytes)):
        raise DiagnosticImportError("archive size limits must be positive integers")
    if expected_sha256 is not None and (
        len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256)
    ):
        raise DiagnosticImportError("expected_sha256 must be a SHA-256 hex digest")
    entries: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    total = 0

    def add(name: str, raw: bytes) -> None:
        nonlocal total
        name_path = PurePosixPath(name)
        if name_path.is_absolute() or ".." in name_path.parts or "\\" in name:
            raise DiagnosticImportError(f"unsafe archive member: {name}")
        if name in seen:
            raise DiagnosticImportError(f"duplicate archive member: {name}")
        seen.add(name)
        total += len(raw)
        if len(raw) > max_member_bytes or total > max_total_bytes:
            raise DiagnosticImportError("diagnostic archive exceeds size limit")
        entries.append((name, raw))

    if root.is_dir():
        candidates = root / "candidate_pool" if (root / "candidate_pool").is_dir() else root
        if candidates.is_symlink():
            raise DiagnosticImportError("candidate_pool must not be a symlink")
        for item in sorted(candidates.glob("*.json")):
            if item.is_symlink() or not item.is_file():
                raise DiagnosticImportError(f"unsafe source member: {item.name}")
            if item.stat().st_size > max_member_bytes:
                raise DiagnosticImportError("diagnostic member exceeds size limit")
            add(item.relative_to(root).as_posix(), item.read_bytes())
        identity = [{"member": name, "sha256": hashlib.sha256(raw).hexdigest()}
                    for name, raw in entries]
        source_hash = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
        source_kind = "directory_snapshot"
    else:
        source_hash = _file_hash(root)
        if expected_sha256 and source_hash != expected_sha256.lower():
            raise DiagnosticImportError("diagnostic source SHA-256 mismatch")
        if root.suffix == ".json":
            if root.stat().st_size > max_member_bytes:
                raise DiagnosticImportError("diagnostic member exceeds size limit")
            add(root.name, root.read_bytes())
            source_kind = "json_file"
        else:
            source_kind = "tar_archive"
            try:
                # tarfile can stop at the tar EOF blocks before gzip verifies
                # its trailer. Drain gzip separately so a truncated .tgz is
                # not accepted merely because the final JSON was complete.
                with root.open("rb") as handle:
                    compressed = handle.read(2) == b"\x1f\x8b"
                if compressed:
                    expanded = 0
                    with gzip.open(root, "rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            expanded += len(chunk)
                            if expanded > max_total_bytes + 1024 * 1024:
                                raise DiagnosticImportError("diagnostic archive exceeds expanded size limit")
                with tarfile.open(root, "r:*") as archive:
                    for member in archive:
                        member_path = PurePosixPath(member.name)
                        if member_path.is_absolute() or ".." in member_path.parts or "\\" in member.name:
                            raise DiagnosticImportError(f"unsafe archive member: {member.name}")
                        if member.issym() or member.islnk():
                            raise DiagnosticImportError(f"archive links are not allowed: {member.name}")
                        if not member.isfile() or member_path.suffix != ".json":
                            continue
                        if "candidate_pool" not in member_path.parts:
                            continue
                        if member.size > max_member_bytes or total + member.size > max_total_bytes:
                            raise DiagnosticImportError("diagnostic archive exceeds size limit")
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise DiagnosticImportError(f"unreadable archive member: {member.name}")
                        raw = stream.read(max_member_bytes + 1)
                        if len(raw) != member.size:
                            raise DiagnosticImportError(f"truncated archive member: {member.name}")
                        add(member.name, raw)
            except (tarfile.TarError, EOFError, OSError) as exc:
                raise DiagnosticImportError("invalid or truncated diagnostic archive") from exc
        if _file_hash(root) != source_hash:
            raise DiagnosticImportError("diagnostic source changed while being read")
    if expected_sha256 and source_hash != expected_sha256.lower():
        raise DiagnosticImportError("diagnostic source SHA-256 mismatch")
    if not entries:
        raise DiagnosticImportError("no candidate_pool JSON artifacts found")
    artifacts = []
    for name, raw in sorted(entries):
        value = _json_object(raw, name)
        if not isinstance(value.get("search"), Mapping) or not isinstance(value.get("selection"), Mapping):
            raise DiagnosticImportError(f"missing search/selection artifact: {name}")
        artifacts.append({"artifact": value, "provenance": {
            "source_kind": source_kind, "source_sha256": source_hash,
            "member": name, "member_sha256": hashlib.sha256(raw).hexdigest(),
        }})
    return {"schema_version": 1, "source_kind": source_kind,
            "source_sha256": source_hash, "artifacts": artifacts}


def import_historical_scores(
    artifact: Mapping[str, Any], *, provenance: Mapping[str, Any] | None = None,
    score_space: str = "unit_interval", conflict_atol: float = 1e-12,
) -> dict[str, Any]:
    """Recover direct scalar observations, retaining all literal sources.

    No scores are reconstructed by subtraction, interpolation, or defaults.
    Duplicate observations are validated and retained, not last-write-wins.
    """
    if score_space not in {"unit_interval", "logit_difference"}:
        raise DiagnosticImportError("unsupported score_space")
    conflict_atol = _number(conflict_atol, "conflict_atol")
    if conflict_atol < 0:
        raise DiagnosticImportError("conflict_atol must be nonnegative")
    search, selection = artifact.get("search"), artifact.get("selection")
    if not isinstance(search, Mapping) or not isinstance(selection, Mapping):
        raise DiagnosticImportError("artifact must contain search and selection objects")
    scores: dict[tuple[str, ...], dict[str, Any]] = {}
    source = dict(provenance or {})
    # This is deliberately not an unfiltered copy of artifact metadata/gold.
    source.update({key: artifact[key] for key in ("task_id", "method_id", "attempt") if key in artifact})
    search_ids: set[tuple[str, ...]] = set()

    def put(ids: Sequence[str], raw: Any, pointer: str, stage: str) -> None:
        key = canonical_ids(ids)
        value = _number(raw, f"score at {pointer}")
        if score_space == "unit_interval" and not 0 <= value <= 1:
            raise DiagnosticImportError(f"unit interval score out of range at {pointer}")
        observation = {**source, "origin": "historical_literal", "score": value,
                       "json_pointer": pointer, "stage": stage}
        if key in scores and abs(scores[key]["score"] - value) > conflict_atol:
            raise HistoricalScoreConflict(f"conflicting historical scores for {key}: {pointer}")
        entry = scores.setdefault(key, {"ids": list(key), "score": value, "observations": []})
        entry["observations"].append(observation)
        if stage == "search":
            search_ids.add(key)

    activations = search.get("activations")
    rounds = selection.get("rounds")
    if not isinstance(activations, list) or not isinstance(rounds, list):
        raise DiagnosticImportError("missing activation or selection round list")
    for index, row in enumerate(activations):
        try:
            p, group = list(canonical_ids(row["premise_ids"])), list(canonical_ids(row["group_ids"]))
            target = row["target_id"]
            if not isinstance(target, str) or not target or not group:
                raise DiagnosticImportError("invalid activation target/group")
            if target in p or target in group or set(p) & set(group):
                raise DiagnosticImportError("activation groups must be disjoint")
            for ids, field in ((p, "P"), (p + [target], "Pe"), (p + group, "PG"),
                               (p + group + [target], "PGe")):
                put(ids, row[field], f"/search/activations/{index}/{field}", "search")
            for field, expected in (
                ("activation", row["PGe"] - row["PG"] - row["Pe"] + row["P"]),
                ("context_marginal", row["PGe"] - row["Pe"]),
            ):
                if field in row and abs(_number(row[field], field) - expected) > conflict_atol:
                    raise DiagnosticImportError(f"inconsistent literal {field}")
        except (KeyError, TypeError) as exc:
            raise DiagnosticImportError(f"malformed activation record {index}") from exc
    feasibility: dict[tuple[str, ...], bool] = {}
    for round_index, round_record in enumerate(rounds):
        if not isinstance(round_record, Mapping) or not isinstance(round_record.get("comparisons"), list):
            raise DiagnosticImportError("malformed selection round")
        for index, row in enumerate(round_record["comparisons"]):
            try:
                current = canonical_ids(row["current_ids"])
                union = canonical_ids(row["union_ids"])
                bundle = canonical_ids(row["bundle_ids"])
                if union != canonical_ids((*current, *bundle)):
                    raise DiagnosticImportError("selection union is not current union bundle")
                feasible = row["feasible"]
                if not isinstance(feasible, bool):
                    raise DiagnosticImportError("selection feasibility must be boolean")
                if union in feasibility and feasibility[union] != feasible:
                    raise DiagnosticImportError("conflicting historical feasibility")
                feasibility[union] = feasible
                pointer = f"/selection/rounds/{round_index}/comparisons/{index}"
                for ids, field in ((current, "base_score"), (union, "combined_score")):
                    if row.get(field) is not None:
                        put(ids, row[field], f"{pointer}/{field}", "selection")
                if row.get("marginal") is not None:
                    if row.get("combined_score") is None or row.get("base_score") is None:
                        raise DiagnosticImportError("marginal lacks literal input scores")
                    expected = row["combined_score"] - row["base_score"]
                    if abs(_number(row["marginal"], "marginal") - expected) > conflict_atol:
                        raise DiagnosticImportError("selection marginal disagrees with literal scores")
            except (KeyError, TypeError) as exc:
                raise DiagnosticImportError("malformed selection comparison") from exc
    # Empty input is never sent to the selector feasibility callback. Do not
    # invent generator feasibility for unobserved nonempty subsets.
    all_ids = {value for key in scores for value in key}
    questions = {value.split(":m", 1)[0] for value in all_ids if ":m" in value}
    if len(questions) > 1:
        raise DiagnosticImportError("historical artifact mixes multiple question IDs")
    return {"schema_version": 1, "origin": "historical_literal",
            "task_id": artifact.get("task_id"), "method_id": artifact.get("method_id"),
            "question_id": next(iter(questions), artifact.get("question_id")),
            "score_space": score_space, "score_contract": "pointwise",
            "conflict_atol": conflict_atol, "records": [scores[key] for key in sorted(scores)],
            "search_seen_ids": [list(key) for key in sorted(search_ids)],
            "historical_feasibility": [{"ids": list(key), "feasible": value}
                                       for key, value in sorted(feasibility.items())],
            "unique_score_count": len(scores),
            "literal_observation_count": sum(len(row["observations"]) for row in scores.values())}
