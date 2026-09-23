"""Time-marked transition kernels for the TMIC retriever.

The temporal operator is intentionally a measure, rather than a recency
weight.  Message indices are used when a source has no explicit event time;
unknown times are neutral (overlap one) and are surfaced in diagnostics.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .math_utils import angular_rank_kernel, normalize_rows
from .types import TemporalMark as SemanticTemporalMark


@dataclass(frozen=True)
class TimeMark:
    # Values are intentionally ``Any`` at the boundary: the semantic schema
    # permits textual event dates while the numerical kernel operates on
    # values that can be converted by ``_numeric``.  Numeric/date-like values
    # are canonicalized to float; opaque textual dates remain provenance-only
    # and make the temporal factor explicitly unavailable rather than being
    # guessed or silently treated as zero.
    observed_start: Any = None
    observed_end: Any = None
    event_start: Any = None
    event_end: Any = None
    validity: str = "unknown"
    time_source: str = "unknown"

    def __post_init__(self) -> None:
        validity = str(self.validity).strip().lower()
        source = str(self.time_source).strip().lower()
        if validity in {"event", "point"}:
            validity = "instant" if validity == "point" else "durative"
        if source in {"index", "observed"}:
            source = "message_index"
        if validity not in {"instant", "durative", "unknown"}:
            raise ValueError("time validity must be instant, durative, or unknown")
        if source not in {"message_index", "explicit", "unknown"}:
            raise ValueError("time source must be message_index, explicit, or unknown")
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "time_source", source)
        for name in ("observed_start", "observed_end", "event_start", "event_end"):
            value = getattr(self, name)
            if value is None:
                continue
            numeric = _numeric(value)
            if numeric is not None:
                if not np.isfinite(numeric):
                    raise ValueError(f"{name} must be finite when present")
                object.__setattr__(self, name, float(numeric))
            elif not isinstance(value, str):
                raise ValueError(f"{name} must be numeric, date-like, or a string")

    @property
    def unavailable(self) -> bool:
        # A caller constructing ``TimeMark`` directly often omits the source
        # while still supplying an unambiguous point/interval.  The temporal
        # measure is unavailable only when its validity is unknown or no
        # usable endpoint exists; ``time_source=unknown`` is retained as
        # provenance but does not erase explicit numeric evidence.
        return self.validity == "unknown" or (self.start is None and self.end is None)

    # A few integrations use the shorter names from the data specification.
    # Keep these read-only aliases so old and new serialized records can be
    # consumed by the same kernel.
    @property
    def is_unknown(self) -> bool:
        return self.unavailable

    @property
    def is_durative(self) -> bool:
        return self.validity == "durative"

    @property
    def start(self) -> float | None:
        value = self.event_start if self.event_start is not None else self.observed_start
        return _numeric(value)

    @property
    def end(self) -> float | None:
        value = self.event_end if self.event_end is not None else self.observed_end
        return _numeric(value)

    @property
    def points(self) -> tuple[float, ...]:
        # Event validity is the temporal object being compared whenever an
        # explicit event endpoint exists.  Observation timestamps describe
        # when the record was seen and must not silently add a second point to
        # an explicitly dated instant.  Legacy/observed marks fall back to
        # their observed endpoints.
        if self.event_start is not None or self.event_end is not None:
            values = [self.event_start, self.event_end]
        else:
            values = [self.observed_start, self.observed_end]
        # Preserve insertion order while removing duplicate endpoints.
        numeric = [_numeric(value) for value in values if value is not None]
        return tuple(dict.fromkeys(float(value) for value in numeric if value is not None))

    def public_dict(self) -> dict[str, Any]:
        return {
            "observed_start": self.observed_start,
            "observed_end": self.observed_end,
            "event_start": self.event_start,
            "event_end": self.event_end,
            "validity": self.validity,
            "time_source": self.time_source,
        }

    @classmethod
    def instant(
        cls,
        point: Any,
        *,
        source: str = "explicit",
        observed: bool = False,
    ) -> "TimeMark":
        """Construct a point measure without making callers spell endpoints."""
        value = _numeric(point)
        if value is None:
            raise ValueError("instant time point must be numeric or date-like")
        return cls(
            observed_start=value if observed else None,
            observed_end=value if observed else None,
            event_start=None if observed else value,
            event_end=None if observed else value,
            validity="instant",
            time_source="message_index" if observed else source,
        )

    @classmethod
    def durative(
        cls,
        start: Any,
        end: Any,
        *,
        source: str = "explicit",
    ) -> "TimeMark":
        """Construct an event interval measure."""
        left, right = _numeric(start), _numeric(end)
        if left is None or right is None:
            raise ValueError("durative endpoints must be numeric or date-like")
        return cls(event_start=left, event_end=right, validity="durative", time_source=source)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None, timestamp: Any = None) -> "TimeMark":
        return _mark_from_metadata(metadata, timestamp)


# Public spelling from the semantic design.  ``TimeMark`` remains the
# historical name and is deliberately an alias (rather than a second class)
# so ``isinstance`` checks continue to work in integrations written against
# either version.
TemporalMark = TimeMark


class TransitionCache:
    """In-memory deterministic transition cache with explicit provenance key."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._values: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        self.root = None
        if root is not None:
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(
        query_hash: str,
        bank_hash: str,
        temporal_measure: bool,
        time_hash: str = "",
    ) -> str:
        payload = json.dumps(
            {
                "schema": 1,
                "query_hash": query_hash,
                "bank_hash": bank_hash,
                "temporal_measure": bool(temporal_measure),
                "time_hash": time_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[np.ndarray, dict[str, Any]] | None:
        value = self._values.get(key)
        if value is not None:
            matrix, diagnostics = value
            return matrix.copy(), dict(diagnostics)
        if self.root is not None:
            path = self.root / f"{key}.npz"
            if path.is_file():
                try:
                    with np.load(path, allow_pickle=False) as loaded:
                        matrix = np.asarray(loaded["matrix"], dtype=np.float64)
                        diagnostics_raw = str(loaded["diagnostics"].item()) if "diagnostics" in loaded else "{}"
                    diagnostics = json.loads(diagnostics_raw)
                    if (
                        matrix.ndim != 2
                        or matrix.shape[0] != matrix.shape[1]
                        or not np.all(np.isfinite(matrix))
                    ):
                        return None
                    self._values[key] = (matrix.copy(), dict(diagnostics))
                    return matrix, diagnostics
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    return None
        return None

    def put(self, key: str, matrix: np.ndarray, diagnostics: Mapping[str, Any] | None = None) -> None:
        value = np.asarray(matrix, dtype=np.float64).copy()
        if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
            raise ValueError("transition cache values must be a finite square matrix")
        detail = dict(diagnostics or {})
        self._values[key] = (value, detail)
        if self.root is not None:
            path = self.root / f"{key}.npz"
            temporary = path.with_name(f"{path.name}.{__import__('os').getpid()}.tmp")
            # Diagnostics may include NumPy arrays; normalize recursively for
            # a portable JSON sidecar inside the npz container.
            def jsonable(item: Any) -> Any:
                if isinstance(item, np.ndarray):
                    return item.tolist()
                if isinstance(item, (np.integer, np.floating, np.bool_)):
                    return item.item()
                if isinstance(item, Mapping):
                    return {str(k): jsonable(v) for k, v in item.items()}
                if isinstance(item, (list, tuple)):
                    return [jsonable(v) for v in item]
                return item
            with temporary.open("wb") as handle:
                np.savez(handle, matrix=value, diagnostics=np.asarray(json.dumps(jsonable(detail), ensure_ascii=False)))
            temporary.replace(path)

    def get_or_build(self, key: str, builder) -> tuple[np.ndarray, dict[str, Any], bool]:
        cached = self.get(key)
        if cached is not None:
            return cached[0], cached[1], True
        matrix, diagnostics = builder()
        self.put(key, matrix, diagnostics)
        return np.asarray(matrix, dtype=np.float64), dict(diagnostics), False


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (_datetime.datetime, _datetime.date)):
        # ``date.timestamp`` is not available on every Python date-like
        # implementation; converting through midnight UTC keeps the mapping
        # deterministic and makes date strings and date objects agree.
        if isinstance(value, _datetime.datetime):
            # Naive datetimes otherwise inherit the host's local timezone,
            # which would make cache keys and overlaps machine-dependent.
            aware = value if value.tzinfo is not None else value.replace(tzinfo=_datetime.timezone.utc)
            return float(aware.timestamp())
        return float(_datetime.datetime(value.year, value.month, value.day, tzinfo=_datetime.timezone.utc).timestamp())
    # ``numpy.datetime64`` is common in dataframe-derived metadata but is not
    # a subclass of ``datetime.date``.  Convert through a nanosecond integer
    # only after rejecting the ``NaT`` sentinel; this keeps the result stable
    # across hosts and avoids NumPy's local-time assumptions.
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        try:
            return float(value.astype("datetime64[ns]").astype(np.int64)) / 1_000_000_000.0
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            try:
                parsed = _datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
                # ``datetime.timestamp`` interprets a naive value in the
                # host's local timezone.  Metadata/cache identities must be
                # machine independent, so treat date-like strings exactly as
                # naive ``datetime`` objects above: midnight UTC.
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=_datetime.timezone.utc)
                return parsed.timestamp()
            except ValueError:
                return None
    return None


def _mark_from_metadata(metadata: Mapping[str, Any] | None, timestamp: Any = None) -> TimeMark:
    if isinstance(metadata, TimeMark):
        return metadata
    if isinstance(metadata, SemanticTemporalMark):
        return TimeMark(
            observed_start=metadata.observed_start,
            observed_end=metadata.observed_end,
            event_start=metadata.event_start,
            event_end=metadata.event_end,
            validity=metadata.validity,
            time_source=metadata.time_source,
        )
    raw = metadata or {}
    # Mapping-shaped memory records frequently look like
    # ``{"memory_id": ..., "metadata": {"time": ...}}``.  Unwrap that
    # envelope before interpreting fields; retain top-level temporal keys so a
    # flat record can override its nested metadata deterministically.
    if isinstance(raw, Mapping) and isinstance(raw.get("metadata"), Mapping):
        nested = dict(raw.get("metadata", {}))
        for key, value in raw.items():
            if key != "metadata":
                nested[key] = value
        raw = nested
    time_raw = raw.get("time", raw) if isinstance(raw, Mapping) else {}
    if isinstance(time_raw, Mapping) and isinstance(time_raw.get("metadata"), Mapping):
        nested_time = dict(time_raw.get("metadata", {}))
        for key, value in time_raw.items():
            if key != "metadata":
                nested_time[key] = value
        time_raw = nested_time
    if isinstance(time_raw, TimeMark):
        return time_raw
    if isinstance(time_raw, SemanticTemporalMark):
        return _mark_from_metadata(time_raw)
    if not isinstance(time_raw, Mapping):
        # Compact records often use ``{"time": "2024-01-01"}`` rather than
        # spelling a nested metadata object.  Treat a scalar as an explicit
        # instant; an absent scalar remains unknown and therefore neutral.
        scalar = _numeric(time_raw)
        if scalar is not None:
            return TimeMark(
                event_start=scalar,
                event_end=scalar,
                validity="instant",
                time_source="explicit",
            )
        # Compact interval records occasionally use ``time: [start, end]``.
        # Preserve that as a durative mark rather than treating the list as an
        # opaque/unknown value.
        if isinstance(time_raw, (list, tuple)) and len(time_raw) >= 2:
            left, right = _numeric(time_raw[0]), _numeric(time_raw[1])
            if left is not None and right is not None:
                return TimeMark(
                    event_start=left,
                    event_end=right,
                    validity="durative",
                    time_source="explicit",
                )
        time_raw = {}
    validity = str(time_raw.get("validity", "")).strip().lower()
    source = str(time_raw.get("time_source", "")).strip().lower()
    observed_value = time_raw.get("observed")
    if isinstance(observed_value, Mapping):
        observed_start = _numeric(observed_value.get("observed_start", observed_value.get("start")))
        observed_end = _numeric(observed_value.get("observed_end", observed_value.get("end")))
    elif isinstance(observed_value, (list, tuple)) and len(observed_value) >= 2:
        observed_start = _numeric(observed_value[0])
        observed_end = _numeric(observed_value[1])
    else:
        observed_start = _numeric(time_raw.get("observed_start", observed_value))
        observed_end = _numeric(time_raw.get("observed_end", observed_value))
    event_start = _numeric(time_raw.get("event_start", time_raw.get("start")))
    event_end = _numeric(time_raw.get("event_end", time_raw.get("end")))
    if event_start is None and event_end is None:
        for alias in ("query_time", "query_date", "cutoff", "date"):
            if alias in time_raw:
                event_start = event_end = _numeric(time_raw.get(alias))
                if event_start is not None:
                    validity = validity or "instant"
                    source = source or "explicit"
                    break
    explicitly_unknown = validity == "unknown" and "validity" in time_raw
    # Be permissive for hand-authored metadata: explicit event endpoints are
    # unambiguously durative, while an observed point is an instant.  A caller
    # that wants neutral semantics can still state ``validity=unknown``.
    if not explicitly_unknown and validity not in {"instant", "durative"}:
        if event_start is not None or event_end is not None:
            validity = "durative" if event_start != event_end else "instant"
        elif observed_start is not None or observed_end is not None:
            validity = "instant"
    if validity not in {"instant", "durative", "unknown"}:
        validity = "unknown"
    if source not in {"message_index", "explicit", "unknown"}:
        source = "unknown"
    # Legacy Memory objects only carry ``timestamp``.  It is an observed
    # message-index point, not a calendar timestamp or a recency signal.
    if validity == "unknown" and timestamp is None and not explicitly_unknown:
        timestamp = time_raw.get("timestamp")
    if validity == "unknown" and timestamp is not None and not explicitly_unknown:
        point = _numeric(timestamp)
        if point is not None:
            observed_start = observed_end = point
            # A legacy ``timestamp`` is an observation-order fallback, not an
            # explicit event time.  Keep the mark unknown so k_T remains
            # neutral while the endpoint is still available for diagnostics
            # and inferred query envelopes.
            validity = "unknown"
            source = "message_index"
    if validity != "unknown" and source == "unknown":
        source = (
            "message_index"
            if observed_start is not None and event_start is None and event_end is None
            else "explicit"
        )
    return TimeMark(observed_start, observed_end, event_start, event_end, validity, source)


def _memory_parts(item: Any, index: int) -> tuple[str, Mapping[str, Any], Any]:
    if hasattr(item, "memory_id"):
        return str(item.memory_id), getattr(item, "metadata", {}) or {}, getattr(item, "timestamp", index)
    if isinstance(item, Mapping):
        identifier = str(item.get("memory_id", item.get("id", index)))
        # Mapping-shaped records commonly keep the legacy timestamp beside
        # ``metadata``.  Use it as the observed message-index fallback rather
        # than the enumeration position.
        timestamp = item.get("timestamp", item.get("observed", index))
        return identifier, item, timestamp
    # A bare scalar sequence (for example ``["2024-01-01", "2024-01-02"]``)
    # is a useful compact time-bank spelling.  Preserve the scalar as the
    # observed point when it is date/numeric-like; opaque objects retain the
    # deterministic enumeration fallback.
    return str(index), {}, item if _numeric(item) is not None else index


def build_time_marks(
    memories: Sequence[Any] | Mapping[str, Any] | Any,
    query_cutoff: Any = None,
    *,
    query: Any = None,
    query_metadata: Mapping[str, Any] | None = None,
    return_diagnostics: bool = False,
) -> Any:
    """Build deterministic :class:`TimeMark` objects.

    For a sequence the return value is an ``id -> TimeMark`` mapping.  Passing
    one metadata mapping (or one Memory) returns a single mark.  The optional
    diagnostics form is ``(value, diagnostics)`` and is useful to callers that
    need to record ``time_unavailable`` without changing the numerical API.
    """
    if query_cutoff is None and query is not None:
        query_cutoff = query
    if isinstance(memories, TimeMark):
        result = memories
    elif hasattr(memories, "memory_id") or (
        isinstance(memories, Mapping)
        and any(
            key in memories
            for key in (
                # A mapping-shaped memory envelope may contain no temporal
                # key at the top level, e.g. ``{"metadata": {"time": ...}}``.
                # Recognize that form as one record before treating a mapping
                # as an ID-indexed bank.
                "metadata",
                "time",
                "validity",
                "observed",
                "observed_start",
                "observed_end",
                "event_start",
                "event_end",
                "start",
                "end",
                "memory_id",
                "id",
                "query_time",
                "query_date",
                "cutoff",
                "date",
                "timestamp",
            )
        )
    ):
        identifier, metadata, timestamp = _memory_parts(memories, 0)
        del identifier
        result: Any = _mark_from_metadata(metadata, timestamp)
    elif isinstance(memories, Mapping):
        result = {}
        for index, (identifier, value) in enumerate(memories.items()):
            normalized_identifier = str(identifier)
            if normalized_identifier in result:
                raise ValueError("time-mark IDs must remain unique after string normalization")
            if isinstance(value, TimeMark):
                mark = value
            elif hasattr(value, "metadata"):
                # Support the convenient ``{memory_id: Memory}`` form as
                # well as ``{memory_id: metadata}``; the former is common in
                # callers that already maintain an ID-indexed bank.
                mark = _mark_from_metadata(
                    getattr(value, "metadata", {}) or {},
                    getattr(value, "timestamp", index),
                )
            elif isinstance(value, Mapping):
                mark = _mark_from_metadata(value, value.get("timestamp", index))
            else:
                mark = _mark_from_metadata({}, value if value is not None else index)
            result[normalized_identifier] = mark
    else:
        # Treat a scalar/date-like value (including a single string) as one
        # compact record.  Iterating a date string character-by-character is
        # both surprising and makes the resulting time hash depend on text
        # length.  Ordinary sequences retain their positional IDs.
        if isinstance(memories, (str, bytes)) or not hasattr(memories, "__iter__"):
            result = {"0": _mark_from_metadata({}, memories)}
        else:
            result = {}
        iterable = () if result else memories
        for index, item in enumerate(iterable):
            if isinstance(item, TimeMark):
                identifier = str(index)
                if identifier in result:
                    raise ValueError("time-mark IDs must remain unique")
                result[identifier] = item
                continue
            identifier, metadata, timestamp = _memory_parts(item, index)
            if identifier in result:
                raise ValueError("time-mark IDs must remain unique after string normalization")
            result[identifier] = _mark_from_metadata(metadata, timestamp)

    unavailable = (
        [key for key, mark in result.items() if mark.unavailable]
        if isinstance(result, dict)
        else (["single"] if result.unavailable else [])
    )
    diagnostics: dict[str, Any] = {
        "time_unavailable": bool(unavailable),
        "time_unavailable_ids": unavailable,
        "query_cutoff_explicit": query_cutoff is not None or query_metadata is not None,
    }
    if query_cutoff is not None or query_metadata is not None:
        diagnostics["query_mark"] = query_time_mark(query_cutoff, query_metadata).public_dict()
    if return_diagnostics:
        return result, diagnostics
    return result


def query_time_mark(query_cutoff: Any = None, query_metadata: Mapping[str, Any] | None = None) -> TimeMark:
    """Normalize an explicit query date/cutoff or an observable cutoff range."""
    if query_metadata is not None:
        mark = _mark_from_metadata(query_metadata)
        if not mark.unavailable:
            return mark
    if isinstance(query_cutoff, TimeMark):
        return query_cutoff
    if isinstance(query_cutoff, Mapping):
        return _mark_from_metadata(query_cutoff)
    # Date-like strings are intentionally handled by ``_numeric`` below;
    # keeping this branch explicit documents that a non-mapping query alias is
    # a cutoff rather than an opaque query-text value.
    if isinstance(query_cutoff, (tuple, list)) and len(query_cutoff) >= 2:
        start, end = _numeric(query_cutoff[0]), _numeric(query_cutoff[1])
        if start is not None and end is not None:
            return TimeMark(start, end, start, end, "durative", "explicit")
    point = _numeric(query_cutoff)
    if point is not None:
        return TimeMark(point, point, point, point, "instant", "explicit")
    return TimeMark()


def _interval(mark: TimeMark) -> tuple[float, float] | None:
    start, end = mark.start, mark.end
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    return (min(float(start), float(end)), max(float(start), float(end)))


def _interval_union_length(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted((min(left, right), max(left, right)) for left, right in intervals)
    total = 0.0
    current_left, current_right = ordered[0]
    for left, right in ordered[1:]:
        if left <= current_right:
            current_right = max(current_right, right)
        else:
            total += max(0.0, current_right - current_left)
            current_left, current_right = left, right
    return float(total + max(0.0, current_right - current_left))


def _overlap_measure(marks: Sequence[TimeMark]) -> float:
    if any(mark.unavailable for mark in marks):
        return 1.0
    if not marks:
        return 1.0
    # A durative mark whose endpoints coincide is observationally a point.
    # Treating it as a zero-length Lebesgue interval would make two distinct
    # observations spuriously overlap at one.  Normalize such marks to the
    # discrete convention before handling genuinely durative intervals.
    normalized_marks = [
        TimeMark(
            mark.observed_start,
            mark.observed_end,
            mark.event_start,
            mark.event_end,
            (
                "instant"
                if (
                    mark.validity == "durative"
                    and mark.start is not None
                    and mark.end is not None
                    and abs(mark.start - mark.end) <= 1e-12
                )
                else mark.validity
            ),
            mark.time_source,
        )
        for mark in marks
    ]
    marks = normalized_marks
    if all(mark.validity == "instant" for mark in marks):
        sets = [set(mark.points) for mark in marks]
        intersection = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        return float(len(intersection) / len(union)) if union else 1.0
    # Mixed instant/durative marks use a discrete point measure for the
    # instants.  A point inside a query/event interval has full membership;
    # treating it as a zero-length Lebesgue interval would incorrectly erase
    # ordinary message observations under a cutoff window.
    if any(mark.validity == "instant" for mark in marks):
        point_sets = [set(mark.points) for mark in marks if mark.validity == "instant"]
        intervals = [_interval(mark) for mark in marks if mark.validity == "durative"]
        intervals = [item for item in intervals if item is not None]
        common = set.intersection(*point_sets) if point_sets else set()
        if intervals:
            common = {
                point
                for point in common
                if all(interval[0] <= point <= interval[1] for interval in intervals)
            }
            # Keep the discrete atom in the hybrid union even when its point
            # lies inside a continuous interval.  Otherwise a short interval
            # (length < 1) could yield a Jaccard value greater than one.  The
            # atom/interval intersection is still counted when the point is
            # contained by every interval, matching the documented mixed
            # membership convention.
            union_size = _interval_union_length(intervals) + len(set.union(*point_sets))
        else:
            union_size = float(len(set.union(*point_sets))) if point_sets else 0.0
        return float(np.clip(len(common) / union_size, 0.0, 1.0)) if union_size > 0.0 else 1.0
    intervals = [_interval(mark) for mark in marks]
    if any(interval is None for interval in intervals):
        return 1.0
    concrete = [interval for interval in intervals if interval is not None]
    left = max(interval[0] for interval in concrete)
    right = min(interval[1] for interval in concrete)
    intersection = max(0.0, right - left)
    union = max(interval[1] for interval in concrete) - min(interval[0] for interval in concrete)
    # All genuinely durative intervals have positive length here.  A zero
    # denominator can only arise from malformed/degenerate input; the
    # measure definition specifies the neutral value in that case.
    return float(np.clip(intersection / union, 0.0, 1.0)) if union > 0.0 else 1.0


def temporal_overlap_kernel(
    left: Any,
    right: Any | None = None,
    query: Any | None = None,
    *,
    return_diagnostics: bool = False,
) -> Any:
    """Return Jaccard overlap of temporal measures, with unknown neutral."""
    def as_mark(value: Any) -> TimeMark:
        if isinstance(value, TimeMark):
            return value
        if isinstance(value, SemanticTemporalMark):
            return _mark_from_metadata(value)
        if hasattr(value, "metadata"):
            return _mark_from_metadata(getattr(value, "metadata", {}), getattr(value, "timestamp", None))
        if isinstance(value, Mapping):
            if isinstance(value.get("time"), TimeMark):
                return value["time"]
            return _mark_from_metadata(value)
        return query_time_mark(value)

    marks = [as_mark(left)]
    if right is not None:
        marks.append(as_mark(right))
    if query is not None:
        marks.append(as_mark(query))
    value = _overlap_measure(marks)
    diagnostics = {
        "value": value,
        "time_unavailable": any(mark.unavailable for mark in marks),
        "validities": [mark.validity for mark in marks],
    }
    return (value, diagnostics) if return_diagnostics else value


def _query_cutoff_for_marks(marks: Sequence[TimeMark]) -> TimeMark:
    concrete = [_interval(mark) for mark in marks if not mark.unavailable]
    concrete = [item for item in concrete if item is not None]
    if not concrete:
        return TimeMark()
    return TimeMark(
        min(item[0] for item in concrete),
        max(item[1] for item in concrete),
        min(item[0] for item in concrete),
        max(item[1] for item in concrete),
        "durative",
        "message_index",
    )


def build_transition_matrix(
    vectors: np.ndarray,
    memories: Sequence[Any] | Mapping[str, Any] | None = None,
    query_cutoff: Any = None,
    *,
    query: Any | None = None,
    temporal_measure: bool = False,
    ids: Sequence[str] | None = None,
    return_diagnostics: bool = False,
    return_kernel: bool = False,
    return_details: bool = False,
    query_metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Construct ``K`` and row-normalized ``P_q`` for T=0/T=1.

    The default return is the transition matrix ``P_q``.  Set
    ``return_diagnostics=True`` to receive ``(P_q, diagnostics)`` including the
    unnormalized kernel, unknown-time flags, and zero-row uniformization.
    """
    if query_metadata is not None:
        query = query_metadata
    if "cutoff" in kwargs and query_cutoff is None:
        query_cutoff = kwargs["cutoff"]
    raw_vectors = np.asarray(vectors, dtype=np.float64)
    if raw_vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if not np.all(np.isfinite(raw_vectors)):
        raise ValueError("vectors must be finite")
    value = normalize_rows(raw_vectors)
    count = len(value)
    # Keep a real ``None`` sentinel here.  An omitted memory collection still
    # has a perfectly well-defined numeric row universe (0..n-1) when the
    # caller supplies only a vector bank; replacing it with an empty tuple
    # used to make the subsequent label-length check fail for every such
    # call.  The sentinel also lets T=1 record neutral unknown marks without
    # pretending that an empty collection was supplied explicitly.
    if memories is None:
        memory_sequence: Sequence[Any] | Mapping[str, Any] | None = None
    elif isinstance(memories, Mapping):
        memory_sequence = memories
    else:
        try:
            memory_sequence = list(memories)
        except TypeError:
            memory_sequence = (memories,)
        memories = memory_sequence
    if ids is not None:
        labels = [str(item) for item in ids]
    elif memory_sequence is None:
        labels = [str(index) for index in range(count)]
    elif isinstance(memory_sequence, Mapping):
        # An ID-indexed memory/metadata mapping already carries the stable
        # row labels.  Iterating its values and falling back to 0..n would
        # make a transition cache silently depend on dictionary position.
        labels = [str(item) for item in memory_sequence]
    else:
        labels = [
            str(
                item.get("memory_id", item.get("id", index))
                if isinstance(item, Mapping)
                else getattr(item, "memory_id", index)
            )
            for index, item in enumerate(memory_sequence)
        ]
    if len(labels) != count:
        raise ValueError("transition IDs and vectors must have equal length")
    if len(set(labels)) != len(labels):
        raise ValueError("transition IDs must be unique")
    if memories is not None and not isinstance(memory_sequence, Mapping):
        memory_count = len(memory_sequence)
        if memory_count != count:
            raise ValueError("memories and vectors must have equal length")
    if not temporal_measure:
        kernel = np.maximum(value @ value.T, 0.0)
        time_diagnostics: dict[str, Any] = {"temporal_measure": False, "time_unavailable": False}
    else:
        kernel = angular_rank_kernel(value, labels)
        if memory_sequence is None:
            marks = {str(index): TimeMark() for index in range(count)}
        else:
            marks = build_time_marks(memories)
            if not isinstance(marks, dict):
                marks = {str(index): marks for index in range(count)}
        ordered_marks = [marks.get(label, TimeMark()) for label in labels]
        # ``query`` is accepted as a convenient alias for either metadata or
        # a scalar/date-like cutoff.  Do not discard a string/date query just
        # because it is not mapping-shaped.
        query_metadata_value = query_metadata
        cutoff_value = query_cutoff
        if query_metadata_value is None and isinstance(query, Mapping):
            query_metadata_value = query
        elif cutoff_value is None and query is not None and not isinstance(query, Mapping):
            cutoff_value = query
        explicit_query_mark = (
            cutoff_value is not None
            or query_metadata_value is not None
            or isinstance(query, TimeMark)
        )
        qmark = query_time_mark(cutoff_value, query_metadata_value)
        if isinstance(query, TimeMark) and query_cutoff is None and query_metadata is None:
            qmark = query
            explicit_query_mark = True
        inferred_query_cutoff = False
        if qmark.unavailable:
            if not explicit_query_mark:
                # A missing query date still has an observable message-index
                # envelope, which is useful provenance (and is retained in
                # the query_mark field below).  It is a cutoff constraint,
                # not an additional Lebesgue/counting mass in the pairwise
                # Jaccard denominator.  Applying the envelope as a durative
                # measure to every observed instant would make a memory's
                # diagonal overlap depend on total conversation length.  Use
                # the neutral mark for the numerical kernel while recording
                # the inferred envelope explicitly.
                qmark = _query_cutoff_for_marks(ordered_marks)
                inferred_query_cutoff = True
                measure_query_mark = TimeMark()
            else:
                # Explicitly unknown/malformed query metadata follows the
                # documented neutral-unknown convention.
                measure_query_mark = TimeMark()
        else:
            measure_query_mark = qmark
        overlap = np.ones((count, count), dtype=np.float64)
        unavailable_pairs: list[tuple[str, str]] = []
        for row in range(count):
            for col in range(count):
                # An unavailable query mark is a neutral *query* factor; do
                # not pass it as a third unknown mark, because the latter
                # would short-circuit the whole pairwise overlap to one and
                # erase the temporal relation between two known memories.
                query_for_overlap = (
                    None if measure_query_mark.unavailable else measure_query_mark
                )
                overlap[row, col], detail = temporal_overlap_kernel(
                    ordered_marks[row], ordered_marks[col], query_for_overlap, return_diagnostics=True
                )
                if detail["time_unavailable"]:
                    unavailable_pairs.append((labels[row], labels[col]))
        kernel *= overlap
        time_diagnostics = {
            "temporal_measure": True,
            "angular_rank_kernel": True,
            "time_marks": {label: mark.public_dict() for label, mark in zip(labels, ordered_marks)},
            "time_unavailable": bool(unavailable_pairs),
            "time_unavailable_pairs": [list(pair) for pair in unavailable_pairs],
            "query_mark": qmark.public_dict(),
            "query_cutoff_inferred": inferred_query_cutoff,
        }
    row_sums = kernel.sum(axis=1)
    transition = np.zeros_like(kernel)
    zero_rows: list[int] = []
    for row, total in enumerate(row_sums):
        if total <= 0.0:
            transition[row, :] = 1.0 / count if count else 0.0
            zero_rows.append(row)
        else:
            transition[row, :] = kernel[row, :] / total
    diagnostics = {
        **time_diagnostics,
        "kernel": kernel,
        "K": kernel,
        "transition": transition,
        "P": transition,
        "row_sums": row_sums,
        "zero_row_uniformized": zero_rows,
        "zero_row_uniformized_ids": [labels[row] for row in zero_rows],
        "ids": labels,
        "transition_hash": hashlib.sha256(transition.tobytes()).hexdigest(),
    }
    # ``return_kernel`` is useful to audits which need to verify the T=0
    # element-wise regression before row normalization.  The historical
    # return value remains P_q unless an explicit richer form is requested.
    if return_details:
        return kernel, transition, diagnostics
    if return_kernel:
        if return_diagnostics:
            return kernel, diagnostics
        return kernel
    return (transition, diagnostics) if return_diagnostics else transition


def build_transition_kernel(
    vectors: np.ndarray,
    memories: Sequence[Any] | Mapping[str, Any] | None = None,
    query_cutoff: Any = None,
    *,
    query: Any | None = None,
    temporal_measure: bool = False,
    ids: Sequence[str] | None = None,
    query_metadata: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(K_q, P_q, diagnostics)`` in one explicit call.

    This companion avoids ambiguity in callers that need both the unnormalized
    relation and the stochastic transition while retaining the compact legacy
    ``build_transition_matrix`` API.
    """
    return build_transition_matrix(
        vectors,
        memories,
        query_cutoff,
        query=query,
        temporal_measure=temporal_measure,
        ids=ids,
        query_metadata=query_metadata,
        return_details=True,
    )


# Descriptive aliases used by downstream experiment notebooks.  Keeping these
# as thin wrappers avoids multiple implementations of the T operator while
# making the mathematical ``K_q``/``P_q`` terminology discoverable.
def temporal_kernel(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("temporal_measure", True)
    kwargs.setdefault("return_kernel", True)
    return build_transition_matrix(*args, **kwargs)


def transition_matrix(*args: Any, **kwargs: Any) -> Any:
    return build_transition_matrix(*args, **kwargs)


def build_time_mark(value: Any, timestamp: Any = None) -> TimeMark:
    """Singular convenience wrapper for integrations handling one memory."""
    if isinstance(value, TimeMark):
        return value
    if isinstance(value, SemanticTemporalMark):
        return _mark_from_metadata(value)
    if hasattr(value, "metadata"):
        return _mark_from_metadata(getattr(value, "metadata", {}) or {}, getattr(value, "timestamp", timestamp))
    if isinstance(value, Mapping):
        return _mark_from_metadata(value, timestamp)
    # A bare scalar passed to this singular helper is conventionally an
    # explicit time value.  Legacy memory objects take the separate metadata
    # path above, where their ``timestamp`` remains an observation fallback.
    point = _numeric(value if timestamp is None else timestamp)
    if point is not None:
        return TimeMark.instant(point, source="explicit")
    return TimeMark()


def time_overlap(*args: Any, **kwargs: Any) -> Any:
    return temporal_overlap_kernel(*args, **kwargs)


def compute_transition_matrix(*args: Any, **kwargs: Any) -> Any:
    return build_transition_matrix(*args, **kwargs)


def transition_hash(transition: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(transition, dtype=np.float64).tobytes()).hexdigest()


def time_marks_hash(marks: Mapping[str, TimeMark] | Sequence[TimeMark]) -> str:
    if isinstance(marks, Mapping):
        payload = {
            str(key): (
                value.public_dict()
                if isinstance(value, TimeMark)
                else _mark_from_metadata(value if isinstance(value, Mapping) else {}).public_dict()
            )
            for key, value in sorted(marks.items(), key=lambda item: str(item[0]))
        }
    else:
        payload = [
            mark.public_dict()
            if isinstance(mark, TimeMark)
            else _mark_from_metadata(mark if isinstance(mark, Mapping) else {}).public_dict()
            for mark in marks
        ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def bank_hash(vectors: np.ndarray, ids: Sequence[str] | None = None) -> str:
    value = np.asarray(vectors, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("bank vectors must be a two-dimensional matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError("bank vectors must be finite")
    labels = [str(item) for item in (list(ids) if ids is not None else range(len(value)))]
    if len(labels) != len(value):
        raise ValueError("bank IDs and vectors must have equal length")
    if len(set(labels)) != len(labels):
        raise ValueError("bank IDs must be unique")
    payload = {
        "shape": list(value.shape),
        "ids": labels,
        "vectors": value.tobytes().hex(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


__all__ = [
    "TimeMark",
    "TemporalMark",
    "TransitionCache",
    "build_time_marks",
    "build_time_mark",
    "query_time_mark",
    "temporal_overlap_kernel",
    "time_overlap",
    "angular_rank_kernel",
    "build_transition_matrix",
    "build_transition_kernel",
    "temporal_kernel",
    "transition_matrix",
    "compute_transition_matrix",
    "transition_hash",
    "time_marks_hash",
    "bank_hash",
]
