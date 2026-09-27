"""Source-ID mapping with finite local recovery and retained valid assessments."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from . import prompts
from .budget import BudgetExceeded
from .evidence_spans import build_source_spans
from .reasoning import InputOverflow, ProtocolError
from .support import make_span, validate_span
from .transport import digest


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


_TEXT = {"type": "string"}
_IDS = {"type": "array", "items": _TEXT}
MAP_SCHEMA = _object({"units": {"type": "array", "items": _object({
    "unit_id": _TEXT, "assessments": {"type": "array", "items": _object({
        "span_ids": _IDS, "node_id": _TEXT, "kind": {"type": "string", "enum": ["explicit", "implicit"]},
        "stance": {"type": "string", "enum": ["support", "partial", "contradiction"]},
        "claim": _TEXT, "entity_scope": _TEXT, "event_time": {"type": ["string", "null"]},
        "time_span_ids": _IDS, "reason": _TEXT})}, "irrelevance_reason": _TEXT})}})


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ProtocolError(name + " fields differ from the mapping schema", category="evidence_relation")


def _ids(value, name, *, empty=True):
    if (not isinstance(value, list) or (not empty and not value)
            or any(not isinstance(v, str) or not v for v in value) or len(value) != len(set(value))):
        raise ProtocolError(name + " must contain unique nonempty IDs", category="evidence_relation")
    return value


def _assessment_identity(fact):
    # Reason wording is diagnostic, not a replacement for a previously
    # failed semantic assessment. Whitespace edits do not create a new fact.
    fields = {k: deepcopy(fact[k]) for k in
              ("doc_id", "source_ids", "node_id", "kind", "stance", "claim", "entity_scope",
               "event_time", "time_source_ids")}
    for key in ("claim", "entity_scope"):
        fields[key] = " ".join(fields[key].split())
    return digest(fields)


class EvidenceMapper:
    def __init__(self, engine):
        self.engine = engine
        self.sources = {}             # Stable short alias -> exact source coordinates.
        self.source_aliases = {}      # Content-bound ID -> stable short alias.
        self.units = {}               # Stable unit ID -> source alias/chunk index.
        self.completed = defaultdict(set)
        self.unavailable = defaultdict(dict)
        self.attempts = defaultdict(int)  # (unit ID, node ID), never reset by map_pending.
        self.unit_facts = defaultdict(set)
        self.feedback = {}
        self.failed_slots = defaultdict(dict)  # (unit, frozen target IDs) -> unresolved assessment slots.
        self.node_results = defaultdict(dict)
        self.active_targets = {}

    @property
    def schema(self):
        return getattr(prompts, "MAP_SCHEMA", MAP_SCHEMA)

    def _node_ids(self):
        return [step["output_slot"] for step in self.engine.steps]

    def add_candidates(self, ids):
        e = self.engine
        for doc_id in ids:
            if doc_id not in e.docs:
                raise ProtocolError("Discovery returned invisible document " + str(doc_id))
            if doc_id in e.candidates:
                continue
            sources = build_source_spans(doc_id, e.docs[doc_id], min(400, e.s["max_quote_chars"]))
            e.candidates.append(doc_id)
            for source in sources:
                alias = "s" + str(len(self.sources) + 1)
                unit_id = "u" + str(len(self.units) + 1)
                if source["id"] in self.source_aliases:
                    raise ProtocolError("Source identity collision")
                self.source_aliases[source["id"]] = alias
                self.sources[alias] = {**deepcopy(source), "source_id": source["id"], "id": alias}
                chunk_index = len(e.chunks)
                self.units[unit_id] = {"unit_id": unit_id, "source_span_id": alias,
                                       "doc_id": doc_id, "chunk_index": chunk_index}
                e.chunks.append({"unit_id": unit_id, "source_span_id": alias, "doc_id": doc_id,
                                 "start": source["start"], "text": source["text"]})

    def _missing(self, unit_id):
        return set(self._node_ids()) - self.completed[unit_id] - set(self.unavailable[unit_id])

    def _targets(self, unit_id):
        missing = self._missing(unit_id)
        current = missing & set(self.active_targets.get(unit_id, ()))
        if not current:
            current = missing
            self.active_targets[unit_id] = tuple(sorted(current))
        return current

    def _sync_chunks(self):
        all_nodes = set(self._node_ids())
        for unit_id, unit in self.units.items():
            if all_nodes <= self.completed[unit_id]:
                self.engine.mapped_chunks.add(unit["chunk_index"])
            else:
                self.engine.mapped_chunks.discard(unit["chunk_index"])

    def _visible(self, unit_ids):
        visible = set()
        for unit_id in unit_ids:
            source = self.sources[self.units[unit_id]["source_span_id"]]
            visible.add(source["id"])
            for key in ("previous_span_id", "next_span_id"):
                alias = self.source_aliases.get(source.get(key))
                if alias is not None and self.sources[alias]["doc_id"] == source["doc_id"]:
                    visible.add(alias)
        return {alias: self.sources[alias] for alias in self.sources if alias in visible}

    def _payload(self, unit_ids, targets, feedback=None):
        nodes = set().union(*(targets[unit_id] for unit_id in unit_ids))
        data = {"original_question": self.engine.q["question"],
                "nodes": [deepcopy(step) for step in self.engine.steps if step["output_slot"] in nodes],
                "units": [{k: self.units[u][k] for k in ("unit_id", "source_span_id", "doc_id")}
                          | {"node_ids": sorted(targets[u])} for u in unit_ids],
                "source_spans": [{k: source[k] for k in
                    ("id", "doc_id", "start", "end", "text", "source_role", "source_message_indices")}
                    for source in self._visible(unit_ids).values()]}
        if feedback is not None:
            retained = {}
            failed = {}
            for u in unit_ids:
                retained[u] = [{"evidence_id": f, "node_id": self.engine.spans[f]["node_id"],
                                "claim": self.engine.spans[f]["claim"],
                                "span_ids": [self.source_aliases[s] for s in self.engine.spans[f]["source_ids"]]}
                               for f in sorted(self.unit_facts[u])
                               if self.engine.spans[f]["node_id"] in targets[u]]
                failed[u] = list(deepcopy(self.failed_slots[u, tuple(sorted(targets[u]))]).values())
            data["repair_scope"] = {
                "unit_ids": list(unit_ids), "validation_feedback": str(feedback)[:1200],
                "retained_evidence_ids": sorted(set().union(*(self.unit_facts[u] for u in unit_ids))),
                "retained_assessments": retained, "failed_assessment_slots": failed,
                "instruction": "Correct requested failed assessment slots. Each needs a new valid replacement; "
                               "repeating a retained assessment does not repair it. Valid earlier evidence is frozen."}
        return data

    def _fits(self, payload):
        count = self.engine.reasoner.estimate("map", prompts.MAP, payload, self.schema)
        s = self.engine.s
        return (count <= s["map_batch_tokens"] and
                count + s["reasoning_output_tokens"] + s.get("input_margin", 256) + 8 <= s["context_tokens"])

    def _assessment(self, item, unit_id, targets, visible):
        _keys(item, ("span_ids", "node_id", "kind", "stance", "claim", "entity_scope",
                     "event_time", "time_span_ids", "reason"), "assessment")
        for key in ("node_id", "kind", "stance", "claim", "entity_scope", "reason"):
            if not isinstance(item[key], str) or (key in {"node_id", "claim"} and not item[key].strip()):
                raise ProtocolError("Invalid assessment " + key, category="evidence_relation")
        if item["node_id"] not in targets:
            raise ProtocolError("Assessment node is outside this request", category="evidence_relation")
        if item["kind"] not in {"explicit", "implicit"} or item["stance"] not in {"support", "partial", "contradiction"}:
            raise ProtocolError("Invalid assessment kind/stance", category="evidence_relation")
        aliases = _ids(item["span_ids"], "span_ids", empty=False)
        unit = self.units[unit_id]
        own = unit["source_span_id"]
        if own not in aliases or set(aliases) - set(visible):
            raise ProtocolError("Assessment must cite own source and only visible source IDs", category="evidence_relation")
        allowed = set(self._visible([unit_id]))
        if set(aliases) - allowed or any(visible[a]["doc_id"] != unit["doc_id"] for a in aliases):
            raise ProtocolError("Assessment may add only adjacent source spans from the same document", category="evidence_relation")
        if any(not visible[a]["text"].strip() for a in aliases):
            raise ProtocolError("Empty source text cannot establish evidence", category="evidence_relation")
        time_ids = _ids(item["time_span_ids"], "time_span_ids")
        event_time = item["event_time"]
        if event_time is not None:
            if not isinstance(event_time, str) or not event_time.strip() or not time_ids or set(time_ids) - set(aliases):
                raise ProtocolError("Event time requires exact cited visible source IDs", category="evidence_relation")
        elif time_ids:
            raise ProtocolError("Unknown event time must not claim time sources", category="evidence_relation")
        ordered = sorted((visible[a] for a in aliases), key=lambda source: source["start"])
        fragments = [make_span(source["source_id"], unit["doc_id"], source["text"], self.engine.docs,
            start=source["start"], source_role=source["source_role"],
            source_message_indices=source["source_message_indices"],
            premise_group_ids=source["premise_group_ids"], source_alias=source["id"])
            for source in ordered]
        identity = {**{k: deepcopy(v) for k, v in item.items() if k not in {"span_ids", "time_span_ids"}},
                    "source_ids": [source["source_id"] for source in ordered],
                    "time_source_ids": sorted(visible[a]["source_id"] for a in time_ids), "doc_id": unit["doc_id"]}
        identifier = "ev_" + digest(identity)[:24]
        roles = {source["source_role"] for source in ordered}
        value = {**identity, "id": identifier, "assessment_id": identifier, "fragments": fragments,
                 "node_ids": [item["node_id"]], "raw_text_hash": ordered[0]["source_hash"],
                 "source_role": next(iter(roles)) if len(roles) == 1 else "ambiguous",
                 "source_message_indices": sorted({i for s in ordered for i in s["source_message_indices"]}),
                 "premise_group_ids": sorted({g for s in ordered for g in s["premise_group_ids"]}),
                 "time_fragments": [deepcopy(f) for f in fragments if f["source_alias"] in time_ids]}
        return validate_span(value, self.engine.docs)

    def _validate(self, raw, unit_ids, targets, visible):
        _keys(raw, ("units",), "map response")
        if not isinstance(raw["units"], list):
            raise ProtocolError("units must be a list", category="evidence_relation")
        rows = defaultdict(list)
        issues = []
        for row in raw["units"]:
            if not isinstance(row, dict) or row.get("unit_id") not in unit_ids:
                issues.append("Unknown or malformed unit row")
            else:
                rows[row["unit_id"]].append(row)
        for unit_id in unit_ids:
            own_rows = rows[unit_id]
            if len(own_rows) != 1:
                issues.append(unit_id + ": missing or duplicate unit row")
                continue
            row = own_rows[0]
            try:
                _keys(row, ("unit_id", "assessments", "irrelevance_reason"), "unit row")
                if not isinstance(row["assessments"], list) or not isinstance(row["irrelevance_reason"], str):
                    raise ProtocolError("Invalid assessments/irrelevance_reason", category="evidence_relation")
                retained_here = any(self.engine.spans[f]["node_id"] in targets[unit_id] for f in self.unit_facts[unit_id])
                if not row["assessments"] and (not row["irrelevance_reason"].strip() or retained_here):
                    raise ProtocolError("Empty mapping requires explicit irrelevance and cannot erase retained evidence", category="evidence_relation")
                if row["assessments"] and row["irrelevance_reason"].strip():
                    raise ProtocolError("Mapped evidence cannot also be marked irrelevant", category="evidence_relation")
                valid = True
                slots = self.failed_slots[unit_id, tuple(sorted(targets[unit_id]))]
                old_slots = set(slots)
                prior_facts = set(self.unit_facts[unit_id])
                prior_assessments = {_assessment_identity(self.engine.spans[f]) for f in prior_facts}
                replacements = []
                failed_now = set()
                for assessment_index, item in enumerate(row["assessments"]):
                    try:
                        fact = self._assessment(item, unit_id, targets[unit_id], visible)
                    except (ValueError, KeyError, TypeError) as exc:
                        valid = False
                        issues.append(unit_id + ": " + str(exc))
                        node = item.get("node_id") if isinstance(item, dict) else None
                        node = node if isinstance(node, str) and node in targets[unit_id] else None
                        # A changed but still invalid repair is another attempt
                        # at an existing obligation, not a new missing fact.
                        # Match at most one response item to each prior slot,
                        # respecting any already-known target node.
                        matches = [sid for sid in old_slots - failed_now
                                   if node is None or slots[sid]["node_id"] in {None, node}]
                        matches.sort(key=lambda sid: (slots[sid]["node_id"] != node, sid))
                        if matches:
                            slot_id = matches[0]
                            if slots[slot_id]["node_id"] is None and node is not None:
                                slots[slot_id]["node_id"] = node
                            slots[slot_id]["error"] = str(exc)[:300]
                        else:
                            slot_id = "failed_" + digest([unit_id, sorted(targets[unit_id]),
                                                         assessment_index, item])[:16]
                            slots[slot_id] = {"slot_id": slot_id, "node_id": node,
                                "allowed_node_ids": sorted(targets[unit_id]), "error": str(exc)[:300]}
                        failed_now.add(slot_id)
                        continue
                    self.engine.spans[fact["id"]] = fact
                    self.unit_facts[unit_id].add(fact["id"])
                    identity = _assessment_identity(fact)
                    if identity not in prior_assessments and identity not in {_assessment_identity(f) for f in replacements}:
                        replacements.append(fact)
                # Only new legal assessments on a later request can replace
                # prior failed slots; a duplicate retained row never hides a
                # missing correction. Match constrained node slots first.
                for slot_id in sorted(old_slots - failed_now, key=lambda sid: slots[sid]["node_id"] is None):
                    expected = slots[slot_id]["node_id"]
                    replacement = next((f for f in replacements if expected is None or f["node_id"] == expected), None)
                    if replacement is not None:
                        replacements.remove(replacement)
                        del slots[slot_id]
                if slots:
                    valid = False
                    issues.append(unit_id + ": unresolved failed assessment slots require new valid replacements")
                if valid:
                    self.completed[unit_id].update(targets[unit_id])
                    for node_id in targets[unit_id]:
                        has_fact = any(self.engine.spans[f]["node_id"] == node_id for f in self.unit_facts[unit_id])
                        self.node_results[unit_id][node_id] = {
                            "status": "evidence_mapped" if has_fact else "no_assessment_emitted",
                            "unit_irrelevance_reason": row["irrelevance_reason"]}
            except (ValueError, KeyError, TypeError) as exc:
                issues.append(unit_id + ": " + str(exc))
        self._sync_chunks()
        if issues:
            raise ProtocolError("; ".join(issues)[:1200], category="evidence_relation")
        return {"completed_unit_ids": list(unit_ids)}

    def _unavailable(self, unit_id, targets, reason):
        for node_id in targets:
            self.unavailable[unit_id][node_id] = reason
        self.engine.event({"event": "mapping_unit_unavailable", "unit_id": unit_id,
                           "node_ids": sorted(targets), "reason": reason})

    def map_pending(self, remaining_nodes=1):
        if type(remaining_nodes) is not int or remaining_nodes < 1:
            raise ValueError("remaining_nodes must be a positive integer")
        e, s = self.engine, self.engine.s
        self._sync_chunks()
        reserve = s["reserved_audit_calls"] + int(s["selection"] == "flat") + remaining_nodes
        available = max(0, e.ledger.remaining("llm") - reserve)
        quota = max(1, available // remaining_nodes) if available else 0
        max_attempts = 1 + s.get("max_repairs_per_request", 2)
        pending = []
        for unit_id in self.units:
            targets = self._missing(unit_id)
            exhausted = {n for n in targets if self.attempts[unit_id, n] >= max_attempts}
            if exhausted:
                self._unavailable(unit_id, exhausted, "per_unit_attempt_limit")
            if self._missing(unit_id):
                pending.append(unit_id)
        if not quota:
            if pending:
                e.event({"event": "mapping_deferred", "unmapped_chunks": len(e.chunks) - len(e.mapped_chunks),
                         "reason": "shared_budget_fairness"})
            return
        queue = []
        while pending:
            batch, targets = [], {}
            while pending:
                unit_id = pending[0]
                trial_targets = {**targets, unit_id: self._targets(unit_id)}
                if not self._fits(self._payload(batch + [unit_id], trial_targets)):
                    if not batch:
                        raise InputOverflow("One whole source unit, adjacent context, schema and fixed mapping fields exceed budget")
                    break
                batch.append(pending.pop(0))
                targets = trial_targets
            queue.append((batch, targets, None))
        while queue and quota > 0 and e.ledger.remaining("llm") > reserve:
            batch, targets, feedback = queue.pop(0)
            batch = [u for u in batch if self._missing(u) & targets[u]]
            if not batch:
                continue
            targets = {u: self._missing(u) & targets[u] for u in batch}
            for unit_id in list(batch):
                exhausted = {n for n in targets[unit_id] if self.attempts[unit_id, n] >= max_attempts}
                if exhausted:
                    self._unavailable(unit_id, exhausted, "per_unit_attempt_limit")
                    targets[unit_id] -= exhausted
                if not targets[unit_id]:
                    batch.remove(unit_id)
            if not batch:
                continue
            retry = any(self.attempts[u, n] for u in batch for n in targets[u])
            if retry and e.ledger.remaining("json_repairs") < 1:
                for u in batch:
                    self._unavailable(u, targets[u], "global_repair_budget")
                continue
            if retry and feedback is None:
                feedback = "; ".join(dict.fromkeys(self.feedback.get(u, "Correct only still-pending mapping units") for u in batch))
            payload = self._payload(batch, targets, feedback if retry else None)
            if not self._fits(payload):
                if feedback is not None:
                    # Shrink diagnostic text only; source and frozen nodes remain complete.
                    payload = self._payload(batch, targets, "Correct invalid unit/source/node references; retained evidence is frozen.")
                if not self._fits(payload):
                    if len(batch) > 1:
                        middle = len(batch) // 2
                        queue[:0] = [(batch[:middle], targets, feedback), (batch[middle:], targets, feedback)]
                        continue
                    raise InputOverflow("Whole mapping repair unit and fixed fields exceed budget")
            if retry:
                e.ledger.reserve("json_repairs", "map_repair")
            for unit_id in batch:
                for node_id in targets[unit_id]:
                    self.attempts[unit_id, node_id] += 1
            quota -= 1
            visible = self._visible(batch)
            try:
                e.reasoner.request("map_repair" if retry else "map", prompts.MAP, payload,
                    lambda raw: self._validate(raw, batch, targets, visible), self.schema,
                    reserve=0, extra_reserve=reserve)
            except BudgetExceeded:
                break
            except ProtocolError as exc:
                error = {"stage": "map", "unit_ids": list(batch), "type": type(exc).__name__,
                         "failure_category": exc.category, "error": str(exc)}
                e.errors.append(error)
                e.event({"event": "mapping_partial_failure", **error})
                for u in batch:
                    self.feedback[u] = str(exc)
                remaining = [u for u in batch if self._missing(u) & targets[u]]
                if exc.category == "refusal":
                    for u in remaining:
                        self._unavailable(u, targets[u], "refusal")
                elif remaining:
                    for u in list(remaining):
                        exhausted = {n for n in self._missing(u) & targets[u] if self.attempts[u, n] >= max_attempts}
                        if exhausted:
                            self._unavailable(u, exhausted, "per_unit_attempt_limit")
                        if not self._missing(u) & targets[u]:
                            remaining.remove(u)
                    split = exc.category in {"json_syntax", "json_wrapper", "output_truncated"} and len(remaining) > 1
                    if split:
                        middle = len(remaining) // 2
                        queue[:0] = [(remaining[:middle], targets, str(exc)), (remaining[middle:], targets, str(exc))]
                    elif remaining:
                        queue.insert(0, (remaining, targets, str(exc)))
                    e.event({"event": "mapping_recovery_queued", "unit_ids": remaining, "split_for_retry": split})
            else:
                e.event({"event": "evidence_mapped", "unit_ids": list(batch),
                         "chunk_indices": [self.units[u]["chunk_index"] for u in batch],
                         "spans": [deepcopy(e.spans[f]) for f in sorted(set().union(*(self.unit_facts[u] for u in batch)))]})
                newly_pending = [u for u in batch if self._missing(u)]
                if newly_pending:
                    queue.append((newly_pending, {u: self._targets(u) for u in newly_pending}, None))
        self._sync_chunks()
        if any(self._missing(u) for u in self.units):
            e.event({"event": "mapping_deferred", "unmapped_chunks": len(e.chunks) - len(e.mapped_chunks),
                     "reason": "shared_budget_fairness"})

    def diagnostics(self):
        nodes = set(self._node_ids())
        unmapped = [u for u in self.units if not nodes <= self.completed[u]]
        unavailable = [u for u in self.units if self.unavailable[u]]
        return {"mapping_incomplete": bool(unmapped), "unmapped_unit_ids": unmapped,
                "unavailable_unit_ids": unavailable, "source_span_registry": deepcopy(self.sources),
                "source_alias_to_id": {alias: source["source_id"] for alias, source in self.sources.items()},
                "unit_statuses": {u: {"mapped_node_ids": sorted(self.completed[u]),
                    "unavailable_nodes": dict(self.unavailable[u]),
                    "attempts_by_node": {n: self.attempts[u, n] for n in self._node_ids()},
                    "node_results": deepcopy(self.node_results[u]),
                    "failed_assessment_slots": [deepcopy(slot) for (owner, _), slots in self.failed_slots.items()
                                                if owner == u for slot in slots.values()],
                    "evidence_ids": sorted(self.unit_facts[u])} for u in self.units}}
