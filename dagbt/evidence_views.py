"""Whole-record, visible-ID evidence views for fusion reasoning v2.

The budget/round-robin policy adapts BridgeTree 16809bd to DAG support records.
This changes only a model's input view; callers retain the complete evidence
registry and support graph. Dropped evidence is unavailable, never irrelevant.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

from .reasoning import InputOverflow, ProtocolError


SPAN_FIELDS = frozenset({"source_span_ids", "guard_span_ids", "partial_span_ids", "span_ids",
                         "resolution_span_ids", "addressed_conflict_span_ids"})
ALTERNATIVE_FIELDS = frozenset({"alternative_ids", "target_alternative_ids"})
POLICY_VERSION = "dag_node_document_round_robin_atomic_support_v2"


def _ids(value, name):
    if (not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value)
            or len(set(value)) != len(value)):
        raise ProtocolError(name + " must contain unique nonempty IDs")
    return value


def _dependencies(alternative):
    return set(_ids(alternative.get("source_span_ids", []), "source_span_ids")
               + _ids(alternative.get("guard_span_ids", []), "guard_span_ids"))


def _established_route(node, alternative):
    """Only compiled, currently eligible support may establish a parent.

    An invalid/partial route can still be a useful audit subject. Visibility
    never upgrades that route to an established conclusion, and missing
    eligibility metadata is not an affirmative eligibility claim.
    """
    return (node.get("status") == "supported"
            and alternative.get("eligible") is True
            and alternative.get("semantic_status") == "supported"
            and not alternative.get("invalidated_by")
            and not alternative.get("disputed_by"))


def _span_references(value, field=None):
    if field in SPAN_FIELDS:
        return set(_ids(value, field))
    if isinstance(value, Mapping):
        return set().union(*(_span_references(item, key) for key, item in value.items()))
    if isinstance(value, list):
        return set().union(*(_span_references(item) for item in value))
    return set()


def _round_robin(records, node_order):
    """Stable contradiction-first node/document interleaving; no model score."""
    result, emitted = [], set()
    for contrary in (True, False):
        groups = OrderedDict((node, OrderedDict()) for node in node_order)
        for record in records:
            if (record.get("stance") == "contradiction") != contrary:
                continue
            nodes = record.get("node_ids") or ["__unassigned__"]
            for node in nodes:
                groups.setdefault(node, OrderedDict()).setdefault(record["doc_id"], []).append(record["id"])
        per_node = []
        for documents in groups.values():
            queue = []
            while any(documents.values()):
                for values in documents.values():
                    if values:
                        queue.append(values.pop(0))
            per_node.append(queue)
        while any(per_node):
            for queue in per_node:
                while queue and queue[0] in emitted:
                    queue.pop(0)
                if queue:
                    identifier = queue.pop(0)
                    emitted.add(identifier)
                    result.append(identifier)
    return result


@dataclass
class EvidenceView:
    data: dict
    audit: dict
    span_alias_to_id: dict
    alternative_alias_to_id: dict

    @property
    def visible_span_ids(self):
        return frozenset(self.span_alias_to_id.values())

    @property
    def visible_alternative_ids(self):
        return frozenset(self.alternative_alias_to_id.values())

    @property
    def visible_doc_ids(self):
        return frozenset(self.audit["visible_doc_ids"])

    def _convert(self, value, decode, field=None):
        table = (self.span_alias_to_id if field in SPAN_FIELDS else
                 self.alternative_alias_to_id if field in ALTERNATIVE_FIELDS else None)
        if table is not None:
            ids = _ids(value, field)
            conversion = table if decode else {real: alias for alias, real in table.items()}
            unknown = [identifier for identifier in ids if identifier not in conversion]
            if unknown:
                raise ProtocolError(f"{field} cites unknown or invisible IDs {unknown}; "
                                    f"allowed={list(conversion)}", category="evidence_relation")
            return [conversion[identifier] for identifier in ids]
        if isinstance(value, Mapping):
            return {key: self._convert(item, decode, key) for key, item in value.items()}
        if isinstance(value, list):
            return [self._convert(item, decode) for item in value]
        # Literal source text/claims/explanations are never rewritten merely
        # because they happen to contain a persisted ID or a short alias.
        return deepcopy(value)

    def decode(self, response):
        return self._convert(response, True)

    def encode(self, response):
        return self._convert(response, False)


def build_view(reasoner, operation, system, fixed, records, settings, event=None, schema=None, *,
               audit_nodes=None, node_order=None, evidence_key="evidence", document_costs=None,
               input_limit=None):
    """Return a payload/registry whose exact estimated wire fits its allowance.

    ``fixed`` excludes evidence. ``audit_nodes`` moves audit alternatives out
    of the irreducible header: each visible alternative retains every direct
    source/guard span and a complete preceding-parent route atomically.
    Resolver supported parents are kept only if at least one full support
    alternative and all recursively used parents remain visible.

    ``document_costs`` enables the flat selector's coupled
    candidate_doc_ids/document_token_counts fields; only visible documents
    survive. The full available/visible/omitted IDs are in ``view.audit`` and
    the emitted event, not repeated as an unbounded model-input header.
    """
    if not isinstance(fixed, Mapping) or evidence_key in fixed:
        raise ValueError("fixed must be a mapping excluding the evidence field")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    fixed = deepcopy(dict(fixed))
    records = deepcopy(list(records))
    by_id = OrderedDict()
    for record in records:
        if not isinstance(record, dict):
            raise ProtocolError("Evidence record must be an object")
        identifier, document = record.get("id"), record.get("doc_id")
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ProtocolError("Evidence records require distinct nonempty IDs")
        if not isinstance(document, str) or not document:
            raise ProtocolError("Evidence record requires a document ID")
        if "node_ids" in record:
            _ids(record["node_ids"], "node_ids")
        by_id[identifier] = record
    aliases = {identifier: f"e{index + 1}" for index, identifier in enumerate(by_id)}
    known_nodes = deepcopy(list(audit_nodes)) if audit_nodes is not None else []
    if audit_nodes is not None and "nodes" in fixed:
        raise ValueError("Pass audit nodes through audit_nodes, not fixed.nodes")
    parent_nodes = fixed.get("supported_parents", [])
    if not isinstance(parent_nodes, list):
        raise ProtocolError("supported_parents must be a list")
    prior = fixed.get("prior_state")
    all_nodes = [*known_nodes, *parent_nodes, *([prior] if isinstance(prior, dict) else [])]
    all_alternatives, alternative_nodes = OrderedDict(), {}
    for node in all_nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise ProtocolError("Support nodes require IDs")
        for alternative in node.get("alternatives", []):
            identifier = alternative.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ProtocolError("Alternative requires an ID")
            if identifier in all_alternatives and all_alternatives[identifier] != alternative:
                raise ProtocolError("Conflicting duplicate alternative ID")
            if not _dependencies(alternative) <= set(by_id):
                raise ProtocolError("Alternative cites unknown evidence")
            all_alternatives[identifier] = deepcopy(alternative)
            alternative_nodes[identifier] = node["id"]
    alt_aliases = {identifier: f"a{index + 1}" for index, identifier in enumerate(all_alternatives)}
    order = list(node_order or [node["id"] for node in all_nodes])
    ordered_ids = _round_robin(records, list(dict.fromkeys(order)))
    documents = list(dict.fromkeys(record["doc_id"] for record in records))
    if document_costs is not None:
        if not isinstance(document_costs, Mapping):
            raise ValueError("document_costs must be a mapping")
        if any(document not in document_costs for document in documents):
            raise ProtocolError("Missing visible document cost")

    def node_view(nodes, visible, allowed_alternatives=None, ancestors=None):
        result, accepted, included = [], set(ancestors or ()), set()
        for node in nodes:
            shown = []
            for alternative in node.get("alternatives", []):
                if allowed_alternatives is not None and alternative["id"] not in allowed_alternatives:
                    continue
                if not _dependencies(alternative) <= visible:
                    continue
                if not set(alternative.get("used_parent_ids", [])) <= accepted:
                    continue
                shown.append(deepcopy(alternative))
                included.add(alternative["id"])
            value = {**deepcopy(node), "alternatives": shown,
                     "omitted_alternative_count": len(node.get("alternatives", [])) - len(shown),
                     "established_in_view": any(_established_route(node, alt) for alt in shown)}
            if "partial_span_ids" in value:
                value["partial_span_ids"] = [sid for sid in value["partial_span_ids"] if sid in visible]
            if value["established_in_view"]:
                accepted.add(node["id"])
            result.append(value)
        return result, accepted, included

    def raw_payload(visible, chosen_alternatives):
        data = deepcopy(fixed)
        shown_ids = [identifier for identifier in ordered_ids if identifier in visible]
        data[evidence_key] = [deepcopy(by_id[identifier]) for identifier in shown_ids]
        shown_docs = list(dict.fromkeys(by_id[identifier]["doc_id"] for identifier in shown_ids))
        if document_costs is not None:
            data["candidate_doc_ids"] = shown_docs
            data["document_token_counts"] = {doc: document_costs[doc] for doc in shown_docs}
        included_alternatives = set()
        if audit_nodes is not None:
            nodes, _, included = node_view(known_nodes, visible, chosen_alternatives)
            data["nodes"] = nodes
            included_alternatives.update(included)
        if "supported_parents" in data:
            nodes, accepted, included = node_view(parent_nodes, visible)
            data["supported_parents"] = [node for node in nodes if node["id"] in accepted]
            # A wholly unavailable parent is absent from the actual payload;
            # its audit-only invalid route must not remain a visible alias.
            included_alternatives.update(alt["id"] for node in data["supported_parents"]
                                         for alt in node["alternatives"])
            data["omitted_supported_parent_ids"] = [node["id"] for node in parent_nodes if node["id"] not in accepted]
        else:
            accepted = set()
        if isinstance(prior, dict):
            nodes, _, included = node_view([prior], visible, ancestors=accepted)
            data["prior_state"] = nodes[0]
            included_alternatives.update(included)
        for field in ("known_conflicts", "previous_conflicts"):
            if field in data:
                original_conflicts = data[field]
                data[field] = [conflict for conflict in original_conflicts
                               if _span_references(conflict) <= visible]
                data["omitted_" + field + "_count"] = len(original_conflicts) - len(data[field])
        data["evidence_visibility"] = {
            "available_record_count": len(records), "visible_record_count": len(visible),
            "omitted_record_count": len(records) - len(visible),
            "available_document_count": len(documents), "visible_document_count": len(shown_docs),
            "selection_input_truncated": len(visible) < len(records),
            "note": "Omitted/unassessed evidence is unavailable, not irrelevant. Cite only displayed IDs.",
        }
        return data, included_alternatives

    def encode_input(value, visible, included, field=None, context=None):
        # Only structural ID fields are aliases. Exact fragments, quotations,
        # claims and natural-language explanations remain byte-for-byte text.
        if field in SPAN_FIELDS:
            return [aliases[sid] for sid in value if sid in visible]
        if field in ALTERNATIVE_FIELDS:
            # Historical conflicts can name replaced alternatives. Keep those
            # real IDs as historical metadata; decode still rejects their use
            # as a new citation because they are absent from this registry.
            return [alt_aliases[aid] if aid in included else aid for aid in value]
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if key == "id" and context == "evidence_record":
                    result[key] = aliases[item]
                elif key == "id" and context == "alternative":
                    result[key] = alt_aliases[item]
                else:
                    result[key] = encode_input(item, visible, included, key)
            return result
        if isinstance(value, list):
            item_context = ("evidence_record" if field == evidence_key else
                            "alternative" if field == "alternatives" else None)
            return [encode_input(item, visible, included, context=item_context) for item in value]
        return deepcopy(value)

    def payload_for(visible, chosen_alternatives):
        raw, included = raw_payload(visible, chosen_alternatives)
        return encode_input(raw, visible, included), included

    limit = settings["context_tokens"] - settings["reasoning_output_tokens"] - 8 - settings.get("input_margin", 256)
    if input_limit is not None:
        if isinstance(input_limit, bool) or not isinstance(input_limit, int) or input_limit < 1:
            raise ValueError("input_limit must be a positive integer")
        limit = min(limit, input_limit)
    empty, _ = payload_for(set(), set())
    empty_tokens = reasoner.estimate(operation, system, empty, schema)
    if empty_tokens > limit:
        if event:
            event({"event": "evidence_view_irreducible_overflow", "operation": operation,
                   "input_tokens_local": empty_tokens, "input_token_limit": limit})
        raise InputOverflow(f"{operation}: irreducible fixed fields use {empty_tokens} estimated tokens, limit {limit}")

    # Audit support records are atomic with their transitive preceding-parent
    # route. Deterministic first complete route follows the graph's own order;
    # no relevance score or discovery ancestry defines proof membership.
    node_by_id = {node["id"]: node for node in known_nodes}
    def support_atom(identifier, stack=()):
        if identifier in stack:
            raise ProtocolError("Cyclic support route in evidence view")
        alternative = all_alternatives[identifier]
        spans, alternatives = set(_dependencies(alternative)), {identifier}
        for parent in alternative.get("used_parent_ids", []):
            node = node_by_id.get(parent)
            route = None
            if node is not None:
                for candidate in node.get("alternatives", []):
                    if not _established_route(node, candidate):
                        continue
                    route = support_atom(candidate["id"], (*stack, identifier))
                    if route is not None:
                        break
            if route is None:
                return None
            spans.update(route[0]); alternatives.update(route[1])
        return spans, alternatives

    atoms = []
    if audit_nodes is not None:
        for identifier in ordered_ids:
            if by_id[identifier].get("stance") == "contradiction":
                atoms.append(({identifier}, set()))
        groups = []
        for node in known_nodes:
            group = []
            for alternative in node.get("alternatives", []):
                atom = support_atom(alternative["id"])
                if atom is not None:
                    group.append(atom)
            groups.append(group)
        while any(groups):
            for group in groups:
                if group:
                    atoms.append(group.pop(0))
    atoms.extend(({identifier}, set()) for identifier in ordered_ids)
    full_ids = set(by_id)
    full_alts = set(all_alternatives) if audit_nodes is not None else set()
    full, full_included = payload_for(full_ids, full_alts)
    before = reasoner.estimate(operation, system, full, schema)
    if before <= limit:
        kept, selected_alts, data, included = full_ids, full_alts, full, full_included
    else:
        kept, selected_alts = set(), set()
        for span_ids, alternative_ids in atoms:
            trial_ids, trial_alts = kept | span_ids, selected_alts | alternative_ids
            trial, _ = payload_for(trial_ids, trial_alts)
            if reasoner.estimate(operation, system, trial, schema) <= limit:
                kept, selected_alts = trial_ids, trial_alts
        data, included = payload_for(kept, selected_alts)
    after = reasoner.estimate(operation, system, data, schema)
    visible_ids = [identifier for identifier in ordered_ids if identifier in kept]
    shown_docs = list(dict.fromkeys(by_id[identifier]["doc_id"] for identifier in visible_ids))
    audit = {
        "event": "evidence_view_prepared", "operation": operation, "policy_version": POLICY_VERSION,
        "input_tokens_before": before, "input_tokens_after": after, "input_token_limit": limit,
        "truncated": len(kept) < len(records) or len(included) < len(all_alternatives),
        "available_span_ids": list(by_id), "visible_span_ids": visible_ids,
        "omitted_span_ids": [identifier for identifier in by_id if identifier not in kept],
        "available_alternative_ids": list(all_alternatives),
        "visible_alternative_ids": [identifier for identifier in all_alternatives if identifier in included],
        "omitted_alternative_ids": [identifier for identifier in all_alternatives if identifier not in included],
        "available_doc_ids": documents, "visible_doc_ids": shown_docs,
        "omitted_doc_ids": [document for document in documents if document not in shown_docs],
        "span_alias_to_id": {aliases[identifier]: identifier for identifier in visible_ids},
        "alternative_alias_to_id": {alt_aliases[identifier]: identifier for identifier in all_alternatives if identifier in included},
    }
    if event:
        event(deepcopy(audit))
    return EvidenceView(data, audit, audit["span_alias_to_id"], audit["alternative_alias_to_id"])
