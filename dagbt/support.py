"""Evidence support compilation and bounded AND/OR closure selection.

This module checks provenance and structural consistency, not natural-language
entailment. ``supported`` is a model judgment which passes these checks. Search
navigation edges and planned parents never automatically become support edges.
All public data structures are JSON-serializable dictionaries.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import unicodedata


STATUSES = {"unknown", "partial", "supported", "ambiguous", "invalidated"}


class SupportError(ValueError):
    """Invalid evidence protocol; callers must log or repair, never silently drop."""


def text_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def normalized_answer(answer):
    return " ".join(unicodedata.normalize("NFKC", str(answer)).casefold().split())


def _passage(document):
    if hasattr(document, "passage"):
        return document.passage
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        if "passage" in document:
            return document["passage"]
        return "\n".join(x for x in (document.get("title", ""), document.get("text", "")) if x)
    raise SupportError("unsupported_document_type")


def _ids(value, field):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise SupportError(field + ": expected string ID list")
    if len(set(value)) != len(value):
        raise SupportError(field + ": duplicate ID")
    return value


def make_span(span_id, doc_id, quote, documents, start=None, event_time=None, **metadata):
    """Ground an exact quote; repeated strings require an explicit start offset."""
    if doc_id not in documents:
        raise SupportError("quote_document_not_visible: " + str(doc_id))
    raw = _passage(documents[doc_id])
    if not isinstance(quote, str) or not quote:
        raise SupportError("empty_exact_quote")
    if start is None:
        start = raw.find(quote)
        if start < 0:
            raise SupportError("quote_not_exact")
        if raw.find(quote, start + 1) >= 0:
            raise SupportError("ambiguous_quote_offset")
    result = {**metadata, "id": span_id, "doc_id": doc_id, "start": start,
              "end": start + len(quote), "exact_quote": quote,
              "raw_text_hash": text_hash(raw), "event_time": event_time}
    return validate_span(result, documents)


def validate_span(span, documents):
    """Return a normalized copy, preserving unknown event time as null."""
    span = deepcopy(span)
    if 'fragments' in span:
        for field in ('id', 'doc_id'):
            if not isinstance(span.get(field), str) or not span[field]:
                raise SupportError('invalid_span_' + field)
        fragments = span['fragments']
        if not isinstance(fragments, list) or not fragments:
            raise SupportError('empty_evidence_fragments')
        if any(not isinstance(f, dict) or 'fragments' in f or f.get('doc_id') != span['doc_id'] for f in fragments):
            raise SupportError('fragment_source_mismatch')
        normalized = [validate_span(fragment, documents) for fragment in fragments]
        if len({(f['start'], f['end']) for f in normalized}) != len(normalized):
            raise SupportError('duplicate_evidence_fragment')
        span['fragments'] = normalized
        raw_hash = text_hash(_passage(documents[span['doc_id']]))
        if span.get('raw_text_hash', raw_hash) != raw_hash:
            raise SupportError('quote_source_hash_changed: ' + span['id'])
        span['raw_text_hash'] = raw_hash
        span.setdefault('event_time', None)
        # Each fragment remains an exact contiguous quotation. Never pretend
        # separated excerpts form one continuous span or independent premises.
        if len(normalized) == 1:
            for field in ('start', 'end', 'exact_quote'):
                if field in span and span[field] != normalized[0][field]:
                    raise SupportError('fragment_summary_mismatch')
                span[field] = normalized[0][field]
        elif any(field in span for field in ('start', 'end', 'exact_quote')):
            raise SupportError('noncontiguous_evidence_has_no_single_quote')
        return span
    for field in ("id", "doc_id", "exact_quote"):
        if not isinstance(span.get(field), str) or not span[field]:
            raise SupportError("invalid_span_" + field)
    if span["doc_id"] not in documents:
        raise SupportError("quote_document_not_visible: " + span["doc_id"])
    raw = _passage(documents[span["doc_id"]])
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
        raise SupportError("quote_offsets_invalid: " + span["id"])
    if raw[start:end] != span["exact_quote"]:
        raise SupportError("quote_not_exact: " + span["id"])
    raw_hash = text_hash(raw)
    if span.get("raw_text_hash", raw_hash) != raw_hash:
        raise SupportError("quote_source_hash_changed: " + span["id"])
    span["raw_text_hash"] = raw_hash
    span.setdefault("event_time", None)
    return span


def _scope_key(scope):
    return json.dumps(scope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compile_graph(nodes, spans, requirements, documents, max_nodes=8, max_alternatives=2,
                  derivation_verifier=None):
    """Compile a topologically ordered, bounded support graph.

    Required node fields: id, answer, status, version, alternatives. Each support
    alternative names source_span_ids, guard_span_ids, used_parent_ids and exact
    used_parent_versions. Missing optional fields use explicit safe defaults.
    Alternative answers/scopes must agree; competing conclusions belong in an
    ambiguous node's separate conflict records, never in its OR alternatives.

    Requirement fields: id, necessary, terminal_node_ids, terminal_mode (all/any).
    These are supplied by the frozen question-only plan, not inferred from hits.
    """
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= max_nodes or max_nodes > 8:
        raise SupportError("support_node_bound")
    if not 1 <= max_alternatives <= 2:
        raise SupportError("support_alternative_bound")
    span_list = [validate_span(span, documents) for span in spans]
    span_map = {span["id"]: span for span in span_list}
    if len(span_map) != len(span_list):
        raise SupportError("duplicate_span_id")
    node_list, node_map, alt_ids = [], {}, set()
    all_node_ids = [node.get("id") for node in nodes]
    if any(not isinstance(x, str) or not x for x in all_node_ids) or len(set(all_node_ids)) != len(nodes):
        raise SupportError("duplicate_or_invalid_node_id")
    for raw_node in nodes:
        node = deepcopy(raw_node)
        node_id = node["id"]
        if node.get("status") not in STATUSES:
            raise SupportError("invalid_node_status: " + node_id)
        if node.get("answer") is not None and not isinstance(node["answer"], str):
            raise SupportError("invalid_node_answer: " + node_id)
        if type(node.get("version")) is not int or node["version"] < 0:
            raise SupportError("invalid_node_version: " + node_id)
        node.setdefault("planned_parent_ids", [])
        if any(x not in node_map for x in _ids(node["planned_parent_ids"], "planned_parent_ids")):
            raise SupportError("planned_parent_unknown_or_forward: " + node_id)
        alternatives = node.get("alternatives", [])
        if not isinstance(alternatives, list) or len(alternatives) > max_alternatives:
            raise SupportError("support_alternative_bound: " + node_id)
        scope_key = None
        for alternative in alternatives:
            aid = alternative.get("id")
            if not isinstance(aid, str) or not aid or aid in alt_ids:
                raise SupportError("duplicate_or_invalid_alternative_id")
            alt_ids.add(aid)
            alternative.setdefault("answer", node.get("answer"))
            alternative.setdefault("applicable_scope", node.get("applicable_scope"))
            alternative.setdefault("semantic_status", node["status"])
            alternative.setdefault("used_parent_ids", [])
            alternative.setdefault("used_parent_versions", {})
            alternative.setdefault("source_span_ids", [])
            alternative.setdefault("guard_span_ids", [])
            alternative.setdefault("invalidated_by", [])
            alternative.setdefault("disputed_by", [])
            if alternative["semantic_status"] not in STATUSES:
                raise SupportError("invalid_alternative_status: " + aid)
            if normalized_answer(alternative["answer"]) != normalized_answer(node.get("answer")):
                raise SupportError("alternative_conclusion_mismatch: " + aid)
            current_scope = _scope_key(alternative["applicable_scope"])
            if scope_key is not None and current_scope != scope_key:
                raise SupportError("alternative_scope_mismatch: " + aid)
            scope_key = current_scope
            parents = _ids(alternative["used_parent_ids"], "used_parent_ids")
            if any(pid not in node_map for pid in parents):
                raise SupportError("support_parent_unknown_forward_or_cycle: " + aid)
            versions = alternative["used_parent_versions"]
            if not isinstance(versions, dict) or set(versions) != set(parents):
                raise SupportError("parent_version_bindings_incomplete: " + aid)
            if any(type(version) is not int or version < 0 for version in versions.values()):
                raise SupportError("invalid_parent_version: " + aid)
            for field in ("source_span_ids", "guard_span_ids"):
                if any(sid not in span_map for sid in _ids(alternative[field], field)):
                    raise SupportError("unknown_source_span: " + aid)
            for field in ("invalidated_by", "disputed_by"):
                _ids(alternative[field], field)
            if alternative["semantic_status"] == "supported" and (
                not normalized_answer(node.get("answer") or "")
                or not (alternative["source_span_ids"] or alternative["guard_span_ids"] or parents)
            ):
                if not (normalized_answer(node.get('answer') or '') and derivation_verifier
                        and derivation_verifier(node, alternative)):
                    raise SupportError("supported_without_answer_or_evidence: " + aid)
            if ('derivation' in alternative or 'binding_derivations' in alternative) and (not derivation_verifier or not derivation_verifier(node, alternative)):
                raise SupportError('unvalidated_symbolic_derivation: ' + aid)
            alternative["structure_valid"] = True
        _ids(node.setdefault("partial_span_ids", []), "partial_span_ids")
        if any(sid not in span_map for sid in node["partial_span_ids"]):
            raise SupportError("unknown_partial_span: " + node_id)
        # Recompilation must preserve the resolver's semantic judgment separately
        # from a derived unavailable/ambiguous status caused by graph constraints.
        # A fresh resolver result must explicitly replace declared_status.
        node.setdefault("declared_status", node["status"])
        if node["declared_status"] not in STATUSES:
            raise SupportError("invalid_declared_status: " + node_id)
        node_list.append(node)
        node_map[node_id] = node
    req_list = deepcopy(requirements)
    seen_requirements = set()
    for requirement in req_list:
        rid = requirement.get("id")
        if not isinstance(rid, str) or not rid or rid in seen_requirements:
            raise SupportError("duplicate_or_invalid_requirement_id")
        seen_requirements.add(rid)
        if type(requirement.get("necessary")) is not bool:
            raise SupportError("requirement_necessary_must_be_boolean")
        terminals = _ids(requirement.get("terminal_node_ids"), "terminal_node_ids")
        if not terminals or any(node_id not in node_map for node_id in terminals):
            raise SupportError("requirement_terminal_missing")
        if requirement.get("terminal_mode") not in ("all", "any"):
            raise SupportError("requirement_terminal_mode")
    if not req_list:
        raise SupportError("frozen_requirements_required")
    visible = {span["doc_id"] for span in span_list}
    graph = {"schema_version": "dagbt_support_v1", "nodes": node_list,
             "spans": span_list, "requirements": req_list,
             "document_order": [doc_id for doc_id in documents if doc_id in visible],
             "conflicts": [], "revision": 0,
             "requirements_hash": text_hash(json.dumps(req_list, ensure_ascii=False, sort_keys=True)),
             "semantic_validation": "model_judgment_not_formal_entailment"}
    _refresh_statuses(graph)
    return graph


def _refresh_statuses(graph):
    nodes = {}
    changed = []
    for node in graph["nodes"]:
        old_status = node["status"]
        for alternative in node["alternatives"]:
            reasons = []
            if alternative["semantic_status"] != "supported":
                reasons.append("semantic_" + alternative["semantic_status"])
            if alternative["invalidated_by"]:
                reasons.append("invalidated")
            if alternative["disputed_by"]:
                reasons.append("unresolved_conflict")
            for pid in alternative["used_parent_ids"]:
                parent = nodes[pid]
                if parent["status"] != "supported":
                    reasons.append("parent_unavailable:" + pid)
                if alternative["used_parent_versions"][pid] != parent["version"]:
                    reasons.append("parent_version_changed:" + pid)
            alternative["eligible"] = not reasons
            alternative["ineligible_reasons"] = reasons
        if any(a["eligible"] for a in node["alternatives"]) and node["declared_status"] == "supported":
            node["status"] = "supported"
        elif node["declared_status"] == "ambiguous" or any(a["disputed_by"] for a in node["alternatives"]):
            node["status"] = "ambiguous"
        elif node["declared_status"] == "supported" or any(a["invalidated_by"] for a in node["alternatives"]):
            node["status"] = "invalidated"
        else:
            node["status"] = node["declared_status"]
        if node["status"] != "supported":
            for alternative in node["alternatives"]:
                if alternative["eligible"]:
                    alternative["eligible"] = False
                    alternative["ineligible_reasons"].append("node_" + node["status"])
        if old_status != node["status"]:
            changed.append(node["id"])
        nodes[node["id"]] = node
    return changed


def _all_support_docs(graph, node_id, alternative_id=None):
    """Conservative disputed group: include all explicitly used parent evidence."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    spans = {s["id"]: s for s in graph["spans"]}
    def visit(nid, only=None):
        found = set()
        for alternative in nodes[nid]["alternatives"]:
            if only is not None and alternative["id"] != only:
                continue
            found.update(spans[sid]["doc_id"] for sid in alternative["source_span_ids"] + alternative["guard_span_ids"])
            for parent in alternative["used_parent_ids"]:
                found.update(visit(parent))
        return found
    return visit(node_id, alternative_id)


def invalidate_support(graph, alternative_ids, conflict_span_ids=(), reason="conflict", *, disputed=False):
    """Copy-on-write local invalidation; a valid OR alternative remains usable.

    Use disputed=True for an unresolved contradiction. Its evidence is protected
    as a paired partial group; it never contributes complete coverage. This API
    does not decide semantic conflict scope: callers name the audited targets.
    Parent answers remain the same version when only proof alternatives change.
    """
    graph = deepcopy(graph)
    targets = set(alternative_ids)
    alternatives = {a["id"]: (n, a) for n in graph["nodes"] for a in n["alternatives"]}
    if not targets or not targets <= set(alternatives):
        raise SupportError("unknown_invalidation_target")
    span_map = {s["id"]: s for s in graph["spans"]}
    if any(sid not in span_map for sid in conflict_span_ids):
        raise SupportError("unknown_conflict_span")
    graph["revision"] += 1
    conflict_id = "conflict_" + str(graph["revision"])
    field = "disputed_by" if disputed else "invalidated_by"
    protected_docs = {span_map[sid]["doc_id"] for sid in conflict_span_ids}
    target_nodes = sorted({alternatives[aid][0]["id"] for aid in targets})
    for aid in sorted(targets):
        alternatives[aid][1][field].append(conflict_id)
        protected_docs.update(_all_support_docs(graph, alternatives[aid][0]["id"], aid))
    graph["conflicts"].append({"id": conflict_id, "target_alternative_ids": sorted(targets),
                               "source_span_ids": list(conflict_span_ids), "reason": reason,
                               "target_node_ids": target_nodes,
                               "target_conclusions": [{"node_id": alternatives[aid][0]["id"],
                                    "answer": alternatives[aid][0].get("answer"),
                                    "applicable_scope": alternatives[aid][1].get("applicable_scope")}
                                   for aid in sorted(targets)],
                               "protected_doc_ids": [doc_id for doc_id in graph["document_order"] if doc_id in protected_docs],
                               "resolution_status": "unresolved" if disputed else "invalidated"})
    status_changes = _refresh_statuses(graph)
    descendants = _affected_descendants(graph, targets)
    statuses = {node["id"]: node["status"] for node in graph["nodes"]}
    graph["last_update"] = {"target_alternative_ids": sorted(targets),
                            "status_changed_node_ids": status_changes,
                            "affected_support_descendant_ids": descendants,
                            "requires_semantic_recheck": [nid for nid in descendants if statuses[nid] != "supported"]}
    return graph


def resolve_conflict(graph, conflict_id, resolution_span_ids, addressed_conflict_span_ids,
                     reason, resolution_kind="scope_distinction"):
    """Record an explicitly evidenced semantic resolution, not inferred recency.

    This checks source provenance and that *all* opposing citations were addressed;
    the model/auditor remains responsible for truth of the proposed distinction.
    Resolver may then re-resolve affected nodes. Calling this does not silently
    mark a partial/ambiguous node supported or transfer old proofs to new answers.
    """
    graph = deepcopy(graph)
    conflict = next((c for c in graph["conflicts"] if c["id"] == conflict_id), None)
    if conflict is None or conflict["resolution_status"] != "unresolved":
        raise SupportError("unresolved_conflict_required")
    spans = {s["id"] for s in graph["spans"]}
    proof = _ids(list(resolution_span_ids), "resolution_span_ids")
    addressed = _ids(list(addressed_conflict_span_ids), "addressed_conflict_span_ids")
    if not proof or not set(proof) <= spans:
        raise SupportError("conflict_resolution_requires_visible_proof")
    if set(addressed) != set(conflict["source_span_ids"]):
        raise SupportError("conflict_resolution_must_address_all_opposing_quotes")
    if not isinstance(reason, str) or not reason.strip() or resolution_kind not in (
        "scope_distinction", "entity_distinction", "time_distinction", "source_correction", "retracted_claim"
    ):
        raise SupportError("explicit_conflict_resolution_reason_required")
    conflict.update(resolution_status="resolved", resolution={"source_span_ids": proof,
                    "addressed_conflict_span_ids": addressed, "reason": reason,
                    "kind": resolution_kind, "semantic_validation": "model_judgment"})
    for node in graph["nodes"]:
        for alternative in node["alternatives"]:
            if conflict_id in alternative["disputed_by"]:
                # The distinction/correction is a condition of this proof's
                # validity, so its exact raw evidence is part of the closure.
                alternative["guard_span_ids"] = list(dict.fromkeys(
                    alternative["guard_span_ids"] + proof))
            alternative["disputed_by"] = [cid for cid in alternative["disputed_by"] if cid != conflict_id]
    graph["revision"] += 1
    graph["last_update"] = {"resolved_conflict_id": conflict_id,
                            "status_changed_node_ids": _refresh_statuses(graph),
                            "requires_semantic_recheck": conflict.get("target_node_ids", [])}
    return graph


def _affected_descendants(graph, targets):
    affected, descendants = set(), []
    for node in graph["nodes"]:
        if any(a["id"] in targets for a in node["alternatives"]):
            affected.add(node["id"])
        elif any(set(a["used_parent_ids"]) & affected for a in node["alternatives"]):
            affected.add(node["id"])
            descendants.append(node["id"])
    return descendants


def update_node_version(graph, node_id, answer, version):
    """Invalidate old node proofs and stale descendant bindings after a new value.

    Engine must re-resolve affected nodes and recompile with new supports before
    treating the revised answer as supported. This deliberately avoids silently
    transferring an old proof to a changed answer.
    """
    graph = deepcopy(graph)
    node = next((n for n in graph["nodes"] if n["id"] == node_id), None)
    if node is None or type(version) is not int or version <= node["version"]:
        raise SupportError("new_node_version_required")
    if answer is not None and not isinstance(answer, str):
        raise SupportError("invalid_node_answer")
    node["answer"], node["version"], node["declared_status"] = answer, version, "partial"
    for alternative in node["alternatives"]:
        alternative["invalidated_by"].append("node_value_changed:" + str(version))
    graph["revision"] += 1
    graph["last_update"] = {"changed_node_id": node_id,
                            "status_changed_node_ids": _refresh_statuses(graph),
                            "requires_semantic_recheck": _affected_descendants(
                                graph, {a["id"] for a in node["alternatives"]})}
    return graph


def _protected_groups(graph):
    span_map = {s["id"]: s for s in graph["spans"]}
    alt_nodes = {a["id"]: n["id"] for n in graph["nodes"] for a in n["alternatives"]}
    groups = []
    for conflict in graph["conflicts"]:
        if conflict["resolution_status"] != "unresolved":
            continue
        docs = set(conflict.get("protected_doc_ids", []))
        docs.update(span_map[sid]["doc_id"] for sid in conflict["source_span_ids"])
        for aid in conflict["target_alternative_ids"]:
            if aid in alt_nodes:
                docs.update(_all_support_docs(graph, alt_nodes[aid], aid))
            elif not conflict.get("protected_doc_ids"):
                raise SupportError("historical_conflict_missing_source_snapshot")
        groups.append({"id": conflict["id"], "doc_ids": sorted(docs),
                       "kind": "disputed", "node_ids": conflict.get("target_node_ids") or
                       sorted({alt_nodes[aid] for aid in conflict["target_alternative_ids"] if aid in alt_nodes})})
    return groups


def with_navigation_closure(graph, navigation_parents, documents):
    """Ablation only: force first-discovery ancestors into the final context.

    ``navigation_parents`` maps each discovered document to the real passages
    used by its frozen first discovery probe. It is not a support relation and
    never changes node semantic status, AND/OR alternatives, or coverage rules.
    The engine applies this only to final selection, not to parent grounding or
    subsequent discovery, so this isolates the cost of retaining navigation.
    """
    graph = deepcopy(graph)
    if not isinstance(navigation_parents, dict):
        raise SupportError("navigation_parent_mapping_required")
    known = set(documents)
    parents = {}
    for candidate, source_ids in navigation_parents.items():
        if candidate not in known or not set(_ids(source_ids, "navigation_source_ids")) <= known:
            raise SupportError("navigation_document_not_visible")
        if candidate in source_ids:
            raise SupportError("navigation_first_discovery_cycle")
        parents[candidate] = list(source_ids)
    if any(source not in parents for sources in parents.values() for source in sources):
        raise SupportError("navigation_source_first_discovery_missing")
    ancestors, visiting = {}, set()
    def visit(candidate):
        if candidate in visiting:
            raise SupportError("navigation_first_discovery_cycle")
        if candidate in ancestors:
            return ancestors[candidate]
        visiting.add(candidate)
        result = set(parents[candidate])
        for source in parents[candidate]:
            result.update(visit(source))
        visiting.remove(candidate)
        ancestors[candidate] = result
        return result
    for candidate in parents:
        visit(candidate)
    used = set(graph["document_order"]) | set(parents)
    graph["document_order"] = [doc_id for doc_id in documents if doc_id in used]
    graph["navigation_closure"] = {"enabled": True, "policy": "frozen_first_discovery",
                                   "parents": parents,
                                   "ancestors": {doc_id: sorted(values) for doc_id, values in ancestors.items()},
                                   "dependency_claim": False}
    return graph


def select_support(graph, feasibility, max_states=10000, partial_groups=(), fill_partial=False):
    """Exhaustively select AND/OR supports within this compiled, finite graph.

    Objective: necessary terminal requirements, optional terminal requirements,
    fewer raw-context tokens, then stable IDs. No intermediate-node reward.
    Callback feasibility(ordered_doc_ids) -> {feasible: bool, token_count: int}
    must count the actual reader template and output reserve. Shared documents
    are counted once. A cached callback result is reused for identical unions.
    """
    if type(max_states) is not int or max_states < 1:
        raise SupportError("invalid_enumeration_budget")
    nodes = graph["nodes"]
    if len(nodes) > 8 or any(len(n["alternatives"]) > 2 for n in nodes):
        raise SupportError("uncompiled_support_bound")
    if text_hash(json.dumps(graph["requirements"], ensure_ascii=False, sort_keys=True)) != graph["requirements_hash"]:
        raise SupportError("frozen_requirements_changed")
    spans = {s["id"]: s for s in graph["spans"]}
    doc_order = graph["document_order"]
    doc_set = set(doc_order)
    protected = _protected_groups(graph)
    navigation = graph.get("navigation_closure", {})
    navigation_ancestors = navigation.get("ancestors", {}) if navigation.get("enabled") else {}
    partial = deepcopy(list(partial_groups)) + deepcopy(protected)
    for group in partial:
        if not isinstance(group.get("id"), str) or not group["id"]:
            raise SupportError("partial_group_id_required")
        if not set(_ids(group.get("doc_ids"), "partial_doc_ids")) <= doc_set:
            raise SupportError("partial_doc_not_validated")
    protected_sets = [set(group["doc_ids"]) for group in protected]
    cache = {}
    def ordered(docs):
        return [doc_id for doc_id in doc_order if doc_id in docs]
    def check(docs):
        key = tuple(ordered(docs))
        if key not in cache:
            result = feasibility(list(key))
            if not isinstance(result, dict) or type(result.get("feasible")) is not bool or type(result.get("token_count")) is not int or result["token_count"] < 0:
                raise SupportError("invalid_feasibility_callback")
            cache[key] = result
        return cache[key]
    # All contested raw documents enter context together, even if one also
    # supports an unrelated conclusion. Otherwise raw-only reader sees one side.
    def protect(docs):
        docs = set(docs)
        while True:
            before = len(docs)
            for doc_id in list(docs):
                docs.update(navigation_ancestors.get(doc_id, ()))
            for group in protected_sets:
                if docs & group:
                    docs.update(group)
            if len(docs) == before:
                return docs
    choices = [[None] + [a for a in node["alternatives"] if a.get("eligible")]
               for node in nodes]
    assignments_total = 1
    for options in choices:
        assignments_total *= len(options)
    best, best_key = None, None
    examined, structurally_valid, feasible_count = 0, 0, 0
    for assignment in product(*choices):
        if examined >= max_states:
            break
        examined += 1
        selected, closure, valid = {}, {}, True
        for node, alternative in zip(nodes, assignment):
            if alternative is None:
                continue
            if any(parent not in selected for parent in alternative["used_parent_ids"]):
                valid = False
                break
            docs = {spans[sid]["doc_id"] for sid in alternative["source_span_ids"] + alternative["guard_span_ids"]}
            for parent in alternative["used_parent_ids"]:
                docs.update(closure[parent])
            closure[node["id"]] = docs
            selected[node["id"]] = alternative["id"]
        if not valid:
            continue
        structurally_valid += 1
        covered = []
        for requirement in graph["requirements"]:
            present = [node_id in selected for node_id in requirement["terminal_node_ids"]]
            if (all(present) if requirement["terminal_mode"] == "all" else any(present)):
                covered.append(requirement["id"])
        support_docs = set().union(*closure.values()) if closure else set()
        docs = protect(support_docs)
        check_result = check(docs)
        if not check_result["feasible"]:
            continue
        feasible_count += 1
        necessary = sum(r["necessary"] for r in graph["requirements"] if r["id"] in covered)
        optional = len(covered) - necessary
        key = (-necessary, -optional, check_result["token_count"], tuple(sorted(docs)), tuple(sorted(selected.items())))
        if best_key is None or key < best_key:
            best_key = key
            best = {"selected_doc_ids": ordered(docs), "chosen_alternatives": selected,
                    "support_selected_doc_ids": ordered(support_docs),
                    "covered_requirement_ids": covered, "token_count": check_result["token_count"],
                    "necessary_covered": necessary, "optional_covered": optional,
                    "node_closures": {nid: ordered(ds) for nid, ds in closure.items()}}
    if best is None:
        best = {"selected_doc_ids": [], "chosen_alternatives": {}, "covered_requirement_ids": [],
                "support_selected_doc_ids": [],
                "token_count": check(set())["token_count"], "necessary_covered": 0, "optional_covered": 0,
                "node_closures": {}}
    best["base_selected_doc_ids"] = list(best["selected_doc_ids"])
    best["base_token_count"] = best["token_count"]
    best["partial_group_ids"] = []
    # Optional raw supplementation never changes structural coverage. Existing
    # complete supports cannot be displaced to make room for partial evidence.
    if fill_partial:
        selected_docs = set(best["selected_doc_ids"])
        for group in sorted(partial, key=lambda item: item["id"]):
            candidate = protect(selected_docs | set(group["doc_ids"]))
            result = check(candidate)
            if result["feasible"]:
                selected_docs = candidate
                best["token_count"] = result["token_count"]
                best["partial_group_ids"].append(group["id"])
        best["selected_doc_ids"] = ordered(selected_docs)
    exhaustive = examined == assignments_total
    necessary_total = sum(r["necessary"] for r in graph["requirements"])
    best["complete_required"] = best["necessary_covered"] == necessary_total
    best["feasible"] = check(set(best["selected_doc_ids"]))["feasible"]
    best["status"] = ("context_infeasible" if not best["feasible"] else
                      "supported" if best["complete_required"] and exhaustive else "partial")
    best["uncovered_requirement_ids"] = [r["id"] for r in graph["requirements"] if r["id"] not in best["covered_requirement_ids"]]
    best["protected_partial_groups"] = protected
    navigation_required = set().union(*(set(navigation_ancestors.get(doc_id, ()))
                                       for doc_id in best["selected_doc_ids"])) if best["selected_doc_ids"] else set()
    best["forced_navigation_doc_ids"] = ordered(navigation_required - set(best["support_selected_doc_ids"]))
    best["navigation_closure_enabled"] = bool(navigation.get("enabled"))
    best["diagnostics"] = {"assignments_total": assignments_total, "assignments_examined": examined,
                           "structurally_valid_assignments": structurally_valid,
                           "feasible_assignments": feasible_count, "unique_context_checks": len(cache),
                           "exhaustive": exhaustive,
                           "finite_graph_optimum": exhaustive and feasible_count > 0 and best["selected_doc_ids"] == best["base_selected_doc_ids"],
                           "base_support_finite_graph_optimum": exhaustive and feasible_count > 0,
                           "partial_supplementation_outside_optimization": fill_partial,
                           "objective": "necessary_terminal_coverage,optional_terminal_coverage,raw_tokens,stable_ids",
                           "semantic_correctness_guaranteed": False}
    if navigation.get("enabled"):
        best["diagnostics"]["navigation_policy"] = "frozen_first_discovery; final_context_only; not_semantic_support"
    return best
