"""Atomic, visibility-bounded edits to an existing compiled support graph.

The reviewer supplies semantic judgments; this adapter checks exact quotations,
route identity, current parent bindings and conflict continuity. It never turns
raw-document supplementation into a proof or infers dependencies from documents.
"""
from __future__ import annotations

from copy import deepcopy
import json

from .evidence_spans import SourceSpanError, build_source_spans, document_source_metadata
from .support import (SupportError, compile_graph, invalidate_support, make_span,
                      normalized_answer, resolve_conflict, text_hash, update_node_version)


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties),
            "additionalProperties": False}


_TEXT = {"type": "string"}
_IDS = {"type": "array", "items": _TEXT}
_ANSWER = {"type": ["string", "null"]}
_ALTERNATIVE = _object({
    "source_span_ids": _IDS, "guard_span_ids": _IDS, "used_parent_ids": _IDS,
    "applicable_scope": _TEXT,
    "semantic_status": {"type": "string", "enum": ["supported", "partial"]},
})
REVIEW_SCHEMA = _object({
    "new_spans": {"type": "array", "items": _object({
        "id": _TEXT, "doc_id": _TEXT, "start": {"type": "integer", "minimum": 0},
        "quote": _TEXT, "node_id": _TEXT,
        "kind": {"type": "string", "enum": ["explicit", "implicit"]},
        "stance": {"type": "string", "enum": ["support", "partial", "contradiction"]},
        "claim": _TEXT, "entity_scope": _TEXT})},
    "invalidations": {"type": "array", "items": _object({
        "alternative_ids": _IDS, "source_span_ids": _IDS,
        "reason": _TEXT, "disputed": {"type": "boolean"}})},
    "resolutions": {"type": "array", "items": _object({
        "conflict_id": _TEXT, "resolution_span_ids": _IDS,
        "addressed_conflict_span_ids": _IDS, "reason": _TEXT,
        "resolution_kind": {"type": "string", "enum": ["scope_distinction", "entity_distinction",
            "time_distinction", "source_correction", "retracted_claim"]}})},
    "node_updates": {"type": "array", "items": _object({
        "node_id": _TEXT, "answer": _ANSWER,
        "status": {"type": "string", "enum": ["unknown", "partial", "supported", "ambiguous"]},
        "applicable_scope": _TEXT, "retained_alternative_ids": _IDS,
        "alternatives": {"type": "array", "items": _ALTERNATIVE},
        "unresolved_inputs": _IDS, "unresolved_guards": _IDS, "reason": _TEXT})},
    "supplemental_doc_ids": _IDS, "reason": _TEXT,
})


def _keys(value, schema, name):
    if (not isinstance(value, dict) or not set(schema["required"]) <= set(value)
            or set(value) - set(schema["properties"])):
        raise SupportError(name + ": fields differ from proof review schema")


def _text(value, name, *, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise SupportError(name + ": expected " + ("nonempty " if nonempty else "") + "string")
    return value


def _ids(value, name, *, nonempty=False):
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))):
        raise SupportError(name + ": expected unique nonempty string IDs")
    return value


def _visible(value, known, name):
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise SupportError(name + ": expected ID collection")
    ids = set(_ids(list(value), name))
    if not ids <= set(known):
        raise SupportError(name + ": unknown visible ID")
    return ids


def _digest(value):
    return text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))[:24]


def _recompile(graph, documents, max_nodes, max_alternatives):
    compiled = compile_graph(graph["nodes"], graph["spans"], graph["requirements"], documents,
                             max_nodes=max_nodes, max_alternatives=max_alternatives)
    # Compilation deliberately resets the conflict ledger. Review must retain
    # it, frozen navigation metadata, and unrelated engine diagnostics.
    result = deepcopy(graph)
    result.update({key: value for key, value in compiled.items()
                   if key not in {"conflicts", "revision", "document_order"}})
    known = set(graph.get("document_order", [])) | set(compiled["document_order"])
    result["document_order"] = [doc_id for doc_id in documents if doc_id in known]
    return result


def _scope(node):
    return node.get("applicable_scope", node["alternatives"][0].get("applicable_scope")
                    if node["alternatives"] else None)


def _proof_signature(node_id, alternative, spans):
    """Bind a route to source coordinates, not assessment aliases or wording.

    Adjacent/overlapping excerpts normalize to their exact raw-source union, so
    splitting an old quote or issuing it a new assessment ID cannot cleanse an
    invalidated proof. Genuinely different context or parent versions can.
    """
    def source_union(ids):
        grouped = {}
        for sid in ids:
            span = spans[sid]
            for fragment in span.get("fragments", [span]):
                key = (fragment["doc_id"], fragment["raw_text_hash"])
                grouped.setdefault(key, []).append((fragment["start"], fragment["end"]))
        result = []
        for (doc_id, source_hash), ranges in sorted(grouped.items()):
            merged = []
            for start, end in sorted(ranges):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            result.append([doc_id, source_hash, merged])
        return result

    return _digest({"node_id": node_id, "answer": normalized_answer(alternative["answer"]),
                    "applicable_scope": alternative["applicable_scope"],
                    "sources": source_union(alternative["source_span_ids"]),
                    "guards": source_union(alternative["guard_span_ids"]),
                    "parent_versions": alternative["used_parent_versions"]})


def apply_review(graph, payload, documents, *, visible_doc_ids, visible_alternative_ids,
                 visible_span_ids, max_nodes=8, max_alternatives=2, max_quote_chars=400):
    """Validate and apply one complete review without mutating any caller input.

    Existing invisible routes must be explicitly retained. New routes cite only
    visible evidence (or exact new quotations) and fully visible eligible parent
    proofs. A changed answer/scope requires fresh routes and a new node version.
    """
    _keys(payload, REVIEW_SCHEMA, "review")
    _text(payload["reason"], "review reason")
    if type(max_quote_chars) is not int or max_quote_chars < 1:
        raise SupportError("invalid_max_quote_chars")
    for field in ("new_spans", "invalidations", "resolutions", "node_updates"):
        if not isinstance(payload[field], list):
            raise SupportError(field + ": expected list")
        row_schema = REVIEW_SCHEMA["properties"][field]["items"]
        for row in payload[field]:
            _keys(row, row_schema, field)

    original_spans = {span["id"]: span for span in graph["spans"]}
    original_nodes = {node["id"]: node for node in graph["nodes"]}
    original_alternatives = {alt["id"]: alt for node in graph["nodes"] for alt in node["alternatives"]}
    visible_docs = _visible(visible_doc_ids, documents, "visible_doc_ids")
    visible_alts = _visible(visible_alternative_ids, original_alternatives, "visible_alternative_ids")
    visible_spans = _visible(visible_span_ids, original_spans, "visible_span_ids")
    supplements = _ids(payload["supplemental_doc_ids"], "supplemental_doc_ids")
    if not set(supplements) <= visible_docs:
        raise SupportError("supplemental_document_not_visible")

    working = deepcopy(graph)
    span_map = {span["id"]: span for span in working["spans"]}
    aliases, added_spans, source_cache = {}, [], {}
    for row in payload["new_spans"]:
        for field in ("id", "doc_id", "node_id", "kind", "stance", "claim", "entity_scope"):
            _text(row[field], "new_span " + field, nonempty=field != "entity_scope")
        alias, doc_id, node_id = row["id"], row["doc_id"], row["node_id"]
        if alias in aliases or alias in original_spans:
            raise SupportError("duplicate_or_existing_new_span_alias")
        if doc_id not in visible_docs:
            raise SupportError("new_span_document_not_visible")
        if node_id not in original_nodes:
            raise SupportError("new_span_unknown_node")
        if row["kind"] not in ("explicit", "implicit") or row["stance"] not in ("support", "partial", "contradiction"):
            raise SupportError("invalid_new_span_assessment")
        if type(row["start"]) is not int:
            raise SupportError("quote_offsets_invalid")
        if not isinstance(row["quote"], str) or not row["quote"].strip() or len(row["quote"]) > min(400, max_quote_chars):
            raise SupportError("invalid_or_oversize_review_quote")
        grounded = make_span(alias, doc_id, row["quote"], documents, start=row["start"])
        if doc_id not in source_cache:
            try:
                source_cache[doc_id] = (document_source_metadata(documents[doc_id]),
                                        build_source_spans(doc_id, documents[doc_id]))
            except SourceSpanError as exc:
                raise SupportError("invalid_review_source_metadata: " + str(exc)) from exc
        metadata, sources = source_cache[doc_id]
        pieces = [source for source in sources
                  if source["start"] < grounded["end"] and grounded["start"] < source["end"]]
        segments = [deepcopy(segment) for segment in metadata["source_segments"]
                    if segment["start"] < grounded["end"] and grounded["start"] < segment["end"]]
        roles = {source["source_role"] for source in pieces}
        identity = {key: deepcopy(value) for key, value in grounded.items() if key != "id"}
        identity.update(node_id=node_id, node_ids=[node_id], kind=row["kind"], stance=row["stance"],
                        claim=" ".join(row["claim"].split()), entity_scope=" ".join(row["entity_scope"].split()),
                        assessment_origin="proof_review", source_segments=segments,
                        source_role=next(iter(roles)) if len(roles) == 1 else "ambiguous",
                        source_message_indices=sorted({i for source in pieces for i in source["source_message_indices"]}),
                        premise_group_ids=sorted({g for source in pieces for g in source["premise_group_ids"]}),
                        source_ids=[source["id"] for source in pieces],
                        time_metadata=deepcopy(metadata.get("time", {})))
        canonical = "review_ev_" + _digest(identity)
        span = {**identity, "id": canonical, "assessment_id": canonical}
        if canonical in span_map and span_map[canonical] != span:
            raise SupportError("review_span_identity_collision")
        if canonical not in span_map:
            working["spans"].append(span)
            span_map[canonical] = span
            added_spans.append(canonical)
        aliases[alias] = canonical

    def refs(value, field, node_id=None, nonempty=False):
        result = []
        for reference in _ids(value, field, nonempty=nonempty):
            if reference in aliases:
                sid = aliases[reference]
            elif reference in visible_spans:
                sid = reference
            else:
                raise SupportError(field + ": span_not_visible")
            if node_id is not None:
                assessment_nodes = span_map[sid].get("node_ids", [span_map[sid].get("node_id")])
                if node_id not in assessment_nodes:
                    raise SupportError(field + ": evidence_not_mapped_to_node")
            result.append(sid)
        if len(set(result)) != len(result):
            raise SupportError(field + ": duplicate canonical span")
        return result

    working = _recompile(working, documents, max_nodes, max_alternatives)
    actions, added_alts, dropped_alts, updated_nodes = [], [], [], []
    for row in payload["invalidations"]:
        targets = _ids(row["alternative_ids"], "alternative_ids", nonempty=True)
        if not set(targets) <= visible_alts:
            raise SupportError("invalidation_target_not_visible")
        _text(row["reason"], "invalidation reason")
        if type(row["disputed"]) is not bool:
            raise SupportError("invalidation_disputed_must_be_boolean")
        proof = refs(row["source_span_ids"], "invalidation source_span_ids", nonempty=True)
        working = invalidate_support(working, targets, proof, row["reason"], disputed=row["disputed"])
        actions.append({"action": "invalidate", "alternative_ids": list(targets),
                        "conflict_id": working["conflicts"][-1]["id"]})
    for row in payload["resolutions"]:
        _text(row["conflict_id"], "conflict_id")
        _text(row["reason"], "resolution reason")
        affected = {alt["id"] for node in working["nodes"] for alt in node["alternatives"]
                    if row["conflict_id"] in alt["disputed_by"]}
        if not affected <= visible_alts:
            raise SupportError("resolution_would_modify_invisible_alternative")
        proof = refs(row["resolution_span_ids"], "resolution_span_ids", nonempty=True)
        opposing = refs(row["addressed_conflict_span_ids"], "addressed_conflict_span_ids")
        working = resolve_conflict(working, row["conflict_id"], proof, opposing,
                                   row["reason"], resolution_kind=row["resolution_kind"])
        actions.append({"action": "resolve", "conflict_id": row["conflict_id"]})

    updates = {}
    for row in payload["node_updates"]:
        _text(row["node_id"], "node_id")
        if row["node_id"] not in original_nodes or row["node_id"] in updates:
            raise SupportError("duplicate_or_unknown_review_node")
        updates[row["node_id"]] = row
    available_route_ids = set(visible_alts)
    visible_proof_ids = visible_spans | set(aliases.values())
    reserved_ids = set(original_alternatives)
    reserved_ids.update(aid for conflict in working["conflicts"] for aid in conflict["target_alternative_ids"])
    invalidated_proofs = set(graph.get("proof_review", {}).get("invalidated_proof_signatures", []))
    invalidated_proofs.update(_proof_signature(node["id"], alternative, span_map)
        for node in working["nodes"] for alternative in node["alternatives"] if alternative["invalidated_by"])

    def has_visible_proof(node_id, nodes):
        current = nodes[node_id]
        if current["status"] != "supported":
            return False
        return any(alt["id"] in available_route_ids and alt["eligible"]
                   and set(alt["source_span_ids"] + alt["guard_span_ids"]) <= visible_proof_ids
                   and all(has_visible_proof(parent, nodes) for parent in alt["used_parent_ids"])
                   for alt in current["alternatives"])

    # Freeze node order before applying edits: payload order never establishes a
    # dependency, and child routes bind the version after the parent's update.
    for node_id in original_nodes:
        if node_id not in updates:
            continue
        row = updates[node_id]
        nodes = {node["id"]: node for node in working["nodes"]}
        old = nodes[node_id]
        _text(row["reason"], "node update reason")
        _text(row["applicable_scope"], "applicable_scope", nonempty=False)
        if row["answer"] is not None:
            _text(row["answer"], "answer")
        if row["status"] not in ("unknown", "partial", "supported", "ambiguous"):
            raise SupportError("invalid_review_node_status")
        if row["status"] == "unknown" and row["answer"] is not None:
            raise SupportError("unknown_answer_must_be_null")
        missing = _ids(row["unresolved_inputs"], "unresolved_inputs")
        guards = _ids(row["unresolved_guards"], "unresolved_guards")
        retained = _ids(row["retained_alternative_ids"], "retained_alternative_ids")
        old_alts = {alt["id"]: alt for alt in old["alternatives"]}
        if not set(retained) <= set(old_alts):
            raise SupportError("retained_alternative_not_owned_by_node")
        hidden = set(old_alts) - visible_alts
        if not hidden <= set(retained):
            raise SupportError("cannot_drop_invisible_alternative")
        changed = (normalized_answer(old.get("answer") or "") != normalized_answer(row["answer"] or "")
                   or _scope(old) != row["applicable_scope"])
        hidden_state_change = bool(hidden and (row["status"] != old["declared_status"]
            or missing != old.get("unresolved_inputs", []) or guards != old.get("unresolved_guards", [])))
        if hidden and changed:
            raise SupportError("cannot_mutate_node_with_invisible_alternative")
        # An independently established new route may bypass gaps belonging to
        # an old hidden partial route. Mere node metadata edits may not do so;
        # verify the fresh route's actual eligibility after compilation below.
        if hidden_state_change and (row["status"] != "supported" or missing or guards
                                    or not row["alternatives"]):
            raise SupportError("cannot_mutate_node_with_invisible_alternative")
        if changed and retained:
            raise SupportError("changed_node_requires_fresh_alternatives")
        if not isinstance(row["alternatives"], list):
            raise SupportError("alternatives: expected list")
        if len(retained) + len(row["alternatives"]) > max_alternatives:
            raise SupportError("support_alternative_bound")
        if row["status"] == "supported" and (not row["answer"] or missing or guards or not (retained or row["alternatives"])):
            raise SupportError("supported_conclusion_has_unresolved_premises")
        if changed:
            working = update_node_version(working, node_id, row["answer"], old["version"] + 1)
            nodes = {node["id"]: node for node in working["nodes"]}
            old = nodes[node_id]
        replacement = deepcopy(old)
        replacement.update(answer=row["answer"], applicable_scope=row["applicable_scope"],
                           status=row["status"], declared_status=row["status"],
                           unresolved_inputs=list(missing), unresolved_guards=list(guards),
                           alternatives=[deepcopy(old_alts[aid]) for aid in retained])
        predecessors = list(nodes)[:list(nodes).index(node_id)]
        conflicts = [conflict for conflict in working["conflicts"]
                     if node_id in conflict.get("target_node_ids", [])]
        inherited_disputes = [conflict["id"] for conflict in conflicts if conflict["resolution_status"] == "unresolved"]
        resolution_guards = list(dict.fromkeys(sid for conflict in conflicts
            if conflict["resolution_status"] == "resolved" for sid in conflict["resolution"]["source_span_ids"]))
        fresh_route_ids = set()
        for proposal in row["alternatives"]:
            _keys(proposal, _ALTERNATIVE, "alternative")
            _text(proposal["applicable_scope"], "alternative applicable_scope",
                  nonempty=proposal["semantic_status"] == "supported")
            if proposal["applicable_scope"] != row["applicable_scope"]:
                raise SupportError("alternative_scope_mismatch")
            if proposal["semantic_status"] not in ("supported", "partial"):
                raise SupportError("invalid_alternative_semantic_status")
            sources = refs(proposal["source_span_ids"], "source_span_ids", node_id)
            conditions = refs(proposal["guard_span_ids"], "guard_span_ids", node_id)
            if not set(resolution_guards) <= visible_proof_ids:
                raise SupportError("inherited_resolution_proof_not_visible")
            conditions = list(dict.fromkeys(conditions + resolution_guards))
            parents = _ids(proposal["used_parent_ids"], "used_parent_ids")
            if any(parent not in predecessors or not has_visible_proof(parent, nodes) for parent in parents):
                raise SupportError("parent_unknown_unavailable_nonpreceding_or_invisible")
            alternative = {"answer": row["answer"], "applicable_scope": row["applicable_scope"],
                "semantic_status": proposal["semantic_status"], "source_span_ids": sources,
                "guard_span_ids": conditions, "used_parent_ids": parents,
                "used_parent_versions": {parent: nodes[parent]["version"] for parent in parents},
                "disputed_by": list(inherited_disputes), "invalidated_by": []}
            if _proof_signature(node_id, alternative, span_map) in invalidated_proofs:
                raise SupportError("cannot_recreate_invalidated_proof_with_new_identity")
            identity = {**alternative, "node_id": node_id, "node_version": replacement["version"]}
            for field in ("source_span_ids", "guard_span_ids", "used_parent_ids", "disputed_by"):
                identity[field] = sorted(identity[field])
            aid = "review_alt_" + _digest(identity)
            if aid in reserved_ids:
                raise SupportError("review_alternative_identity_already_exists_use_retained_id")
            alternative["id"] = aid
            reserved_ids.add(aid)
            available_route_ids.add(aid)
            fresh_route_ids.add(aid)
            added_alts.append(aid)
            replacement["alternatives"].append(alternative)
        removed = [aid for aid in old_alts if aid not in retained]
        dropped_alts.extend(removed)
        available_route_ids.difference_update(removed)
        # Old partial evidence remains diagnostic; new quotes do not become an
        # established route unless explicitly used in the alternatives above.
        working["nodes"] = [replacement if node["id"] == node_id else node for node in working["nodes"]]
        working["revision"] += 1
        working = _recompile(working, documents, max_nodes, max_alternatives)
        revised_nodes = {node["id"]: node for node in working["nodes"]}
        revised_alts = {alt["id"]: alt for alt in revised_nodes[node_id]["alternatives"]}
        materialized = set()
        if hidden_state_change:
            complete_fresh_route = any(alt["id"] in fresh_route_ids and alt["eligible"]
                and set(alt["source_span_ids"] + alt["guard_span_ids"]) <= visible_proof_ids
                and all(has_visible_proof(parent, revised_nodes) for parent in alt["used_parent_ids"])
                for alt in revised_alts.values())
            if not complete_fresh_route:
                raise SupportError("hidden_node_upgrade_requires_new_complete_visible_proof")
            for aid in hidden:
                if not old_alts[aid]["eligible"] and revised_alts[aid]["eligible"]:
                    # Legacy routes can look supported while blocked by their
                    # node's incomplete semantic judgment. Preserve that known
                    # incompleteness locally before the independent route raises
                    # the node's status. This is a mechanical narrowing, never a
                    # fresh semantic judgment about an unseen proof.
                    if old["declared_status"] == "supported":
                        raise SupportError("node_upgrade_would_activate_unreviewed_hidden_proof")
                    revised_alts[aid]["semantic_status"] = "partial"
                    materialized.add(aid)
                    actions.append({"action": "materialize_prior_node_incompleteness",
                        "node_id": node_id, "alternative_id": aid,
                        "prior_semantic_status": old_alts[aid]["semantic_status"],
                        "semantic_status": "partial", "prior_declared_status": old["declared_status"],
                        "unresolved_inputs": deepcopy(old.get("unresolved_inputs", [])),
                        "unresolved_guards": deepcopy(old.get("unresolved_guards", []))})
            if materialized:
                working = _recompile(working, documents, max_nodes, max_alternatives)
                revised_nodes = {node["id"]: node for node in working["nodes"]}
                revised_alts = {alt["id"]: alt for alt in revised_nodes[node_id]["alternatives"]}
        for aid in hidden:
            # Eligibility is derived from the complete graph, while all actual
            # provenance, bindings and conflict state are frozen. The sole
            # semantic exception materializes already-known incompleteness above.
            proof_fields = lambda alt: {key: value for key, value in alt.items()
                                        if key not in {"eligible", "ineligible_reasons", "structure_valid"}}
            expected = {**old_alts[aid], **({"semantic_status": "partial"} if aid in materialized else {})}
            if aid not in revised_alts or proof_fields(revised_alts[aid]) != proof_fields(expected):
                raise SupportError("review_modified_invisible_alternative")
        updated_nodes.append(node_id)
        actions.append({"action": "update_node", "node_id": node_id, "version_changed": changed,
                        "retained_alternative_ids": list(retained), "dropped_alternative_ids": removed})

    working = _recompile(working, documents, max_nodes, max_alternatives)
    working["supplemental_doc_ids"] = list(supplements)
    working["proof_review"] = {"reason": payload["reason"], "new_span_ids": dict(aliases),
        "added_span_ids": added_spans, "added_alternative_ids": added_alts,
        "dropped_alternative_ids": dropped_alts, "updated_node_ids": updated_nodes,
        "invalidated_proof_signatures": sorted(invalidated_proofs),
        "actions": actions, "semantic_validation": "model_judgment_not_formal_entailment"}
    return working
