"""Independent adversarial review of input-only durable-review text sharing.

All data are synthetic. The inverse below does not import production iterators,
normalizers, or unpack functions. Integration substitutes fixed offline request
counts and scripted HTTP-shaped replies; it never calls a tokenizer or model.
"""
from copy import deepcopy
import json
import random
import socket

import pytest

from dagbt import history_wire
from dagbt import review_wire as codec
from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from dagbt.reasoning import InputOverflow, ProtocolError
from test_personamem_joint_protocol import fixture, observation, visible


def exact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


TEXT = 'Literal history 🙂 e\u0301 "quoted" \\ path\nnext line; ' * 80
REASON = 'Grounded interpretation with unchanged source and temporal scope. ' * 36
RAW = ' { "slots" : [{"value":"\\u0061","value":"a"}], "tail": '
POOL = "shared_review_texts"
REFS = {"review_value_ref": "value", "review_reason_ref": "reason"}


# This is an independent declared protocol grammar, not production traversal.
ROW_PATHS = [
    ("supported_parents", "{*}"),
    ("known_observations", "[*]"),
    ("failed_batch_reviews", "[*]", "validated_observations", "[*]"),
]
for failed in (
    ("failed_batch_reviews", "[*]", "failed", "[*]"),
    ("repair_scope", "failed", "[*]"),
):
    ROW_PATHS.extend([
        failed + ("observation",),
        failed + ("slot", "observations", "[*]"),
        *[failed + (name, "[*]") for name in (
            "observations", "replacement_observations", "required_counter_observations")],
    ])
for whole in (
    ("failed_batch_reviews", "[*]", "whole_value_reviews", "[*]"),
    ("repair_scope", "whole_value_reviews", "[*]"),
):
    ROW_PATHS.extend(whole + (name, "[*]") for name in (
        "draft_observations", "known_observations"))


def walk_declared(node, path):
    if not path:
        if type(node) is dict:
            yield node
        return
    part, rest = path[0], path[1:]
    if part == "[*]" and type(node) is list:
        for item in node:
            yield from walk_declared(item, rest)
    elif part == "{*}" and type(node) is dict:
        for item in node.values():
            yield from walk_declared(item, rest)
    elif type(node) is dict and part in node:
        yield from walk_declared(node[part], rest)


def declared_entries(wire):
    for path in ROW_PATHS:
        if path[0] == "known_observations" and "known_observation_columns" in wire:
            continue
        fields = {"value"} if path[0] == "supported_parents" else {"value", "reason"}
        for row in walk_declared(wire, path):
            yield row, fields


def declared_rows(wire):
    for row, _ in declared_entries(wire):
        yield row


def independent_inverse(wire):
    """Resolve only this codec's own fields, once, preserving key order."""
    result = deepcopy(wire)
    if POOL not in result:
        if any(set(row) & set(REFS) for row in declared_rows(result)):
            raise ValueError("orphan review reference")
        return result
    pool = result.pop(POOL)
    if type(pool) is not dict or not pool or any(
        type(k) is not str or type(v) is not str for k, v in pool.items()
    ):
        raise ValueError("invalid literal text pool")
    for row, allowed in declared_entries(result):
        rebuilt = {}
        for key, value in row.items():
            if key not in REFS:
                rebuilt[key] = value
                continue
            field = REFS[key]
            if field not in allowed or field in row or type(value) is not str or value not in pool:
                raise ValueError("ambiguous or unknown reference")
            # The retrieved string is never parsed or recursively interpreted.
            rebuilt[field] = pool[value]
        row.clear()
        row.update(rebuilt)
    return result


def put_row(wire, path, row):
    node = wire
    for index, part in enumerate(path):
        last = index == len(path) - 1
        following = path[index + 1] if not last else None
        if part == "[*]":
            if not node:
                node.append(deepcopy(row) if last else {})
            if last:
                node[-1] = deepcopy(row)
            else:
                node = node[-1]
        elif part == "{*}":
            node.setdefault("fixture-parent", deepcopy(row) if last else {})
            if last:
                node["fixture-parent"] = deepcopy(row)
            else:
                node = node["fixture-parent"]
        elif last:
            node[part] = deepcopy(row)
        else:
            node = node.setdefault(part, [] if following == "[*]" else {})


def fake_row(identifier="r", *, value=TEXT, reason=REASON):
    return {
        "id": identifier, "variable": "memory", "stance": "support",
        "value": value, "source_ids": ["s1", "s2"],
        "entity_scope": "user", "time_scope": "earlier", "parent_versions": {"p": 2},
        "retract_ids": [], "reason": reason, "version": 3,
        "revoked": [{"source_ids": ["s3"], "reason": "A real counter", "retract_ids": [identifier]}],
        "summary_revision": {"previous_version": 1, "supersedes_route_ids": ["old"]},
        "superseded_by": ["later"],
        "opaque_extra": {"review_allowed": True, "source_ids": ["literal-not-source"]},
    }


def all_paths_fixture():
    wire = {
        "grounded_subquestion": "Original unchanged query.",
        "source_spans": [{"id": "s1", "text": TEXT, "source_role": "unknown"}],
        "request_context": {"query_identity": "query", "program_identity": "program"},
        "evidence_obligations": [{"id": "pending", "source_ids": ["s3"]}],
        "public_options": None,
    }
    for index, path in enumerate(ROW_PATHS):
        put_row(wire, path, fake_row("role-" + str(index)))
    record = wire["failed_batch_reviews"][0]
    record.update(id="failed-original", variable="memory", review_allowed=False,
                  source_ids=["s1", "s2", "s3"], phase="history_evidence",
                  task_id="owner", query_identity="original-query", invalid_response=RAW)
    wire["repair_scope"]["invalid_response"] = RAW + "\n"
    return wire


@pytest.fixture(autouse=True)
def forbid_services(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent review must not contact services")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("dagbt.reasoning.estimate_evidence_tokens", lambda value: 0)


def check_roundtrip(original, expected=True):
    frozen = exact(original)
    encoded, applied = codec.pack_with_status(original)
    assert applied is expected
    assert exact(original) == frozen
    assert encoded is not original
    assert exact(independent_inverse(encoded)) == frozen
    assert exact(codec.unpack(encoded)) == frozen
    again, again_applied = codec.pack_with_status(original)
    assert again_applied is applied and exact(again) == exact(encoded)
    return encoded


def test_complete_explicit_roles_have_lossless_independent_inverse():
    original = all_paths_fixture()
    encoded = check_roundtrip(original)
    assert len(list(declared_rows(encoded))) == len(list(declared_rows(original)))
    assert encoded["failed_batch_reviews"][0]["review_allowed"] is False
    assert encoded["failed_batch_reviews"][0]["invalid_response"] == RAW
    assert encoded["repair_scope"]["invalid_response"] == RAW + "\n"
    assert encoded["source_spans"] == original["source_spans"]
    for (before, allowed), after in zip(declared_entries(original), declared_rows(encoded)):
        assert "value" not in after and "review_value_ref" in after
        if "reason" in allowed:
            assert "reason" not in after and "review_reason_ref" in after
        for field in set(before) - allowed:
            assert exact(after[field]) == exact(before[field])
    encoded["failed_batch_reviews"][0]["source_ids"].append("mutated-copy")
    assert original["failed_batch_reviews"][0]["source_ids"] == ["s1", "s2", "s3"]


@pytest.mark.parametrize("path", ROW_PATHS, ids=lambda p: "/".join(p))
def test_each_registered_path_can_share_without_supported_parent(path):
    original = {"known_observations": [fake_row("anchor")]}
    if path == ("known_observations", "[*]"):
        original["known_observations"].append(fake_row("second"))
    else:
        put_row(original, path, fake_row("separate-role"))
    encoded = check_roundtrip(original)
    assert all("review_value_ref" in row for row in walk_declared(encoded, path))
    if path[0] != "supported_parents":
        assert "supported_parents" not in encoded


@pytest.mark.parametrize("path", ROW_PATHS, ids=lambda p: "/".join(p))
@pytest.mark.parametrize("collision", ["review_value_ref", "review_reason_ref"])
def test_collision_in_uncompressed_row_rolls_back_entire_layer(path, collision):
    original = {"known_observations": [fake_row("a"), fake_row("b")]}
    tiny = fake_row("literal", value="tiny", reason="short")
    tiny[collision] = "untrusted-literal"
    if path == ("known_observations", "[*]"):
        original["known_observations"].append(tiny)
    else:
        put_row(original, path, tiny)
    before = exact(original)
    encoded, applied = codec.pack_with_status(original)
    assert applied is False and exact(encoded) == before and exact(original) == before
    assert POOL not in encoded


@pytest.mark.parametrize("pool", [{}, {"r0": "untrusted"}, None, [], "literal"])
def test_preexisting_top_pool_never_activates_encoder(pool):
    original = all_paths_fixture()
    original[POOL] = pool
    before = exact(original)
    result, applied = codec.pack_with_status(original)
    assert applied is False and exact(result) == before
    assert result is not original


def test_literal_reserved_dictionaries_and_typed_values_stay_opaque():
    original = all_paths_fixture()
    literals = [
        {"review_value_ref": "r0", "review_allowed": True},
        {"shared_review_texts": {"r0": "literal"}, "source_ids": ["not-evidence"]},
        {"nested": [{"shared": 0}, {"value_ref": "t0"}, {"text_ref": "t0"}]},
        True, 1, 1.0, False, 0, None, ["r0", {"review_reason_ref": "r1"}],
    ]
    for index, value in enumerate(literals):
        original["known_observations"].append(fake_row("literal-" + str(index), value=value))
    original["opaque_application"] = {
        "failed_batch_reviews": [{"validated_observations": [
            {"review_value_ref": "caller-text", "value": TEXT}]}],
        "observations": [{"value": TEXT, "reason": REASON}],
    }
    encoded = check_roundtrip(original)
    assert exact(encoded["opaque_application"]) == exact(original["opaque_application"])
    for before, after in zip(original["known_observations"][1:], encoded["known_observations"][1:]):
        assert exact(before["value"]) == exact(after["value"])


def test_raw_json_and_structured_draft_differences_remain_visible():
    original = all_paths_fixture()
    target = original["failed_batch_reviews"][0]["failed"][0]["slot"]["observations"][0]
    target["value"] = TEXT + " "
    target["reason"] = REASON + "\r\n"
    target["id"] = original["known_observations"][0]["id"]
    raw = ' { "value" : ' + json.dumps(TEXT, ensure_ascii=True) + ', "value":"DIFFERENT", "tail":'
    original["failed_batch_reviews"][0]["invalid_response"] = raw
    encoded = check_roundtrip(original)
    restored = independent_inverse(encoded)
    assert restored["failed_batch_reviews"][0]["invalid_response"] == raw
    assert target["value"] != restored["known_observations"][0]["value"]
    actual = encoded["failed_batch_reviews"][0]["failed"][0]["slot"]["observations"][0]
    assert actual["value"] == TEXT + " " and actual["reason"] == REASON + "\r\n"


def test_known_column_table_is_never_expanded_or_recursively_decoded():
    original = all_paths_fixture()
    columns = ["id", "value", "reason"]
    rows = [["first", {"shared": 0}, {"review_value_ref": "literal"}],
            ["second", {"shared": True}, {"shared": -1}]]
    original["known_observation_columns"] = columns
    original["known_observation_values"] = [TEXT, {"shared": 0}]
    original["known_observations"] = rows
    encoded = check_roundtrip(original)
    for key in ("known_observation_columns", "known_observation_values", "known_observations"):
        assert exact(encoded[key]) == exact(original[key])


def test_nonbeneficial_unique_short_and_nonstring_values_keep_old_wire():
    for original in (
        {"known_observations": [fake_row("a", value="short", reason="small")]},
        {"known_observations": [fake_row("a", value="unique" + TEXT, reason="unique" + REASON)]},
        {"known_observations": [fake_row("a", value={"x": TEXT}, reason=None),
                                fake_row("b", value={"x": TEXT}, reason=None)]},
    ):
        check_roundtrip(original, expected=False)


def test_reencoding_is_nonmutating_fallback_and_returned_objects_do_not_alias():
    original = all_paths_fixture()
    encoded = check_roundtrip(original)
    again, applied = codec.pack_with_status(encoded)
    assert applied is False and exact(again) == exact(encoded)
    again[POOL][next(iter(again[POOL]))] = "changed"
    assert exact(again) != exact(encoded)
    assert exact(original) == exact(all_paths_fixture())


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, None, [], {}, "missing"])
def test_invalid_reference_types_and_missing_ids_are_rejected(bad):
    encoded = check_roundtrip(all_paths_fixture())
    next(declared_rows(encoded))["review_value_ref"] = bad
    with pytest.raises(ValueError):
        independent_inverse(encoded)
    with pytest.raises(ValueError):
        codec.unpack(encoded)


@pytest.mark.parametrize("bad", ["both", "orphan", "empty_pool", "nonstring_pool", "list_pool"])
def test_ambiguous_reference_encodings_are_rejected(bad):
    encoded = check_roundtrip(all_paths_fixture())
    first = next(declared_rows(encoded))
    if bad == "both":
        first["value"] = "do not overwrite"
    elif bad == "orphan":
        encoded.pop(POOL)
    elif bad == "empty_pool":
        encoded[POOL] = {}
    elif bad == "nonstring_pool":
        encoded[POOL][first["review_value_ref"]] = {"review_value_ref": "recursive"}
    else:
        encoded[POOL] = list(encoded[POOL].values())
    with pytest.raises(ValueError):
        independent_inverse(encoded)
    with pytest.raises(ValueError):
        codec.unpack(encoded)


def test_seeded_unicode_exactness_without_normalization_or_substring_sharing():
    rng = random.Random(972310)
    alphabet = ["é", "e\u0301", "🙂", "\u200d", "\u2028", "\r", "\n", "\t", '"', "\\", " ", "x"]
    for _ in range(10):
        value = "".join(rng.choice(alphabet) for _ in range(800))
        original = {"known_observations": [fake_row("a", value=value), fake_row("b", value=value),
                                          fake_row("c", value=value + " ")]}
        encoded = check_roundtrip(original)
        assert encoded["known_observations"][2]["value"] == value + " "


def reader_fixture(monkeypatch):
    _, engine, initial, calls = fixture(max_repairs_per_request=2)
    program = deepcopy(initial.initial_program)
    program["variables"]["history"]["type"] = "text"
    state = FactState(program, initial.contract, engine.steps, "suggestion")
    reader = JointReader(engine, state)
    reader.current_identity = "synthetic-review"
    data = reader._payload(engine.steps[0], "Read the current scoped history", ["history"], ["s1"])
    row = observation("frozen-known", ["s1"], value=TEXT, reason=REASON)
    data["known_observations"] = [{**deepcopy(row), "variable": "history", "version": 1}]
    data["failed_batch_reviews"] = [{
        "id": "blocked", "variable": "history", "task_id": "other-owner",
        "phase": "history", "query_identity": "old-query", "batch_identity": "old-batch",
        "review_allowed": False, "source_ids": ["s1"],
        "failed": [{"variable": "history", "id": "*", "feedback": "Original failed slot",
                    "observation": None, "slot": {"variable": "history",
                    "observations": [deepcopy(row)], "unresolved": ["Actual pending issue"]}}],
        "validated_observations": [deepcopy(row)], "whole_value_reviews": [],
        "invalid_response": RAW,
    }]
    # Counts below are explicit deterministic stubs, not actual tokenization.
    counts = []
    def count(operation, request):
        counts.append((operation, deepcopy(request)))
        return {"count": 100, "token_count_is_estimate": True}
    monkeypatch.setattr(engine.reasoner, "_count_payload", count)
    return reader, engine, state, calls, data, counts


@pytest.mark.parametrize("operation", [
    "joint_read", "joint_read_repair", "semantic_compose", "semantic_compose_repair",
    "audit_joint", "audit_joint_repair"])
@pytest.mark.parametrize("mode", ["plain", "json_object", "json_schema"])
def test_estimate_and_actual_request_have_identical_complete_wire_and_schema(operation, mode, monkeypatch):
    reader, engine, state, calls, data, counts = reader_fixture(monkeypatch)
    engine.s["response_format"] = mode
    frozen = exact(data)
    state_before = exact(state.public_dict())
    settings = deepcopy(engine.s)
    identity = reader._selection_data_identity(data)
    schema = deepcopy(reader._schema(data))
    assert reader._estimate(operation, data) == 100
    wire, system = reader._prepare_wire(data)
    assert POOL in wire and system.count(codec.INSTRUCTION) == 1
    assert exact(independent_inverse(wire)) == exact(reader._wire_data(data))
    received = []
    def offline_get(stage, url, request, **options):
        received.append(deepcopy(request))
        engine.ledger.reserve("llm", "/".join(stage))
        return {"response_ref": "offline", "response": {"choices": [
            {"finish_reason": "stop", "message": {"content": '{"slots":[]}'}}]}}
    monkeypatch.setattr(calls, "get", offline_get)
    engine.reasoner.request(operation, system, wire, schema=schema)
    assert len(received) == 1 and all(payload == received[0] for _, payload in counts)
    assert exact(data) == frozen and exact(state.public_dict()) == state_before
    assert engine.s == settings and reader._selection_data_identity(data) == identity
    assert reader._schema(data) == schema
    assert "review_value_ref" not in exact(schema) and "review_reason_ref" not in exact(schema)
    assert engine.ledger.used["llm"] == 1 and engine.ledger.used["json_repairs"] == 0


def test_actual_repair_capacity_failure_retains_guard_and_does_not_charge(monkeypatch):
    reader, engine, state, calls, data, _ = reader_fixture(monkeypatch)
    before = exact(data), exact(state.public_dict()), deepcopy(engine.ledger.used)
    seen = []
    def oversized(operation, request):
        seen.append(request)
        return {"count": engine.s["context_tokens"] - request["max_tokens"] - engine.s["input_margin"] - 7}
    monkeypatch.setattr(engine.reasoner, "_count_payload", oversized)
    assert not reader._repair_fits("joint_read_repair", data, engine.steps[0], "synthetic-review", 1)
    assert POOL in json.loads(seen[0]["messages"][1]["content"])
    wire, system = reader._prepare_wire(data)
    with pytest.raises(InputOverflow):
        engine.reasoner.request("joint_read_repair", system, wire, schema=reader._schema(data))
    assert not calls.requests
    assert (exact(data), exact(state.public_dict()), engine.ledger.used) == before


@pytest.mark.parametrize("collision", ["top", "noncandidate"])
def test_prepare_collision_adds_no_review_instruction_or_permission(collision, monkeypatch):
    reader, engine, state, calls, data, _ = reader_fixture(monkeypatch)
    if collision == "top":
        data[POOL] = {"r0": "untrusted-caller"}
    else:
        data["failed_batch_reviews"][0]["validated_observations"].append(
            {"value": "short", "review_reason_ref": "untrusted-caller"})
    expected = reader._wire_data(data)
    wire, system = reader._prepare_wire(data)
    assert exact(wire) == exact(expected)
    assert codec.INSTRUCTION not in system
    assert wire["failed_batch_reviews"][0]["review_allowed"] is False
    assert not calls.requests and engine.ledger.used["llm"] == 0


def test_permission_and_output_contract_are_not_extended_by_input_references(monkeypatch):
    reader, engine, state, _, data, _ = reader_fixture(monkeypatch)
    original_visible = visible(reader)
    wire, _ = reader._prepare_wire(data)
    restored = independent_inverse(wire)
    reader.failed_batch_reviews["blocked"] = deepcopy(data["failed_batch_reviews"][0])
    failures = {}
    resolved = reader._failed_review_candidates(
        {"history": {"resolved_failed_batch_ids": ["blocked"]}}, [], restored, failures)
    assert not resolved and ("history", "*") in failures
    assert "blocked" in reader.failed_batch_reviews
    for ref, field in REFS.items():
        row = observation("bad-output", ["s1"], value=TEXT)
        row[ref] = "r0"
        with pytest.raises(ProtocolError, match="fields"):
            reader._observation(row, "history", original_visible, engine.steps[0])
        del row[field]
        with pytest.raises(ProtocolError, match="fields"):
            reader._observation(row, "history", original_visible, engine.steps[0])
    assert not state.routes and engine.ledger.used["llm"] == 0


def test_history_then_review_are_independent_layers_with_separate_instructions(monkeypatch):
    reader, engine, state, _, data, _ = reader_fixture(monkeypatch)
    data["supported_parents"] = {"history": {
        "type": "text", "value": TEXT, "version": 4, "source_ids": ["s1"],
        "scope": {"entity_scope": "user", "time_scope": "history"}}}
    data["grounded_subquestion"] = "Read: " + TEXT
    base = reader._wire_data(data)
    previous, history_applied = history_wire.pack_with_status(base)
    assert history_applied
    wire, system = reader._prepare_wire(data)
    assert "shared_history_texts" in wire and POOL in wire
    assert exact(independent_inverse(wire)) == exact(previous)
    assert system.count(history_wire.INSTRUCTION) == system.count(codec.INSTRUCTION) == 1
    # The review inverse alone must preserve all old-history references verbatim.
    assert wire["known_observations"][0]["value_ref"] == previous["known_observations"][0]["value_ref"]
    assert wire["grounded_subquestion_parts"] == previous["grounded_subquestion_parts"]
    assert engine.ledger.used["llm"] == 0 and not state.routes


def test_review_collision_does_not_disable_applied_history_or_activate_review(monkeypatch):
    reader, engine, state, _, data, _ = reader_fixture(monkeypatch)
    data["supported_parents"] = {"history": {
        "type": "text", "value": TEXT, "version": 4, "source_ids": ["s1"]}}
    data["grounded_subquestion"] = "Read: " + TEXT
    data[POOL] = {"r0": "This caller-owned table remains literal."}
    base = reader._wire_data(data)
    expected, applied = history_wire.pack_with_status(base)
    assert applied
    wire, system = reader._prepare_wire(data)
    assert exact(wire) == exact(expected)
    assert system.count(history_wire.INSTRUCTION) == 1
    assert codec.INSTRUCTION not in system
    assert wire[POOL] == data[POOL]
    assert engine.ledger.used["llm"] == 0 and not state.routes


@pytest.mark.parametrize("collision", ["shared_history_texts", "grounded_subquestion_parts",
                                       "parent_value_ref", "known_value_ref"])
def test_history_collision_remains_literal_when_independent_review_layer_applies(collision, monkeypatch):
    reader, engine, state, _, data, _ = reader_fixture(monkeypatch)
    data["supported_parents"] = {"history": {
        "type": "text", "value": TEXT, "version": 4, "source_ids": ["s1"]}}
    data["grounded_subquestion"] = "Read: " + TEXT
    if collision == "parent_value_ref":
        data["supported_parents"]["history"]["value_ref"] = "untrusted-literal"
    elif collision == "known_value_ref":
        data["known_observations"][0]["value_ref"] = "untrusted-literal"
    else:
        data[collision] = {"untrusted": "This field is not an encoding instruction."}
    base = reader._wire_data(data)
    untouched, history_applied = history_wire.pack_with_status(base)
    assert not history_applied and exact(untouched) == exact(base)
    wire, system = reader._prepare_wire(data)
    assert POOL in wire and codec.INSTRUCTION in system
    assert history_wire.INSTRUCTION not in system
    assert exact(independent_inverse(wire)) == exact(base)
    assert engine.ledger.used["llm"] == 0 and not state.routes


@pytest.mark.parametrize("malformation", [
    "parents_list", "known_nondict", "failed_batch_nondict", "validated_nonlist",
    "failed_nonlist", "failed_observation_nonobject", "failed_slot_nonobject",
    "slot_observations_nonlist", "whole_nonlist", "whole_group_nonobject",
    "known_columns_duplicate", "known_short_row", "orphan_known_values",
])
def test_unsupported_shapes_fall_back_without_repairing_or_dropping_input(malformation):
    original = all_paths_fixture()
    record = original["failed_batch_reviews"][0]
    failed = record["failed"][0]
    if malformation == "parents_list":
        original["supported_parents"] = []
    elif malformation == "known_nondict":
        original["known_observations"].append(["not-a-row"])
    elif malformation == "failed_batch_nondict":
        original["failed_batch_reviews"].append(None)
    elif malformation == "validated_nonlist":
        record["validated_observations"] = {"value": TEXT}
    elif malformation == "failed_nonlist":
        record["failed"] = {}
    elif malformation == "failed_observation_nonobject":
        failed["observation"] = "invalid original draft"
    elif malformation == "failed_slot_nonobject":
        failed["slot"] = ["original malformed slot"]
    elif malformation == "slot_observations_nonlist":
        failed["slot"]["observations"] = {"value": TEXT}
    elif malformation == "whole_nonlist":
        record["whole_value_reviews"] = {}
    elif malformation == "whole_group_nonobject":
        record["whole_value_reviews"].append("literal")
    elif malformation == "known_columns_duplicate":
        original["known_observation_columns"] = ["value", "value"]
        original["known_observations"] = [[TEXT, TEXT]]
    elif malformation == "known_short_row":
        original["known_observation_columns"] = ["id", "value"]
        original["known_observations"] = [["id-only"]]
    else:
        original["known_observation_values"] = [TEXT]
    before = exact(original)
    encoded, applied = codec.pack_with_status(original)
    assert applied is False and exact(encoded) == before and exact(original) == before
    with pytest.raises(ValueError):
        codec.unpack(encoded)


def test_nullable_producer_observation_and_slot_are_preserved():
    original = all_paths_fixture()
    failure = original["failed_batch_reviews"][0]["failed"][0]
    failure["observation"] = None
    failure["slot"] = None
    encoded = check_roundtrip(original)
    actual = encoded["failed_batch_reviews"][0]["failed"][0]
    assert actual["observation"] is None and actual["slot"] is None


def test_decoder_does_not_allow_reason_reference_on_parent_role():
    encoded = check_roundtrip(all_paths_fixture())
    parent = next(iter(encoded["supported_parents"].values()))
    parent["review_reason_ref"] = parent["review_value_ref"]
    parent.pop("reason")
    with pytest.raises(ValueError):
        independent_inverse(encoded)
    with pytest.raises(ValueError):
        codec.unpack(encoded)


def test_real_durable_recovery_charges_existing_budget_and_keeps_unresolved_obligation(monkeypatch):
    reader, engine, state, calls, _, counts = reader_fixture(monkeypatch)
    source_id = engine.mapper.sources["s1"]["source_id"]
    row = observation("saved-observation", [source_id], value=TEXT, reason=REASON)
    original_record = {
        "id": "durable-review", "variable": "history", "task_id": "history",
        "phase": "history", "query_identity": "old-query", "batch_identity": "old-batch",
        "source_ids": [source_id],
        "failed": [{"variable": "history", "id": "*", "feedback": "Original unresolved slot",
                    "observation": None, "slot": {"variable": "history",
                    "observations": [deepcopy(row)], "unresolved": ["Original issue"]}}],
        "validated_observations": [deepcopy(row)], "whole_value_reviews": [],
        "invalid_response": RAW, "seen_source_ids": [], "seen_parent_signatures": [],
    }
    reader.failed_batch_reviews["durable-review"] = deepcopy(original_record)
    reader.failed_obligations[("old-query", "old-batch")] = {"history"}
    frozen_sources = exact(engine.mapper.sources)
    settings = deepcopy(engine.s)
    received = []
    def offline_get(stage, url, request, **options):
        received.append((stage[0], deepcopy(request)))
        data = independent_inverse(json.loads(request["messages"][1]["content"]))
        assert data["failed_batch_reviews"][0]["review_allowed"] is True
        assert data["failed_batch_reviews"][0]["invalid_response"] == RAW
        assert POOL in json.loads(request["messages"][1]["content"])
        engine.ledger.reserve("llm", "/".join(stage))
        partial = observation("partial-current", ["s1"], stance="partial")
        partial["value"] = None
        raw = {"slots": [{"variable": "history", "observations": [partial],
                          "unresolved": ["Original evidence still does not resolve the issue"]}]}
        return {"response_ref": "offline-durable", "response": {"choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]}}
    monkeypatch.setattr(calls, "get", offline_get)
    reader.read(engine.steps[0], "Review the failed historical slot", ["history"], ["piano"],
                "new-source-window")
    assert len(received) == 1 and received[0][0] == "joint_read_repair"
    assert received[0] in counts
    assert engine.ledger.used["llm"] == 1 and engine.ledger.used["json_repairs"] == 1
    assert "durable-review" in reader.failed_batch_reviews
    current = reader.failed_batch_reviews["durable-review"]
    for key in ("failed", "validated_observations", "invalid_response", "source_ids"):
        assert exact(current[key]) == exact(original_record[key])
    assert reader.failed_obligations[("old-query", "old-batch")] == {"history"}
    assert "history" in reader.pending_variables() and not state.valid()[0]
    assert exact(engine.mapper.sources) == frozen_sources and engine.s == settings
