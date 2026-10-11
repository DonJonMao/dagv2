"""Typed per-slot wire contracts, with the local reader remaining authoritative.

All evidence and responses are synthetic. JSON Schema checks use the optional
test-only jsonschema package; local-validator regressions need no extra package.
"""
from copy import deepcopy

import pytest

from dagbt.joint_reading import JointReader
from dagbt.reasoning import ProtocolError
from dagbt.residual import ProgramError, UNITS
from test_personamem_joint_protocol import fixture, observation, visible


@pytest.fixture
def schema_validator():
    jsonschema = pytest.importorskip("jsonschema")

    def build(data):
        schema = JointReader._schema(data)
        jsonschema.Draft202012Validator.check_schema(schema)
        return jsonschema.Draft202012Validator(schema)

    return build


def typed_request(kind="set", requested=("history",), *, known=False):
    reader, engine, state, calls = fixture()
    state.program["variables"]["history"]["type"] = kind
    if known:
        stable = engine.mapper.sources["s1"]["source_id"]
        state.integrate([{**observation(sources=[stable]), "variable": "history"}])
        state.integrate([{**observation("suggestion-proof", [stable], "Play piano"),
                          "variable": "suggestion", "parent_versions": {"history": state.versions["history"]}}])
    data = reader._payload(engine.steps[1], "Use past activities", requested, ["s1", "s2"])
    assert not calls.requests
    return reader, engine, state, data


def answer(variable="history", value=None, *, stance="support", retract_ids=()):
    row = observation(value=value, stance=stance, retract_ids=list(retract_ids))
    # The shared fixture uses None as its default; preserve an explicit null.
    row["value"] = deepcopy(value)
    return {"slots": [{"variable": variable, "observations": [row], "unresolved": []}]}


VALID_VALUES = [
    ("bool", True), ("bool", False),
    ("entity", "fictional museum"), ("text", "  Preserve raw whitespace.  "),
    ("set", []), ("set", [{"event": "fictional event"}]),
    ("set", [1, 1, {"source_ids": ["application-id"]}, None]),
    ("record", {}), ("record", {"source_ids": ["application-id"]}),
    ("rule", {"variables": {}, "expression": {"op": "const", "value": True},
              "constraints": [], "rules_complete": True}),
    ("number", 0), ("number", -1.25),
    ("number", {"number": 3, "unit": "min"}),
    ("number", {"number": 3, "unit": None}),
    ("number", {"lower": 1, "upper": 2}),
    ("number", {"lower": None, "upper": None}),
    ("number", {"lower": 1, "upper": 2, "lower_closed": False, "upper_closed": True, "unit": "s"}),
]


INVALID_VALUES = [
    ("bool", 1), ("bool", "true"),
    ("entity", {"name": "fictional museum"}), ("text", ["text"]),
    ("set", {"event": "fictional event"}), ("set", "event"),
    ("record", [{"event": "fictional event"}]), ("rule", []),
    ("number", True), ("number", "3"), ("number", []),
    ("number", {"number": 3}),
    ("number", {"number": True, "unit": "s"}),
    ("number", {"number": 3, "unit": "invented-unit"}),
    ("number", {"number": 3, "unit": "s", "extra": 1}),
    ("number", {"lower": 1}), ("number", {"upper": 2}),
    ("number", {"lower": True, "upper": 2}),
    ("number", {"lower": 1, "upper": 2, "lower_closed": 1}),
    ("number", {"lower": 1, "upper": 2, "unit": "invented-unit"}),
    ("number", {"lower": 1, "upper": 2, "extra": 1}),
]


@pytest.mark.parametrize("kind,value", VALID_VALUES)
def test_declared_support_type_accepts_lossless_json_values(schema_validator, kind, value):
    _, _, _, data = typed_request(kind)
    result = answer(value=value)
    before = deepcopy(result)
    assert schema_validator(data).is_valid(result)
    assert result == before


@pytest.mark.parametrize("kind,value", INVALID_VALUES)
def test_declared_support_type_rejects_other_json_shapes(schema_validator, kind, value):
    _, _, _, data = typed_request(kind)
    assert not schema_validator(data).is_valid(answer(value=value))


@pytest.mark.parametrize("kind", ["bool", "entity", "text", "set", "record", "rule", "number"])
@pytest.mark.parametrize("stance", ["support", "partial", "unknown", "irrelevant"])
def test_null_is_reserved_for_non_support_stances(schema_validator, kind, stance):
    _, _, _, data = typed_request(kind)
    assert schema_validator(data).is_valid(answer(value=None, stance=stance)) is (stance != "support")


@pytest.mark.parametrize("stance", ["partial", "unknown", "irrelevant"])
def test_non_support_cannot_bind_an_otherwise_typed_value(schema_validator, stance):
    _, _, _, data = typed_request()
    assert not schema_validator(data).is_valid(answer(value=["piano"], stance=stance))


@pytest.mark.parametrize("unit", [*UNITS, None])
def test_all_supported_numeric_units_are_available(schema_validator, unit):
    _, _, _, data = typed_request("number")
    validator = schema_validator(data)
    assert validator.is_valid(answer(value={"number": 2, "unit": unit}))
    assert validator.is_valid(answer(value={"lower": 1, "upper": 2, "unit": unit}))


def test_mixed_slots_bind_value_shape_to_each_variable(schema_validator):
    _, _, _, data = typed_request(requested=("history", "suggestion"))
    validator = schema_validator(data)
    result = {"slots": [answer(value=[{"event": "piano"}])["slots"][0],
                        answer("suggestion", "Play piano")["slots"][0]]}
    assert validator.is_valid(result)
    swapped = deepcopy(result)
    swapped["slots"][0]["observations"][0]["value"] = "Play piano"
    swapped["slots"][1]["observations"][0]["value"] = [{"event": "piano"}]
    assert not validator.is_valid(swapped)


@pytest.mark.parametrize("targets", [[], ["invented-route"]])
def test_no_existing_route_disallows_contradiction(schema_validator, targets):
    _, _, _, data = typed_request()
    assert not schema_validator(data).is_valid(answer(stance="contradiction", retract_ids=targets))


@pytest.mark.parametrize("stance", ["support", "partial", "unknown", "irrelevant"])
def test_non_counter_cannot_retract_even_a_real_route(schema_validator, stance):
    _, _, _, data = typed_request(known=True)
    route = next(row["id"] for row in data["known_observations"] if row["variable"] == "history")
    value = ["piano"] if stance == "support" else None
    assert not schema_validator(data).is_valid(answer(value=value, stance=stance, retract_ids=[route]))


def test_counter_requires_nonempty_targets_for_its_own_variable(schema_validator):
    _, _, _, data = typed_request(requested=("history", "suggestion"), known=True)
    routes = {row["variable"]: row["id"] for row in data["known_observations"]}
    validator = schema_validator(data)
    other_slot = {"variable": "suggestion", "observations": [], "unresolved": []}
    for targets, expected in [([], False), (["invented-route"], False),
                              ([routes["suggestion"]], False), ([routes["history"]], True)]:
        result = answer(stance="contradiction", retract_ids=targets)
        result["slots"].append(other_slot)
        assert validator.is_valid(result) is expected
    result = answer(value=["piano"], stance="contradiction", retract_ids=[routes["history"]])
    result["slots"].append(other_slot)
    assert not validator.is_valid(result)


def test_repair_subset_uses_requested_variable_not_background_task(schema_validator):
    _, _, _, data = typed_request(requested=("history",), known=True)
    assert data["task"]["output_slot"] == "suggestion"
    route = data["known_observations"][0]["id"]
    data["repair_scope"] = {"failed": [{"variable": "history", "id": "counter-id", "observation": None}]}
    validator = schema_validator(data)
    assert validator.is_valid(answer(value=["piano"]))
    assert validator.is_valid(answer(stance="contradiction", retract_ids=[route]))
    assert not validator.is_valid(answer(value="Play piano"))
    assert not validator.is_valid(answer("suggestion", "Play piano"))


def test_schema_preserves_shared_fields_and_input_without_duplicating_each_slot():
    _, _, _, data = typed_request(requested=("history", "suggestion"), known=True)
    before = deepcopy(data)
    schema = JointReader._schema(data)
    item = schema["properties"]["slots"]["items"]
    assert item["properties"]["variable"]["enum"] == data["requested_slots"]
    assert {"variable", "observations", "unresolved"} <= set(item["required"])
    assert set(item["properties"]["observations"]["items"]["required"]) == set(observation())
    assert {branch["properties"]["variable"]["const"] for branch in item["oneOf"]} == set(data["requested_slots"])
    assert data == before


def test_empty_requested_slots_still_produce_a_valid_schema(schema_validator):
    _, _, _, data = typed_request(requested=())
    assert schema_validator(data).is_valid({"slots": []})


@pytest.mark.parametrize("value", [[], [{"event": "fictional event"}], [1, 1, {"source_ids": ["application-id"]}]])
def test_local_set_validator_keeps_whole_values_and_duplicate_members(value):
    reader, engine, state, _ = fixture()
    draft = observation(value=value)
    before = deepcopy(draft)
    row = reader._observation(draft, "history", visible(reader), engine.steps[0])
    state.integrate([row])
    assert row["value"] == value == state.valid()[0]["history"]
    assert draft == before
    assert len(state.routes) == 1


def test_local_validator_does_not_wrap_a_dictionary_into_a_set():
    reader, engine, state, _ = fixture()
    draft = observation(value={"event": "fictional event"})
    before = deepcopy(draft)
    with pytest.raises(ProgramError, match="declared type: set"):
        reader._observation(draft, "history", visible(reader), engine.steps[0])
    assert draft == before and not state.routes


def test_local_domain_validation_remains_authoritative():
    reader, engine, state, _ = fixture()
    state.program["variables"]["history"]["domain"] = [["piano"]]
    with pytest.raises(ProgramError, match="finite declared domain"):
        reader._observation(observation(value=["painting"]), "history", visible(reader), engine.steps[0])
    assert not state.routes


@pytest.mark.parametrize("stance", ["partial", "unknown", "irrelevant"])
def test_local_validator_rejects_non_support_value_without_dropping_it(stance):
    reader, engine, state, _ = fixture()
    draft = observation(value=["piano"], stance=stance)
    before = deepcopy(draft)
    with pytest.raises(ProtocolError, match="Non-support observation"):
        reader._observation(draft, "history", visible(reader), engine.steps[0])
    assert draft == before and not state.routes
