"""Independent nonempty response-floor review; no model or tokenizer network.

The reference is a metering artifact, never an answer or an obligation release.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.evidence_spans import build_source_spans
from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from test_personamem_joint_protocol import observation, visible
from test_v4_summary_revision_review import OLD, summary_fixture


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('State-preserving floor review is strictly offline')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def setup_reader():
    reader, engine, state, calls, original = summary_fixture(
        context_tokens=16384, reasoning_output_tokens=4096)
    engine.s['max_repairs_per_request'] = 2
    reader.current_identity = 'state-preserving-floor'
    return reader, engine, state, calls, original


def payload(reader, engine, variables=None):
    return reader._payload(engine.steps[0], 'Read new history while preserving prior evidence',
                           variables or ['history'], ['s2'])


def meter(monkeypatch, reader, data, *, count=741):
    captured = []
    def exact_count(operation, wire):
        assert operation == 'joint_read_structure'
        assert len(wire['messages']) == 1 and wire['messages'][0]['role'] == 'assistant'
        captured.append(deepcopy(wire))
        return {'count': count, 'token_count_is_estimate': False}
    monkeypatch.setattr(reader.e.reasoner, '_count_payload', exact_count)
    result = reader._fresh_structure_floor('joint_read', data, output_token_cap=3000)
    reference = json.loads(captured[0]['messages'][0]['content']) if captured else None
    return result, reference, captured


def add_source(engine, alias, text=None):
    text = text or f'This is the original, unresolved experience recorded for {alias}.'
    name = 'memory-' + alias
    document = SimpleNamespace(doc_id=name, passage=text, metadata={
        'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                             'source_message_indices': [len(engine.docs)]}]})
    engine.docs[name] = document
    span = build_source_spans(name, document, 400)[0]
    engine.mapper.sources[alias] = {**span, 'id': alias, 'source_id': span['id']}
    engine.mapper.source_aliases[span['id']] = alias
    return span['id']


def add_obligation(reader, identifier, aliases, descriptions):
    reader.evidence_obligations[identifier] = {
        'id': identifier, 'variable': 'history', 'entity_scope': 'user', 'time_scope': 'history',
        'source_ids': [reader.e.mapper.sources[alias]['source_id'] for alias in aliases],
        'descriptions': list(descriptions), 'origins': ['an-earlier-reading'],
    }


def test_nonempty_metering_reference_preserves_the_complete_current_summary_and_exact_target(monkeypatch):
    reader, engine, state, calls, original = setup_reader()
    data = payload(reader, engine)
    before = (deepcopy(data), deepcopy(state.public_dict()), deepcopy(reader.obligation_snapshot()),
              deepcopy(engine.ledger.used), deepcopy(reader.batches), deepcopy(engine.spans))
    result, reference, captured = meter(monkeypatch, reader, data)
    assert result == 741 and set(reference) == {'slots'}
    slot, = reference['slots']
    assert slot['variable'] == 'history' and slot['unresolved'] == []
    supports = [row for row in slot['observations'] if row['stance'] == 'support']
    support, = supports
    assert support['value'] == OLD and support['source_ids'] == ['s1']
    assert support['entity_scope'] == 'user' and support['time_scope'] == 'history'
    assert support['parent_versions'] == {} and support['retract_ids'] == []
    assert support['summary_revision']['supersedes_route_ids'] == [original]
    assert support['summary_revision']['previous_version'] == state.versions['history']
    assert support['summary_revision']['reason'].strip()
    assert captured[0]['max_tokens'] == 3000
    assert (data, state.public_dict(), reader.obligation_snapshot(), engine.ledger.used,
            reader.batches, engine.spans) == before
    assert not calls.requests and not engine.reasoner.requests and engine.reasoner.sequence == 0


@pytest.mark.parametrize('case', ['no_route', 'conflicting_values', 'revoked', 'superseded', 'stale_parent'])
def test_no_valid_current_summary_cannot_be_replaced_by_an_empty_or_arbitrarily_selected_floor(case, monkeypatch):
    reader, engine, state, calls, original = setup_reader()
    if case == 'no_route':
        state.routes.clear()
    elif case == 'conflicting_values':
        different = observation('different-complete-history', ['s2'], value='A different whole history.')
        state.integrate([reader._observation(different, 'history', visible(reader, ['s2']), engine.steps[0])])
    elif case == 'revoked':
        state.routes[original]['revoked'] = [{'source_ids': [engine.mapper.sources['s2']['source_id']],
                                             'reason': 'The original statement was withdrawn.'}]
    elif case == 'superseded':
        state.routes[original]['superseded_by'] = ['a-retired-representation']
    else:
        state.routes[original]['parent_versions'] = {'suggestion': 999}
    assert 'history' not in state.valid()[0]
    data = payload(reader, engine)
    before = deepcopy(state.public_dict())
    result, reference, captured = meter(monkeypatch, reader, data)
    assert result is None and reference is None and not captured
    assert not calls.requests and state.public_dict() == before


def test_missing_requested_slot_never_disappears_from_the_response_workload(monkeypatch):
    reader, engine, state, calls, _ = setup_reader()
    state.program['variables']['missing_history'] = {
        'type': 'text', 'demand': 'history', 'evidence_semantics': 'personal_history_summary',
        'scope': {'entity_scope': 'user', 'time_scope': 'history'},
    }
    data = payload(reader, engine, ['history', 'missing_history'])
    result, reference, captured = meter(monkeypatch, reader, data)
    assert result is None and reference is None and not captured and not calls.requests


def test_or_alternative_sources_are_not_turned_into_a_required_and_union(monkeypatch):
    reader, engine, state, _, original = setup_reader()
    other = observation('independent-larger-proof', ['s2', 's3'], value=OLD)
    state.integrate([reader._observation(other, 'history', visible(reader, ['s2', 's3']), engine.steps[0])])
    assert state.valid()[2]['history'] == [original]
    data = payload(reader, engine)
    assert len(data['summary_revision_contracts']['history']['eligible_route_ids']) == 2
    assert set(data['summary_revision_contracts']['history']['required_source_ids']) == {'s1', 's2', 's3'}
    _, reference, _ = meter(monkeypatch, reader, data)
    support, = [row for row in reference['slots'][0]['observations'] if row['stance'] == 'support']
    assert support['source_ids'] == ['s1']
    assert support['summary_revision']['supersedes_route_ids'] == [original]


def test_counter_source_closure_is_metered_without_revoking_or_replacing_current_support(monkeypatch):
    reader, engine, state, _, original = setup_reader()
    correction = add_source(engine, 's4', 'My earlier statement about painting was wrong; I did not enjoy it.')
    old = observation('withdrawn-painting-account', ['s2'], value='The user enjoyed painting.')
    state.integrate([reader._observation(old, 'history', visible(reader, ['s2']), engine.steps[0])])
    other = next(rid for rid, row in state.routes.items() if row['value'] != OLD)
    counter = observation('painting-counter', ['s4'], stance='contradiction', retract_ids=[other])
    counter['value'] = None
    state.integrate([reader._observation(counter, 'history', visible(reader, ['s4']), engine.steps[0])])
    assert state.valid()[2]['history'] == [original]
    before = deepcopy(state.public_dict())
    _, reference, _ = meter(monkeypatch, reader, payload(reader, engine))
    support, = [row for row in reference['slots'][0]['observations'] if row['stance'] == 'support']
    assert support['value'] == OLD and set(support['source_ids']) == {'s1', 's4'}
    assert support['summary_revision']['supersedes_route_ids'] == [original]
    assert correction in state.summary_revision_sources('history', [original])
    assert state.public_dict() == before and state.routes[other]['revoked']


def test_actual_parent_versions_and_original_parent_sources_are_included_in_the_revision_budget(monkeypatch):
    _, engine, previous, _, _ = setup_reader()
    engine.steps = deepcopy(engine.steps)
    engine.steps.insert(0, {'output_slot': 'context', 'question': 'Read the user context',
                           'inputs': [], 'execution': 'retrieval', 'answer_type': 'fact'})
    engine.steps[1]['inputs'] = ['context']
    program = deepcopy(previous.initial_program)
    program['variables']['context'] = {
        'type': 'text', 'demand': 'context', 'evidence_semantics': 'personal_atomic_fact',
        'scope': {'entity_scope': 'user', 'time_scope': 'history'},
    }
    state = FactState(program, previous.contract, engine.steps, 'suggestion')
    reader = JointReader(engine, state)
    reader.current_identity = 'parent-closure-reference'
    parent = observation('current-context', ['s2'], value='The user described a past creative activity.')
    state.integrate([reader._observation(parent, 'context', visible(reader, ['s2']), engine.steps[0])])
    history = observation('parent-grounded-summary', ['s1'], value=OLD,
                          parent_versions={'context': state.versions['context']})
    state.integrate([reader._observation(history, 'history', visible(reader, ['s1', 's2']), engine.steps[1])])
    data = reader._payload(engine.steps[1], 'Review history in its actual context', ['history', 'context'], ['s3'])
    before = deepcopy(state.public_dict())
    _, reference, _ = meter(monkeypatch, reader, data)
    slots = {slot['variable']: slot for slot in reference['slots']}
    assert set(slots) == {'history', 'context'}
    support, = slots['history']['observations']
    assert set(support['source_ids']) == {'s1', 's2'}
    assert support['parent_versions'] == {'context': state.versions['context']}
    assert support['summary_revision']['supersedes_route_ids'] == state.valid()[2]['history']
    context, = slots['context']['observations']
    assert context['source_ids'] == ['s2'] and context['value'] == parent['value']
    assert 'summary_revision' not in context, 'A factual parent does not acquire summary revision authority'
    assert state.public_dict() == before


def test_existing_53_source_obligation_is_preserved_separately_not_cited_as_settled_summary(monkeypatch):
    reader, engine, state, calls, _ = setup_reader()
    aliases = ['s' + str(index) for index in range(4, 57)]
    for alias in aliases:
        add_source(engine, alias)
    descriptions = ['The negative painting experience remains unresolved.',
                    'All original memories need an explicit scoped review.']
    add_obligation(reader, 'unresolved-fifty-three', aliases, descriptions)
    state.integrate([], {'history'})
    data = payload(reader, engine)
    before = (deepcopy(state.public_dict()), deepcopy(reader.obligation_snapshot()), deepcopy(data))
    _, reference, _ = meter(monkeypatch, reader, data)
    slot, = reference['slots']
    support, = [row for row in slot['observations'] if row['stance'] == 'support']
    assert support['source_ids'] == ['s1']
    unsettled = [row for row in slot['observations'] if row['stance'] in {'unknown', 'partial'}]
    assert unsettled and {alias for row in unsettled for alias in row['source_ids']} == set(aliases)
    assert all(row['value'] is None and row['retract_ids'] == [] for row in unsettled)
    assert all(any(description in row['reason'] for row in unsettled) for description in descriptions)
    assert set(descriptions) <= set(slot['unresolved']) and slot['resolved_obligation_ids'] == []
    assert (state.public_dict(), reader.obligation_snapshot(), data) == before
    assert state.pending == {'history'} and not calls.requests


def test_independent_sourced_and_unsourced_obligations_keep_their_own_descriptions(monkeypatch):
    reader, engine, _, _, _ = setup_reader()
    add_obligation(reader, 'painting-question', ['s2'], ['Painting history remains unresolved.'])
    add_obligation(reader, 'assistant-suggestion', ['s3'], ['An assistant suggestion is not a user preference.'])
    add_obligation(reader, 'search-gap', [], ['No settled account exists for this additional question.'])
    data = payload(reader, engine)
    _, reference, _ = meter(monkeypatch, reader, data)
    slot, = reference['slots']
    for original in reader.evidence_obligations.values():
        expected = {engine.mapper.source_aliases[sid] for sid in original['source_ids']}
        matching = [row for row in slot['observations'] if row['stance'] in {'unknown', 'partial'}
                    and all(description in row['reason'] for description in original['descriptions'])]
        assert matching and all(set(row['source_ids']) == expected for row in matching)
        assert set(original['descriptions']) <= set(slot['unresolved'])
    assert slot['resolved_obligation_ids'] == []


@pytest.mark.parametrize('tampering', ['omit_all', 'omit_source', 'omit_description', 'change_scope', 'duplicate'])
def test_request_cannot_lower_the_reference_workload_by_hiding_or_altering_trusted_obligations(tampering, monkeypatch):
    reader, engine, _, calls, _ = setup_reader()
    add_obligation(reader, 'trusted-unresolved', ['s2', 's3'], ['First unresolved issue.', 'Second unresolved issue.'])
    data = payload(reader, engine)
    if tampering == 'omit_all':
        del data['evidence_obligations']
    elif tampering == 'omit_source':
        data['evidence_obligations'][0]['source_ids'].pop()
    elif tampering == 'omit_description':
        data['evidence_obligations'][0]['descriptions'].pop()
    elif tampering == 'change_scope':
        data['evidence_obligations'][0]['time_scope'] = 'a different period'
    else:
        data['evidence_obligations'].append(deepcopy(data['evidence_obligations'][0]))
    before = deepcopy(reader.obligation_snapshot())
    result, reference, captured = meter(monkeypatch, reader, data)
    assert result is None and reference is None and not captured
    assert reader.obligation_snapshot() == before and not calls.requests


@pytest.mark.parametrize('case', ['missing_summary_alias', 'missing_obligation_alias', 'unknown_visible_alias'])
def test_reference_never_falls_back_to_hidden_stable_ids_or_invented_aliases(case, monkeypatch):
    reader, engine, _, calls, _ = setup_reader()
    add_obligation(reader, 'separate-source', ['s3'], ['This source remains unresolved.'])
    data = payload(reader, engine)
    if case == 'missing_summary_alias':
        data['source_spans'] = [row for row in data['source_spans'] if row['id'] != 's1']
    elif case == 'missing_obligation_alias':
        data['source_spans'] = [row for row in data['source_spans'] if row['id'] != 's3']
    else:
        data['source_spans'][0]['id'] = 'not-an-authoritative-alias'
    result, reference, captured = meter(monkeypatch, reader, data)
    assert result is None and reference is None and not captured and not calls.requests


def test_metering_reference_and_its_revision_placeholders_never_enter_the_real_generation_prompt(monkeypatch):
    reader, engine, state, calls, _ = setup_reader()
    data = payload(reader, engine)
    before = deepcopy(state.public_dict())
    _, reference, _ = meter(monkeypatch, reader, data)
    wire = reader._wire_data(data)
    request = engine.reasoner._payload('joint_read', reader._system(wire), wire,
                                       reader._schema(data), output_token_cap=3000)
    generated_data = json.loads(request['messages'][1]['content'])
    assert generated_data == wire and 'slots' not in generated_data
    assert json.dumps(reference, ensure_ascii=False, separators=(',', ':')) not in request['messages'][1]['content']
    assert state.public_dict() == before and not calls.requests and not engine.reasoner.requests
