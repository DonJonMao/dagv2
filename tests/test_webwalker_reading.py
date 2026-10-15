"""Source-only Web reading regressions; no labels or external model calls."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest

from dagbt.budget import Ledger
from dagbt.evidence_spans import build_source_spans
from dagbt.fact_state import FactState
from dagbt.methods import JOINT, RESIDUAL_V2
from dagbt.reasoning import Reasoner
from dagbt.transport import digest
from dagbt.webwalker_engine import WebEngine
from dagbt.webwalker_reading import WebJointReader
import test_webwalker as browser


def observation(identifier='history-1', sources=('s1',), value=None, **changes):
    return {'id': identifier, 'stance': 'support', 'value': ['piano'] if value is None else value,
            'source_ids': list(sources), 'entity_scope': 'user', 'time_scope': 'history',
            'parent_versions': {}, 'retract_ids': [], 'reason': 'Explicit original source.', **changes}


def response(*rows):
    return {'slots': [{'variable': 'history', 'observations': list(rows), 'unresolved': []}]}


def expand_wire_data(data):
    data = deepcopy(data)
    columns = data.pop('source_span_columns', None)
    if columns:
        data['source_spans'] = [dict(zip(columns, row)) for row in data['source_spans']]
    documents = data.pop('source_documents', None)
    if documents:
        for source in data['source_spans']:
            source.update(deepcopy(documents[source['doc_id']]))
    return data


def visible(reader):
    data = reader._payload(reader.e.steps[0], 'Read historical activities', ['history'], ['s1'])
    return {source['id']: source for source in data['source_spans']}


def web_fixture(respond=None, **settings):
    steps = [{'output_slot': 'history', 'question': 'Read historical activities', 'inputs': [],
              'execution': 'retrieval', 'answer_type': 'fact'},
             {'output_slot': 'suggestion', 'question': 'Respond from {history}', 'inputs': ['history'],
              'execution': 'compose', 'answer_type': 'fact'}]
    program = {'variables': {'history': {'type': 'set', 'demand': 'history'},
                             'suggestion': {'type': 'text', 'demand': 'suggestion'}},
               'expression': {'op': 'var', 'id': 'suggestion'}}
    contract = {'mode': 'semantic', 'result_type': 'text', 'entity_scope': 'user',
                'time_scope': 'history', 'explanation': 'semantic'}
    state = FactState(program, contract, steps, 'suggestion')
    documents, sources, source_aliases = {}, {}, {}
    for identifier, text in [('piano', 'I played piano last year.'),
                             ('painting', 'I enjoyed painting last spring.'),
                             ('advice', 'A project page describes guitar lessons.')]:
        document = documents[identifier] = SimpleNamespace(doc_id=identifier, passage=text, metadata={})
        for source in build_source_spans(identifier, document, 400):
            alias = 's' + str(len(sources) + 1)
            sources[alias] = {**source, 'id': alias, 'source_id': source['id']}
            source_aliases[source['id']] = alias
    policy = {'algorithm_version': RESIDUAL_V2, 'max_quote_chars': 400, 'reasoning_output_tokens': 512,
              'input_margin': 256, 'context_tokens': 16384, 'response_format': 'plain',
              'max_repairs_per_request': 2, 'reserved_audit_calls': 1, **settings}
    ledger, events = Ledger({'llm': 24, 'json_repairs': 6}), []
    class Calls:
        def __init__(self): self.requests = []; self.respond = respond or (lambda i, data: response(observation()))
        def get(self, stage, url, payload, **kwargs):
            data = expand_wire_data(json.loads(payload['messages'][1]['content']))
            self.requests.append({'stage': stage, 'data': data, 'payload': deepcopy(payload)})
            ledger.reserve('llm', '/'.join(stage))
            value = self.respond(len(self.requests), data)
            return {'response_ref': digest(payload), 'response': {'choices': [
                {'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}}
    calls = Calls()
    config = {'llm_base_url': 'http://offline.invalid/v1', 'llm_model': 'scripted', 'fusion': policy}
    reasoner = Reasoner(calls, None, config, policy, ledger, events.append)
    engine = SimpleNamespace(q={'question': 'Suggest an activity using historical evidence.'}, steps=steps,
                             docs=documents, mapper=SimpleNamespace(sources=sources, source_aliases=source_aliases),
                             s=policy, ledger=ledger, reasoner=reasoner, event=events.append, events=events,
                             errors=[], spans={}, reader_question='Suggest an activity.', output_options=None)
    return WebJointReader(engine, state), engine, state, calls


def gap_response(text='Continue to an unvisited archive.'):
    return {'slots': [{'variable': 'history', 'observations': [], 'unresolved': [],
                       'search_gaps': [text]}]}


def read(reader, doc='piano', identity='local-web'):
    return reader.read(reader.e.steps[0], 'Find activity evidence', ['history'], [doc], identity,
                       discovery_doc_ids=[doc])


def test_explicit_gap_keeps_raw_archive_without_binding_or_retaining_old_page():
    reader, engine, state, calls = web_fixture(lambda i, data: gap_response())
    before = deepcopy(engine.ledger.limits)
    read(reader)
    assert state.routes == {} and state.valid()[0] == {} and not state.pending
    assert not reader.evidence_obligations and not reader.failed_batch_reviews
    assert reader.read_states['local-web', 's1'] == 'read'
    next_page = reader._payload(engine.steps[0], 'Find activity evidence', ['history'], ['s2'])
    assert [source['id'] for source in next_page['source_spans']] == ['s2']
    assert reader.public_dict()['web_search_gaps'][0]['search_gaps'] == ['Continue to an unvisited archive.']
    assert json.loads(engine.reasoner.last_response['raw_output']) == gap_response()
    assert 'search_gaps' in engine.reasoner.last_response['raw_output']
    assert engine.ledger.limits == before and engine.ledger.used['llm'] == 1
    assert engine.ledger.used['json_repairs'] == 0 and len(calls.requests) == 1
    assert not getattr(engine, 'personal_response_selection', False)
    assert 'personal_response_selection' not in calls.requests[0]['data']


@pytest.mark.parametrize('message', ['Not found in this page.', '本页没有答案。', 'Conflicting dates remain.'])
def test_legacy_unresolved_is_never_reclassified_by_its_text(message):
    value = {'slots': [{'variable': 'history', 'observations': [], 'unresolved': [message]}]}
    reader, engine, state, _ = web_fixture(lambda i, data: value)
    read(reader)
    assert state.pending == {'history'}
    assert next(iter(reader.evidence_obligations.values()))['source_ids'] == [engine.mapper.sources['s1']['source_id']]
    assert not reader.web_search_gaps
    assert {source['id'] for source in reader._payload(engine.steps[0], 'query', ['history'], ['s2'])['source_spans']} == {'s1', 's2'}


def test_new_gap_never_discharges_real_sourced_issue_or_restored_obligation():
    issue = response(observation('ambiguous', ['s1'], stance='unknown', value=None,
                                 reason='The original source leaves its date ambiguous.'))
    issue['slots'][0]['observations'][0]['value'] = None
    issue['slots'][0]['unresolved'] = ['Resolve the ambiguity in this source.']
    reader, engine, state, calls = web_fixture(lambda i, data: issue if i == 1 else gap_response())
    read(reader)
    obligations = deepcopy(reader.obligation_snapshot())
    restored = WebJointReader(engine, state)
    restored.restore_obligations(obligations)
    assert restored.evidence_obligations == reader.evidence_obligations
    read(reader, 'painting')
    assert reader.evidence_obligations and state.pending == {'history'}
    assert not state.valid()[0] and reader.failed_batch_reviews == {}
    assert engine.mapper.sources['s1']['source_id'] in next(iter(reader.evidence_obligations.values()))['source_ids']
    assert len(calls.requests) == 2


@pytest.mark.parametrize('gaps', [True, 'continue', [], [''], [' '], [7]])
def test_malformed_gap_stays_in_failed_draft_instead_of_becoming_empty_read(gaps):
    value = gap_response()
    value['slots'][0]['search_gaps'] = gaps
    reader, _, state, _ = web_fixture(lambda i, data: value, max_repairs_per_request=0)
    read(reader)
    assert not reader.web_search_gaps and not state.valid()[0]
    assert state.pending == {'history'} and reader.failed_batch_reviews
    record = next(iter(reader.failed_batch_reviews.values()))
    assert record['failed'][0]['slot']['search_gaps'] == gaps


@pytest.mark.parametrize('change', ['support', 'unresolved', 'resolution', 'extra', 'duplicate', 'missing'])
def test_invalid_gap_envelope_cannot_write_valid_gap_diagnostics(change):
    value = gap_response()
    slot = value['slots'][0]
    if change == 'support':
        slot['observations'] = [observation()]
    elif change == 'unresolved':
        slot['unresolved'] = ['A concrete factual issue.']
    elif change == 'resolution':
        slot['resolved_obligation_ids'] = ['invented']
    elif change == 'extra':
        slot['undeclared'] = True
    elif change == 'duplicate':
        value['slots'].append(deepcopy(slot))
    else:
        value['slots'] = []
    reader, _, state, _ = web_fixture(lambda i, data: value, max_repairs_per_request=0)
    read(reader)
    assert not reader.web_search_gaps and not state.valid()[0]
    assert state.pending == {'history'} and reader.failed_batch_reviews


@pytest.mark.parametrize('repair_counter', [False, True])
def test_gap_extension_never_bypasses_invalid_envelope_counter_capture(repair_counter):
    reader, engine, state, calls = web_fixture(max_repairs_per_request=1)
    state.integrate([reader._observation(observation(), 'history', visible(reader), engine.steps[0])])
    route = next(iter(state.routes))
    counter = observation('counter', ['s2'], stance='contradiction', value=None,
                          retract_ids=[route], reason='The original source corrects the history.')
    counter['value'] = None
    invalid = response(counter)
    invalid['slots'][0]['search_gaps'] = ['Look elsewhere.']
    invalid['bad_envelope'] = True
    calls.respond = lambda i, data: invalid if i == 1 else response(counter) if repair_counter else gap_response()
    read(reader, 'painting')
    assert len(calls.requests) == 2
    assert json.loads(engine.reasoner.requests[0]['raw_output']) == invalid
    if repair_counter:
        assert state.routes[route]['revoked']
    else:
        assert state.pending == {'history'} and reader.failed_batch_reviews
        protected = [counter for record in reader.failed_batch_reviews.values()
                     for failure in record['failed'] for counter in failure.get('required_counter_observations', [])]
        assert protected and protected[0]['retract_ids'] == [route]
        assert not state.routes[route]['revoked']


def test_diagnostic_archives_do_not_count_as_material_progress_or_checkpoint_obligations():
    reader, _, _, _ = web_fixture()
    before = reader.material_signature()
    reader.web_search_gaps.append({'variable': 'history', 'search_gaps': ['Explore archive.']})
    reader.web_capacity_admissions.append({'source_count': 1})
    assert reader.material_signature() == before
    assert 'web_search_gaps' not in reader.obligation_snapshot()
    assert 'web_capacity_admissions' not in reader.obligation_snapshot()


def test_valid_gap_in_sibling_slot_cannot_hide_malformed_counter_envelope():
    reader, engine, old, calls = web_fixture(max_repairs_per_request=0)
    program = deepcopy(old.program)
    program['variables']['peer'] = {'type': 'set', 'demand': 'history'}
    state = FactState(program, old.contract, engine.steps, 'suggestion')
    reader = WebJointReader(engine, state)
    state.integrate([reader._observation(observation(), 'peer', visible(reader), engine.steps[0])])
    route = next(iter(state.routes))
    counter = observation('peer-counter', ['s2'], stance='contradiction', retract_ids=[route])
    counter['value'] = None
    invalid = gap_response()
    invalid['slots'].append({'variable': 'peer', 'observations': [counter], 'unresolved': [], 'extra': True})
    calls.respond = lambda i, data: invalid
    reader.read(engine.steps[0], 'Find activity evidence', ['history', 'peer'], ['painting'],
                'sibling-envelope', discovery_doc_ids=['painting'])
    assert not reader.web_search_gaps and state.pending == {'history', 'peer'}
    peer = next(record for record in reader.failed_batch_reviews.values() if record['variable'] == 'peer')
    assert any(item.get('required_counter_observations') for item in peer['failed'])
    assert not state.routes[route]['revoked']


def test_empty_gap_repair_does_not_unfreeze_or_erase_source_validated_unknown():
    reader, engine, state, calls = web_fixture(max_repairs_per_request=1)
    unknown = observation('frozen-unknown', ['s1'], stance='unknown', reason='The source omits the year.')
    unknown['value'] = None
    invalid = observation('repair-me', ['invented-source'])
    calls.respond = lambda i, data: response(unknown, invalid) if i == 1 else gap_response()
    read(reader)
    assert len(calls.requests) == 2 and not state.valid()[0] and state.pending == {'history'}
    assert calls.requests[1]['data']['repair_scope']['frozen_observation_ids'] == {'history': ['frozen-unknown']}
    assert reader.evidence_obligations and reader.failed_batch_reviews
    retained = [row for record in reader.failed_batch_reviews.values() for row in record['validated_observations']]
    assert any(row['id'] == 'frozen-unknown' and row['stance'] == 'unknown' for row in retained)


@pytest.mark.parametrize('form', ['mixed_slot', 'top_level', 'duplicate_slots', 'malformed_list',
                                 'legacy_top_level', 'legacy_duplicate'])
def test_invalid_gap_then_empty_gap_repair_cannot_erase_ordinary_unresolved_source(form):
    issue = 'The original source gives conflicting dates.'
    initial = gap_response()
    initial['slots'][0]['unresolved'] = [issue]
    if form == 'top_level':
        initial['invalid_envelope'] = True
    elif form == 'duplicate_slots':
        initial['slots'] = [gap_response()['slots'][0],
                            {'variable': 'history', 'observations': [], 'unresolved': [issue]}]
    elif form == 'malformed_list':
        initial['slots'][0]['unresolved'].append(7)
    elif form.startswith('legacy_'):
        initial['slots'][0].pop('search_gaps')
        if form == 'legacy_top_level':
            initial['invalid_envelope'] = True
        else:
            initial['slots'].append(deepcopy(initial['slots'][0]))
    reader, engine, state, calls = web_fixture(
        lambda i, data: initial if i == 1 else gap_response(), max_repairs_per_request=1)
    read(reader)
    assert len(calls.requests) == 2 and not state.valid()[0]
    assert state.pending == {'history'} and reader.evidence_obligations
    obligation = next(iter(reader.evidence_obligations.values()))
    assert issue in obligation['descriptions']
    assert obligation['source_ids'] == [engine.mapper.sources['s1']['source_id']]
    repair = calls.requests[1]['data']
    assert repair['evidence_obligations'][0]['descriptions'] == [issue]
    assert json.loads(engine.reasoner.requests[0]['raw_output']) == initial
    # A gap is diagnostic only; neither it nor a corrected empty envelope
    # authorizes the sourced review required to close the original issue.
    assert not state.routes and all(row['search_gaps'] for row in reader.web_search_gaps)


class ExactCounts:
    def __init__(self, forecast=None):
        self.forecast = forecast
        self.records = []

    def __call__(self, stage, url, payload):
        data = expand_wire_data(json.loads(payload['messages'][1]['content']))
        scope = data.get('repair_scope', {})
        is_forecast = 'invalid_response' in scope and scope['invalid_response'] == ''
        count = (self.forecast if self.forecast is not None else 2200 + 2000 * len(data['source_spans'])) if is_forecast else 1000
        self.records.append((deepcopy(payload), is_forecast))
        return {'count': count, 'token_count_is_estimate': False,
                'tokenization_wire_sha256': digest(payload), 'token_estimator_id': 'offline-exact-fixture'}


@pytest.mark.parametrize('forecast,allowed', [(7928, True), (7929, False)])
def test_fresh_headroom_exact_boundary_preserves_fixed_context_output_and_sources(forecast, allowed):
    reader, engine, _, calls = web_fixture(reasoning_output_tokens=4096)
    counter = ExactCounts(forecast)
    calls.count_reasoning_tokens = counter
    data = reader._payload(engine.steps[0], 'query', ['history'], ['s1'])
    original, settings, limits = deepcopy(data), deepcopy(engine.s), deepcopy(engine.ledger.limits)
    assert reader._fresh_window_fits('joint_read', data) is allowed
    assert data == original and engine.s == settings and engine.ledger.limits == limits
    assert all(payload['max_tokens'] == 4096 for payload, _ in counter.records)
    assert not calls.requests and not reader.web_capacity_admissions
    assert reader._fresh_window_fits('joint_read', data, fresh_sources=False)
    assert reader._adaptive_fresh_cap('joint_read', data) is None


def test_headroom_selects_smaller_untrimmed_prefix_without_spending_calls_or_repairs():
    reader, engine, _, calls = web_fixture(reasoning_output_tokens=4096)
    calls.count_reasoning_tokens = ExactCounts()
    before = deepcopy(engine.mapper.sources)
    pending = [[alias] for alias in engine.mapper.sources]
    reader.current_identity = 'headroom'
    selected, consumed = reader._fitting_prefix('joint_read', engine.steps[0], 'query', ['history'], pending, False, False)
    assert selected == ['s1', 's2'] and consumed == 2 and pending == [['s1'], ['s2'], ['s3']]
    assert engine.mapper.sources == before and not calls.requests
    assert engine.ledger.used['llm'] == engine.ledger.used['json_repairs'] == 0
    assert not reader._web_headroom_candidates and not reader.web_capacity_admissions


def test_headroom_uses_admission_measurement_without_extra_tokenizer_call_after_response():
    reader, engine, _, calls = web_fixture(lambda i, data: gap_response(), reasoning_output_tokens=4096)
    counter = ExactCounts(2000)
    def count(stage, url, payload):
        assert not calls.requests, 'No new fallible tokenizer operation after the model response'
        return counter(stage, url, payload)
    calls.count_reasoning_tokens = count
    read(reader)
    assert len(calls.requests) == 1 and reader.web_capacity_admissions
    assert reader.web_capacity_admissions[0]['within_budget']


def test_tokenizer_failure_on_attempt_keeps_common_source_exposure_and_unread_scope():
    reader, _, state, calls = web_fixture(lambda i, data: gap_response(), reasoning_output_tokens=4096)
    counter = ExactCounts(2000)
    def count(stage, url, payload):
        if reader.source_exposures:
            raise ConnectionError('Offline tokenization service failure at attempted request')
        return counter(stage, url, payload)
    calls.count_reasoning_tokens = count
    read(reader)
    assert not calls.requests and reader.source_exposures
    assert reader.source_exposures[0]['request_status'] == 'not_sent'
    assert reader.source_exposures[0]['error_type'] == 'ConnectionError'
    assert state.pending == {'history'} and reader.failed_batch_reviews
    assert not reader.web_search_gaps and not reader.web_capacity_admissions


def single_plan():
    return {'steps': [browser.demo.step('primary', 'Find the primary package')],
            'final_node_id': 'primary',
            'output_contract': browser.demo.contract('value', 'entity', 'package'),
            'program': {'variables': {'p': {'type': 'entity', 'demand': 'primary'}},
                        'expression': {'op': 'var', 'id': 'p'}}}


@pytest.mark.parametrize('method', [JOINT, RESIDUAL_V2])
@pytest.mark.parametrize('has_target', [True, False])
def test_multiple_empty_pages_continue_to_observed_target_or_finish_without_false_certificate(method, has_target, tmp_path):
    class Handler(BaseHTTPRequestHandler):
        hits = []
        def do_GET(self):
            self.hits.append(self.path)
            if self.path == '/':
                content = '<title>Home</title><p>Site navigation.</p><a href="/archive">Archive</a>'
            elif self.path == '/archive':
                content = '<title>Archive</title><p>Browse project records.</p><a href="/target">Project</a>'
            else:
                content = '<title>Project</title><p>' + ('Primary package: Star E3.' if has_target else 'Contact the webmaster.') + '</p>'
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(content.encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    class Model:
        def __init__(self): self.requests = []
        def get(self, stage, url, payload):
            self.requests.append((stage[0], deepcopy(payload)))
            data = json.loads(payload['messages'][1]['content'])
            if isinstance(data, dict):
                data = expand_wire_data(data)
            if stage[0] == 'planner':
                value = single_plan()
            elif stage[0] == 'web_navigate':
                value = {'link_id': data['links'][0]['id']}
            elif stage[0].startswith('audit_joint'):
                value = {'slots': [{'variable': 'p', 'observations': [], 'unresolved': []}],
                         'audit_verdict': 'supported'}
            else:
                sources = [source['id'] for source in data['source_spans'] if 'Primary package: Star E3.' in source['text']]
                row = []
                if sources:
                    row = [{'id': 'package', 'stance': 'support', 'value': 'Star E3', 'source_ids': sources,
                            **data['requested_scopes']['p'], 'parent_versions': {}, 'retract_ids': [],
                            'reason': 'Explicit project statement.'}]
                slot = {'variable': 'p', 'observations': row, 'unresolved': []}
                if not row: slot['search_gaps'] = ['Follow another observed project link.']
                value = {'slots': [slot]}
            return {'response_ref': 'offline-' + str(len(self.requests)),
                    'response': {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}}
    try:
        model = Model()
        config = {'_test_transport': True, 'llm_base_url': 'http://fixture/v1', 'llm_model': 'fixture',
                  'webwalker': {'max_actions': 36, 'http_attempts': 1}}
        engine = WebEngine({'id': 'navigation', 'question': 'What is the primary package?'},
                           ({}, [], np.empty((0, 0), dtype=np.float32), None, browser.demo.Tokenizer()),
                           model, config, method, root_url='http://127.0.0.1:' + str(server.server_port),
                           page_cache=tmp_path, allow_local=True)
        result = engine.run()
        assert Handler.hits == ['/', '/archive', '/target']
        assert model.requests[0][0] == 'planner'
        assert result['diagnostics']['web']['actions'] == 3
        assert result['diagnostics']['ledger']['used'].get('json_repairs', 0) == 0
        assert not engine.residual_control.reader.evidence_obligations
        assert len(result['diagnostics']['joint_reading']['web_search_gaps']) == (2 if has_target else 3)
        if has_target:
            assert result['answer']['status'] == 'ok' and result['answer']['prediction'] == 'Star E3'
            assert result['answer']['certificate'] and not engine.fact_state.pending
        else:
            assert result['answer']['prediction'] is None and result['answer']['certificate'] is None
            assert not engine.fact_state.valid()[0] and result['answer']['status'] != 'ok'
    finally:
        server.shutdown(); server.server_close(); thread.join()
