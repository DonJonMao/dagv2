"""Web feedback and real controller scheduling, using only invented sources."""
import ast
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
from threading import Thread
import textwrap

import numpy as np
import pytest

from dagbt.methods import JOINT, RESIDUAL_V2
from dagbt.residual_control import ResidualControl
from dagbt.webwalker_control import WebResidualControl
from dagbt.webwalker_engine import WebEngine
from test_webwalker import demo
from test_webwalker_reading import (expand_wire_data, gap_response, observation, read,
                                   response, web_fixture)


def test_web_run_preserves_common_control_except_documented_scheduling_hooks():
    common = inspect.getsource(ResidualControl.run)
    expected = common.replace('for step in self.e.steps:', 'for step in self._web_step_order():')
    expected = expected.replace('self.read_progress(step,query,variables,local_docs,key,',
                                'self.web_read_existing(step,query,variables,local_docs,key,')
    expected = expected.replace('executed=True;break', 'self._web_completed_turn(nid);executed=True;break')
    assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(WebResidualControl.run)))) == ast.dump(
        ast.parse(textwrap.dedent(expected)))


def test_old_issue_new_window_and_frozen_repair_have_readonly_structural_metadata():
    issue = observation('old-partial', ['s1'], stance='partial',
                        reason='The activity is explicit but its date remains incomplete.')
    issue['value'] = None
    repeated = {**deepcopy(issue), 'id': 'current-partial'}
    def respond(index, data):
        if index == 1:
            return response(issue)
        if index == 2:
            result = response(repeated)
            result['slots'][0]['resolved_obligation_ids'] = [data['evidence_obligations'][0]['id']]
            return result
        return response()
    reader, engine, state, calls = web_fixture(respond, max_repairs_per_request=1)
    read(reader, 'piano', 'same-query')
    original = deepcopy(reader.evidence_obligations)
    sources = deepcopy(engine.mapper.sources)
    read(reader, 'painting', 'same-query')

    initial, repair = calls.requests[1:]
    context = initial['data']['web_reading_context']
    assert context['window_source_ids'] == context['new_window_source_ids'] == ['s2']
    assert context['retained_mandatory_source_ids'] == ['s1']
    assert context['pending_source_issues'] == [
        {'obligation_id': next(iter(original)), 'source_ids': ['s1'],
         'state': 'unresolved_controller_owned', 'ordinary_restatement_needed': False}]
    assert context['grants_review_authority'] is False
    assert 'repair_scope' not in initial['data']
    assert 'WEB REPAIR FIELD CHECK' in initial['payload']['messages'][0]['content']
    frozen = repair['data']['web_reading_context']['frozen_observations']
    assert frozen == [{'variable': 'history', 'id': 'current-partial',
                       'immutable': True, 'grants_review_authority': False}]
    assert not any(entry.get('frozen_observation_reviews') for entry in repair['data']['repair_scope']['failed'])
    assert reader.evidence_obligations == original and state.pending == {'history'}
    assert not state.routes and not reader.failed_batch_reviews
    assert engine.mapper.sources == sources
    assert engine.ledger.used['llm'] == 3 and engine.ledger.used['json_repairs'] == 1
    # The final empty repair withdraws a resolution claim; it does not erase the
    # valid partial frozen earlier in this same batch or authorize a page yield.
    assert reader.batches[-1]['complete'] and reader.batches[-1]['valid_rows'] == 1
    assert reader.web_search_gaps and not reader.web_navigation_reads


def test_accepted_empty_feedback_is_same_query_readonly_and_never_fact_progress():
    reader, engine, state, calls = web_fixture(lambda index, data: gap_response('Try the observed archive.'))
    read(reader, 'piano', 'one-query')
    signature, checkpoint = reader.material_signature(), deepcopy(reader.obligation_snapshot())
    feedback = reader.navigation_feedback('history', identity='one-query')
    assert feedback['evidence_semantics'] == 'navigation_hints_not_facts'
    assert feedback['assessed_doc_ids'] == ['piano']
    assert feedback['accepted_empty_reads'][0]['search_gaps'] == ['Try the observed archive.']
    assert feedback['accepted_empty_reads'][0]['source_ids'] == [engine.mapper.sources['s1']['source_id']]
    assert reader.navigation_feedback('other', identity='one-query')['accepted_empty_reads'] == []
    assert reader.navigation_feedback('history', identity='changed-parents')['accepted_empty_reads'] == []
    feedback['accepted_empty_reads'][0]['search_gaps'].append('Mutated caller copy.')
    assert reader.navigation_feedback('history', identity='one-query')['accepted_empty_reads'][0]['search_gaps'] == [
        'Try the observed archive.']
    assert reader.material_signature() == signature and reader.obligation_snapshot() == checkpoint
    assert not state.routes and not state.pending and len(calls.requests) == 1


def test_incomplete_read_cannot_publish_accepted_empty_navigation_feedback():
    invalid = gap_response()
    invalid['bad_envelope'] = True
    reader, _, state, _ = web_fixture(lambda index, data: invalid, max_repairs_per_request=0)
    read(reader, 'piano', 'failed-query')
    assert reader.failed_batch_reviews and state.pending == {'history'}
    assert reader.navigation_feedback('history', identity='failed-query')['accepted_empty_reads'] == []


def test_rejected_empty_repair_is_not_accepted_when_final_repair_has_evidence():
    invalid = response(observation('required-draft', ['invented-source']))
    corrected = response(observation('required-draft', ['s1']))
    reader, _, state, calls = web_fixture(
        lambda index, data: invalid if index == 1 else gap_response() if index == 2 else corrected,
        max_repairs_per_request=2)
    read(reader, 'piano', 'corrected-evidence')
    assert len(calls.requests) == 3 and reader.batches[-1]['complete']
    assert state.valid()[0]['history'] == ['piano']
    assert reader.web_search_gaps  # Raw typed disposition remains archived.
    assert reader.navigation_feedback('history', identity='corrected-evidence')['accepted_empty_reads'] == []


def test_dom_priority_reorders_only_unoffered_discovery_and_keeps_every_original():
    reader, engine, state, _ = web_fixture()
    reader.current_identity = 'dom-priority'
    for alias, role in [('s1', 'utility_ui'), ('s2', 'main'), ('s3', 'hidden_ui')]:
        source = engine.mapper.sources[alias]
        engine.docs[source['doc_id']].metadata['web_regions'] = [
            {'start': source['start'], 'end': source['end'], 'role': role}]
        reader.discovery_frontier['dom-priority', alias] = reader._frontier_entry(
            engine.steps[0], ['history'], alias, 'dom-priority')
    sources, frontier = deepcopy(engine.mapper.sources), deepcopy(reader.discovery_frontier)
    pending = [['s1'], ['s2'], ['s3']]
    reader._prioritize_discovery_regions(pending)
    assert pending == [['s2'], ['s1'], ['s3']]
    assert engine.mapper.sources == sources and reader.discovery_frontier == frontier and not state.routes
    # Mandatory/unowned work retains its position even if it is UI text.
    reader.discovery_frontier.pop(('dom-priority', 's1'))
    pending = [['s1'], ['s3'], ['s2']]
    reader._prioritize_discovery_regions(pending)
    assert pending == [['s1'], ['s2'], ['s3']]


def flag_plan():
    return {'steps': [demo.step('first', 'Find the first flag.'),
                      demo.step('second', 'Find the second flag.'),
                      demo.step('decision', 'Are both flags true?', ['first', 'second'], 'compose')],
            'final_node_id': 'decision', 'output_contract': demo.contract('value', 'bool', 'flags'),
            'program': {'variables': {'a': {'type': 'bool', 'demand': 'first'},
                                      'b': {'type': 'bool', 'demand': 'second'}},
                        'expression': demo.op('and', demo.var('a'), demo.var('b'))}}


@pytest.mark.parametrize('method', [JOINT, RESIDUAL_V2])
def test_real_web_control_rotates_independent_leaves_and_yields_then_resumes_long_page(method, tmp_path):
    long_text = ' '.join('Archive entry %d describes an unrelated project.' % index for index in range(1800))
    class Handler(BaseHTTPRequestHandler):
        hits = []
        def do_GET(self):
            self.hits.append(self.path)
            pages = {'/': '<main><p>Second flag is true.</p><a href="/long">Archive</a></main>',
                     '/long': '<main><a href="/side">Appendix</a><a href="/target">Flag report</a><p>'
                              + long_text + '</p></main>',
                     '/side': '<main><p>Appendix lists unrelated archive projects.</p></main>',
                     '/target': '<main><p>First flag is true.</p></main>'}
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(pages[self.path].encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    class Model:
        def __init__(self):
            self.requests = []
            self.navigations = 0
        def get(self, stage, url, payload):
            data = json.loads(payload['messages'][1]['content'])
            if isinstance(data, dict): data = expand_wire_data(data)
            self.requests.append((stage[0], deepcopy(data)))
            if stage[0] == 'planner':
                value = flag_plan()
            elif stage[0] == 'web_navigate':
                target = ['/long', '/side', '/target'][self.navigations]
                self.navigations += 1
                value = {'link_id': next(link['id'] for link in data['links'] if link['url'].endswith(target))}
            else:
                slots = []
                for variable in data['variables']:
                    marker = 'First flag is true.' if variable == 'a' else 'Second flag is true.'
                    cited = [source['id'] for source in data['source_spans'] if marker in source['text']]
                    rows = []
                    if cited and not stage[0].startswith('audit_joint'):
                        rows = [{'id': variable + '-fact', 'stance': 'support', 'value': True,
                                 'source_ids': cited, **data['requested_scopes'][variable],
                                 'parent_versions': {}, 'retract_ids': [], 'reason': 'Explicit invented source statement.'}]
                    slot = {'variable': variable, 'observations': rows, 'unresolved': []}
                    if data['task']['execution'] == 'retrieval' and not stage[0].startswith('audit_joint'):
                        slot['reading_status'] = 'evidence_assessment' if rows else 'no_relevant_evidence'
                    slots.append(slot)
                value = {'slots': slots}
                if stage[0].startswith('audit_joint'): value['audit_verdict'] = 'supported'
            return {'response_ref': 'offline-' + str(len(self.requests)),
                    'response': {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}}
    try:
        model = Model()
        config = {'_test_transport': True, 'llm_base_url': 'http://fixture/v1', 'llm_model': 'fixture',
                  'webwalker': {'max_actions': 36, 'http_attempts': 1}}
        engine = WebEngine({'id': 'fair-frontier', 'question': 'Are the first and second flags both true?'},
                           ({}, [], np.empty((0, 0), dtype=np.float32), None, demo.Tokenizer()), model, config,
                           method, root_url='http://127.0.0.1:' + str(server.server_port),
                           page_cache=tmp_path, allow_local=True)
        result = engine.run()
        assert isinstance(engine.residual_control, WebResidualControl)
        assert engine.steps == flag_plan()['steps']
        reading = [data for op, data in model.requests if op == 'joint_read']
        assert [data['task']['output_slot'] for data in reading[:2]] == ['first', 'second']
        events = result['diagnostics']['events']
        yielded = [event for event in events if event['event'] == 'web_document_window_yielded']
        resumed = [event for event in events if event['event'] == 'web_document_window_resumed']
        assert yielded and resumed
        assert yielded[0]['sequence'] < resumed[0]['sequence']
        assert Handler.hits == ['/', '/long', '/side', '/target']
        assert result['answer']['status'] == 'ok' and result['answer']['prediction'] == 'true'
        reader = engine.residual_control.reader
        assert reader.discovery_frontier  # Deferred raw archive text was never deleted or called irrelevant.
        assert all(entry['status'] == 'unoffered' for entry in reader.discovery_frontier.values())
        assert not reader.failed_batch_reviews and not reader.evidence_obligations
        assert engine.s['context_tokens'] == 16384 and engine.s['reasoning_output_tokens'] == 4096
        assert engine.ledger.limits['llm'] == 24 and engine.ledger.limits['web_actions'] == 36
        assert engine.ledger.used['llm'] < 24 and not engine.ledger.used['json_repairs']
        assert next(doc for doc in engine.docs.values() if doc.metadata.get('url', '').endswith('/long')).passage.endswith(
            'Archive entry 1799 describes an unrelated project.')
    finally:
        server.shutdown(); server.server_close(); thread.join()
