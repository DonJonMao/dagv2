"""Mechanism and protocol tests: production search executes against local transports."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dagbt.bridge import BridgeSession, _Scorer
from dagbt.budget import Ledger
from vendor.bridgetree.dependency_scoring import SetResponseError


class Calls:
    def __init__(self, fail_rerank=False):
        self.requests = []
        self.fail_rerank = fail_rerank

    def get(self, stage, url, payload):
        self.requests.append((stage, url, payload))
        if url.endswith('/embeddings'):
            # Deterministic server substitute, with distinct bridge discoveries.
            x = len(self.requests) % 4
            return {'response': {'data': [{'index': 0, 'embedding': [float(i == x) for i in range(4)]}]}}
        if self.fail_rerank:
            raise ConnectionError('test reranker unavailable')
        # Genuine source activation code receives complete indexed responses.
        return {'response': {'results': [
            {'index': i, 'relevance_score': min(.9, .1 + .08 * text.count('[Passage '))}
            for i, text in reversed(list(enumerate(payload['documents'])))
        ]}}

    post_rerank = get


def session(ann=36, sets=128, calls=None, **settings):
    ids = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']
    vectors = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [.7,.7,0,0], [0,0,.7,.7]], np.float32)
    docs = {d: SimpleNamespace(passage=f'Title {i}\nEntity fact number {i}.') for i,d in enumerate(ids)}
    config = {
        'embedding_base_url':'http://local/v1', 'embedding_model':'test-embedding',
        'reranker':{'url':'http://local/rerank','model':'test-pointwise'},
        'fusion':{'reserved_gap_ann_calls':2,'initial_width':2,'proposal_width':2, **settings},
    }
    ledger = Ledger({'ann':ann,'set_score':sets,'llm':24,'reader':1})
    return BridgeSession('Which country is the author from?', docs, ids, vectors, None,
                         calls or Calls(), config, ledger)


def requirement():
    return [{'id':'r1','description':'birthplace and its country','time_scope':'unknown','necessary':True}]


def test_vendored_source_matches_working_tree_snapshot_manifest():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'vendor/bridgetree_manifest.json').read_text())
    assert manifest['snapshot_kind'] == 'working_tree_including_uncommitted_files'
    assert len(manifest['files']) > 40
    for entry in manifest['files']:
        assert hashlib.sha256((root/'vendor/bridgetree'/entry['path']).read_bytes()).hexdigest() == entry['sha256']


def test_latest_search_executes_and_all_discoveries_are_preserved():
    s = session()
    result = s.discover('Where was {author} born?', 'birthplace', requirement(), ['d0'])
    trace = result['trace']
    assert trace['search_archive']['method_version'] == 'evidence_bridge_v1'
    assert trace['search_archive']['scheduler_events']
    assert trace['search_archive']['measured_sets']
    opened = {e['target_id'] for e in trace['search_archive']['scheduler_events']
              if e.get('event') == 'quantum_started' and e.get('action') == 'open_root'}
    assert len(opened) >= 2
    assert isinstance(s.scorer, _Scorer)
    assert s.scorer.scored_sets == s.ledger.used['set_score']
    assert s.ledger.used['ann'] <= 34
    assert s.ledger.used['set_score'] <= 128
    discovered = {d for b in trace['retrieval']['proposal_batches'] for d in b['candidate_ids']}
    assert discovered == set(result['candidate_ids'])
    assert all(e['dependency_claim'] is False for e in trace['retrieval']['source_graph'])
    embeddings = [p for _, u, p in s.calls.requests if u.endswith('/embeddings')]
    assert all('Entity fact number 0.' in p['input'][0] for p in embeddings)
    reranks = [p for _, u, p in s.calls.requests if u.endswith('/rerank')]
    assert reranks and all(p['query'] == s.original_query for p in reranks)
    assert all('Personal memories' not in text for p in reranks for text in p['documents'])


def test_multiple_nodes_share_global_budget_and_leave_gap_reservation():
    s = session(ann=12, sets=48)
    first = s.discover('Find author', 'author', requirement(), remaining_nodes=2)
    assert first['trace']['allocated_ann'] == 5
    assert first['trace']['initial_ann_cap'] == 2
    assert first['trace']['allocated_new_sets'] == 24
    before = set(first['candidate_ids'])
    second = s.discover('Find birthplace', 'birthplace', requirement(), remaining_nodes=1)
    assert before <= set(second['candidate_ids'])
    assert s.ledger.used['ann'] <= 10
    assert s.ledger.used['set_score'] <= 48
    assert s.ledger.remaining('ann') >= 2
    assert second['trace']['allocated_new_sets'] <= 48


def test_gap_budget_is_global_and_excludes_already_discovered():
    s = session(ann=4)
    s.discover('Find author', 'a', requirement(), mode='dense')
    before = set(s.candidate_ids)
    result = s.discover('What is missing?', 'a-gap', requirement(), feedback=True)
    assert not before.intersection(result['new_candidate_ids'])
    s.discover('What is missing?', 'a-gap2', requirement(), feedback=True)
    n = len(s.calls.requests)
    result = s.discover('What is missing?', 'a-gap3', requirement(), feedback=True)
    assert result['stop_reason'] == 'ann_budget_exhausted'
    assert len(s.calls.requests) == n
    assert s.gap_calls == 2
    assert s.ledger.used['ann'] <= 4


def test_dense_control_needs_no_reranker_and_has_no_set_scoring():
    s = session()
    s.config.pop('reranker')
    result = s.discover('Find author', 'a', requirement(), mode='dense')
    assert result['candidate_ids']
    assert s.ledger.used['set_score'] == 0
    assert s.scorer is None
    assert len(s.calls.requests) == 1


def test_missing_reranker_is_explicit_before_any_calls():
    s = session()
    s.config.pop('reranker')
    with pytest.raises(ValueError, match='reranker.url'):
        s.discover('Find author', 'a', requirement())
    assert not s.calls.requests


def test_reranker_failure_preserves_partial_candidates_and_consumed_cost():
    s = session(calls=Calls(fail_rerank=True))
    with pytest.raises(ConnectionError):
        s.discover('Find author', 'a', requirement())
    trace = s.traces[-1]
    assert trace['stop_reason'] == 'execution_error'
    assert trace['retrieval']['proposal_batches']
    assert s.candidate_ids
    assert s.ledger.used['set_score'] > 0
    assert s.ledger.used['ann'] > 0
    assert trace['search_archive']['stop_reason'] == 'execution_error'


def test_shared_cache_charges_new_unique_sets_once():
    s = session()
    scorer = s._score_backend()
    scorer.score_sets([['d0'], ['d1'], ['d0']])
    spent = s.ledger.used['set_score']
    assert spent == 2
    scorer.score_sets([['d1'], ['d0']])
    assert s.ledger.used['set_score'] == spent
    assert scorer.scored_sets == spent


def test_reranker_contract_rejects_truncation_no_surrogate_fallback():
    s = session()
    s.calls.post_rerank = lambda *args: {'response': {'truncated': True, 'results': []}}
    with pytest.raises(SetResponseError, match='truncation'):
        s._score_backend().score_sets([['d0']])
    assert s.ledger.used['set_score'] == 1


def test_navigation_parents_must_be_real_and_no_implicit_vector_encoding():
    s = session()
    with pytest.raises(ValueError, match='parent source'):
        s.discover('Find author', 'a', requirement(), ['invented'])
    assert not s.calls.requests


def test_raw_passage_text_is_never_genericized_or_rewritten():
    s = session()
    literal = 'Bridge memory real visible history Missing historical information'
    from vendor.bridgetree.types import Memory
    original = s.records['d0']
    s.records['d0'] = Memory('d0', literal, original.timestamp, 'd0', original.metadata)
    s.memories = tuple(s.records.values())
    s.discover('Find author', 'a', requirement(), ['d0'], mode='dense')
    assert literal in s.calls.requests[0][2]['input'][0]
