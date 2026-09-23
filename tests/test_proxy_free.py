"""Day2 none mode is an explicit raw-conditioned bridge control, not fake scores."""
import pytest

from dagbt.config import resolve
from test_bridge import session, requirement


def test_proxy_free_runs_conditional_multihop_without_any_reranker():
    s = session(ann=12, sets=32, proxy_mode='none')
    s.config.pop('reranker')
    result = s.discover('Find the author birthplace country', 'country', requirement(), ['d0'])
    archive = result['trace']['search_archive']
    assert archive['method_version'] == 'day2_proxy_free_requirements_ann_v1'
    assert archive['uses_set_proxy'] is False and archive['signal_kind'] == 'none'
    assert archive['measured_sets'] == archive['activations'] == []
    assert s.scorer is None and s.ledger.used['set_score'] == 0
    assert all(url.endswith('/embeddings') for _,url,_ in s.calls.requests)
    conditional = [b for b in result['trace']['retrieval']['proposal_batches'] if b['stage'] == 'conditional']
    assert len(conditional) >= 2
    assert any(b['premise_ids'] for b in conditional), 'Expected actual continuation beyond a root proposal'
    for batch in conditional:
        assert s.records[batch['target_id']].text in batch['probe_text']
        for premise in batch['premise_ids']:
            assert s.records[premise].text in batch['probe_text']
        assert s.records['d0'].text in batch['probe_text']
    discovered = {d for b in result['trace']['retrieval']['proposal_batches'] for d in b['candidate_ids']}
    assert set(result['candidate_ids']) == discovered
    assert result['trace']['allocated_new_sets'] == 0
    assert s.ledger.used['ann'] <= 10 and s.ledger.remaining('ann') >= 2


def test_necessary_gaps_take_turns_and_optional_needs_do_not_steal_turns():
    s = session(ann=16, proxy_mode='none')
    needs = [
        {'id':'author','description':'Find author identity','necessary':True,'time_scope':'unknown'},
        {'id':'country','description':'Find birthplace country','necessary':True,'time_scope':'unknown'},
        {'id':'extra','description':'Optional painting hobby','necessary':False,'time_scope':'unknown'},
    ]
    result = s.discover('Answer the relation', 'answer', needs)
    events = result['trace']['search_archive']['scheduler_events']
    turns = [e['requirement_id'] for e in events if e['event'] == 'proxy_free_conditional_started']
    assert len(turns) >= 4 and turns[:4] == ['author','country','author','country']
    assert 'extra' not in turns
    states = result['trace']['search_archive']['states']
    for state in states:
        assert len(state['navigation_path']) == len(set(state['navigation_path']))
    pairs=[(s['requirement_id'],s['target_id'],tuple(s['premise_ids'])) for s in states]
    assert len(pairs) == len(set(pairs))


def test_proxy_free_shares_nodes_and_feedback_budget_without_resetting():
    s = session(ann=12, sets=32, proxy_mode='none')
    first = s.discover('Find author','author',requirement(),remaining_nodes=2)
    before = set(first['candidate_ids'])
    second = s.discover('Find country','country',requirement(),remaining_nodes=1)
    assert before <= set(second['candidate_ids'])
    assert s.ledger.used['ann'] <= 10
    for i in range(3):
        result = s.discover('Missing factual relation',f'gap{i}',requirement(),feedback=True)
    assert s.gap_calls == 2 and result['stop_reason'] == 'ann_budget_exhausted'
    assert s.ledger.used['ann'] <= 12 and s.ledger.used['set_score'] == 0


def test_default_full_bt_and_named_proxy_free_are_distinct_frozen_factors():
    assert resolve({})['proxy_mode'] == 'activation'
    assert resolve({},'fusion_proxy_free')['proxy_mode'] == 'none'
    assert resolve({},'fusion_proxy_free')['retrieval'] == 'bridge'
    with pytest.raises(ValueError,match='proxy_mode'):
        resolve({'fusion':{'proxy_mode':'cosine_surrogate'}})
    s = session(ann=12)
    result=s.discover('Find country','country',requirement())
    assert result['trace']['search_archive']['method_version'] == 'evidence_bridge_v1'
    assert s.scorer is not None and s.ledger.used['set_score'] > 0


def test_score_dependent_knobs_are_logged_disabled_and_have_no_proxy_free_effect():
    a = session(ann=12, proxy_mode='none')
    b = session(ann=12, proxy_mode='none', search={
        'coverage_roots':1,'exploration_roots':0,'quantum_new_sets':4,
        'quantum_measurements':1,'max_pivots':0,'max_pairs_per_state':0,
        'max_speculative_states':0,'max_speculative_depth':0,
    })
    ra=a.discover('Find country','country',requirement())
    rb=b.discover('Find country','country',requirement())
    signature=lambda r:[(x['stage'],x['target_id'],x['premise_ids'],x['candidate_ids'])
                        for x in r['trace']['retrieval']['proposal_batches']]
    assert signature(ra) == signature(rb)
    assert 'score_quantum' in rb['trace']['search_archive']['disabled_modules']
    assert 'pair_tests' in rb['trace']['search_archive']['disabled_modules']
