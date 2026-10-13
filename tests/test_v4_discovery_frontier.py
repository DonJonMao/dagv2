"""Offline source-work transitions through the actual reader and transport."""
from copy import deepcopy
import json

import pytest

from dagbt.joint_reading import JointReader
from dagbt.reasoning import InputOverflow, ProtocolError, RefusalError
from test_personamem_joint_protocol import fixture, observation, response, visible


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Discovery frontier tests must not contact services')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def discover(reader, docs=None, identity='discovery'):
    docs=list(reader.e.docs) if docs is None else docs
    return reader.read(reader.e.steps[0], 'Review the original historical evidence', ['history'],
                       docs, identity, discovery_doc_ids=docs)


def aliases(record):
    return {source['source_alias'] for source in record['sources']}


def test_uncited_full_wire_is_conditionally_assessed_without_becoming_irrelevant_or_unseen():
    reader, e, state, calls=fixture(max_repairs_per_request=0,context_tokens=100000)
    discover(reader)
    assert len(calls.requests)==1 and not reader.discovery_frontier
    assert aliases(reader.source_exposures[0])=={'s1','s2','s3'}
    assert reader.source_exposures[0]['status']=='batch_assessed'
    assert reader.source_exposures[0]['request_status']=='returned'
    assert reader.source_exposures[0]['logical_call_index']==1
    assert reader.read_states=={('discovery',source):'read' for source in ('s1','s2','s3')}
    assert not state.pending and not reader.unread_obligations
    assert state.valid()[1]['history']==[e.mapper.sources['s1']['source_id']]
    data=calls.requests[0]['data']
    assert 'reviewed_scope_id' not in data
    assert set(calls.respond(1,data))=={'slots'}, 'Existing slot protocol alone performs the assessment'


def test_preflight_only_discovery_registration_does_not_create_semantic_progress():
    reader, e, state, calls=fixture(max_repairs_per_request=0,context_tokens=100000)
    before=reader.material_signature()
    ledger=deepcopy(e.ledger.used)
    reader._fits=lambda operation,data:False
    with pytest.raises(InputOverflow):
        discover(reader)
    assert not calls.requests and not reader.source_exposures
    assert len(reader.discovery_frontier)==3
    assert all(item['status']=='unoffered' for item in reader.discovery_frontier.values())
    assert not reader.pending_variables() and not state.pending
    assert reader.material_signature()==before and e.ledger.used==ledger
    # Queue reordering/removal is scheduling only, not a new reading result.
    reader.discovery_frontier=dict(reversed(list(reader.discovery_frontier.items())))
    reader.discovery_frontier.pop(('discovery','s3'))
    assert reader.material_signature()==before


def test_required_old_proof_promotes_frontier_and_preserves_original_discovery_provenance():
    reader, e, state, calls=fixture(max_repairs_per_request=0,context_tokens=100000)
    reader._fits=lambda operation,data:False
    with pytest.raises(InputOverflow):
        discover(reader)
    old_origin=deepcopy(reader.discovery_frontier['discovery','s2'])
    state.integrate([reader._observation(observation('old-proof',['s2']),
                     'history',visible(reader,['s2']),e.steps[0])])
    reader._fits=lambda operation,data:True
    discover(reader,['piano'])
    assert reader.batches[-1]['source_aliases']==['s1']
    assert aliases(reader.source_exposures[-1])=={'s1','s2'}
    assert reader.read_states['discovery','s2']=='read'
    assert ('discovery','s2') not in reader.discovery_frontier
    assert reader.discovery_promotions==[{'status':'required_evidence','origin':old_origin}]
    assert ('discovery','s3') in reader.discovery_frontier
    assert not state.pending
    restored=JointReader(e,state)
    restored.restore_obligations(json.loads(json.dumps(reader.obligation_snapshot())))
    assert restored.obligation_snapshot()==reader.obligation_snapshot()


def test_known_unoffered_joint_premises_stay_hard_on_capacity_failure():
    reader, e, state, calls=fixture(max_repairs_per_request=0,context_tokens=100000)
    state.integrate([reader._observation(observation('joint-premise',['s1','s2']),
                     'history',visible(reader,['s1','s2']),e.steps[0])])
    reader._fits=lambda operation,data:False
    with pytest.raises(InputOverflow):
        discover(reader)
    assert not calls.requests and not reader.source_exposures
    assert {('discovery','s1'),('discovery','s2')}<=set(reader.unread_obligations)
    assert set(reader.discovery_frontier)=={('discovery','s3')}
    assert state.pending=={'history'}


@pytest.mark.parametrize('error',[OSError,RefusalError])
def test_failed_request_keeps_full_wire_hard_without_inventing_read_results(error):
    reader, e, state, calls=fixture(max_repairs_per_request=0,context_tokens=100000)
    def fail(index,data):
        assert {source['id'] for source in data['source_spans']}=={'s1','s2','s3'}
        raise error('offline interruption')
    calls.respond=fail
    if error is RefusalError:
        with pytest.raises(RefusalError):
            discover(reader)
    else:
        discover(reader)
    assert not reader.discovery_frontier and not state.valid()[0]
    assert state.pending==reader.pending_variables()=={'history'}
    record=reader.source_exposures[0]
    assert aliases(record)=={'s1','s2','s3'} and record['request_status']=='failed'
    assert record['logical_call_index']==1
    assert record['status']!='batch_assessed'
    assert all(value!='read' for value in reader.read_states.values())
    assert reader.unread_obligations if error is RefusalError else reader.failed_batch_reviews


@pytest.mark.parametrize('change',['scope','program','parent_versions'])
def test_same_query_identity_cannot_reuse_frontier_under_changed_scope_or_version(change):
    reader, e, state, calls=fixture(max_repairs_per_request=0,context_tokens=100000)
    reader._fits=lambda operation,data:False
    with pytest.raises(InputOverflow):
        discover(reader)
    if change=='scope':
        state.contract['time_scope']='different-history'
    elif change=='program':
        state.revision+=1
    else:
        e.steps[0]['inputs']=['suggestion']
    reader._fits=lambda operation,data:True
    before=deepcopy(reader.obligation_snapshot())
    with pytest.raises(ProtocolError,match='identity cannot change'):
        discover(reader)
    assert not calls.requests and reader.obligation_snapshot()==before


@pytest.mark.parametrize('corruption',['missing_scope','parent_type','exposed','read'])
def test_checkpoint_cannot_reclassify_observed_or_unknown_scope_work_as_discovery(corruption):
    reader, e, state, calls=fixture(max_repairs_per_request=0,context_tokens=100000)
    reader._fits=lambda operation,data:False
    with pytest.raises(InputOverflow):
        discover(reader)
    snapshot=reader.obligation_snapshot()
    if corruption=='missing_scope':
        del snapshot['discovery_frontier'][0][2]['scopes']
    elif corruption=='parent_type':
        snapshot['discovery_frontier'][0][2]['parent_versions']={'parent':False}
    elif corruption=='read':
        snapshot['read_states']=[['discovery','s1','read']]
    else:
        reader._fits=lambda operation,data:True
        discover(reader,['piano'])
        snapshot['source_exposures']=deepcopy(reader.source_exposures)
    restored=JointReader(e,state)
    with pytest.raises(ProtocolError):
        restored.restore_obligations(snapshot)


def test_alias_permutation_preserves_material_progress_and_hard_source_ownership():
    reader, e, state, calls=fixture(max_repairs_per_request=0,context_tokens=100000)
    discover(reader)
    reader.unread_obligations['later-scope','s3']={'history'}
    state.integrate([],reader.pending_variables())
    before=reader.material_signature()
    snapshot={**reader.obligation_snapshot(),
        'read_states':[[*key,value] for key,value in reader.read_states.items()],
        'clues':deepcopy(reader.clues)}
    rename={'s1':'s3','s2':'s1','s3':'s2'}
    e.mapper.sources={rename[alias]:{**source,'id':rename[alias]} for alias,source in e.mapper.sources.items()}
    e.mapper.source_aliases={source['source_id']:alias for alias,source in e.mapper.sources.items()}
    restored=JointReader(e,state)
    rebound=restored.restore_obligations(snapshot)
    restored.read_states={(i,alias):value for i,alias,value in rebound['read_states']}
    restored.clues=rebound['clues']
    assert restored.unread_obligations=={('later-scope','s2'):{'history'}}
    assert restored.material_signature()==before, 'Temporary alias changes cannot buy semantic progress'
    assert not restored.discovery_frontier
