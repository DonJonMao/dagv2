"""Multi-issue feedback and exact same-batch reference lifecycle checks."""
from copy import deepcopy

import pytest

from dagbt.joint_reading import JointReader, SYSTEM
from dagbt.reasoning import ProtocolError
from test_personamem_joint_protocol import observation
from test_v4_obligation_review_feedback_review import (
    OID, PROOF, claim, failed_slot, irrelevant, obligation_fixture, read_issue,
)


def test_multiple_coverage_only_issues_report_every_exact_missing_set():
    reader,engine,state,calls,aliases=obligation_fixture()
    second=deepcopy(reader.evidence_obligations[OID])
    second['id']='second-scoped-issue'
    second['source_ids']=second['source_ids'][1:10]
    reader.evidence_obligations[second['id']]=second
    def reply(index,data):
        if index==2:
            details=failed_slot(data)['obligation_review_sources']
            assert len(details)==2
            by_id={item['obligation_id']:item for item in details}
            assert set(by_id[OID]['missing_source_ids'])==set(aliases[3:53])
            assert set(by_id[second['id']]['missing_source_ids'])==set(aliases[3:10])
            result=claim(irrelevant('independent-original-review',aliases[3:53]),refs=[PROOF])
        else:
            result=claim(observation(PROOF,aliases[:3]))
        result['slots'][0]['resolved_obligation_ids']=[OID,second['id']]
        return result
    calls.respond=reply
    read_issue(reader,engine)
    assert len(calls.requests)==2 and not state.pending
    assert not reader.evidence_obligations


def test_other_issue_failure_suppresses_coverage_feedback_and_frozen_authorization():
    reader,engine,state,calls,aliases=obligation_fixture()
    def reply(index,data):
        if index==2:
            item=failed_slot(data)
            assert 'obligation_review_sources' not in item
            assert 'frozen_observation_reviews' not in item
        result=claim(observation(PROOF,aliases[:3]))
        result['slots'][0]['resolved_obligation_ids']=[OID,'not-an-offered-issue']
        return result
    calls.respond=reply
    read_issue(reader,engine)
    assert OID in reader.evidence_obligations and state.pending


def test_next_repair_transmits_complete_frozen_row_when_previous_slot_only_referenced_it():
    reader,engine,state,calls,aliases=obligation_fixture(repairs=2)
    def reply(index,data):
        if index==1:
            return claim(observation(PROOF,aliases[:3]))
        item=failed_slot(data)
        if index==2:
            return claim(irrelevant('first-forty-nine',aliases[3:52]),refs=[PROOF])
        assert index==3
        offer=item['frozen_observation_reviews']
        assert offer['slot_observation_ids']==['first-forty-nine']
        assert len(offer['observations'])==1
        preserved=offer['observations'][0]
        assert preserved==observation(PROOF,aliases[:3])
        assert item['obligation_review_sources'][0]['missing_source_ids']==[aliases[52]]
        return claim(irrelevant('last-independent-source',aliases[52:53]),refs=[PROOF,'first-forty-nine'])
    calls.respond=reply
    read_issue(reader,engine)
    assert len(calls.requests)==3 and engine.ledger.used['json_repairs']==2
    assert not state.pending and not reader.evidence_obligations
    assert state.valid()[0]['history']==['piano']
    assert len(state.valid()[1]['history'])==3


def test_controller_feedback_rejects_mapping_two_originals_to_one_visible_alias():
    feedback=[{'obligation_id':'scoped-issue','required_source_ids':['first','second'],
        'covered_source_ids':['first'],'missing_source_ids':['second']}]
    with pytest.raises(ProtocolError,match='ambiguous'):
        JointReader._remap_obligation_review_sources(feedback,{'first':'same','second':'same'})


def test_source_remapping_is_unambiguous_across_separate_obligations_too():
    feedback=[{'obligation_id':source,'required_source_ids':[source],
        'covered_source_ids':[],'missing_source_ids':[source]} for source in ('first','second')]
    with pytest.raises(ProtocolError,match='ambiguous'):
        JointReader._remap_obligation_review_sources(feedback,{'first':'same','second':'same'})


def test_initial_prompt_distinguishes_issue_review_coverage_from_support_premises():
    assert 'cover EVERY source_ids' in SYSTEM
    assert 'Review coverage is not a claim that every source' in SYSTEM
    assert 'Do not assert hypothetical' in SYSTEM
    assert 'without having read every memory' in SYSTEM
