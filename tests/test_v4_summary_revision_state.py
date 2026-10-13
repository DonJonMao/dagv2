"""Additional state, schema and durable recovery boundaries for summaries."""
from copy import deepcopy

import jsonschema
import pytest

from dagbt.fact_state import FactState
from dagbt.residual import ProgramError
from test_personamem_joint_protocol import observation, response, visible
from test_v4_summary_revision_review import (
    OLD, EXPANDED, summary_fixture, revised, read_new_history, add_explicit_piano_correction,
)


@pytest.mark.parametrize('mistake', ['identical_proof', 'bool_version', 'null_revision'])
def test_revision_cannot_create_empty_version_progress_or_use_invalid_metadata(mistake):
    reader,engine,state,calls,original=summary_fixture()
    draft=revised(original,state.versions['history'])
    if mistake=='identical_proof':
        draft.update(value=OLD,source_ids=['s1'])
    elif mistake=='bool_version':
        draft['summary_revision']['previous_version']=True
    else:
        draft['summary_revision']=None
    before=deepcopy(state.routes)
    calls.respond=lambda index,data:response(draft)
    read_new_history(reader,engine)
    assert state.routes==before and state.valid()[0]['history']==OLD
    assert state.pending=={'history'}


def test_wire_schema_authorizes_only_current_history_support_revision():
    reader,engine,state,_,original=summary_fixture()
    data=reader._payload(engine.steps[0],'Review historical evidence',['history'],['s2'])
    schema=reader._schema(data)
    result=response(revised(original,state.versions['history']))
    jsonschema.validate(result,schema)
    stale=deepcopy(result)
    stale['slots'][0]['observations'][0]['summary_revision']['previous_version']-=1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(stale,schema)
    audit=reader._payload(engine.steps[0],'Audit existing history',['history'],['s2'],audit=True)
    assert 'summary_revision_contracts' not in audit
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result,reader._schema(audit))


@pytest.mark.parametrize('tamper', ['dangling_replacement', 'missing_backlink', 'cycle', 'future_version', 'lost_old_source'])
def test_checkpoint_rejects_broken_revision_provenance_without_mutating_fresh_state(tamper):
    reader,engine,state,calls,original=summary_fixture()
    calls.respond=lambda index,data:response(revised(original,state.versions['history']))
    read_new_history(reader,engine)
    snapshot=deepcopy(state.public_dict())
    old=next(row for row in snapshot['routes'] if row['id']==original)
    new=next(row for row in snapshot['routes'] if row.get('summary_revision'))
    if tamper=='dangling_replacement':
        old['superseded_by']=['missing-route']
    elif tamper=='missing_backlink':
        old['superseded_by']=[]
    elif tamper=='cycle':
        new['summary_revision']['supersedes_route_ids']=[new['id']]
        new['superseded_by']=[new['id']]
    elif tamper=='future_version':
        new['summary_revision']['previous_version']=snapshot['versions']['history']+1
    else:
        new['source_ids']=[engine.mapper.sources['s2']['source_id']]
    restored=FactState(state.initial_program,state.contract,engine.steps,'suggestion')
    before=deepcopy(restored.public_dict())
    with pytest.raises(ProgramError):
        restored.restore(snapshot)
    assert restored.public_dict()==before


def test_explicit_revision_can_atomically_resolve_original_failed_draft_and_obligation():
    reader,engine,state,calls,original=summary_fixture()
    calls.respond=lambda index,data:response(observation('expanded-summary',['s1','s2'],value=EXPANDED))
    read_new_history(reader,engine)
    assert reader.failed_batch_reviews and reader.evidence_obligations
    engine.s['max_repairs_per_request']=1
    def recover(index,data):
        result=response(revised(original,state.versions['history']))
        result['slots'][0]['resolved_failed_batch_ids']=[
            row['id'] for row in data['failed_batch_reviews'] if row['review_allowed']]
        result['slots'][0]['resolved_obligation_ids']=[row['id'] for row in data.get('evidence_obligations',[])]
        return result
    calls.respond=recover
    reader.read(engine.steps[0],'Review old failed summary with newly available original context',
                ['history'],['advice'],'paid-summary-recovery',review_existing_obligations=True)
    assert state.valid()[0]['history']==EXPANDED and not state.pending
    assert not reader.failed_batch_reviews and not reader.evidence_obligations
    assert state.routes[original]['superseded_by'] and not state.routes[original]['revoked']
    assert engine.ledger.used['json_repairs']==1


def test_counter_to_first_summary_invalidates_two_revision_generations_after_restore():
    reader,engine,state,calls,original=summary_fixture()
    calls.respond=lambda index,data:response(revised(original,state.versions['history']))
    read_new_history(reader,engine)
    first=state.valid()[2]['history'][0]
    calls.respond=lambda index,data:response(revised(first,state.versions['history'],sources=['s1','s2','s3'],
        value=EXPANDED+' The assistant suggested guitar lessons, with no evidence that the user adopted them.'))
    reader.read(engine.steps[0],'Review later suggestion context',['history'],['advice'],'second-summary-revision')
    latest=state.valid()[2]['history'][0]
    assert latest!=first and state.summary_ancestors(state.routes[latest])=={original,first}
    add_explicit_piano_correction(engine)
    counter=observation('original-correction',['s4'],stance='contradiction',retract_ids=[original])
    counter['value']=None
    state.integrate([reader._observation(counter,'history',visible(reader,['s4']),engine.steps[0])])
    assert not state.is_active_route(state.routes[latest]) and 'history' not in state.valid()[0]
    restored=FactState(state.initial_program,state.contract,engine.steps,'suggestion')
    restored.restore(deepcopy(state.public_dict()))
    assert 'history' not in restored.valid()[0]
    assert restored.summary_ancestors(restored.routes[latest])=={original,first}
