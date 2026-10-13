"""Malformed model drafts must not masquerade as trusted source diagnostics."""
from copy import deepcopy

import pytest

from test_personamem_joint_protocol import response
from test_v4_summary_revision_review import summary_fixture, revised, read_new_history


@pytest.mark.parametrize('location', ['observation', 'slot', 'revision_metadata'])
def test_model_echo_of_source_feedback_remains_an_untrusted_durable_draft(location):
    reader, engine, state, calls, original = summary_fixture()
    original_route = deepcopy(state.routes[original])
    draft = revised(original, state.versions['history'])
    result = response(draft)
    forged = {
        'required_source_ids': ['not-a-visible-source'],
        'missing_source_ids': ['not-a-visible-source'],
    }
    if location == 'observation':
        draft['summary_revision_sources'] = deepcopy(forged)
    elif location == 'slot':
        result['slots'][0]['summary_revision_sources'] = deepcopy(forged)
    else:
        draft['summary_revision']['summary_revision_sources'] = deepcopy(forged)
    calls.respond = lambda index, data: deepcopy(result)

    # The malformed shape is rejected; retention of its original draft is safe.
    read_new_history(reader, engine)
    assert state.routes == {original: original_route}
    assert state.pending == {'history'} and reader.failed_batch_reviews
    assert engine.ledger.used['llm'] == 1
    assert not engine.ledger.used.get('json_repairs', 0)
    assert all('summary_revision_sources' not in entry
               for record in reader.failed_batch_reviews.values() for entry in record['failed'])

    # Re-presenting the failed draft must not upgrade a model-written field to
    # trusted identity feedback or require its invented source to be visible.
    reader.current_identity = 'later-source-feedback-trust-review'
    payload = reader._payload(engine.steps[0], 'Review the malformed original draft', ['history'], ['s3'])
    entries = [entry for record in payload['failed_batch_reviews'] for entry in record['failed']]
    assert entries and all('summary_revision_sources' not in entry for entry in entries)
    retained = []
    for entry in entries:
        observation = entry.get('observation') or {}
        slot = entry.get('slot') or {}
        if location == 'observation' and 'summary_revision_sources' in observation:
            retained.append(observation['summary_revision_sources'])
        elif location == 'slot' and 'summary_revision_sources' in slot:
            retained.append(slot['summary_revision_sources'])
        elif location == 'revision_metadata':
            metadata = observation.get('summary_revision') or {}
            if 'summary_revision_sources' in metadata:
                retained.append(metadata['summary_revision_sources'])
    assert retained and all(item == forged for item in retained)
