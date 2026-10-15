"""An old sourced issue closes only through an explicit original-text review."""
from copy import deepcopy

from dagbt.evidence_spans import build_source_spans
from test_webwalker_reading import web_fixture


def observation(identifier, sources, *, stance, value, reason):
    return {'id': identifier, 'stance': stance, 'value': value, 'source_ids': list(sources),
            'entity_scope': 'user', 'time_scope': 'history', 'parent_versions': {},
            'retract_ids': [], 'reason': reason}


def menu_fixture(respond):
    reader, engine, state, calls = web_fixture(respond, max_repairs_per_request=0)
    for doc_id, alias, text in [('piano', 's1', 'Navigation menu: About and Contact.'),
                                ('painting', 's2', 'Navigation menu: Search and Help.')]:
        doc = engine.docs[doc_id]
        doc.passage = text
        old = engine.mapper.sources[alias]['source_id']
        source = build_source_spans(doc_id, doc, 400)[0]
        engine.mapper.sources[alias] = {**source, 'id': alias, 'source_id': source['id']}
        del engine.mapper.source_aliases[old]
        engine.mapper.source_aliases[source['id']] = alias
    return reader, engine, state, calls


def slot(row, **fields):
    return {'slots': [{'variable': 'history', 'reading_status': 'evidence_assessment',
                       'observations': [row], 'unresolved': [], **fields}]}


def execute(reader, docs, identity):
    return reader.read(reader.e.steps[0], 'Find historical activity evidence', ['history'],
                       docs, identity, discovery_doc_ids=docs)


def test_sourced_irrelevant_review_can_close_an_earlier_mistaken_unknown_without_binding():
    def respond(index, data):
        if index == 1:
            return slot(observation('initial-uncertainty', ['s1'], stance='unknown', value=None,
                                    reason='The navigation page does not provide an activity fact.'))
        obligation = data['evidence_obligations'][0]
        return slot(observation('original-text-review', obligation['source_ids'], stance='irrelevant',
                                value=None, reason='Re-read the original menu: these are navigation labels, '
                                                   'with no relevant activity statement.'),
                    resolved_obligation_ids=[obligation['id']])
    reader, engine, state, calls = menu_fixture(respond)
    execute(reader, ['piano'], 'first-assessment')
    old = deepcopy(reader.evidence_obligations)
    assert old and state.pending == {'history'}
    execute(reader, ['piano'], 'original-text-review')
    assert not reader.evidence_obligations and not state.pending
    assert state.valid()[0] == {} and reader.web_search_gaps == []
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 0
    review = calls.requests[1]['data']
    assert review['evidence_obligations'][0]['id'] == next(iter(old))
    assert {source['id'] for source in review['source_spans']} == {'s1'}
    assert reader.read_states['original-text-review', 's1'] == 'read'


def test_partial_original_source_review_cannot_close_a_multi_source_issue():
    def respond(index, data):
        if index == 1:
            return slot(observation('initial-uncertainty', ['s1', 's2'], stance='unknown', value=None,
                                    reason='These navigation pages do not provide an activity fact.'))
        obligation = data['evidence_obligations'][0]
        return slot(observation('incomplete-review', ['s1'], stance='irrelevant', value=None,
                                reason='This first menu contains only navigation labels.'),
                    resolved_obligation_ids=[obligation['id']])
    reader, engine, state, calls = menu_fixture(respond)
    execute(reader, ['piano', 'painting'], 'first-assessment')
    original = deepcopy(reader.evidence_obligations)
    execute(reader, ['piano'], 'incomplete-review')
    assert reader.evidence_obligations == original and state.pending == {'history'}
    assert state.valid()[0] == {} and len(calls.requests) == 2
    assert {engine.mapper.sources[alias]['source_id'] for alias in ('s1', 's2')} == set(
        next(iter(reader.evidence_obligations.values()))['source_ids'])
