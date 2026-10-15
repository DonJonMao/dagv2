"""Structural, source-verifiable Web navigation; no models or oracle routes."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dagbt.budget import BudgetExceeded, Ledger
from dagbt.reasoning import InputOverflow
from dagbt.transport import digest
from dagbt.webwalker_browser import PageParser, WebBridge


def parsed(html, url='https://fixture.example/'):
    parser = PageParser(url)
    parser.feed(html)
    return parser.result()


def bridge_fixture(html, *, maximum_links=128, llm_limit=24, response=None, oversized_url=None,
                   paid_overflow=False, oversized_feedback=False):
    root = 'https://fixture.example/'
    document = parsed(html, root)
    pages = {root: document}
    for link in document['links']:
        pages[link['url']] = parsed('<title>Branch</title><main>Unchanged branch original.</main>', link['url'])
    events, requests, fetched = [], [], []
    ledger = Ledger({'llm': llm_limit})
    class Store:
        settings, events = {}, []
        def get(self, url):
            fetched.append(url)
            doc = pages[url]
            return {'url': url, 'document': doc, 'document_hash': digest(doc),
                    'raw_sha256': 'frozen-raw-hash', 'fetched_unix': 1}
    class Reasoner:
        def json(self, operation, system, data, validate, schema):
            if (len(data['links']) > maximum_links or any(link['url'] == oversized_url for link in data['links'])
                    or (oversized_feedback and data.get('reading_feedback', {}).get('accepted_empty_reads'))):
                raise InputOverflow('Controlled complete-wire capacity')
            ledger.reserve('llm', operation)
            requests.append(deepcopy(data))
            if paid_overflow:
                raise InputOverflow('Actual paid repair wire cannot fit')
            selected = response(data, ledger) if response else data['links'][0]['id']
            return validate({'link_id': selected})
    reader = SimpleNamespace(navigation_feedback=lambda node_id, identity=None: {
        'protocol': 'web_reading_feedback_v1', 'evidence_semantics': 'navigation_hints_not_facts',
        'query_identity': identity if isinstance(identity, str) else digest(identity), 'accepted_empty_reads': [
            {'doc_ids': ['previous-original'], 'source_ids': ['source-original'],
             'search_gaps': ['Inspect a different observed project branch.']}]})
    engine = SimpleNamespace(docs={}, ledger=ledger, s={'reserved_audit_calls': 1},
                             event=events.append, navigation_provenance={},
                             reasoner=Reasoner(), residual_control=SimpleNamespace(reader=reader))
    bridge = WebBridge(engine, root, Store(), {'max_actions': 36, 'link_window': 128})
    identity = {'node': 'project', 'parent_versions': {}}
    bridge.step('Find project details', 'project', identity=identity)
    return bridge, engine, requests, fetched, events, identity


def test_structural_views_preserve_all_original_text_and_all_observed_links():
    html = ('<title>Research</title><header><a href="/menu">Menu</a></header>'
            '<div role="dialog"><p>Consent details stay archived.</p>'
            '<a href="/consent">Manage consent</a><a href="[#PLACEHOLDER#]">Template</a></div>'
            '<main><p>The source identifies the project program.</p><a href="/details">Project details</a></main>'
            '<aside hidden><p>Hidden source remains archived.</p><a href="/hidden">Hidden settings</a></aside>'
            '<footer>Footer original.</footer>')
    page = parsed(html)
    for original in ('Consent details stay archived.', 'Hidden source remains archived.',
                     'The source identifies the project program.', 'Footer original.'):
        assert original in page['text']
    assert len(page['links']) == 5
    roles = {page['text'][region['start']:region['end']]: region['role']
             for region in page['text_regions']}
    assert any('Consent details' in text and role == 'utility_ui' for text, role in roles.items())
    assert any('project program' in text and role == 'main' for text, role in roles.items())
    assert any('Hidden source' in text and role == 'hidden_ui' for text, role in roles.items())
    assert page['link_contexts']['https://fixture.example/details']['region'] == 'main'
    assert any(context['region'] == 'template_ui' for context in page['link_contexts'].values())


def test_minified_dialog_and_main_on_one_line_keep_distinct_exact_body_regions():
    page = parsed('<div role="dialog">' + 'Consent setting. ' * 600 + '</div>'
                  '<main>Substantive original body. <a href="/details">Useful branch</a></main>')
    body = [page['text'][region['start']:region['end']] for region in page['text_regions']
            if region['role'] == 'main']
    assert 'Substantive original body. Useful branch' in ''.join(body)
    assert not any('Consent setting.' in text for text in body)
    assert page['link_contexts']['https://fixture.example/details']['region'] == 'main'


def test_navigation_uses_original_body_and_link_context_beyond_long_utility_prefix():
    html = ('<title>Research</title><div role="dialog">' + 'Consent storage settings. ' * 500
            + '<a href="/service">Consent provider</a></div><header><a href="/menu">Menu</a></header>'
            '<main><p>Substantive original body beyond the first six thousand characters.</p>'
            '<a href="/project">Project details</a></main>')
    bridge, engine, requests, fetched, events, identity = bridge_fixture(html)
    root_doc = next(iter(engine.docs.values()))
    assert len(root_doc.passage) > 6000 and 'Substantive' not in root_doc.passage[:6000]
    bridge.step('Find project details', 'project', identity=identity)
    request = requests[0]
    assert fetched[-1] == 'https://fixture.example/project'
    assert 'Substantive original' in request['source_pages'][0]['text']
    assert request['reading_feedback']['evidence_semantics'] == 'navigation_hints_not_facts'
    assert request['reading_feedback']['query_identity'] == digest(identity)
    for source in request['source_pages']:
        original = engine.docs[source['doc_id']].passage
        for interval in source['source_ranges']:
            assert source['text'][interval['excerpt_start']:interval['excerpt_end']] == original[interval['start']:interval['end']]
    for link in request['links']:
        context = link.get('source_context')
        if context:
            assert context['text'] == engine.docs[link['source_doc_id']].passage[context['start']:context['end']]
    assert 'Consent storage settings.' in root_doc.passage
    assert bridge.pages[root_doc.doc_id]['document_hash'] == digest(bridge.pages[root_doc.doc_id]['document'])


def test_capacity_shortened_frontier_rotates_without_dropping_unselected_urls():
    html = ('<title>Index</title><div role="dialog">' + ''.join(
        '<a href="/utility-%d">Consent control %d</a>' % (i, i) for i in range(4))
        + '</div><main>' + ''.join('<a href="/project-%d">Project %d</a>' % (i, i) for i in range(6)) + '</main>')
    bridge, engine, requests, fetched, events, identity = bridge_fixture(html, maximum_links=2)
    root_doc_id = next(iter(engine.docs))
    original_frontier = {link['url'] for link in bridge.pages[root_doc_id]['document']['links']}
    for _ in range(3):
        bridge.step('Find project details', 'project', identity=identity)
    assert all(len(row['links']) <= 2 for row in requests)
    # Unselected project-1 from window one cannot keep project-2/3 behind it.
    assert {link['url'] for link in requests[0]['links']}.isdisjoint(
        {link['url'] for link in requests[1]['links']})
    assert all('/project-' in url for url in fetched[1:])
    state = bridge.sessions[digest(identity)]
    remaining = {link['url'] for link in bridge._observed_links(state)}
    assert remaining | set(fetched[1:]) == original_frontier
    assert state['skipped'] == [] and bridge.has_unvisited_links('project', identity=identity)
    assert any('/utility-' in url for url in remaining)


def test_unvisited_link_probe_is_read_only_and_budget_prevents_fetch_without_read_call():
    bridge, engine, requests, fetched, events, identity = bridge_fixture(
        '<main><a href="/project">Project</a></main>', llm_limit=2)
    before = deepcopy(bridge.snapshot()), deepcopy(engine.ledger.public_dict()), deepcopy(events)
    assert bridge.has_unvisited_links('project', identity=identity)
    assert before == (bridge.snapshot(), engine.ledger.public_dict(), events)
    with pytest.raises(BudgetExceeded, match='web_navigation_and_followup_read'):
        bridge.step('Find project details', 'project', identity=identity)
    assert len(fetched) == 1 and not requests
    assert engine.ledger.limits == {'llm': 2, 'web_actions': 36}


def test_navigation_repair_cannot_spend_followup_read_then_fetch_a_new_page():
    def consume_last(data, ledger):
        ledger.reserve('llm', 'bounded-navigation-repair')
        return data['links'][0]['id']
    bridge, engine, requests, fetched, events, identity = bridge_fixture(
        '<main><a href="/project">Project</a></main>', llm_limit=3, response=consume_last)
    with pytest.raises(BudgetExceeded, match='web_navigation_followup_read'):
        bridge.step('Find project details', 'project', identity=identity)
    assert len(fetched) == 1 and len(requests) == 1
    assert engine.ledger.remaining('llm') == 1


def test_null_declines_only_this_window_and_next_probe_exposes_other_observed_links():
    bridge, engine, requests, fetched, events, identity = bridge_fixture(
        '<main>' + ''.join('<a href="/p%d">Project %d</a>' % (i, i) for i in range(4)) + '</main>',
        maximum_links=2, response=lambda data, ledger: None)
    result = bridge.step('Find project details', 'project', identity=identity)
    assert result['trace']['search_status'] == 'empty_probe' and len(fetched) == 1
    assert bridge.has_unvisited_links('project', identity=digest(identity))
    bridge.step('Find project details', 'project', identity=identity)
    assert {link['url'] for link in requests[0]['links']}.isdisjoint(
        {link['url'] for link in requests[1]['links']})
    assert not bridge.has_unvisited_links('project', identity='different-parent-scope')
    assert engine.ledger.used['llm'] == 2


def test_one_unfittable_link_does_not_hide_a_fitting_sibling_or_remove_the_original():
    bridge, engine, requests, fetched, events, identity = bridge_fixture(
        '<main><a href="/oversized">Long original anchor</a><a href="/fitting">Another branch</a></main>',
        oversized_url='https://fixture.example/oversized')
    bridge.step('Find project details', 'project', identity=identity)
    assert fetched[-1] == 'https://fixture.example/fitting'
    state = bridge.sessions[digest(identity)]
    assert any(link['url'].endswith('/oversized') for link in bridge._observed_links(state))
    assert len(requests) == 1 and engine.ledger.used['llm'] == 1


def test_paid_repair_overflow_never_reopens_the_navigation_attempt_allowance():
    bridge, engine, requests, fetched, events, identity = bridge_fixture(
        '<main><a href="/first">First branch</a><a href="/other">Other branch</a></main>',
        paid_overflow=True)
    with pytest.raises(InputOverflow, match='Actual paid repair'):
        bridge.step('Find project details', 'project', identity=identity)
    assert len(requests) == 1 and engine.ledger.used['llm'] == 1 and len(fetched) == 1


def test_oversized_optional_reading_feedback_is_windowed_without_editing_its_archive():
    bridge, engine, requests, fetched, events, identity = bridge_fixture(
        '<main><a href="/first">First branch</a></main>', oversized_feedback=True)
    original = engine.residual_control.reader.navigation_feedback('project', identity=identity)
    bridge.step('Find project details', 'project', identity=identity)
    offered = requests[0]['reading_feedback']
    assert offered['accepted_empty_reads'] == [] and offered['omitted_accepted_empty_read_count'] == 1
    assert offered['evidence_semantics'] == 'navigation_hints_not_facts'
    assert engine.residual_control.reader.navigation_feedback('project', identity=identity) == original
    assert len(fetched) == 2 and engine.ledger.used['llm'] == 1
