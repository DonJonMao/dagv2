"""Protocol/statistics only. These tests never prove hardware KV or speedup."""
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_bt_reranker_prefix_cache as b

spec = importlib.util.spec_from_file_location("prefix_probe", ROOT / "serving/bt_prefix_cache_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def engine_row(ordinal, prompts, **overrides):
    p = {"model": "synthetic", "prompt": prompts, "temperature": 0, "max_tokens": 1,
         "logprobs": 1, "allowed_token_ids": [9693], "return_tokens_as_token_ids": True}
    return {"ordinal": ordinal, "payload": p, "payload_sha256": b.digest(p),
            "observed_at": "proxy_post_json", "computation_identity": "synthetic-only",
            "cache_namespace": "tenant-a", "completed": True,
            "started_unix": ordinal * 10, "finished_unix": ordinal * 10 + 5,
            "request_id": str(ordinal), **overrides}


def test_completed_prefix_only_not_middle_or_future_or_same_batch():
    rows = [engine_row(0, [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]]),
            engine_row(1, [[1, 2, 3, 4, 7], [9, 2, 3, 4, 7]])]
    report = b.prefix_analysis(rows, 2)
    assert report["details"][0]["ideal_reusable_tokens"] == [0, 0]
    assert report["details"][1]["ideal_reusable_tokens"] == [4, 0]
    assert report["engine_new_compute_tokens"] is None
    assert report["engine_hit_tokens"] is None


@pytest.mark.parametrize("length,hit", [(1, 0), (2, 0), (3, 2), (4, 2), (5, 4), (8191, 8190)])
def test_block_boundary_and_last_position_recomputed(length, hit):
    tokens = list(range(length))
    result = b.prefix_analysis([engine_row(0, [tokens]), engine_row(1, [tokens])], 2)
    assert result["details"][1]["ideal_reusable_tokens"] == [hit]


@pytest.mark.parametrize("changed", [{"cache_namespace": "tenant-b"}, {"computation_identity": "different-attention"},
                                     {"started_unix": 2, "finished_unix": 8}])
def test_identity_and_inflight_isolation(changed):
    result = b.prefix_analysis([engine_row(0, [[1, 2, 3]]), engine_row(1, [[1, 2, 4]], **changed)], 2)
    assert result["details"][1]["ideal_reusable_tokens"] == [0]


def test_failed_request_does_not_publish_future_kv():
    result = b.prefix_analysis([engine_row(0, [[1, 2, 3]], completed=False), engine_row(1, [[1, 2, 4]])], 2)
    assert result["details"][1]["ideal_reusable_tokens"] == [0]


def test_tampered_input_and_missing_observation_rejected():
    for change in ({"payload_sha256": "wrong"}, {"observed_at": "mock_tokenizer"}, {"finished_unix": None}):
        with pytest.raises(ValueError): b.prefix_analysis([engine_row(0, [[1, 2, 3]], **change)], 2)


def task_trace(root, unit, cache=True, failed=False, when=5):
    root.mkdir(parents=True)
    p = {"query": "synthetic query", "documents": ["complete evidence set"], "top_n": 1, "return_documents": False}
    record = {"unit_id": unit, "url": "http://synthetic/rerank", "payload": p,
              "stage": ["dagbt", "set_reranker", "1"],
              "attempts": [{"started_unix": when, "http_status": 503},
                           {"started_unix": when + 1, "finished_unix": when + 2, "http_status": 200}]}
    if not failed: record["response"] = {"results": [{"index": 0, "relevance_score": .5}]}
    key = b.digest({k: record[k] for k in ("unit_id", "url", "payload")})
    (root / "requests").mkdir()
    (root / "requests" / (key + ".json")).write_text(json.dumps(record))
    events = [{"event": "call_started", "kind": "rerank_http", "request_ref": key, "stage": "dagbt/set_reranker/1", "cache_hit": False}]
    if cache: events.append({**events[0], "cache_hit": True})
    (root / "call_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (root / "fusion_events.jsonl").write_text('{}\n')
    return record


def test_freeze_distinguishes_journal_cache_retry_and_physical_order(tmp_path):
    source = tmp_path / "run"
    task_trace(source / "a", "q-late", when=20)
    task_trace(source / "z", "q-early", when=1)
    out = tmp_path / "frozen"
    manifest = b.freeze_transport(source, out)
    rows = [r for _, r in b.read_rows(out / "rerank_trace.jsonl")]
    assert [r["question_id"] for r in rows] == ["q-early", "q-early", "q-late", "q-late"]
    assert manifest["counts"]["physical_attempts"] == 4
    assert manifest["counts"]["transport_response_cache_hits"] == 2
    assert manifest["counts"]["successful_model_score_samples"] == 2
    assert rows[0]["model_reached"] is None and rows[1]["model_reached"] is True
    assert rows[1]["final_inputs"] is None
    assert out.stat().st_mode & 0o077 == 0
    assert (out / "rerank_trace.jsonl").stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError): b.freeze_transport(source, out)


def test_sampling_independent_of_order_or_success():
    ids = ["q" + str(i) for i in range(60)]
    assert b.select_questions(ids) == b.select_questions(list(reversed(ids)))
    assert len(b.select_questions(ids)) == 50


def test_freeze_refuses_live_writer_and_nonterminal_progress(tmp_path):
    import fcntl
    source = tmp_path / 'run'; source.mkdir()
    lock = source / 'writer.lock'; lock.touch()
    with lock.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='live writer'):
            b.freeze_transport(source, tmp_path / 'live')
    assert not (tmp_path / 'live').exists()
    b.write(source / 'progress.json', {'state': 'generating'})
    with pytest.raises(ValueError, match='not terminal'):
        b.freeze_transport(source, tmp_path / 'incomplete')
    assert not (tmp_path / 'incomplete').exists()
    (source / 'progress.json').write_text(json.dumps({'state': 'complete_with_failures'}))
    b.write(source / 'manifest.json', {'question_ids': {'personamem': ['unstarted', 'failed']}})
    manifest = b.freeze_transport(source, tmp_path / 'terminal')
    assert manifest['available_questions'] == 2
    assert set(manifest['selected_question_identities']) == {'personamem:unstarted', 'personamem:failed'}


def test_copied_journals_not_counted_as_more_model_work(tmp_path):
    import shutil
    source=tmp_path/'run'
    task_trace(source/'original','q')
    shutil.copytree(source/'original',source/'copied')
    manifest=b.freeze_transport(source,tmp_path/'frozen')
    assert manifest['counts']['physical_attempts']==2
    assert manifest['counts']['duplicate_journal_copies']==1
    assert manifest['counts']['logical_requests']==2


def test_main_collection_gate_preserves_failed_questions_and_rejects_identity_changes(tmp_path):
    source=tmp_path/'smoke';source.mkdir()
    config=tmp_path/'config.json';b.write(config,{'synthetic':True})
    selected=[{'id':str(i)} for i in range(50)]
    b.write(source/'manifest.json',{'kind':'real_dagbt_workload_capture_not_accuracy_experiment',
        'selection':'first_10_packaged_questions','question_ids':{'personamem':[q['id'] for q in selected[:10]]},
        'config_file_sha256':b.file_hash(config),'dataset_manifest_sha256':'synthetic-dataset'})
    b.write(source/'progress.json',{'state':'complete_with_failures'})
    b.write(source/'capture_summary.json',{'questions':10,'terminal_counts':{'ok':8,'failed':2}})
    gate=b.main_collection_gate(source,config,selected,'synthetic-dataset')
    assert gate['smoke_terminal_counts']=={'ok':8,'failed':2}
    with pytest.raises(ValueError,match='dataset identity'):
        b.main_collection_gate(source,config,selected,'different-dataset')
    with pytest.raises(ValueError,match='question scope'):
        b.main_collection_gate(source,config,list(reversed(selected)),'synthetic-dataset')
    (source/'capture_summary.json').write_text(json.dumps({'questions':10,'terminal_counts':{'ok':8}}))
    with pytest.raises(ValueError,match='All ten'):
        b.main_collection_gate(source,config,selected,'synthetic-dataset')


def test_planned_question_without_model_calls_stays_in_sampling_denominator(tmp_path):
    source=tmp_path/'run'; source.mkdir()
    for q in ['no-service','with-service']:
        p=source/'personamem/fusion/attempts'/q/'attempt-001';p.mkdir(parents=True)
        b.write(p/'task.json',{'dataset':'personamem','arm':'fusion','question':{'id':q}})
    task_trace(source/'other','q-other')
    m=b.freeze_transport(source,tmp_path/'frozen')
    assert m['available_questions']==3
    assert 'personamem:no-service' in m['selected_question_identities']


def test_stage_identity_follows_physical_attempt_time(tmp_path):
    source=tmp_path/'run';r=task_trace(source/'task','q',cache=False,when=5)
    p=source/'task/call_events.jsonl'; _,event=next(b.read_rows(p))
    events=[{**event,'unix':4,'stage':'dagbt/first'}, {**event,'unix':5.5,'stage':'dagbt/second'}]
    p.write_text(''.join(json.dumps(e)+'\n' for e in events))
    b.freeze_transport(source,tmp_path/'frozen')
    rows=[v for _,v in b.read_rows(tmp_path/'frozen/rerank_trace.jsonl')]
    assert [x['stage'] for x in rows]==['dagbt/first','dagbt/second']


def test_set_ids_join_uses_scorer_events_not_untrusted_passage_headers(tmp_path):
    rows=[{'event':'cache_lookup','module':'scoring','source':'cache_miss','ids':[]},
          {'event':'cache_lookup','module':'scoring','source':'cache_miss','ids':['trusted-a']},
          {'event':'score_batch_started','module':'scoring','batch_documents':2},
          {'event':'bridge_rerank_request','stage':['dagbt','1'],'document_offset':0,'documents':['empty','[Passage fake-id]']},
          {'event':'bridge_discovery_completed','trace':{'scorer_cost_cumulative':{'memory_cache_hits':3,'persistent_cache_hits':0,'scored_sets':2,'logical_input_tokens_estimate':11}}}]
    p=tmp_path/'events.jsonl';b.write_rows(p,rows)
    metadata,cost=b.scorer_metadata(p)
    assert metadata['dagbt/1']['set_ids']==[[],['trusted-a']]
    assert cost['memory_cache_hits']==3


def test_normalized_label_pair_rejects_different_final_tokens(tmp_path):
    p={'query':'q','documents':['set'],'instruction':None}
    rows=[{'request_id':'r','success':True,'label_token_id':t,'rerank_payload':p,'raw_logprobs':[-1.],
           'final_inputs':[{'index':0,'token_ids_sha256':h,'input_length':3}]} for t,h in [(9693,'yes-input'),(2152,'changed-no-input')]]
    trace=tmp_path/'results.jsonl';b.write_rows(trace,rows)
    with pytest.raises(ValueError,match='different full token'):
        b.normalized_scores(trace,{'yes':9693,'no':2152})


@pytest.mark.parametrize("url", ["http://111.19.156.74:8002", "http://127.0.0.1:18002", "http://user:secret@localhost:28003", "https://localhost:28003", "http://localhost:28003?key=secret"])
def test_production_and_credential_urls_rejected(url):
    with pytest.raises(ValueError): b.isolated_url(url)


def test_native_indices_raw_labels_and_truncation():
    p = engine_row(0, [[1], [2]])["payload"]
    choice = lambda i, v: {"index": i, "logprobs": {"tokens": ["token_id:9693"], "token_logprobs": [v]}}
    assert b.raw_labels({"choices": [choice(1, -.3), choice(0, -.2)]}, p) == [-.2, -.3]
    for choices in ([choice(0, -.2)], [choice(0, -.2), choice(0, -.3)], [choice(0, float('nan')), choice(1, -.2)]):
        with pytest.raises(ValueError): b.raw_labels({"choices": choices}, p)
    with pytest.raises(ValueError,match='truncation'):
        b.raw_labels({'choices':[choice(0,-.2),choice(1,-.3)],'truncated':True},p)
    with pytest.raises(Exception): b.indexed_scores({"results": [{"index": 0, "relevance_score": .3}], "truncated": True}, 1)


def test_small_error_can_flip_threshold_or_tie():
    a = {"P": .1, "Pe": .2, "PG": .2, "PGe": .30000000001}
    c = {**a, "PGe": .29999999999}
    assert b.compare_scores(list(a.values()), list(c.values()))["equivalent"]
    assert b.quartet(a, 0)["decision"] != b.quartet(c, 0)["decision"]
    comparison = b.compare_scores([.5, .5], [.5, .500000001])
    assert comparison["equivalent"] and not comparison["rank_equal"] and not comparison["exact_ties_equal"]
    pivot = b.quartet({"P": .4, "Pe": .5, "PG": .6, "PGe": .2}, .01)
    assert pivot["pivot_eligible"]


def test_normalization_requires_both_real_labels_and_same_input(tmp_path):
    payload = {"query": "synthetic", "documents": ["whole set"], "instruction": None}
    rows = [{"request_id": "r", "success": True, "label_token_id": t, "rerank_payload": payload,
             "raw_logprobs": [v]} for t, v in [(9693, -2.), (2152, -3.)]]
    p = tmp_path / "results.jsonl"; b.write_rows(p, rows)
    bank, samples = b.normalized_scores(p, {"yes": 9693, "no": 2152})
    assert samples[0]["score"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert len(bank) == 1
    q = tmp_path / "incomplete.jsonl"; b.write_rows(q, rows[:1])
    with pytest.raises(ValueError): b.normalized_scores(q, {"yes": 9693, "no": 2152})


def test_startup_overlay_only_changes_apc_and_isolated_binding():
    command = ["vllm", "serve", "/models/same", "--dtype", "float16", "--max-num-seqs", "4",
               "--max-model-len", "8192", "--enforce-eager", "--port", "18002", "--enable-prefix-caching"]
    help_text = "--enable-prefix-caching --no-enable-prefix-caching"
    off = probe.build_arm(command, help_text, "off", 28003)
    on = probe.build_arm(command, help_text, "on", 28003)
    assert off[:-1] == on[:-1]
    assert off[2:9] == command[2:9]
    assert off[-1] == "--no-enable-prefix-caching" and on[-1] == "--enable-prefix-caching"
    with pytest.raises(ValueError): probe.build_arm(command, "only enable-prefix-caching", "off", 28003)
    with pytest.raises(ValueError): probe.build_arm(command + ["--api-key", "secret"], help_text, "on", 28003)


def test_metrics_missing_reset_and_multi_device_not_invented():
    definitions = {"kv_hits": {"metric": "observed_metric", "kind": "counter", "unit": "tokens"}}
    before = {"values": {"kv_hits": 10}}
    after = {"values": {"kv_hits": 20}}
    assert b.metrics_delta(before, after, definitions)["kv_hits"]["value"] == 10
    assert b.metrics_delta(after, before, definitions)["kv_hits"] is None
    assert b.metrics_delta({"values": {}}, after, definitions)["kv_hits"] is None


def test_actual_frozen_bt_queues_and_quartets_compared(tmp_path):
    from test_bridge import session, requirement
    s = session(ann=10, sets=32)
    s.discover("Where was the author born?", "birth", requirement(), ["d0"])
    value = {"bridge_traces": s.traces, "events": [], "requests": [], "logical_counts": dict(s.ledger.used)}
    off, on = tmp_path / "off", tmp_path / "on"
    off.mkdir(); on.mkdir()
    b.write(off / "trajectory.json", value); b.write(on / "trajectory.json", value)
    report = b.compare_search(off, on, tmp_path / "equal")
    assert report["trajectory_equal"] and report["quartets"]
    assert all(x["A_error"] == x["M_error"] == 0 for x in report["quartets"])
    value["bridge_traces"][0]["search_archive"]["target_events"][0]["decision"] = "changed"
    (on / "trajectory.json").write_text(json.dumps(value))
    report = b.compare_search(off, on, tmp_path / "divergent")
    assert not report["trajectory_equal"] and report["first_divergence"]["section"] == "bridge_traces"


def test_synthetic_suite_uses_whole_sets_and_preserves_reverse_order():
    rows = b.synthetic_requests()
    cases = {r["case"]: r["payload"] for r in rows}
    assert cases["empty-set"]["documents"] == ["[No evidence passages]"]
    assert cases["native-batch"]["documents"] == list(reversed(cases["reversed-batch"]["documents"]))
    assert len(cases["pair"]["documents"]) == 1
    for row in rows: b.validate_payload(row["payload"])


def test_observer_records_exact_submissions_without_changes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / "proxy.py"
    source.write_text('''
def post_json(url, payload):
    return {"choices": [{"index": 0, "logprobs": {"tokens": ["token_id:" + str(payload["allowed_token_ids"][0])], "token_logprobs": [-2.0]}}]}
def compute_scores(query, documents, instruction):
    for token in [9693, 2152]:
        payload = {"model": "synthetic", "prompt": [[1,2,3]], "allowed_token_ids": [token]}
        post_json("http://127.0.0.1:28003/v1/completions", payload)
    return [.5]
def app():
    assert compute_scores("synthetic", ["complete set"], None) == [.5]
''')
    inventory = {k: "synthetic-only" for k in probe.IDENTITY_FIELDS}
    inventory.update(proxy_source_sha256=b.file_hash(source), private_single_tenant_instance=True, tenant_namespace_hash="synthetic-tenant")
    p = tmp_path / "identity.json"; b.write(p, inventory)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda app, **kwargs: app()))
    args = SimpleNamespace(backend="http://127.0.0.1:28003", port=28004, inventory=p, proxy_source=source, output=tmp_path / "observer")
    probe.observe_proxy(args)
    rows = [r for _, r in b.read_rows(args.output / "engine_trace.jsonl")]
    assert len(rows) == 2 and rows[0]["request_id"] == rows[1]["request_id"]
    assert [r["payload"]["allowed_token_ids"] for r in rows] == [[9693], [2152]]
    assert all(r["payload"]["prompt"] == [[1,2,3]] and r["completed"] for r in rows)
    assert all(r["payload_sha256"] == b.digest(r["payload"]) for r in rows)


def test_search_replay_refuses_new_observations_without_network(tmp_path, monkeypatch):
    import numpy as np
    vectors = tmp_path / "vectors.npy"; np.save(vectors, np.eye(2, dtype=np.float32))
    source = tmp_path / "attempt" / "requests"; source.mkdir(parents=True)
    bundle = {"config": {"model_profile": "bridgetree", "llm_base_url": "http://frozen/v1", "llm_model": "synthetic",
                         "embedding_base_url": "http://frozen/v1", "embedding_model": "synthetic",
                         "reranker": {"url": "http://frozen/rerank"}},
              "question": {"id": "q", "question": "Synthetic question?"},
              "documents": [{"doc_id": "a", "title": "A", "text": "Synthetic A"}, {"doc_id": "b", "title": "B", "text": "Synthetic B"}],
              "ids": ["a", "b"], "vectors_path": str(vectors), "vectors_sha256": b.file_hash(vectors),
              "attempt_dir": str(source.parent)}
    bp = tmp_path / "bundle.json"; b.write(bp, bundle)
    scores = tmp_path / "scores.jsonl"; b.write_rows(scores, [])
    monkeypatch.setattr(b, "http_json", lambda *a, **kw: pytest.fail("Network fallback forbidden"))
    out = tmp_path / "replay"
    with pytest.raises(ValueError, match="TRAJECTORY_DIVERGENCE_UNSEEN_REQUEST"):
        b.search_replay(bp, scores, {"yes": 9693, "no": 2152}, out)
    status = json.loads((out / "replay_status.json").read_text())
    assert not status["completed"] and status["first_divergent_request"] is not None


def test_token_replay_and_comparison_never_upgrade_mock_to_verified(tmp_path, monkeypatch):
    evidence = tmp_path / "runtime.txt"; evidence.write_text('synthetic runtime evidence')
    payload = {"query": "synthetic", "documents": ["set-a", "set-b"], "instruction": None}
    y = engine_row(0, [[1,2,3], [1,2,4]], request_id="r", rerank_payload=payload)
    n = engine_row(1, [[1,2,3], [1,2,4]], request_id="r", rerank_payload=payload)
    n["payload"]["allowed_token_ids"] = [2152]; n["payload_sha256"] = b.digest(n["payload"])
    trace = tmp_path / "engine.jsonl"; b.write_rows(trace, [y,n])
    identity = {"isolated": True, "backend_url": "http://127.0.0.1:28003", "runtime_verified": True,
                "health_verified": True, "runtime_evidence_file": str(evidence), "runtime_evidence_sha256": b.file_hash(evidence),
                "enable_prefix_caching": False, "computation_identity": "synthetic-only", "tenant_namespace_hash": "tenant-a",
                "label_token_ids": {"yes":9693,"no":2152}, "prefix_reset_verified": True}
    ip = tmp_path / "off.json"; b.write(ip, identity)
    seen = []
    def native(url, p=None):
        seen.append((url,p))
        if url.endswith('/reset_prefix_cache'): return True
        token = p['allowed_token_ids'][0]
        return {"choices": [{"index": i, "logprobs": {"tokens": ["token_id:"+str(token)], "token_logprobs": [-1. if token == 9693 else -2.]}} for i in reversed(range(len(p['prompt'])))]}
    monkeypatch.setattr(b, "http_json", native)
    monkeypatch.setattr(b, "http_health", lambda _: None)
    off = b.replay(trace, tmp_path / "off", identity['backend_url'], ip, "off")
    identity['enable_prefix_caching'] = True
    jp = tmp_path / "on.json"; b.write(jp, identity)
    on = b.replay(trace, tmp_path / "on", identity['backend_url'], jp, "cold", True)
    assert off['counts']['model_samples'] == on['counts']['model_samples'] == 4
    assert off['counts']['whole_set_score_samples'] == 2
    assert off['distinct_serialized_samples'] == 2
    assert off['engine_new_compute_tokens'] is None
    assert len([x for x in seen if x[0].endswith('/v1/completions')]) == 4
    report = b.compare_runs(tmp_path / "off", tmp_path / "on", tmp_path / "comparison")
    assert report['normalized_score_equivalence']['equivalent']
    assert report['status'] == 'IMPLEMENTED_NOT_HARDWARE_VERIFIED'
    assert report['action_equivalence'] is None


def test_replay_rejects_launch_plan_as_runtime_identity(tmp_path):
    identity = tmp_path / "plan.json"; b.write(identity, {"isolated":True, "backend_url":"http://127.0.0.1:28003", "health_verified":False})
    with pytest.raises(ValueError, match="launch plan"):
        b.replay(tmp_path / "absent.jsonl", tmp_path / "output", "http://127.0.0.1:28003", identity, "off")
    assert not (tmp_path / "output").exists()


def test_collector_enforces_fixed_smoke_scope_and_original_budgets(tmp_path):
    with pytest.raises(ValueError, match='fixed 10-question'):
        b.collect_workload(ROOT/'configs/paired.example.json',tmp_path/'uncreated',11)
    config=json.loads((ROOT/'configs/paired.example.json').read_text())
    config['fusion']['ann_calls']=35
    p=tmp_path/'changed.json';b.write(p,config)
    with pytest.raises(ValueError,match='budgets changed'):
        b.collect_workload(p,tmp_path/'uncreated')
    assert not (tmp_path/'uncreated').exists()


@pytest.mark.parametrize('personal', [False, True])
def test_full_engine_frozen_nonreranker_responses_replay_without_regeneration(tmp_path, monkeypatch, personal):
    """A full source Engine protocol fixture; fixed-pool fixture has no KV claim."""
    from types import SimpleNamespace
    import threading
    import numpy as np
    from dagbt import engine as e
    from dagbt.model_runtime import load_tokenizer, prepare_request
    from dagbt.transport import Transport
    from test_engine import FakeCalls, step, answer
    docs = {d: SimpleNamespace(doc_id=d, title=d, text="Synthetic fact " + d, passage=d + "\nSynthetic fact " + d) for d in ['a','b']}
    config = {"model_profile":"bridgetree", "llm_base_url":"http://frozen/v1", "llm_model":"synthetic",
              "embedding_base_url":"http://frozen/v1", "embedding_model":"synthetic",
              "reranker":{"url":"http://frozen/rerank"}, "fixed_candidate_pools":{"q":['a','b']}}
    if personal:
        import importlib
        from dagbt.runner import configure_personamem_reader
        sys.path.insert(0,str(ROOT/'dagv2'))
        pipeline=importlib.import_module('pipeline_dagv2')
        reader=importlib.import_module('reader')
        monkeypatch.setattr(reader,'reader_messages',reader.reader_messages)
        monkeypatch.setattr(reader,'parse_answer',reader.parse_answer)
        monkeypatch.setattr(pipeline.e,'CONFIG',{**pipeline.e.CONFIG,**config})
        configure_personamem_reader()
    fake = FakeCalls([step('node')], lambda data: answer(data,'fixture_answer',['a']))
    source = tmp_path / 'original'; source.mkdir(); (source/'requests').mkdir()
    class Recorder(Transport):
        def get(self, stage, url, payload, *, reserve=None, extra_reserve=0):
            body = fake.get(stage,url,payload)['response']
            url, wire, _ = prepare_request(stage,url,payload,self.config)
            identity = {'unit_id':self.unit,'url':url,'payload':wire}; key=b.digest(identity)
            kind='reader' if stage[0]=='reader' else 'llm'
            self._reserve(kind,'/'.join(stage),reserve,extra_reserve)
            b.write(source/'requests'/(key+'.json'), {**identity,'attempts':[{'http_status':200}], 'response':body})
            return {'response':body,'response_ref':key,'request':wire}
        post_rerank=get
    monkeypatch.setattr(e,'Transport',Recorder)
    vectors = np.eye(2,dtype=np.float32)
    resources=(docs,['a','b'],vectors,SimpleNamespace(vectors=dict(zip(['a','b'],vectors)),lock=threading.Lock()),load_tokenizer(config))
    baseline=e.Engine({'id':'q','question':'Synthetic question?'},resources,SimpleNamespace(output=source),config,'fusion',
                      reader_question='Synthetic MCQ?' if personal else None)
    baseline.run()
    old_call_count=len(fake.requests)
    vp=tmp_path/'vectors.npy';np.save(vp,vectors)
    bundle={'config':config,'question':baseline.q,'documents':[{'doc_id':d, 'title':d, 'text':'Synthetic fact '+d} for d in ['a','b']],
            'ids':['a','b'],'vectors_path':str(vp),'vectors_sha256':b.file_hash(vp),'attempt_dir':str(source)}
    if personal:bundle.update(dataset='personamem',reader_question=baseline.reader_question)
    bp=tmp_path/'bundle.json';b.write(bp,bundle)
    scores=tmp_path/'empty_scores.jsonl';b.write_rows(scores,[])
    assert b.search_replay(bp,scores,{'yes':9693,'no':2152},tmp_path/'replayed')['completed']
    assert len(fake.requests)==old_call_count
    result=json.loads((tmp_path/'replayed/replay_status.json').read_text())
    assert result['logical_counts']==dict(baseline.ledger.used)


def test_search_bundle_exports_only_original_visible_scope_and_checks_index_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import threading
    import numpy as np
    from dagbt import runner, resources, personamem
    source=tmp_path/'source';source.mkdir()
    config={'model_profile':'bridgetree','llm_base_url':'http://synthetic/v1','embedding_base_url':'http://synthetic/v1'}
    q={'id':'q','scope_id':'scope','user_question':'Synthetic current request','question':'Synthetic MCQ'}
    hashes={'synthetic-vectors':'identity-v1'}
    docs={d:SimpleNamespace(doc_id=d,title=d,text='Synthetic '+d,passage=d+'\nSynthetic '+d,
          metadata={'source_message_indices':[i]}) for i,d in enumerate(['a','b','future'])}
    ids=list(docs);vectors=np.eye(3,dtype=np.float32)
    native_resources=(docs,ids,vectors,SimpleNamespace(lock=threading.Lock()),None)
    monkeypatch.setattr(runner,'frozen_sources',lambda:{'synthetic-source':'unchanged'})
    monkeypatch.setattr(personamem,'load_questions',lambda root:[q])
    monkeypatch.setattr(personamem,'load_scopes',lambda root:{'scope':['b','a']})
    fake_pipeline=SimpleNamespace(e=SimpleNamespace(CONFIG=config))
    monkeypatch.setitem(sys.modules,'pipeline_dagv2',fake_pipeline)
    monkeypatch.setattr(resources,'prepare_resources',lambda *args:([],native_resources,hashes))
    b.write(source/'manifest.json',{'kind':'real_dagbt_workload_capture_not_accuracy_experiment',
        'datasets':['personamem'],'arms':['fusion'],'config':config,'source_hashes':runner.frozen_sources(),
        'dataset_manifest_sha256':b.file_hash(ROOT/'data/personamem/manifest.json'),'question_ids':{'personamem':['q']}})
    arm=source/'personamem/fusion';attempt=arm/'attempts'/runner.digest('q')/'attempt-001';attempt.mkdir(parents=True)
    b.write(arm/'resource_hashes.json',hashes)
    b.write(attempt/'task.json',{'question':q,'dataset':'personamem','arm':'fusion'})
    b.write(attempt/'result.json',{'answer':{'status':'execution_failed'}})
    out=tmp_path/'bundles';manifest=b.build_search_bundles(source,out)
    assert manifest['planned_questions']==1 and manifest['entries'][0]['source_status']=='execution_failed'
    bundle=json.loads((out/manifest['entries'][0]['bundle']).read_text())
    assert bundle['ids']==['b','a'] and [d['doc_id'] for d in bundle['documents']]==['b','a']
    assert bundle['documents'][0]['metadata']==docs['b'].metadata
    assert np.array_equal(np.load(bundle['vectors_path']),vectors[[1,0]])
    assert Path(bundle['vectors_path']).stat().st_mode & 0o777==0o600
    assert bundle['question']=={'id':'q','question':q['user_question']} and bundle['reader_question']==q['question']
    (arm/'resource_hashes.json').write_text(json.dumps({'synthetic-vectors':'changed'}))
    with pytest.raises(ValueError,match='resource identity changed'):
        b.build_search_bundles(source,tmp_path/'must-not-export')
    assert not (tmp_path/'must-not-export').exists()
