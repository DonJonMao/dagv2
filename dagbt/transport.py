"""OpenAI-compatible transport with exact cached requests and physical-attempt accounting."""
from __future__ import annotations
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from .model_runtime import (prepare_request, normalize_response, resolve_api_key,
                            is_bridgetree, count_request_tokens, token_accounting,
                            REASONING_MARKER, evidence_wire_tokens, evidence_token_accounting)

from .methods import terminal_method, capabilities
from .token_counting import DeploymentTokenCounter, validate_tokenization_config

class ServiceError(RuntimeError): pass
class ResponseError(ValueError): pass

def call_reservation(settings, stage, reserve=None, extra_reserve=0):
    """Keep the existing audit/flat allowance and optionally protect more work."""
    stage_text='/'.join(map(str,stage)) if isinstance(stage,(tuple,list)) else str(stage)
    if reserve is None:
        if terminal_method(settings.get('algorithm_version')):
            reserve = 0 if stage_text.split('/')[0].startswith('audit') else int(settings.get('reserved_audit_calls', 1))
        else:
            reserve = _legacy_reservation(settings, stage_text)
    for name,value in (('reserve',reserve),('extra_reserve',extra_reserve)):
        if isinstance(value,bool) or not isinstance(value,int) or value<0:
            raise ValueError(name+' must be a nonnegative integer')
    return reserve+extra_reserve

def _legacy_reservation(settings, stage_text):
    final=settings.get('final_selection_calls',int(settings.get('selection')=='flat'))
    if isinstance(final,bool) or not isinstance(final,int) or final<0:
        raise ValueError('final_selection_calls must be a nonnegative integer')
    return (0 if stage_text.split('/')[0].startswith(('select','reader')) else final
            if stage_text.split('/')[0].startswith('audit')
            else int(settings.get('reserved_audit_calls',1))+final)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def request_identity(unit, url, payload, config, scoring_context_id=None):
    identity = {'unit_id':unit,'url':url,'payload':payload}
    version = config.get('fusion',{}).get('algorithm_version')
    if terminal_method(version):
        identity['algorithm_version'] = version
        if scoring_context_id is not None:
            identity['scoring_context_id'] = scoring_context_id
    return identity

class Transport:
    """Native Calls return shape. No authorization headers ever written to disk."""
    def __init__(self, unit, output, config, ledger, tokenizer):
        self.unit,self.output,self.config,self.ledger,self.tokenizer=unit,Path(output),config,ledger,tokenizer
        self.events=[]
        self.token_counter = (DeploymentTokenCounter(config, self._record_tokenization)
                              if validate_tokenization_config(config) is not None else None)

    def _record_tokenization(self, event):
        value = {**event, 'unix': time.time()}
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / 'tokenization_events.jsonl').open('a') as journal:
            journal.write(json.dumps(value, ensure_ascii=False) + '\n')
        self.ledger.record(value)

    def count_reasoning_tokens(self, stage, url, payload):
        """The partitioner and final transport check share the same wire count."""
        _, wire, _ = prepare_request(stage, url, payload, self.config)
        if self.token_counter is not None:
            return self.token_counter.count(wire)
        return {'count': evidence_wire_tokens(wire), **evidence_token_accounting()}

    def _verify_token_count(self, body, accounting, record, path, stage):
        if self.token_counter is None:
            return
        if record.get('tokenization_validation', {}).get('status') == 'failed':
            self.ledger.record({'event': 'tokenization_usage_validation', 'stage': stage,
                                'status': 'failed', 'reason': 'prior_validation_failure',
                                'current_expected_prompt_tokens': accounting['count']})
            raise ResponseError('Deployment tokenization does not match generation prompt_tokens (prior validation failed)')
        usage = body.get('usage') if isinstance(body, dict) else None
        observed = usage.get('prompt_tokens') if isinstance(usage, dict) else None
        matched = type(observed) is int and observed == accounting['count']
        validation = {'status': 'matched' if matched else 'failed',
                      'expected_prompt_tokens': accounting['count'],
                      'observed_prompt_tokens': observed if type(observed) is int else None,
                      'tokenization_request_sha256': accounting['tokenization_request_sha256']}
        record['tokenization_validation'] = validation
        save(path, record)
        self.ledger.record({'event': 'tokenization_usage_validation', 'stage': stage, **validation})
        if not matched:
            # The completed generation is preserved, but never retried merely
            # to make an accounting disagreement disappear (including replay).
            raise ResponseError('Deployment tokenization does not match generation prompt_tokens')

    supports_scoring_context = True

    def get(self, stage, url, payload, *, reserve=None, extra_reserve=0, scoring_context_id=None):
        stage_text='/'.join(map(str,stage)) if isinstance(stage,(tuple,list)) else str(stage)
        embed=url.rstrip('/').endswith('/embeddings')
        rerank=url==self.config.get('reranker',{}).get('url') or url.rstrip('/').endswith(('/rerank','/reranks'))
        version=self.config.get('fusion',{}).get('algorithm_version')
        if rerank and terminal_method(version) and not capabilities(version).scoring:
            raise ValueError('Zero-score method prohibits reranker requests')
        reader=stage_text.startswith('reader/')
        kind='embedding_http' if embed else 'rerank_http' if rerank else 'reader' if reader else 'llm'
        evidence_reasoning=bool(payload.get(REASONING_MARKER)) and not (embed or rerank or reader)
        url,payload,legacy_text=prepare_request(stage,url,payload,self.config)
        accounting = None
        if evidence_reasoning or (is_bridgetree(self.config) and not (embed or rerank)):
            accounting = (self.token_counter.count(payload) if self.token_counter is not None else
                          {'count': evidence_wire_tokens(payload), **evidence_token_accounting()}
                          if evidence_reasoning else
                          {'count': count_request_tokens(stage,url,payload,self.config,self.tokenizer),
                           **token_accounting(self.config)})
            count=accounting['count']
            context_limit=self.config.get('fusion',{}).get('context_tokens',16384)
            margin=int(self.config.get('fusion',{}).get('input_margin',256)) if evidence_reasoning else 0
            total=count+payload['max_tokens']+8+margin
            event={'event':'model_context_budget','stage':stage_text,'input_tokens_local':count,
                   'output_token_reserve':payload['max_tokens'],'safety_margin':8,'input_margin':margin,
                   'context_tokens':context_limit,'total_tokens_local':total,
                   'within_estimated_budget':total<=context_limit,
                   **{k:v for k,v in accounting.items() if k != 'count'}}
            self.ledger.record(event)
            if total>context_limit:
                note = (accounting.get('token_estimator_id', 'unknown accounting') if self.token_counter is not None
                        else 'regex estimate, not the deployed model tokenizer')
                raise ResponseError(f'BridgeTree context budget at {stage_text}: '
                                    f'{count}+{payload["max_tokens"]}+8+{margin}>{context_limit}; '
                                    + note)
        if rerank and self.config.get('fusion',{}).get('algorithm_version') == 'dagbt_local_terminal_v1' and not scoring_context_id:
            raise ResponseError('Local rerank requires scoring context identity')
        identity=request_identity(self.unit,url,payload,self.config,scoring_context_id)
        key=digest(identity);path=self.output/'requests'/(key+'.json')
        record=json.loads(path.read_text()) if path.exists() else {**identity,'stage':stage,'attempts':[]}
        self.output.mkdir(parents=True,exist_ok=True)
        with (self.output/'call_events.jsonl').open('a') as journal:
            journal.write(json.dumps({'event':'call_started','request_ref':key,'stage':stage_text,'kind':kind,'unix':time.time(),'cache_hit':'response' in record})+'\n')
        if any(record.get(k)!=v for k,v in identity.items()):raise ResponseError('Request cache identity mismatch')
        if 'response' in record:
            if self.token_counter is not None and not (embed or rerank):
                self._verify_token_count(record['response'], accounting, record, path, stage_text)
            # Replay consumes logical budget so a resumed task follows the same frontier.
            charges=max(1,len(record.get('attempts',[])))
            for _ in range(charges):self._reserve(kind,stage_text,reserve,extra_reserve)
            self.ledger.reserve('cache_hits',stage_text)
            self.ledger.record({'event':'cached_attempt_charge','stage':stage_text,'kind':kind,'replayed_attempts':charges})
            self.ledger.record({'event':'request_cache_hit','stage':stage_text,'response_ref':key,'kind':kind})
            return {'response':normalize_response(record['response'],legacy_text),'response_ref':key,'request':payload}
        secret=resolve_api_key(self.config,'embedding' if embed else 'reranker' if rerank else 'llm')
        limit=int(self.config.get('max_identical_attempts',3))
        attempts_this_call=0
        # A runner explicit retry uses a new attempt dir; old failed attempts remain archived.
        while attempts_this_call < limit:
            self._reserve(kind,stage_text,reserve,extra_reserve)
            self.ledger.reserve('http_attempts',stage_text)
            attempts_this_call+=1
            attempt={'started_unix':time.time(),'retry_index':attempts_this_call,'kind':kind}
            record['attempts'].append(attempt);save(path,record)
            headers={'Content-Type':'application/json'}
            if secret:headers['Authorization']='Bearer '+secret
            request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers,method='POST')
            try:
                timeout=(self.config.get('reranker',{}).get('timeout_seconds',self.config.get('request_timeout_seconds',600))
                         if rerank else self.config.get('request_timeout_seconds',600))
                with urllib.request.urlopen(request,timeout=timeout) as response:
                    body=json.load(response)
                if not isinstance(body,dict):raise ResponseError('Response must be an object')
                if embed and not body.get('data'):raise ResponseError('Embedding data missing')
                if rerank and not (isinstance(body.get('results'),list) or isinstance(body.get('data'),list)):
                    raise ResponseError('Indexed rerank results missing')
                if not(embed or rerank or evidence_reasoning) and not body.get('choices'):raise ResponseError('Choices missing')
            except urllib.error.HTTPError as exc:
                attempt.update(http_status=exc.code,retryable=exc.code in (408,429,500,502,503,504))
                # Body can echo credentials; retain type/status, never echo server secrets.
                attempt['error_type']='HTTPError';save(path,record)
                if not attempt['retryable']:raise ServiceError(f'HTTP {exc.code} at {stage_text}; request {key}') from exc
            except (urllib.error.URLError,TimeoutError,ConnectionError,ValueError) as exc:
                attempt.update(error_type=type(exc).__name__,retryable=True);save(path,record)
            else:
                attempt.update(http_status=200,finished_unix=time.time());record['response']=body;save(path,record)
                event={'event':'http_response','stage':stage_text,'response_ref':key,'kind':kind,
                       'usage':body.get('usage'), 'seconds':attempt['finished_unix']-attempt['started_unix']}
                self.ledger.record(event);self.events.append(event)
                if self.token_counter is not None and not (embed or rerank):
                    self._verify_token_count(body, accounting, record, path, stage_text)
                return {'response':normalize_response(body,legacy_text),'response_ref':key,'request':payload}
            if attempts_this_call < limit:time.sleep(min(2*attempts_this_call,5))
        raise ServiceError(f'Retry limit at {stage_text}; request {key}')

    def _reserve(self,kind,stage,reserve=None,extra_reserve=0):
        settings=self.config.get('fusion',{})
        reserved=call_reservation(settings,stage,reserve,extra_reserve)
        if kind=='llm' and 'llm' in self.ledger.limits and self.ledger.remaining('llm')<=reserved:
            from .budget import BudgetExceeded
            raise BudgetExceeded('llm',stage,1,0)
        self.ledger.reserve(kind,stage)

    post_rerank=get

class StubMeter:
    """Explicit injection path for offline protocol tests, never a production fallback."""
    def __init__(self, base, ledger, config):self.base,self.ledger,self.config=base,ledger,config
    supports_scoring_context = True
    def get(self,stage,url,payload,*,reserve=None,extra_reserve=0,scoring_context_id=None):
        txt='/'.join(map(str,stage)) if isinstance(stage,(tuple,list)) else str(stage)
        k='embedding_http' if url.endswith('/embeddings') else 'rerank_http' if 'rerank' in url else 'reader' if txt.startswith('reader/') else 'llm'
        version=self.config.get('fusion',{}).get('algorithm_version')
        if k=='rerank_http' and terminal_method(version) and not capabilities(version).scoring:
            raise ValueError('Zero-score method prohibits reranker requests')
        reserved=call_reservation(self.config.get('fusion',{}),txt,reserve,extra_reserve)
        if k=='llm' and 'llm' in self.ledger.limits and self.ledger.remaining('llm')<=reserved:
            from .budget import BudgetExceeded
            raise BudgetExceeded('llm',txt,1,0)
        self.ledger.reserve(k,txt)
        return self.base.get(stage,url,payload)
    post_rerank=get
