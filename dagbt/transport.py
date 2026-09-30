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

class ServiceError(RuntimeError): pass
class ResponseError(ValueError): pass

def call_reservation(settings, stage, reserve=None, extra_reserve=0):
    """Keep the existing audit/flat allowance and optionally protect more work."""
    stage_text='/'.join(map(str,stage)) if isinstance(stage,(tuple,list)) else str(stage)
    if reserve is None:
        final=settings.get('final_selection_calls',int(settings.get('selection')=='flat'))
        if isinstance(final,bool) or not isinstance(final,int) or final<0:
            raise ValueError('final_selection_calls must be a nonnegative integer')
        reserve=(0 if stage_text.split('/')[0].startswith(('select','reader')) else final
                 if stage_text.split('/')[0].startswith('audit')
                 else int(settings.get('reserved_audit_calls',1))+final)
    for name,value in (('reserve',reserve),('extra_reserve',extra_reserve)):
        if isinstance(value,bool) or not isinstance(value,int) or value<0:
            raise ValueError(name+' must be a nonnegative integer')
    return reserve+extra_reserve

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

class Transport:
    """Native Calls return shape. No authorization headers ever written to disk."""
    def __init__(self, unit, output, config, ledger, tokenizer):
        self.unit,self.output,self.config,self.ledger,self.tokenizer=unit,Path(output),config,ledger,tokenizer
        self.events=[]

    def get(self, stage, url, payload, *, reserve=None, extra_reserve=0):
        stage_text='/'.join(map(str,stage)) if isinstance(stage,(tuple,list)) else str(stage)
        embed=url.rstrip('/').endswith('/embeddings')
        rerank=url==self.config.get('reranker',{}).get('url') or url.rstrip('/').endswith(('/rerank','/reranks'))
        reader=stage_text.startswith('reader/')
        kind='embedding_http' if embed else 'rerank_http' if rerank else 'reader' if reader else 'llm'
        evidence_reasoning=bool(payload.get(REASONING_MARKER)) and not (embed or rerank or reader)
        url,payload,legacy_text=prepare_request(stage,url,payload,self.config)
        if evidence_reasoning or (is_bridgetree(self.config) and not (embed or rerank)):
            count=(evidence_wire_tokens(payload) if evidence_reasoning else
                   count_request_tokens(stage,url,payload,self.config,self.tokenizer))
            context_limit=self.config.get('fusion',{}).get('context_tokens',16384)
            margin=int(self.config.get('fusion',{}).get('input_margin',256)) if evidence_reasoning else 0
            total=count+payload['max_tokens']+8+margin
            event={'event':'model_context_budget','stage':stage_text,'input_tokens_local':count,
                   'output_token_reserve':payload['max_tokens'],'safety_margin':8,'input_margin':margin,
                   'context_tokens':context_limit,'total_tokens_local':total,
                   'within_estimated_budget':total<=context_limit,
                   **(evidence_token_accounting() if evidence_reasoning else token_accounting(self.config))}
            self.ledger.record(event)
            if total>context_limit:
                raise ResponseError(f'BridgeTree estimated context budget at {stage_text}: '
                                    f'{count}+{payload["max_tokens"]}+8+{margin}>{context_limit}; '
                                    'regex estimate, not the deployed model tokenizer')
        identity={'unit_id':self.unit,'url':url,'payload':payload}
        key=digest(identity);path=self.output/'requests'/(key+'.json')
        record=json.loads(path.read_text()) if path.exists() else {**identity,'stage':stage,'attempts':[]}
        self.output.mkdir(parents=True,exist_ok=True)
        with (self.output/'call_events.jsonl').open('a') as journal:
            journal.write(json.dumps({'event':'call_started','request_ref':key,'stage':stage_text,'kind':kind,'unix':time.time(),'cache_hit':'response' in record})+'\n')
        if any(record.get(k)!=v for k,v in identity.items()):raise ResponseError('Request cache identity mismatch')
        if 'response' in record:
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
    def get(self,stage,url,payload,*,reserve=None,extra_reserve=0):
        txt='/'.join(map(str,stage)) if isinstance(stage,(tuple,list)) else str(stage)
        k='embedding_http' if url.endswith('/embeddings') else 'rerank_http' if 'rerank' in url else 'reader' if txt.startswith('reader/') else 'llm'
        reserved=call_reservation(self.config.get('fusion',{}),txt,reserve,extra_reserve)
        if k=='llm' and 'llm' in self.ledger.limits and self.ledger.remaining('llm')<=reserved:
            from .budget import BudgetExceeded
            raise BudgetExceeded('llm',txt,1,0)
        self.ledger.reserve(k,txt)
        return self.base.get(stage,url,payload)
    post_rerank=get
