"""OpenAI-compatible transport with exact cached requests and physical-attempt accounting."""
from __future__ import annotations
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

class ServiceError(RuntimeError): pass
class ResponseError(ValueError): pass

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

    def get(self, stage, url, payload):
        stage_text='/'.join(map(str,stage)) if isinstance(stage,(tuple,list)) else str(stage)
        embed=url.rstrip('/').endswith('/embeddings')
        rerank=url==self.config.get('reranker',{}).get('url') or url.rstrip('/').endswith(('/rerank','/reranks'))
        reader=stage_text.startswith('reader/')
        kind='embedding_http' if embed else 'rerank_http' if rerank else 'reader' if reader else 'llm'
        model=self.config.get('reranker',{}).get('model') if rerank else self.config['embedding_model' if embed else 'llm_model']
        payload={**payload,'model':model}
        if not (embed or rerank):
            payload={**payload,'temperature':0,'top_p':1,'seed':20260918}
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
            for _ in range(charges):self._reserve(kind,stage_text)
            self.ledger.reserve('cache_hits',stage_text)
            self.ledger.record({'event':'cached_attempt_charge','stage':stage_text,'kind':kind,'replayed_attempts':charges})
            self.ledger.record({'event':'request_cache_hit','stage':stage_text,'response_ref':key,'kind':kind})
            return {'response':record['response'],'response_ref':key,'request':payload}
        env='DAG_EMBED_API_KEY' if embed else 'DAG_RERANK_API_KEY' if rerank else 'DAG_LLM_API_KEY'
        secret=os.environ.get(env,'')
        limit=int(self.config.get('max_identical_attempts',3))
        attempts_this_call=0
        # A runner explicit retry uses a new attempt dir; old failed attempts remain archived.
        while attempts_this_call < limit:
            self._reserve(kind,stage_text)
            self.ledger.reserve('http_attempts',stage_text)
            attempts_this_call+=1
            attempt={'started_unix':time.time(),'retry_index':attempts_this_call,'kind':kind}
            record['attempts'].append(attempt);save(path,record)
            headers={'Content-Type':'application/json'}
            if secret:headers['Authorization']='Bearer '+secret
            request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers,method='POST')
            try:
                with urllib.request.urlopen(request,timeout=self.config.get('request_timeout_seconds',600)) as response:
                    body=json.load(response)
                if not isinstance(body,dict):raise ResponseError('Response must be an object')
                if embed and not body.get('data'):raise ResponseError('Embedding data missing')
                if rerank and not (isinstance(body.get('results'),list) or isinstance(body.get('data'),list)):
                    raise ResponseError('Indexed rerank results missing')
                if not(embed or rerank) and not body.get('choices'):raise ResponseError('Choices missing')
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
                return {'response':body,'response_ref':key,'request':payload}
            if attempts_this_call < limit:time.sleep(min(2*attempts_this_call,5))
        raise ServiceError(f'Retry limit at {stage_text}; request {key}')

    def _reserve(self,kind,stage):
        settings=self.config.get('fusion',{})
        flat_reserve=int(settings.get('selection')=='flat')
        reserved=(0 if stage.startswith('select/') else flat_reserve if stage.startswith('audit/')
                  else int(settings.get('reserved_audit_calls',1))+flat_reserve)
        if kind=='llm' and self.ledger.remaining('llm')<=reserved:
            from .budget import BudgetExceeded
            raise BudgetExceeded('llm',stage,1,0)
        self.ledger.reserve(kind,stage)

    post_rerank=get

class StubMeter:
    """Explicit injection path for offline protocol tests, never a production fallback."""
    def __init__(self, base, ledger, config):self.base,self.ledger,self.config=base,ledger,config
    def get(self,stage,url,payload):
        txt='/'.join(map(str,stage)) if isinstance(stage,(tuple,list)) else str(stage)
        k='embedding_http' if url.endswith('/embeddings') else 'rerank_http' if 'rerank' in url else 'reader' if txt.startswith('reader/') else 'llm'
        self.ledger.reserve(k,txt)
        return self.base.get(stage,url,payload)
    post_rerank=get
