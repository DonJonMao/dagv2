#!/usr/bin/env python3
"""Fixed first-three HotpotQA + first-three PersonaMem; one question attempt.

No evaluation labels, production restart, cache clearing, or full experiment.
Reruns resume terminal rows under an exact config/source/scope manifest.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dagbt import runner
from dagbt.config import resolve
from dagbt.local_terminal import VERSION
from dagbt.resources import ensure_index,inspect_index


def run(config_path, output):
    config=runner.validate_config(runner.load(config_path))
    settings=resolve(config,VERSION)
    if [settings[k] for k in ('ann_calls','set_score_calls','llm_calls','reader_calls')] != [36,512,24,0]:
        raise ValueError('Smoke requires exactly 36/512/24/0 budgets')
    if config.get('experiment',{}).get('max_question_attempts') != 1:
        raise ValueError('Smoke requires one question attempt')
    scopes={dataset:runner.load_questions(dataset,3) for dataset in ('hotpotqa','personamem')}
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True);output.chmod(0o700)
    hosts=[urlsplit(config[k]).hostname for k in ('llm_base_url','embedding_base_url')]
    hosts.append(urlsplit(config['reranker']['url']).hostname)
    no_proxy=','.join(filter(None,[os.environ.get('NO_PROXY',''),*hosts]))
    os.environ['NO_PROXY']=os.environ['no_proxy']=no_proxy
    manifest={'kind':'fixed_local_terminal_smoke','algorithm_version':VERSION,'datasets':list(scopes),
        'arms':[VERSION],'config':config,'question_ids':{d:[q['id'] for q in qs] for d,qs in scopes.items()},
        'questions_digest':{d:runner.digest(qs) for d,qs in scopes.items()},
        'source_hashes':runner.frozen_sources(),'labels_read':False,
        'selection':'first_3_canonical_questions_per_dataset','question_attempts':1}
    with runner.writer_lock(output):
        path=output/'manifest.json'
        if path.exists() and runner.load(path)!=manifest:
            raise ValueError('Smoke config/source/scope changed; use a new output directory')
        runner.save(path,manifest)
        command=subprocess.check_output(['ps','-p',str(os.getpid()),'-o','command='],text=True).strip()
        runner.save(output/'pid.json',{'pid':os.getpid(),'pgid':os.getpgrp(),'output':str(output),
            'started_unix':time.time(),'process_command':command})
        try:
            runner.save(output/'progress.json',{'state':'preflight','updated_unix':time.time()})
            report=runner.preflight(config,list(scopes),endpoints=True,output=output,arms=[VERSION])
            runner.save(output/'preflight.json',report)
            for dataset in scopes:
                ensure_index(config,dataset,output)
            runner.save(output/'index_artifacts.json',{d:inspect_index(config,d) for d in scopes})
            counts={d:runner.generate_dataset(output,d,qs,[VERSION],config,retry_failed=False) for d,qs in scopes.items()}
            summary={'algorithm_version':VERSION,'questions':6,'datasets':counts,'labels_read':False,
                     'apc_hardware_status':'IMPLEMENTED_NOT_HARDWARE_VERIFIED'}
            runner.save(output/'smoke_summary.json',summary)
            runner.save(output/'progress.json',{'state':'complete' if all(v.get('ok',0)==3 for v in counts.values()) else 'complete_with_failures',
                'updated_unix':time.time(),'terminal_counts':counts})
            return summary
        except BaseException as exc:
            runner.save(output/'progress.json',{'state':'failed','updated_unix':time.time(),**runner.redacted_error(exc)})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        result=run(args.config,args.output)
    except Exception as exc:
        print(json.dumps(runner.redacted_error(exc)),file=sys.stderr)
        return 1
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
