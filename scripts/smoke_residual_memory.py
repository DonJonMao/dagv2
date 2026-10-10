#!/usr/bin/env python3
"""Fixed six, single-attempt residual smoke; reuse the existing runner."""
import argparse
import json
from smoke_local_terminal import run
from dagbt.methods import RESIDUAL
from dagbt import runner
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        result=run(args.config,args.output,RESIDUAL)
    except Exception as exc:
        print(json.dumps(runner.redacted_error(exc)));raise SystemExit(1)
    print(json.dumps(result,indent=2))
