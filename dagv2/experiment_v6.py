"""Patch loader: experiment module with no-thinking corpus-wide DAG v2 flow.

Replaces frozen_flow.controller/reader_input/chat_request and run.solve with
frozen_flow_v6 variants. Import before `import experiment` in the pipelines.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # dagv2/
BASE = HERE                                     # experiment.py lives beside this file
PACKAGE = HERE.parent / 'package'               # frozen DAG v1 core
for path in (str(PACKAGE), str(BASE), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import experiment  # noqa: F401  (original module, single sys.modules instance)
import frozen_flow
import frozen_flow_v6
import run as native

frozen_flow.controller = frozen_flow_v6.controller
frozen_flow.reader_input = frozen_flow_v6.reader_input
frozen_flow.chat_request = frozen_flow_v6.chat_request
native.solve = frozen_flow_v6.solve
