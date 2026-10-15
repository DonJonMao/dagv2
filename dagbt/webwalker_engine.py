"""Additive adapter: the PersonaMem Engine and its runner remain unchanged."""
from .engine import Engine
from .residual_control import ResidualControl
import time

from .webwalker_browser import RootPageUnavailable, PageStore, WebBridge
from .webwalker_reasoning import WebReasoner
from .webwalker_reading import WebJointReader


class WebEngine(Engine):
    def __init__(self, question, resources, calls, config, method, *, root_url, page_cache, allow_local=False):
        super().__init__(question, resources, calls, config, method)
        self.reasoner = WebReasoner(self.calls, self.tokenizer, self.config, self.s,
                                    self.ledger, self.event)
        self.web_root, self.page_cache, self.allow_local = root_url, page_cache, allow_local

    def run(self):
        # Planning precedes any website access. Only the public root URL is
        # available; no answer/source website/golden path enters this engine.
        self.plan()
        self.fixed_pool = None
        settings = self.config.get('webwalker', {})
        store = PageStore(self.page_cache, settings, allow_local=self.allow_local, event=self.event)
        self.bridge = WebBridge(self, self.web_root, store, settings)
        self.residual_control = ResidualControl(self)
        self.residual_control.reader = WebJointReader(self, self.fact_state)
        try:
            row = self.residual_control.run()
        except RootPageUnavailable as exc:
            self.event({'event': 'web_question_environment_skipped', **exc.details})
            row = {'unit_id': self.q['id'], 'method': self.method,
                   'answer': {'status': 'skipped_environment', 'prediction': None},
                   'environment_exclusion': {'scope': 'root_page', **exc.details},
                   'seconds': time.time() - self.started,
                   'diagnostics': {'ledger': self.ledger.public_dict(),
                                   'initial_dag': self.steps, 'final_node_id': self.final_node_id}}
        row['dataset'] = 'webwalkerqa'
        row['protocol'] = 'provided_root_observed_link_traversal_v4_access_v2'
        row['root_url'] = self.web_root
        row['diagnostics']['web'] = self.bridge.public_dict()
        row['diagnostics']['web']['http_events'] = store.events
        return row
