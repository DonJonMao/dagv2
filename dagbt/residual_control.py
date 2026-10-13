"""Residual bindings control the existing DAG's next local memory action.

This adapter owns no planner, transport or retrieval implementation. The Engine
still plans once before creating BT and supplies its shared ledger/source bank.
"""
from copy import deepcopy
import json
import time
from collections import Counter

from .residual import possible_outputs, certify_output, ProgramError, references
from .joint_reading import JointReader
from .budget import BudgetExceeded
from .reasoning import ProtocolError, InputOverflow, RefusalError
from .support import validate_span
from .transport import digest, save
from . import local_terminal


def display(value):
    return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,sort_keys=True)


class ResidualControl:
    # A failed audit authorizes another evidence-backed attempt, never publication
    # of its rejected answer. All attempts still share the original ledger.
    MAX_AUDIT_CORRECTIONS = 2

    def __init__(self, engine):
        self.e, self.state = engine, engine.fact_state
        self.reader = JointReader(engine,self.state)
        self.certificate = None
        self.stop_reason = 'not_started'
        self.exhausted, self.visited, self.audited = set(), set(), set()
        self.last_active = None
        self.local_pools = {}
        self.format_identity = None
        self.formatted_prediction = None
        self.format_input = None
        self.audit_corrections = 0
        self.audit_feedback = None
        self.audit_replayed = False
        self.semantic_audit_identity = None
        self.forced_demands = set()
        self.read_attempts, self.read_effects, self.compose_exhausted = set(), set(), set()
        self.e.derivation_verifier = self.verify_derivation

    def candidate(self):
        values,sources,_ = self.state.valid()
        cert=certify_output(self.state.program,values,self.state.contract,
            program_revision=self.state.revision, versions=self.state.versions,sources=sources,
            pending_variables=self.state.pending,max_states=self.e.s['max_enumeration_states'])
        if cert is None:
            return None
        binding=self.state.initial_program.get('rule_binding')
        if binding:
            variable=binding['variable']
            if variable not in values or not self.state.rule_route_ids or variable in self.state.pending:
                return None
            cert['rule_assumptions']={'variable':variable,'version':self.state.versions[variable],
                'identity':digest(values[variable]),'route_ids':list(self.state.rule_route_ids)}
            cert['source_ids']=sorted(set(cert['source_ids'])|set(sources[variable]))
        values, _, routes = self.state.valid()
        dependencies=set(cert['fact_versions'])
        if cert.get('rule_assumptions'):
            dependencies.add(cert['rule_assumptions']['variable'])
        pending=list(dependencies)
        while pending:
            variable=pending.pop()
            for rid in routes.get(variable,[]):
                for parent in self.state.routes[rid]['parent_versions']:
                    if parent not in dependencies:
                        dependencies.add(parent);pending.append(parent)
        if dependencies & self.state.pending:
            return None
        cert['binding_dependencies']={v:{'version':self.state.versions[v], 'identity':digest(values[v]),
            'route_ids':routes[v]} for v in sorted(dependencies)}
        # Revalidate sources at publication, including authoritative role metadata.
        for sid in cert['source_ids']:
            validate_span(self.e.spans[sid],self.e.docs)
            alias=self.e.mapper.source_aliases[sid]
            self.reader._source(alias)
        return cert

    def verify_derivation(self,node,alternative):
        if alternative.get('support_kind')=='scoped_task_projection':
            current=self.task_projection(node['id'])
            return current is not None and alternative.get('derivation')==current and node['answer']==display(current['output'])
        if alternative.get('binding_derivations'):
            for variable,certificate in alternative['binding_derivations'].items():
                if self.binding_certificate(variable)!=certificate:
                    return False
            return True
        if node['id']!=self.e.final_node_id or alternative.get('derivation') is None:
            return False
        current=self.candidate()
        return current is not None and alternative['derivation']==current and node['answer']==display(current['output'])

    def binding_certificate(self,variable):
        expression=self.state.program.get('task_expressions',{}).get(variable)
        if expression is None:
            return None
        values,sources,_=self.state.valid()
        # Never use a previous derived binding as an assumption for itself.
        values.pop(variable,None)
        program={**self.state.program,'expression':expression,'rules_complete':True}
        contract={'mode':'value','result_type':self.state.program['variables'][variable]['type'],
            **self.state.scope(variable),'explanation':'none'}
        return certify_output(program,values,contract,program_revision=self.state.revision,
            versions=self.state.versions,sources=sources,pending_variables=self.state.pending,
            max_states=self.e.s['max_enumeration_states'])

    def deterministic_bindings(self):
        for step in self.e.steps:
            for variable in self.demand_variables(step['output_slot']):
                if variable not in self.state.program.get('task_expressions',{}):
                    continue
                cert=self.binding_certificate(variable)
                if cert is None:
                    continue
                row={'variable':variable,'stance':'support','value':cert['output'],
                    'source_ids':cert['source_ids'],**self.state.scope(variable),'parent_versions':cert['fact_versions'],
                    'retract_ids':[],'reason':'Whitelist program derivation','binding_derivation':cert}
                self.state.integrate([row],self.state.pending)

    def task_projection(self,task):
        """A rule field can resolve its local task without binding unused inputs.

        This is local node solving in both joint/residual arms, not global task
        elimination. In particular, a witnessed universal violation settles D
        while its separate, unsupported positive-compliance input stays unknown.
        """
        expression=self.state.program['expression']
        if expression['op'] not in ('one_reason','failures','record') or task not in expression['fields']:
            return None
        field=expression['fields'][task]
        tasks={s['output_slot']:s for s in self.e.steps}
        allowed={task}
        frontier=[task]
        while frontier:
            t=frontier.pop()
            for p in tasks[t]['inputs']:
                if p not in allowed:
                    allowed.add(p);frontier.append(p)
        if any(self.state.program['variables'][v]['demand'] not in allowed for v in references(field)):
            return None
        values,sources,_=self.state.valid()
        scopes=[self.state.scope(v) for v in self.demand_variables(task)]
        if scopes and any(scope!=scopes[0] for scope in scopes):
            # A single projected bool cannot claim one scope for mixed local
            # declarations. Their ordinary typed bindings remain available.
            return None
        scope=scopes[0] if scopes else {k:self.state.contract[k] for k in ('entity_scope','time_scope')}
        contract={'mode':'value','result_type':'bool',**scope,'explanation':'none'}
        cert=certify_output({**self.state.program,'expression':field,'rules_complete':True},values,contract,
            program_revision=self.state.revision,versions=self.state.versions,sources=sources,
            pending_variables=self.state.pending,max_states=self.e.s['max_enumeration_states'])
        if cert is None:
            return None
        binding=self.state.initial_program.get('rule_binding')
        if binding:
            v=binding['variable']
            if v not in values or v in self.state.pending or not self.state.rule_route_ids:
                return None
            cert['rule_assumptions']={'variable':v,'version':self.state.versions[v],
                'identity':digest(values[v]),'route_ids':self.state.rule_route_ids}
            cert['source_ids']=sorted(set(cert['source_ids'])|set(sources[v]))
        # Local elimination relies on the same actual grounding chain as
        # publication. A settled child cannot hide its pending parent source.
        _,_,routes=self.state.valid()
        dependencies=set(cert['fact_versions'])
        if cert.get('rule_assumptions'):
            dependencies.add(cert['rule_assumptions']['variable'])
        frontier=list(dependencies)
        while frontier:
            for rid in routes.get(frontier.pop(),[]):
                for parent in self.state.routes[rid]['parent_versions']:
                    if parent not in dependencies:
                        dependencies.add(parent);frontier.append(parent)
        if dependencies & self.state.pending:
            return None
        cert['scope']={'task_projection':task,**scope}
        return cert

    def local_complete(self,task):
        variables=self.demand_variables(task)
        return (bool(variables) and all(v in self.state.valid()[0] for v in variables)) or self.task_projection(task) is not None

    def demand_variables(self,task):
        return [v for v,d in self.state.program['variables'].items() if d['demand']==task]

    def semantic_dependencies(self):
        _,_,routes=self.state.valid()
        variables=set(self.demand_variables(self.e.final_node_id))
        pending=list(variables)
        while pending:
            v=pending.pop()
            for rid in routes.get(v,[]):
                for p in self.state.routes[rid]['parent_versions']:
                    if p not in variables:
                        variables.add(p);pending.append(p)
        return variables

    def semantic_binding_identity(self):
        """Identity of the actual typed terminal and its grounding assumptions."""
        values,sources,routes=self.state.valid()
        return digest({v:{'value':values.get(v),'sources':sources.get(v,[]),
            'routes':routes.get(v,[]),'version':self.state.versions.get(v),
            'pending':v in self.state.pending} for v in sorted(self.semantic_dependencies())})

    def semantic_inputs_current(self):
        if self.reader.semantic_inputs is None:
            return False
        values,_,_=self.state.valid()
        dependencies=self.semantic_dependencies()
        current={v:{'version':self.state.versions[v],'identity':digest(values[v])}
            for v in sorted(dependencies) if v in values}
        return (dependencies<=set(values) and not dependencies & self.state.pending and
            current==self.reader.semantic_inputs.get('binding_dependencies'))

    def semantic_publication_identity(self):
        dependencies=self.semantic_dependencies()
        routes=[]
        for route in self.state.routes.values():
            if route['variable'] in dependencies:
                row=deepcopy(route)
                row['revoked']=sorted({digest(r):r for r in row['revoked']}.values(),key=digest)
                routes.append(row)
        return digest([self.semantic_binding_identity(),self.state.program,
            sorted(routes,key=digest),self.reader.semantic_result,self.reader.semantic_prediction])

    def typed_terminal_value(self):
        # Legacy DAG node answers are display strings, not typed bindings. Never
        # round-trip those strings when validating or publishing a typed result.
        values,_,_=self.state.valid()
        expression=self.state.program['expression']
        if expression['op']=='var':
            return deepcopy(values.get(expression['id']))
        outcome=possible_outputs(self.state.program,values,
            max_states=self.e.s['max_enumeration_states'])
        if outcome['status']=='determined':
            return deepcopy(outcome['outputs'][0])
        return None

    def recover_after_audit(self,reason):
        """Keep all counters/obligations and allow bounded local correction."""
        self.e.audit_complete=False
        self.semantic_audit_identity=None
        self.certificate=None
        if (self.audit_corrections>=self.MAX_AUDIT_CORRECTIONS or
                self.audit_replayed or
                self.e.ledger.remaining('llm')<=self.e.s['reserved_audit_calls']):
            return False
        self.audit_corrections+=1
        recent=[b for b in self.reader.batches if b['operation']=='audit_joint']
        self.audit_feedback={'reason':reason,'correction_round':self.audit_corrections,
            'previous_semantic_answer':deepcopy(self.reader.semantic_result),
            'previous_prediction':self.reader.semantic_prediction,
            'audits':[{'verdict':b.get('audit_verdict'),
                'failed_slots':deepcopy(b.get('failed_slots',[]))} for b in recent[-1:]],
            'instruction':'Re-evaluate using the original evidence and all retained counterevidence. '
                'Retract an unsupported old route explicitly before replacing it. '
                'A corrected terminal must pass a fresh audit; do not repeat the rejected answer unchanged.'}
        if self.state.contract['mode']=='semantic':
            self.forced_demands.add(self.e.final_node_id)
            self.reader.semantic_inputs=None
            self.reader.semantic_result=None
            self.reader.semantic_prediction=None
        else:
            # A symbolic certificate may depend on retrieval nodes only. Re-read
            # those local facts, not an unrelated terminal answer generation.
            values,_,_=self.state.valid()
            variables=set(values)|self.state.pending
            self.forced_demands.update(self.state.program['variables'][v]['demand']
                for v in variables if v not in self.state.program.get('task_expressions',{}))
        self.format_identity=None;self.formatted_prediction=None;self.format_input=None
        self.e.event({'event':'audit_correction_scheduled','reason':reason,
            'round':self.audit_corrections,'max_rounds':self.MAX_AUDIT_CORRECTIONS,
            'active_demands':sorted(self.forced_demands)})
        return True

    def grounded(self,step):
        values,_,_=self.state.valid()
        query=step['question']
        binding=self.state.initial_program.get('rule_binding')
        if binding and step['output_slot']==self.state.initial_program['variables'][binding['variable']]['demand'] and self.rules_completion_needed():
            query+='\nRemaining local gap: obtain applicable original evidence confirming the complete rule collection; the currently bound rules are incomplete.'
        for parent in step['inputs']:
            projection=self.task_projection(parent)
            if projection is not None:
                replacement=display(projection['output'])
                query=query.replace('{'+parent+'}',replacement)
                if '{'+parent+'}' not in step['question']:
                    query+='\nEstablished '+parent+': '+replacement
                continue
            variables=self.demand_variables(parent)
            if not variables or any(v not in values for v in variables):
                return None
            bound={v:values[v] for v in variables}
            replacement=display(next(iter(bound.values()))) if len(bound)==1 else display(bound)
            query=query.replace('{'+parent+'}',replacement)
            if '{'+parent+'}' not in step['question']:
                query+='\nEstablished '+parent+': '+replacement
        return query

    def rules_completion_needed(self):
        if not self.state.initial_program.get('rule_binding') or not self.state.rule_route_ids or self.state.program.get('rules_complete',True):
            return False
        values,_,_=self.state.valid()
        # A necessary-condition failure can suffice without completing all
        # rules. Other outputs retain completeness as a residual rule demand.
        return certify_output(self.state.program,values,self.state.contract,
            max_states=self.e.s['max_enumeration_states']) is None

    def active(self,outcome):
        values,_,_=self.state.valid()
        live=set(outcome['live_variables'])|self.state.pending
        tasks={self.state.program['variables'][v]['demand'] for v in live if v in self.state.program['variables'] and (v not in values or v in self.state.pending)}
        binding=self.state.initial_program.get('rule_binding')
        if binding and (not self.state.rule_route_ids or self.rules_completion_needed()):
            tasks.add(self.state.initial_program['variables'][binding['variable']]['demand'])
        if not self.e.capabilities.residual_control:
            tasks|={s['output_slot'] for s in self.e.steps if s['execution']=='retrieval' and s['output_slot'] not in self.visited}
        # A locally certified field has solved its DAG node even when an
        # eliminated input retains an unresolved evidence obligation. The
        # certificate itself rejects pending *used* assumptions. Keep every
        # other obligation stored: invalidating the decisive source removes
        # this projection and reactivates its local task automatically.
        tasks={t for t in tasks if self.task_projection(t) is None}
        tasks |= self.forced_demands
        steps={s['output_slot']:s for s in self.e.steps}
        def parents(task):
            for p in steps[task]['inputs']:
                variables=self.demand_variables(p)
                if self.task_projection(p) is None and any(v not in values for v in variables):
                    tasks.add(p);parents(p)
        for task in list(tasks):
            parents(task)
        return [s['output_slot'] for s in self.e.steps if s['output_slot'] in tasks]

    def sync_nodes(self,certificate=None):
        values,sources,ids=self.state.valid()
        nodes=[]
        for step in self.e.steps:
            nid=step['output_slot'];old=self.e.node_map()[nid]
            variables=self.demand_variables(nid)
            bound={v:values[v] for v in variables if v in values}
            bound_scopes={v:self.state.scope(v) for v in bound}
            node={**deepcopy(old),'answer':None,'status':'unknown','declared_status':'unknown','alternatives':[],
                  'typed_values':bound,'variable_scopes':bound_scopes,
                  'version':max([self.state.versions.get(v,0) for v in variables]+[0])}
            if bound and len(bound)==len(variables):
                answer=display(next(iter(bound.values()))) if len(bound)==1 else display(bound)
                # The fact store retains all OR routes; the bounded legacy graph
                # displays at most two. Typed certificates retain every route.
                source_ids=sorted({sid for v in bound for sid in sources[v]})
                node.update(answer=answer,status='supported',declared_status='supported',alternatives=[{
                    'id':nid+'.typed.'+digest([bound,source_ids])[:16],'answer':answer,
                    'semantic_status':'supported','applicable_scope':(
                        next(iter(bound_scopes.values())) if len(bound_scopes)==1 else {'variables':bound_scopes}),
                    'source_span_ids':source_ids,'guard_span_ids':[], 'used_parent_ids':[], 'used_parent_versions':{},
                    'fact_route_ids':{v:ids[v] for v in bound},'support_kind':'joint_semantic_routes'}])
                derivations={v:self.state.routes[ids[v][0]]['binding_derivation'] for v in bound
                             if 'binding_derivation' in self.state.routes[ids[v][0]]}
                if derivations:
                    node['alternatives'][0].update(support_kind='symbolic_bindings',binding_derivations=derivations)
                elif len(bound)==1:
                    variable=next(iter(bound))
                    alternatives=[]
                    for route in self.state.routes.values():
                        if route['variable']!=variable or route['revoked'] or digest(route['value'])!=digest(bound[variable]):
                            continue
                        if any(route.get(k)!=v for k,v in self.state.scope(variable).items()):
                            continue
                        if 'binding_derivation' in route and route['binding_derivation']['program_revision']!=self.state.revision:
                            continue
                        if any(p not in values or self.state.versions[p]!=ver for p,ver in route['parent_versions'].items()):
                            continue
                        refs=sorted(set(route['source_ids'])|{s for p in route['parent_versions'] for s in sources[p]})
                        alternatives.append({**deepcopy(node['alternatives'][0]),
                            'id':nid+'.route.'+route['id'],'source_span_ids':refs,'fact_route_ids':{variable:[route['id']]}})
                    node['alternatives']=sorted(alternatives,key=lambda a:(len(a['source_span_ids']),a['source_span_ids'],a['id']))[:self.e.s['max_alternatives']]
            if nid==self.e.final_node_id and certificate:
                answer=display(certificate['output'])
                node.update(answer=answer,status='supported',declared_status='supported',
                    version=self.state.revision+sum(certificate['fact_versions'].values()),alternatives=[{
                        'id':nid+'.derivation.'+digest(certificate)[:16],'answer':answer,
                        'semantic_status':'supported','applicable_scope':{
                            k:self.state.contract[k] for k in ('entity_scope','time_scope')},
                        'source_span_ids':list(certificate['source_ids']),'guard_span_ids':[],
                        'used_parent_ids':[],'used_parent_versions':{},'support_kind':'symbolic_derivation',
                        'derivation':deepcopy(certificate)}])
            elif nid!=self.e.final_node_id and self.task_projection(nid) is not None:
                projection=self.task_projection(nid)
                answer=display(projection['output'])
                node.update(answer=answer,status='supported',declared_status='supported',
                    alternatives=[{'id':nid+'.projection.'+digest(projection)[:16], 'answer':answer,
                        'semantic_status':'supported','applicable_scope':{
                            k:projection['scope'][k] for k in ('entity_scope','time_scope')},
                        'source_span_ids':projection['source_ids'],'guard_span_ids':[],
                        'used_parent_ids':[],'used_parent_versions':{},
                        'support_kind':'scoped_task_projection','derivation':projection}])
            nodes.append(node)
        self.e.nodes=nodes
        self.e.compile()

    def update(self):
        self.deterministic_bindings()
        values,_,_=self.state.valid()
        outcome=possible_outputs(self.state.program,values,max_states=self.e.s['max_enumeration_states'])
        active=self.active(outcome)
        if active!=self.last_active:
            prior=set(self.last_active or [])
            for task in prior-set(active):
                self.e.bridge.pause_node(task)
            self.e.event({'event':'residual_state','program_revision':self.state.revision,
                'remaining_expression':outcome['reduction']['expression'],'active_demands':active,
                'paused_demands':sorted(prior-set(active)),'reactivated_demands':sorted(set(active)-prior),
                'elimination_rules':outcome['reduction']['steps'],'outcome':outcome['status']})
            self.last_active=active
        self.sync_nodes()
        return outcome,active

    def audit(self,certificate=None):
        values, sources, _=self.state.valid()
        if certificate:
            variables=set(certificate['binding_dependencies'])-set(self.state.program.get('task_expressions',{}))
            if certificate.get('rule_assumptions'):
                variables.add(certificate['rule_assumptions']['variable'])
            source_ids=certificate['source_ids']
        else:
            variables=self.semantic_dependencies()-set(self.state.program.get('task_expressions',{}))
            source_ids=sorted({s for v in variables for s in sources.get(v,[])})
        # Call counters/history and duplicate counter records do not constitute
        # new evidence. A changed answer, real route or counter does.
        routes=[]
        for route in self.state.routes.values():
            if route['variable'] in variables:
                row=deepcopy(route)
                row['revoked']=sorted({digest(r):r for r in row['revoked']}.values(),key=digest)
                routes.append(row)
        key=digest([certificate,self.semantic_binding_identity(),
            self.reader.semantic_result,self.reader.semantic_prediction,sorted(routes,key=digest)])
        self.audit_replayed=False
        if not variables and certificate:
            # Pure tautology has no semantic facts to audit. Still validate its
            # nonempty worlds and derivation with the free evaluator.
            self.e.audit_complete=True;return True
        if key in self.audited:
            self.e.audit_complete=False
            self.audit_replayed=True
            return False
        self.audited.add(key)
        doc_ids=list(dict.fromkeys(self.e.spans[s]['doc_id'] for s in source_ids))
        # Include every known relevant counter, including revoked routes.
        for r in self.state.routes.values():
            if r['variable'] in variables:
                for sid in r['source_ids']:
                    d=self.e.spans[sid]['doc_id']
                    if d not in doc_ids:doc_ids.append(d)
                for counter in r['revoked']:
                    for sid in counter['source_ids']:
                        d=self.e.spans[sid]['doc_id']
                        if d not in doc_ids:doc_ids.append(d)
        step={**next(s for s in self.e.steps if s['output_slot']==self.e.final_node_id),
              'inputs':[s['output_slot'] for s in self.e.steps if s['output_slot']!=self.e.final_node_id]}
        self.reader.read(step,self.e.q['question'],sorted(variables),doc_ids,key,audit=True)
        relevant=[b for b in self.reader.batches if b['identity']==key and b['operation']=='audit_joint']
        self.e.audit_complete=bool(relevant) and all(b.get('audit_complete') for b in relevant)
        return self.e.audit_complete

    def terminal_format_request(self, certificate):
        """The exact paid formatting envelope, shared with capacity checks."""
        from .personamem import TERMINAL_CHOICE_INSTRUCTION
        sources=[deepcopy(self.e.spans[s]) for s in certificate['source_ids']]
        data={'terminal_node_id':self.e.final_node_id,'certificate':certificate,
              'semantic_result':certificate['output'],'original_source_spans':sources,
              'public_output_question':self.e.reader_question,'public_options':self.e.output_options,
              'output_identity':self.e.output_identity}
        schema={'type':'object','properties':{'semantic_answer':{},'final_prediction':{'type':['string','null']}},
                'required':['semantic_answer','final_prediction'],'additionalProperties':False}
        system=('This is the first and only generation at the designated deterministic DAG terminal. '
            'Return semantic_answer (the exact typed result) and its corresponding final_prediction together. '
            'Use the computed result and its original sources. Return null label if ambiguous. '
            'Never guess or change the semantic result. No subsequent Reader or option matcher will run.'
            + TERMINAL_CHOICE_INSTRUCTION)
        return system,data,schema

    def terminal_format(self, certificate):
        """The designated terminal's output phase, no upstream option exposure."""
        if self.e.output_options is None:
            return True
        from .personamem import parse_choice
        value=certificate['output']
        matches=[chr(97+i) for i,text in enumerate(self.e.output_options) if str(text).strip()==str(value).strip()]
        if len(matches)==1:
            self.formatted_prediction=matches[0]
            self.format_input=deepcopy(self.reader.semantic_inputs)
            return True
        if self.reader.semantic_inputs is not None:
            # The real factual terminal has already produced its semantic answer
            # and label together. Never add an after-the-fact option matcher.
            if (not self.reader.semantic_inputs['validation_complete'] or
                    digest(self.reader.semantic_result)!=digest(value) or
                    parse_choice(self.reader.semantic_prediction,self.e.output_options) is None):
                return False
            self.formatted_prediction=self.reader.semantic_prediction
            self.format_input=deepcopy(self.reader.semantic_inputs)
            return True
        identity=digest([certificate,self.e.output_identity])
        if self.format_identity==identity:
            return self.formatted_prediction is not None
        self.format_identity=identity
        system,data,schema=self.terminal_format_request(certificate)
        sources=data['original_source_spans']
        def validate(response):
            if not isinstance(response,dict) or set(response)!={'semantic_answer','final_prediction'}:
                raise ProtocolError('Terminal output requires semantic_answer and final_prediction together')
            if digest(response['semantic_answer'])!=digest(value):
                raise ProtocolError('Terminal semantic answer differs from the computed result')
            prediction=response['final_prediction']
            if prediction is not None and parse_choice(prediction,self.e.output_options) is None:
                raise ProtocolError('Invalid terminal option label')
            return prediction
        self.formatted_prediction=self.e.reasoner.json('semantic_compose',system,data,validate,schema,reserve_repairs=0)
        self.format_input={'doc_ids':list(dict.fromkeys(s['doc_id'] for s in sources)),
            'validation_complete':self.formatted_prediction is not None,
            'response_ref':self.e.reasoner.last_response.get('response_ref'),
            'request':deepcopy(self.e.reasoner.last_response),'output_identity':self.e.output_identity,
            'certificate_identity':digest(certificate),'scope':'semantic_terminal_output'}
        return self.formatted_prediction is not None

    def read_progress(self, step, query, variables, doc_ids, identity, **options):
        """A repeated local read/effect cannot restart the control loop.

        A parent review can clear a gap that a terminal immediately recreates,
        so comparing only each individual call's before/after state is not
        enough. Keep scoped input/effect identities across those transitions.
        This does not exhaust retrieval: a rejected repeat still falls through
        to the same bridge session, whose frontier may supply genuinely new
        original text. The reader signature includes every new window, fact,
        source, counter and obligation, but excludes logs and call counters.
        """
        before=self.reader.material_signature()
        attempt=digest([identity,before,list(dict.fromkeys(doc_ids)),options])
        if attempt in self.read_attempts:
            self.e.event({'event':'local_read_replay_skipped','node_id':step['output_slot'],
                'identity':identity,'reason':'same_original_text_and_material_state'})
            return False
        self.read_attempts.add(attempt)
        changed=self.reader.read(step,query,variables,doc_ids,identity,**options)
        effect=digest([identity,self.reader.material_signature()])
        fresh=effect not in self.read_effects
        self.read_effects.add(effect)
        if changed and not fresh:
            self.e.event({'event':'local_read_cycle_detected','node_id':step['output_slot'],
                'identity':identity,'reason':'previous_material_state_recreated',
                'pending_variables':sorted(self.state.pending)})
        return changed and fresh

    def review_existing_obligations(self, step, variables, identity):
        if step['output_slot'] in self.forced_demands:
            return True
        values,_,routes=self.state.valid()
        related=set(variables)|{v for v,d in self.state.program['variables'].items()
            if v in values and d['demand'] in step['inputs']}
        frontier=list(related)
        while frontier:
            for rid in routes.get(frontier.pop(),[]):
                for parent in self.state.routes[rid]['parent_versions']:
                    if parent not in related:
                        related.add(parent);frontier.append(parent)
        return (self.reader.failed_review_ready(step,related)
            or any(item['variable'] in related and identity not in item.get('origins',[])
                   for item in self.reader.evidence_obligations.values()))

    def run(self):
        if self.e.audit_complete and self.certificate is not None and self.certificate==self.candidate():
            self.stop_reason='certified'
            return self.publish()
        if self.e.audit_complete and self.stop_reason=='semantic_assessed':
            return self.publish()
        self.e.event({'event':'typed_dag_initialized','dag':deepcopy(self.e.steps),
            'final_node_id':self.e.final_node_id,'output_contract':self.state.contract,
            'F':self.state.initial_program,'planning_before_first_retrieval':True})
        while True:
            outcome,active=self.update()
            cert=self.candidate()
            ready=cert is not None and not self.forced_demands and (self.e.capabilities.residual_control or not active)
            semantic=self.state.contract['mode']=='semantic'
            values,_,_=self.state.valid()
            terminal_variables=self.demand_variables(self.e.final_node_id)
            semantic_ready=(semantic and not self.forced_demands and bool(terminal_variables) and self.reader.semantic_inputs is not None
                and self.reader.semantic_inputs['validation_complete'] and
                self.semantic_inputs_current())
            if ready or semantic_ready:
                binding_identity=self.semantic_binding_identity()
                try:
                    if cert and not self.terminal_format(cert):
                        self.stop_reason='invalid_output_format';break
                    if not self.audit(cert):
                        if self.recover_after_audit('audit_rejected'):
                            continue
                        self.stop_reason='audit_incomplete';break
                except (BudgetExceeded,InputOverflow,ProtocolError,ProgramError) as exc:
                    self.error('audit',exc);self.stop_reason='audit_incomplete';break
                # Audit may correct/revoke facts. Recompute; no old certificate.
                if cert is not None:
                    new=self.candidate()
                    if new!=cert:
                        self.e.audit_complete=False
                        self.e.event({'event':'certificate_invalidated_after_audit','old_identity':digest(cert)})
                        if self.recover_after_audit('certificate_changed'):
                            continue
                        self.stop_reason='audit_incomplete';break
                    self.certificate=new;self.stop_reason='certified';break
                values,_,_=self.state.valid()
                if (binding_identity!=self.semantic_binding_identity() or
                        not all(v in values for v in self.semantic_dependencies())):
                    if self.recover_after_audit('semantic_bindings_changed'):
                        continue
                    self.stop_reason='audit_incomplete';break
                self.semantic_audit_identity=self.semantic_publication_identity()
                self.stop_reason='semantic_assessed';break
            if outcome['status']=='inconsistent':
                self.stop_reason='inconsistent';break
            if outcome['status']=='determined' and not semantic and not active and cert is None:
                self.stop_reason='uncertifiable_output'
                self.e.event({'event':'determined_output_not_certifiable',
                    'mode':self.state.contract['mode'],'result_type':self.state.contract['result_type'],
                    'expression_op':self.state.program['expression']['op'],
                    'rules_complete':self.state.program.get('rules_complete',True),
                    'pending_variables':sorted(self.state.pending),
                    'reason':'No certificate and no remaining admissible evidence action; output is withheld.'})
                break
            executed=False
            for step in self.e.steps:
                nid=step['output_slot']
                if nid not in active:
                    continue
                query=self.grounded(step)
                if query is None:
                    continue
                variables=self.demand_variables(nid)
                # Placeholder rule-dependent tasks are retained in the initial
                # DAG but can execute only after sourced instantiation.
                if not variables:
                    continue
                if any(v in self.state.program.get('task_expressions',{}) for v in variables):
                    continue
                _, sources, _=self.state.valid()
                parent_vars=[v for p in step['inputs'] for v in self.demand_variables(p)]
                parent_docs=list(dict.fromkeys(self.e.spans[s]['doc_id'] for v in parent_vars for s in sources.get(v,[])))
                identity={'algorithm':self.e.s['algorithm_version'],'query':query,'node_id':nid,
                    'program_revision':self.state.revision,'program_identity':digest(self.state.program),
                    'initial_dag_identity':digest(self.e.steps),'bindings':{v:self.state.versions[v] for v in parent_vars},
                    'contract_identity':digest(self.state.contract)}
                if self.audit_feedback is not None:
                    identity['audit_correction']=digest(self.audit_feedback)
                key=digest(identity)
                if key in self.exhausted:
                    continue
                if (step['execution']=='compose' and
                        digest([key,self.reader.material_signature()]) in self.compose_exhausted):
                    continue
                try:
                    before_read=self.reader.material_signature()
                    if self.read_progress(step,query,variables,list(dict.fromkeys(self.local_pools.get(nid,[])+parent_docs)),key,
                            semantic=semantic and nid==self.e.final_node_id and step['execution']=='compose',
                            terminal_output=nid==self.e.final_node_id and self.e.output_options is not None,
                            review_existing_obligations=self.review_existing_obligations(step,variables,key)):
                        self.forced_demands.discard(nid)
                        self.visited.update([nid] if self.local_complete(nid) else []);executed=True;break
                    self.forced_demands.discard(nid)
                    if step['execution']=='compose':
                        blocked=digest([key,self.reader.material_signature()])
                        newly_blocked=blocked not in self.compose_exhausted
                        self.compose_exhausted.add(blocked)
                        if newly_blocked and before_read!=self.reader.material_signature():
                            # This read recreated an old unresolved state. Close
                            # that composition once, then let its newly pending
                            # parents advance their still-live bridge frontiers.
                            # New text/facts may later admit a different attempt.
                            executed=True;break
                        continue
                    if self.e.ledger.remaining('llm')<=self.e.s['reserved_audit_calls']:
                        self.exhausted.add(key);continue
                    if self.reader.failed_review_blocked(step,variables):
                        # Extra ANN windows cannot repair a required failed
                        # batch after its repair budget is unavailable.
                        self.exhausted.add(key);continue
                    if self.e.fixed_pool is not None:
                        local=self.e.fixed_pool
                        if set(local)<=set(self.e.candidates) and all((key,a) in self.reader.read_states for a,s in self.e.mapper.sources.items() if s['doc_id'] in local):
                            self.exhausted.add(key);continue
                    else:
                        found=self.e.bridge.step(query,nid,[{'id':nid,'description':query,'necessary':True}],parent_docs,identity=identity)
                        self.e.discoveries.append(deepcopy(found));self.e.record_navigation(found)
                        local=found['local_candidate_ids']
                        if not local:
                            if found['trace'].get('search_status')=='empty_probe':
                                executed=True;break
                            self.exhausted.add(key);continue
                    self.local_pools[nid]=list(dict.fromkeys(self.local_pools.get(nid,[])+local))
                    self.e.add_candidates(local)
                    self.read_progress(step,query,variables,local,key,semantic=semantic and nid==self.e.final_node_id,
                        terminal_output=nid==self.e.final_node_id and self.e.output_options is not None,
                        review_existing_obligations=False)
                    self.visited.update([nid] if self.local_complete(nid) else []);executed=True;break
                except RefusalError:
                    raise
                except (BudgetExceeded,InputOverflow,ProtocolError,ProgramError,ConnectionError,ValueError) as exc:
                    self.error(nid,exc);self.exhausted.add(key)
            if not executed:
                self.stop_reason='budget_exhausted' if self.e.ledger.remaining('ann')==0 or self.e.ledger.remaining('llm')<=self.e.s['reserved_audit_calls'] else 'unresolved_or_exhausted'
                break
        # Free final evaluation always happens, including at budget zero.
        self.update()
        return self.publish()

    def error(self,task,exc):
        value={'node_id':task,'type':type(exc).__name__,'error':str(exc)}
        self.e.errors.append(value);self.e.event({'event':'residual_action_incomplete',**value})

    def snapshot(self):
        """Checkpoint at an observation boundary; no in-flight HTTP is replayed."""
        return deepcopy({'algorithm_version':self.e.s['algorithm_version'],'settings':self.e.s,
            'question':self.e.q,'output_identity':self.e.output_identity,
            'steps':self.e.steps,'final_node_id':self.e.final_node_id,
            'requirements':self.e.requirements,'requirements_hash':self.e.requirements_hash,
            'fact_state':self.state.public_dict(),'bridge':self.e.bridge.snapshot(),
            'candidates':self.e.candidates,'spans':list(self.e.spans.values()),
            'events':self.e.events,'errors':self.e.errors,'discoveries':self.e.discoveries,
            'reasoner_sequence':self.e.reasoner.sequence,'reasoner_requests':self.e.reasoner.requests,
            'navigation_provenance':self.e.navigation_provenance,
            'local_pools':self.local_pools,'exhausted':list(self.exhausted),
            'visited':list(self.visited),'audited':list(self.audited),'last_active':self.last_active,
            'audit_corrections':self.audit_corrections,'audit_feedback':self.audit_feedback,
            'audit_replayed':self.audit_replayed,'forced_demands':sorted(self.forced_demands),
            'read_attempts':sorted(self.read_attempts),'read_effects':sorted(self.read_effects),
            'compose_exhausted':sorted(self.compose_exhausted),
            'semantic_audit_identity':self.semantic_audit_identity,
            'reader':{'read_states':[[*k,v] for k,v in self.reader.read_states.items()],
                **self.reader.obligation_snapshot(),
                'gaps':[[*k,v] for k,v in self.reader.gaps.items()],'clues':self.reader.clues,
                'batches':self.reader.batches,'semantic_inputs':self.reader.semantic_inputs,
                'semantic_prediction':self.reader.semantic_prediction,'semantic_result':self.reader.semantic_result},
            'certificate':self.certificate,'audit_complete':self.e.audit_complete,
            'format_identity':self.format_identity,'formatted_prediction':self.formatted_prediction,
            'format_input':self.format_input,'stop_reason':self.stop_reason})

    def restore(self,snapshot):
        if snapshot['settings']!=self.e.s or snapshot['question']!=self.e.q or snapshot['output_identity']!=self.e.output_identity:
            raise ProtocolError('Resume method, budget, question or output identity changed')
        if snapshot['steps']!=self.e.steps or snapshot['requirements_hash']!=self.e.requirements_hash:
            raise ProtocolError('Resume initial DAG/requirements changed')
        self.e.bridge.restore(snapshot['bridge'])
        self.state.restore(snapshot['fact_state'])
        self.e.add_candidates(snapshot['candidates'])
        self.e.spans={s['id']:validate_span(s,self.e.docs) for s in snapshot['spans']}
        self.e.events=deepcopy(snapshot['events']);self.e.errors=deepcopy(snapshot['errors'])
        self.e.discoveries=deepcopy(snapshot['discoveries'])
        self.e.navigation_provenance=deepcopy(snapshot['navigation_provenance'])
        self.e.reasoner.sequence=snapshot['reasoner_sequence']
        self.e.reasoner.requests=deepcopy(snapshot['reasoner_requests'])
        self.local_pools=deepcopy(snapshot['local_pools'])
        self.exhausted=set(snapshot['exhausted']);self.visited=set(snapshot['visited'])
        self.audited=set(snapshot['audited']);self.last_active=snapshot['last_active']
        self.audit_corrections=snapshot.get('audit_corrections',0)
        self.audit_feedback=deepcopy(snapshot.get('audit_feedback'))
        self.audit_replayed=snapshot.get('audit_replayed',False)
        self.semantic_audit_identity=snapshot.get('semantic_audit_identity')
        self.forced_demands=set(snapshot.get('forced_demands',[]))
        self.read_attempts=set(snapshot.get('read_attempts',[]))
        self.read_effects=set(snapshot.get('read_effects',[]))
        self.compose_exhausted=set(snapshot.get('compose_exhausted',[]))
        r=snapshot['reader']
        self.reader.restore_obligations(r)
        self.reader.read_states={(i,a):v for i,a,v in r['read_states']}
        self.reader.gaps={(i,a):v for i,a,v in r['gaps']}
        self.reader.clues=deepcopy(r['clues']);self.reader.batches=deepcopy(r['batches'])
        self.reader.semantic_inputs=deepcopy(r['semantic_inputs']);self.reader.semantic_prediction=r['semantic_prediction']
        self.reader.semantic_result=deepcopy(r['semantic_result'])
        self.certificate=deepcopy(snapshot['certificate']);self.e.audit_complete=snapshot['audit_complete']
        self.format_identity=snapshot['format_identity'];self.formatted_prediction=snapshot['formatted_prediction']
        self.format_input=deepcopy(snapshot['format_input'])
        self.stop_reason=snapshot['stop_reason']
        if self.certificate is not None and self.candidate()!=self.certificate:
            self.certificate=None;self.e.audit_complete=False
        self.sync_nodes(self.certificate)

    def publish(self):
        cert=self.candidate() if self.certificate else None
        if cert!=self.certificate:
            self.certificate=None;self.e.audit_complete=False
        self.sync_nodes(self.certificate)
        sources=local_terminal.provenance(self.e.graph,self.e.final_node_id)
        semantic_valid=(self.stop_reason=='semantic_assessed' and self.reader.semantic_inputs is not None
            and self.reader.semantic_inputs['validation_complete'] and sources is not None
            and self.semantic_inputs_current()
            and self.semantic_audit_identity==self.semantic_publication_identity())
        if self.stop_reason=='semantic_assessed' and not semantic_valid:
            self.e.audit_complete=False
            self.stop_reason='audit_incomplete'
        success=self.e.audit_complete and (self.certificate is not None or semantic_valid)
        value=self.certificate['output'] if self.certificate else self.typed_terminal_value()
        prediction=display(value) if value is not None else None
        if success and not self.certificate and digest(self.reader.semantic_result)!=digest(value):
            success=False;self.stop_reason='invalid_output_format'
        if success and self.e.output_options is not None:
            from .personamem import parse_choice
            if self.certificate:
                prediction=self.formatted_prediction
            else:
                prediction=self.reader.semantic_prediction
            if not isinstance(prediction,str) or parse_choice(prediction,self.e.output_options) is None:
                success=False;self.stop_reason='invalid_output_format'
        answer={'status':'ok' if success else self.stop_reason,'prediction':prediction if success else None,
            'semantic_answer':value,'answer_source':'dag_terminal','final_node_id':self.e.final_node_id,
            'output_identity':self.e.output_identity,'sources':sources,
            'dependency_versions':self.certificate['fact_versions'] if self.certificate else self.state.versions,
            'completion_kind':'certified_symbolic' if success and self.certificate else 'semantic_assessment' if success else 'incomplete',
            'certificate':self.certificate,'program_revision':self.state.revision,
            'terminal_input_doc_ids':self.format_input['doc_ids'] if self.format_input else [] if self.certificate or not self.reader.semantic_inputs else self.reader.semantic_inputs['doc_ids'],
            'certificate_source_ids':self.certificate['source_ids'] if self.certificate else []}
        legacy_repairs=Counter(e['operation'] for e in self.e.events if e['event']=='reasoning_repair_prepared')
        cost={'generation_calls':len(self.e.reasoner.requests),'node_resolve_calls':0,'mapping_calls':0,
            'joint_read_calls':sum(r['operation']=='joint_read' for r in self.e.reasoner.requests),
            'repair_calls':sum(r['operation'].endswith('_repair') for r in self.e.reasoner.requests)+sum(legacy_repairs.values()),
            'json_repair_reservations':self.e.ledger.used['json_repairs'],
            'semantic_compose_calls':sum(r['operation']=='semantic_compose' for r in self.e.reasoner.requests)-legacy_repairs['semantic_compose'],
            'audit_calls':sum(r['operation']=='audit_joint' for r in self.e.reasoner.requests),
            'planner_calls':sum(r['operation']=='planner' for r in self.e.reasoner.requests)-legacy_repairs['planner'],
            'global_final_selector_calls':0,'independent_reader_calls':0,'independent_option_matcher_calls':0,
            'ann_calls':self.e.ledger.used['ann'],'set_score_calls':self.e.ledger.used['set_score'],
            'llm_attempts':self.e.ledger.used['llm'],'rerank_http_attempts':self.e.ledger.used['rerank_http']}
        result={'unit_id':self.e.q['id'],'method':self.e.method,'algorithm_version':self.e.s['algorithm_version'],
            'ranking':{'status':'ok' if success else 'partial','nodes':self.e.graph['nodes'],'trace':self.e.discoveries},
            'budgets':{},'answer':answer,'seconds':time.time()-self.e.started,
            'diagnostics':{'settings':self.e.s,'requirements':self.e.requirements,'requirements_hash':self.e.requirements_hash,
                'support_graph':self.e.graph,'spans':list(self.e.spans.values()),'candidate_doc_ids':self.e.candidates,
                'ledger':self.e.ledger.public_dict(),'errors':self.e.errors,'events':self.e.events,
                'fact_state':self.state.public_dict(),'joint_reading':self.reader.public_dict(),
                'bridge_cost':self.e.bridge.public_dict(),'bridge_resume':self.e.bridge.snapshot(),
                'audit_complete':self.e.audit_complete,'terminal_input':self.format_input or self.reader.semantic_inputs,
                'audit_corrections':self.audit_corrections,'audit_feedback':self.audit_feedback,
                'cost':cost,'reasoning_call_count':len(self.e.reasoner.requests),'legacy_reader_metrics':'not_applicable',
                'reliability':{'cohort':'normal' if success else 'partially_read',
                    'mapping_status':'not_used','mapping_complete':False,'mapping_incomplete':False,
                    'input_truncated':False,'joint_reading_complete':not self.state.pending}}}
        self.e.event({'event':'terminal_published','answer':answer,'stop_reason':self.stop_reason})
        if self.e.output:
            save(self.e.output/'fusion_snapshot.json',result);(self.e.output/'fusion_snapshot.json').chmod(0o600)
        return result
