"""Versioned typed observations; provenance routes are OR alternatives."""
from copy import deepcopy
from .residual import validate_value, validate_program, variable_scope, ProgramError
from .residual_plan import validate_bindings
from .transport import digest


class FactState:
    def __init__(self, program, contract, steps, final_node_id):
        validate_bindings(program, steps, final_node_id, contract=contract)
        self.initial_program = deepcopy(program)
        self.program, self.contract = deepcopy(program), deepcopy(contract)
        self.steps, self.final_node_id = deepcopy(steps), final_node_id
        self.routes = {}
        self.versions = {v:0 for v in program['variables']}
        self.pending = set()
        self.history = []
        self.revision = 0
        self.rule_route_ids = []
        self.instantiations = []
        self._identities = {}

    def scope(self, variable):
        return variable_scope(self.program['variables'][variable], self.contract)

    def valid(self):
        # Resolve parent routes in DAG order, never accept unsupported guesses.
        values, sources, route_ids = {}, {}, {}
        for task in self.steps:
            for variable, declaration in self.program['variables'].items():
                if declaration['demand'] != task['output_slot']:
                    continue
                routes = [r for r in self.routes.values() if r['variable']==variable and not r['revoked']
                          and all(r.get(k)==v for k,v in self.scope(variable).items())
                          and ('binding_derivation' not in r or r['binding_derivation']['program_revision']==self.revision)
                          and all(p in values and self.versions[p]==ver for p,ver in r['parent_versions'].items())]
                if not routes or len({digest(r['value']) for r in routes}) != 1:
                    continue
                def closure(route):
                    return set(route['source_ids']) | {s for p in route['parent_versions'] for s in sources[p]}
                # Retain every OR alternative in routes, but certify an actual
                # sufficient route rather than turning all alternatives into AND.
                chosen=min(routes,key=lambda r:(len(closure(r)),sorted(closure(r)),r['id']))
                values[variable] = deepcopy(chosen['value'])
                sources[variable] = sorted(closure(chosen))
                route_ids[variable] = [chosen['id']]
        return values, sources, route_ids

    def _refresh_versions(self):
        # A route change (including a correction with identical display text)
        # invalidates actual dependents, while surviving independent OR routes
        # still provide a fact. Iterate over the finite DAG, no model calls.
        for _ in self.steps:
            values, sources, route_ids = self.valid()
            changed = False
            for v in self.program['variables']:
                identity = digest([values[v], sources[v], route_ids[v]]) if v in values else None
                if self._identities.get(v) != identity:
                    self._identities[v] = identity
                    self.versions[v] = self.versions.get(v,0)+1
                    changed = True
            if not changed:
                break

    def integrate(self, rows, pending=()):
        """Validated complete batch first; no evaluator/publication in this loop."""
        # Validate all supported values before mutating any route, including
        # callers outside JointReader and numeric domain intersections.
        rows=[{**r,'value':validate_value(r['value'],self.program['variables'][r['variable']])}
              if r['stance']=='support' else r for r in rows]
        for row in rows:
            if row['stance'] in ('support','contradiction') and any(
                    row.get(key) != value for key,value in self.scope(row['variable']).items()):
                raise ProgramError('Observation scope does not match its declared variable scope')
        for row in rows:
            if row['stance'] in ('unknown','partial','irrelevant'):
                continue
            if row['stance']=='contradiction':
                for rid in row['retract_ids']:
                    if rid in self.routes:
                        self.routes[rid]['revoked'].append({'source_ids':row['source_ids'], 'reason':row['reason']})
                continue
            identity = digest({k:row[k] for k in ('variable','value','source_ids','entity_scope','time_scope','parent_versions')})
            if 'binding_derivation' in row:
                identity=digest([identity,row['binding_derivation']])
            rid = 'fact_'+identity[:24]
            if rid not in self.routes:
                self.routes[rid] = {**deepcopy(row),'id':rid,'revoked':[], 'semantic_status':'model_judgment'}
        self.pending = set(pending)
        self._refresh_versions()
        self._check_rule_revision()
        self.history.append({'event':'fact_batch_integrated','rows':deepcopy(rows),
            'pending_variables':sorted(self.pending),'versions':deepcopy(self.versions),
            'program_revision':self.revision})

    def _check_rule_revision(self):
        binding = self.initial_program.get('rule_binding')
        if not binding:
            return
        values, _, ids = self.valid()
        variable = binding['variable']
        current_ids = ids.get(variable, [])
        if self.rule_route_ids and self.rule_route_ids != current_ids:
            self.program = deepcopy(self.initial_program)
            self.revision += 1
            self.rule_route_ids = []
        if variable not in values or self.rule_route_ids == current_ids:
            return
        rule = values[variable]
        if set(rule) != {'variables','expression','constraints','rules_complete'}:
            raise ProgramError('A sourced rule must include variables, expression, constraints and completeness')
        if any(d.get('demand') not in binding['condition_demands'] for d in rule['variables'].values()):
            raise ProgramError('Rule instantiation cannot introduce unplanned paid tasks')
        overlap = set(rule['variables']) & set(self.initial_program['variables'])
        if any(rule['variables'][v]!=self.initial_program['variables'][v] for v in overlap):
            raise ProgramError('Rule instantiation cannot change a predeclared variable identity/type/demand')
        candidate = {**deepcopy(rule), 'variables':{**deepcopy(self.initial_program['variables']),**deepcopy(rule['variables'])}}
        validate_program(candidate)
        validate_bindings(candidate, self.steps, self.final_node_id, contract=self.contract)
        mode = self.contract['mode']
        if mode in ('one_reason','all_failures') and candidate['expression']['op'] != ('one_reason' if mode=='one_reason' else 'failures'):
            raise ProgramError('Sourced rule cannot change the fixed output mode')
        self.program = candidate
        self.revision += 1
        self.rule_route_ids = list(current_ids)
        for v in rule['variables']:
            self.versions.setdefault(v,0)
        self.instantiations.append({'program_revision':self.revision,'rule_variable':variable,
            'rule_route_ids':list(current_ids),'rule_value_identity':digest(rule),'program':deepcopy(candidate)})

    def revoke_source(self, source_id, reason):
        for route in self.routes.values():
            if source_id in route['source_ids']:
                route['revoked'].append({'source_ids':[source_id],'reason':reason})
        self._refresh_versions()
        self._check_rule_revision()

    def public_dict(self):
        values, sources, ids = self.valid()
        return {'initial_program':self.initial_program,'program':self.program,'contract':self.contract,
            'program_revision':self.revision,'facts':values,'sources':sources,'route_ids':ids,
            'versions':self.versions,'routes':list(self.routes.values()),'pending_variables':sorted(self.pending),
            'rule_route_ids':self.rule_route_ids,'instantiations':self.instantiations,'history':self.history}

    def restore(self, snapshot):
        if snapshot['initial_program'] != self.initial_program or snapshot['contract'] != self.contract:
            raise ProgramError('Resume cannot change initial program or output contract')
        def validate_restored_program(value):
            program=validate_program(value)
            validate_bindings(program,self.steps,self.final_node_id,contract=self.contract)
            if any(program['variables'].get(v)!=declaration
                   for v,declaration in self.initial_program['variables'].items()):
                raise ProgramError('Resume cannot change a predeclared variable identity or scope')
            return program
        program=validate_restored_program(snapshot['program'])
        historical=[validate_restored_program(row['program']) for row in snapshot['instantiations']]
        if program!=self.initial_program and program not in historical:
            raise ProgramError('Resumed program does not match a recorded rule instantiation')
        for route in snapshot['routes']:
            # Sourced rule revisions retain obsolete routes for provenance.
            # They may match a historical dynamic declaration, but valid()
            # admits only routes with the currently declared scope.
            declarations=[p['variables'][route['variable']] for p in [program,*historical]
                          if route.get('variable') in p['variables']]
            if not any(all(route.get(k)==v for k,v in variable_scope(d,self.contract).items())
                       for d in declarations):
                raise ProgramError('Resumed observation scope differs from its declared variable scope')
        self.program=deepcopy(program)
        self.routes={r['id']:deepcopy(r) for r in snapshot['routes']}
        self.versions=deepcopy(snapshot['versions'])
        self.revision=snapshot['program_revision']
        self.rule_route_ids=list(snapshot['rule_route_ids'])
        self.instantiations=deepcopy(snapshot['instantiations'])
        self.history=deepcopy(snapshot['history'])
        self.pending=set(snapshot['pending_variables'])
        values,sources,route_ids=self.valid()
        self._identities={v:digest([values[v],sources[v],route_ids[v]]) if v in values else None for v in self.program['variables']}
