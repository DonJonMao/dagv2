"""Versioned typed observations; provenance routes are OR alternatives."""
from copy import copy, deepcopy
from .residual import validate_value, validate_program, variable_scope, ProgramError
from .residual_plan import validate_bindings
from .transport import digest


class SummaryRevisionSourcesMissing(ProgramError):
    """An otherwise valid revision omitted specific inherited source identities."""
    def __init__(self, required, provided):
        super().__init__('Summary revision must retain every original premise and relevant counter source')
        self.required_source_ids=tuple(sorted(required))
        self.missing_source_ids=tuple(sorted(set(required)-set(provided)))


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

    @staticmethod
    def support_base_identity(row):
        """A proof's source set has no model-selected citation order."""
        proof={k:row[k] for k in ('variable','value','entity_scope','time_scope','parent_versions')}
        proof['source_ids']=sorted(set(row['source_ids']))
        identity=digest(proof)
        if 'binding_derivation' in row:
            identity=digest([identity,row['binding_derivation']])
        return identity

    @staticmethod
    def support_identity(row):
        identity=FactState.support_base_identity(row)
        if 'summary_revision' in row:
            revision=row['summary_revision']
            identity=digest([identity,{'supersedes_route_ids':sorted(revision['supersedes_route_ids']),
                                       'previous_version':revision['previous_version']}])
        return identity

    def summary_ancestors(self, route):
        """Representation ancestry is separate from factual counterevidence."""
        ancestors=set()
        def visit(current, path):
            for rid in current.get('summary_revision',{}).get('supersedes_route_ids',[]):
                if rid not in self.routes or rid in path:
                    raise ProgramError('Summary revision ancestry is missing or cyclic')
                ancestors.add(rid)
                visit(self.routes[rid],path|{rid})
        visit(route,{route['id']})
        return ancestors

    def is_active_route(self, route):
        if route['revoked'] or route.get('superseded_by'):
            return False
        # A compatible revision inherits the old evidence. A later real counter
        # to that inherited content cannot be laundered through a new summary.
        return not any(self.routes[rid]['revoked'] for rid in self.summary_ancestors(route))

    def summary_revision_sources(self, variable, targets, *, historical=False):
        required={s for rid in targets for s in self.routes[rid]['source_ids']}
        if not historical:
            _,sources,_=self.valid()
            for rid in targets:
                required.update(s for p in self.routes[rid]['parent_versions'] for s in sources.get(p,[]))
            # Counter records remain independent evidence, even on retired
            # routes. A new compatible account must explicitly retain them.
            required.update(s for route in self.routes.values() if route['variable']==variable
                            for counter in route['revoked'] for s in counter['source_ids'])
        return required

    def validate_summary_revision(self, row, *, historical=False):
        revision=row.get('summary_revision')
        if 'summary_revision' not in row:
            return
        variable=row['variable']
        declaration=self.program['variables'][variable]
        if (row['stance']!='support' or row['retract_ids']
                or declaration.get('evidence_semantics')!='personal_history_summary'
                or declaration['type']!='text' or 'domain' in declaration
                or declaration['demand']==self.final_node_id
                or variable in self.program.get('task_expressions',{})):
            raise ProgramError('Summary revision requires an explicitly declared personal historical text summary')
        if (not isinstance(revision,dict)
                or set(revision)!={'supersedes_route_ids','previous_version','reason'}
                or type(revision['previous_version']) is not int or revision['previous_version']<0
                or not isinstance(revision['reason'],str) or not revision['reason'].strip()):
            raise ProgramError('Summary revision fields/version/reason malformed')
        targets=revision['supersedes_route_ids']
        if (not isinstance(targets,list) or not targets
                or any(not isinstance(rid,str) for rid in targets) or len(set(targets))!=len(targets)):
            raise ProgramError('Summary revision needs unique current route IDs')
        if not historical and revision['previous_version']!=self.versions[variable]:
            raise ProgramError('Summary revision previous_version is stale')
        values=self.valid()[0] if not historical else {}
        for rid in targets:
            old=self.routes.get(rid)
            if (old is None or old['variable']!=variable
                    or any(old[k]!=row[k] for k in ('entity_scope','time_scope','parent_versions'))):
                raise ProgramError('Summary revision target must retain the same variable, scope and actual parents')
            if not historical and (not self.is_active_route(old) or any(
                    p not in values or self.versions[p]!=version for p,version in old['parent_versions'].items())):
                raise ProgramError('Summary revision cannot supersede revoked, retired or stale routes')
        required=self.summary_revision_sources(variable,targets,historical=historical)
        if not required<=set(row['source_ids']):
            raise SummaryRevisionSourcesMissing(required,row['source_ids'])

    def summary_revision_is_noop(self, row):
        """Classify only after the full revision/provenance validation succeeds.

        An unchanged explicit review is not a new ordinary support: its current
        routes may inherit earlier revisions and counters. Keep those exact
        routes and their ancestry instead of rebuilding a source-equivalent one.
        """
        revision=row.get('summary_revision')
        return bool(revision) and all(digest(self.routes[rid]['value'])==digest(row['value'])
            and set(self.routes[rid]['source_ids'])==set(row['source_ids'])
            for rid in revision['supersedes_route_ids'])

    def valid(self):
        # Resolve parent routes in DAG order, never accept unsupported guesses.
        values, sources, route_ids = {}, {}, {}
        for task in self.steps:
            for variable, declaration in self.program['variables'].items():
                if declaration['demand'] != task['output_slot']:
                    continue
                routes = [r for r in self.routes.values() if r['variable']==variable and self.is_active_route(r)
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
        existing={self.support_identity(route):rid for rid,route in self.routes.items()}
        countered={rid for row in rows if row['stance']=='contradiction' for rid in row['retract_ids']}
        for row in rows:
            if row['stance'] in ('support','contradiction') and any(
                    row.get(key) != value for key,value in self.scope(row['variable']).items()):
                raise ProgramError('Observation scope does not match its declared variable scope')
            binding = self.initial_program.get('rule_binding')
            if (row['stance'] == 'support' and binding is not None
                    and row['variable'] == binding['variable']):
                # Reject impossible rule programs before adopting any routes
                # or contradictions from this batch.
                self._validated_rule_program(row['value'])
            if 'summary_revision' in row:
                self.validate_summary_revision(row)
                targets=row['summary_revision']['supersedes_route_ids']
                inherited=set(targets)|{rid for target in targets for rid in self.summary_ancestors(self.routes[target])}
                if inherited & countered:
                    raise ProgramError('Summary revision cannot conceal a counter to inherited evidence in the same batch')
        noops={self.support_identity(row) for row in rows
               if 'summary_revision' in row and self.summary_revision_is_noop(row)}
        # Replaying an exact revised proof without revision metadata must not
        # create an independent OR route that loses its inherited counters.
        # Include retired/revoked history and this validated batch so neither
        # later replay nor within-batch order can resurrect the same proof.
        revised_proofs={self.support_base_identity(row)
                        for row in [*self.routes.values(), *rows] if 'summary_revision' in row}
        revisions=[]
        for row in rows:
            if row['stance'] in ('unknown','partial','irrelevant'):
                continue
            if row['stance']=='contradiction':
                for rid in row['retract_ids']:
                    if rid in self.routes:
                        self.routes[rid]['revoked'].append({'source_ids':row['source_ids'], 'reason':row['reason']})
                continue
            identity = self.support_identity(row)
            if identity in noops:
                # Full validation and same-batch counter checks already passed.
                # Preserve every existing route/version/ancestry; only explicit
                # obligation handling outside FactState may count this review as
                # progress. Never strip summary_revision and reinsert the row.
                continue
            if 'summary_revision' not in row and self.support_base_identity(row) in revised_proofs:
                continue
            # Older checkpoints retain their original route IDs. Replaying
            # their source set in another order must neither create another
            # route nor resurrect a route that a sourced counter revoked.
            rid = existing.get(identity,'fact_'+identity[:24])
            if rid not in self.routes:
                self.routes[rid] = {**deepcopy(row),'source_ids':sorted(set(row['source_ids'])),
                    'id':rid,'revoked':[], 'semantic_status':'model_judgment'}
                existing[identity]=rid
            if 'summary_revision' in row:
                revisions.append((rid,row['summary_revision']))
        # All candidates were validated against the unchanged state. Retiring a
        # representation and installing its replacement is one atomic batch.
        for replacement,revision in revisions:
            for previous in revision['supersedes_route_ids']:
                retired=self.routes[previous].setdefault('superseded_by',[])
                if replacement not in retired:
                    retired.append(replacement)
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
        candidate = self._validated_rule_program(rule)
        self.program = candidate
        self.revision += 1
        self.rule_route_ids = list(current_ids)
        for v in rule['variables']:
            self.versions.setdefault(v,0)
        self.instantiations.append({'program_revision':self.revision,'rule_variable':variable,
            'rule_route_ids':list(current_ids),'rule_value_identity':digest(rule),'program':deepcopy(candidate)})

    def _validated_rule_program(self, rule):
        binding = self.initial_program.get('rule_binding')
        if binding is None:
            raise ProgramError('Sourced rule instantiation requires an initial rule binding')
        if not isinstance(rule, dict) or set(rule) != {'variables','expression','constraints','rules_complete'}:
            raise ProgramError('A sourced rule must include variables, expression, constraints and completeness')
        if not isinstance(rule['variables'], dict) or any(not isinstance(d, dict) for d in rule['variables'].values()):
            raise ProgramError('Sourced rule variables must be declarations')
        if any(d.get('demand') not in binding['condition_demands'] for d in rule['variables'].values()):
            raise ProgramError('Rule instantiation cannot introduce unplanned paid tasks')
        overlap = set(rule['variables']) & set(self.initial_program['variables'])
        if any(rule['variables'][v]!=self.initial_program['variables'][v] for v in overlap):
            raise ProgramError('Rule instantiation cannot change a predeclared variable identity/type/demand')
        candidate = {**deepcopy(rule), 'variables':{**deepcopy(self.initial_program['variables']),**deepcopy(rule['variables'])}}
        validate_bindings(candidate, self.steps, self.final_node_id, contract=self.contract,
                          rule_completion_available=True)
        return candidate

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
            if program != self.initial_program and program.get('rule_binding') is not None:
                raise ProgramError('Resume cannot introduce or revise an unbound rule placeholder')
            validate_bindings(program,self.steps,self.final_node_id,contract=self.contract,
                              rule_completion_available=bool(self.initial_program.get('rule_binding')))
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
            matching = [d for d in declarations if all(
                route.get(k)==v for k,v in variable_scope(d,self.contract).items())]
            if not matching:
                raise ProgramError('Resumed observation scope differs from its declared variable scope')
            valid_value = False
            for declaration in matching:
                try:
                    validated = validate_value(route['value'], declaration)
                except ProgramError:
                    continue
                valid_value |= digest(validated) == digest(route['value'])
            if not valid_value:
                raise ProgramError('Resumed observation value differs from its declared type or legal domain')
            binding = self.initial_program.get('rule_binding')
            if binding and route.get('variable') == binding['variable']:
                self._validated_rule_program(route['value'])
        probe=copy(self)
        probe.program=deepcopy(program)
        probe.routes={r['id']:deepcopy(r) for r in snapshot['routes']}
        probe.versions=deepcopy(snapshot['versions'])
        for route in probe.routes.values():
            replacements=route.get('superseded_by',[])
            if (not isinstance(replacements,list) or any(not isinstance(rid,str) for rid in replacements)
                    or len(set(replacements))!=len(replacements)):
                raise ProgramError('Resumed summary replacement references malformed')
            for rid in replacements:
                replacement=probe.routes.get(rid,{})
                if route['id'] not in replacement.get('summary_revision',{}).get('supersedes_route_ids',[]):
                    raise ProgramError('Resumed summary replacement has no matching revision provenance')
            if 'summary_revision' in route:
                probe.validate_summary_revision(route,historical=True)
                if route['summary_revision']['previous_version']>probe.versions[route['variable']]:
                    raise ProgramError('Resumed summary revision version is from the future')
                probe.summary_ancestors(route)
                for rid in route['summary_revision']['supersedes_route_ids']:
                    if route['id'] not in probe.routes[rid].get('superseded_by',[]):
                        raise ProgramError('Resumed summary revision did not retire its prior binding')
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
