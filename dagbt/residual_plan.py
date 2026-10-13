"""One initial DAG plan, with typed bindings and a fixed output contract."""
from copy import deepcopy
from . import local_terminal
from .residual import (validate_program, validate_output_contract, variable_scope,
                       validate_scope, validate_program_contract, ProgramError, references, UNITS)
from .reasoning import ProtocolError

PLAN_SYSTEM = local_terminal.PLAN_SYSTEM.replace('Do not include extra fields.', '') + '''
Also return output_contract and program. Before ANY retrieval, plan every foreseeable
downstream task, its input dependencies and the designated terminal. Unknown entities,
rules and conditions are UNBOUND INPUTS, not grounds to omit downstream tasks.
F is the typed solving relation of this DAG; a reduced F_t never replaces the DAG.
output_contract has mode(value/one_reason/all_failures/semantic), result_type
(bool/entity/number/record/set/text), entity_scope, time_scope and explanation
(none/one/all/semantic), optional fields. Do not reduce the user's output requirements.
program has variables (ID -> declaration), expression, optional constraints,
rules_complete and rule_binding. Declaration has type, demand (a planned task ID),
optional domain, open, unit, scope. Variables may share a task for bounded multi-field reading.
Explicitly declare each variable's scope={entity_scope,time_scope} BEFORE retrieval,
using the meaning of that variable's own planned task, not the final answer's scope.
For example, a historical observation and a future recommendation have different time
scopes; compared entities may differ from the final comparison's entity scope.
The scope of every terminal-task variable MUST equal the immutable output_contract scope.
Supported parents retain their own scopes when used or audited by another task; never
relabel historical evidence as current/future evidence to satisfy the output scope.
Omitting scope is only compatibility for old plans and inherits output_contract scope.
Omit optional fields that do not apply; optional does not mean fill with null.
An unknown/unrestricted domain is represented by OMITTING domain, not null, a string,
a scalar, or an empty array. For an open entity/text variable use, for example,
{"type":"entity","demand":"place","open":true} with no domain field.
A known finite domain is an array of correctly typed possible values; only use
values explicitly constrained by the question, never guessed answers or candidates.
Only number variables may use an interval domain: {"lower":0,"upper":null}.
Interval endpoints are numbers or null (unbounded); lower_closed/upper_closed are
optional booleans. Null endpoints are legal; a null domain itself is not.
A numeric domain states legal possible values, NOT an acceptance threshold. Keep
thresholds in comparison expressions, never exclude a possible failing observation.
Unknown numeric bounds and memory-derived rules must remain unknown until sourced.
Expressions ONLY use {op:const,value:...}, {op:var,id:...}, {op:and/or/not/eq/ne/lt/le/gt/ge/if/tuple,args:[...]},
or {op:record/failures/one_reason,fields:{name:expression}}. No code.
Use symbolic value mode for ordinary sourced facts/entity chains and comparisons.
For genuinely open advice use semantic mode, with the terminal text variable as expression.
one_reason and all_failures are ONLY Boolean-witness operators over named conditions:
one_reason requires expression={op:one_reason,fields:{name:boolean_expression,...}}
and result_type=record; its result has exactly decision and reason. all_failures
requires expression={op:failures,fields:{name:boolean_expression,...}} and result_type=set;
its result lists all failed condition names. These modes do NOT mean an ordinary
natural-language explanation, a historical reason, or a list of personalized reasons.
Use value for sourced explanation text/lists, or semantic for an open synthesis.
The expression's declared result type must match output_contract.result_type,
including an initial rule-bound placeholder; tuple expressions produce arrays (set).
For symbolic modes without a rule_binding completion route, rules_complete=false
is allowed only for necessary-condition and/one_reason programs that can certify
a negative witness; it can never certify a positive or exhaustive result.
Semantic mode is assessed from supported evidence and never receives a symbolic
completeness certificate. Do not mark rules complete
unless their completeness is justified by the task or applicable original evidence.
rules_complete describes the completeness of the solving relation/decision rules,
NOT corpus completeness, retrieval coverage, confidence, or whether factual input
values are already known. For a direct sourced fact/entity lookup, or a fixed
comparison/composition whose full function is specified by the question, the
solving relation is complete even while all its factual inputs are still unbound:
set rules_complete=true (or omit it). Missing facts remain unknown variables and
require sources. Use false only when applicable decision rules themselves must
still be obtained from memory through rule_binding, or for the explicit permitted
negative-witness case. Never invent thresholds, source values, or completeness.
Rules from memory: do NOT invent thresholds. Plan a rule task and all foreseeable
condition-checking tasks with that rule as an unbound parent. Set rules_complete=false.
rule_binding={variable:rule_variable,condition_demands:[preplanned task IDs]} permits a
sourced rule reading to instantiate internal conditions within those tasks. Initially
use an unbound result variable in the expression. Keep the output contract fixed.
Every variable's demand must exist; parameter dependencies use the DAG inputs.
The terminal may be compose even when some parents will later be short-circuited.
Optional program.task_expressions maps intermediate compose variables to whitelist
expressions over declared earlier DAG parents. These are free deterministic bindings;
partial evaluation runs before demanding every planned parent. Do not use this to
invent sourced facts or create paid tasks. Unsupported compose remains semantic reading.
'''


def planner_contract(legacy_schema, *, personal_history=False):
    schema = local_terminal.planner_contract(legacy_schema)
    unit = {'enum': [None, *sorted(UNITS)]}
    interval_schema = {'type':'object', 'properties': {
        'lower': {'type':['number','null']}, 'upper': {'type':['number','null']},
        'lower_closed': {'type':'boolean'}, 'upper_closed': {'type':'boolean'}, 'unit': unit},
        'required':['lower','upper'], 'additionalProperties':False}
    declaration = {'type':'object', 'properties': {
        'type': {'enum':['bool','entity','number','text','set','record','rule']},
        'demand': {'type':'string','minLength':1},
        'domain': {'description':'Optional legal domain; omit if unrestricted. Arrays list finite values; interval objects apply only to number variables.',
                   'anyOf':[{'type':'array','items':{},'uniqueItems':True}, interval_schema]},
        'open': {'type':'boolean'}, 'unit': unit,
        'scope': {'type':'object', 'properties': {
            'entity_scope': {'type':'string','minLength':1},
            'time_scope': {'type':'string','minLength':1}},
            'required':['entity_scope','time_scope'], 'additionalProperties':False}},
        'required':['type','demand'], 'additionalProperties':False}
    if personal_history:
        declaration['properties']['evidence_semantics'] = {
            'enum':['personal_history_summary'],
            'description':'Only an unrestricted text summary of personal history before the terminal; never a scalar fact, rule, deterministic binding or final response.'}
    program = {'type':'object', 'properties': {
        'variables': {'type':'object','additionalProperties':declaration},
        'expression': {'type':'object'},
        'constraints': {'type':'array','items':{'type':'object'}},
        'rules_complete': {'type':'boolean'},
        'rule_binding': {'type':'object','properties': {
            'variable': {'type':'string'},
            'condition_demands': {'type':'array','items':{'type':'string'},'minItems':1,'uniqueItems':True}},
            'required':['variable','condition_demands'],'additionalProperties':False},
        'task_expressions': {'type':'object','additionalProperties':{'type':'object'}}},
        'required':['variables','expression'],'additionalProperties':False}
    contract = {'type':'object','properties': {
        'mode': {'enum':['value','one_reason','all_failures','semantic']},
        'result_type': {'enum':['bool','entity','number','record','set','text']},
        'entity_scope': {'type':'string'}, 'time_scope': {'type':'string'},
        'explanation': {'enum':['none','one','all','semantic']},
        'fields': {'type':'array','items':{'type':'string','minLength':1},'minItems':1,'uniqueItems':True}},
        'required':['mode','result_type','entity_scope','time_scope','explanation'],'additionalProperties':False}
    schema['properties'].update(output_contract=contract, program=program)
    schema['required'] += ['output_contract','program']
    return schema


def validate_plan(value, legacy_validate, *, personal_history=False):
    if not isinstance(value, dict) or set(value) != {'steps','final_node_id','output_contract','program'}:
        raise ProtocolError('Typed plan requires full DAG, terminal, output_contract and program')
    plan = local_terminal.validate_plan({k:value[k] for k in ('steps','final_node_id')}, legacy_validate)
    try:
        contract = validate_output_contract(value['output_contract'])
        program = validate_program(value['program'])
        if not personal_history and any('evidence_semantics' in d for d in program['variables'].values()):
            raise ProgramError('Personal history summary declarations require the explicit PersonaMem task')
        validate_bindings(program, plan['steps'], plan['final_node_id'], contract=contract)
    except ProgramError as exc:
        raise ProtocolError(str(exc)) from exc
    return {**plan, 'output_contract':contract, 'program':program}


def validate_bindings(program, steps, final_node_id, *, contract=None,
                      rule_completion_available=False):
    program = validate_program(program)
    tasks = {s['output_slot']:s for s in steps}
    for declaration in program['variables'].values():
        if set(declaration) - {'type','demand','domain','open','unit','scope','evidence_semantics'}:
            raise ProgramError('Unknown variable declaration metadata')
        if 'scope' in declaration:
            validate_scope(declaration['scope'])
        if not isinstance(declaration.get('demand'), str) or declaration['demand'] not in tasks:
            raise ProgramError('Variable demand must belong to the preplanned DAG')
        if ('evidence_semantics' in declaration and
                (declaration['demand'] == final_node_id or 'scope' not in declaration)):
            raise ProgramError('Personal history summary requires an explicit nonterminal evidence scope')
        if (contract is not None and declaration['demand'] == final_node_id
                and variable_scope(declaration, contract) != {
                    key: contract[key] for key in ('entity_scope','time_scope')}):
            raise ProgramError('Terminal variable scope must match the immutable output contract')
    for variable,expression in program.get('task_expressions',{}).items():
        task=tasks[program['variables'][variable]['demand']]
        if task['execution']!='compose' or any(program['variables'][v]['demand'] not in task['inputs']
                for v in references(expression)):
            raise ProgramError('Deterministic task must use only declared DAG parent variables')
    binding = program.get('rule_binding')
    if binding is not None:
        if not isinstance(binding, dict) or set(binding) != {'variable','condition_demands'}:
            raise ProgramError('Rule binding needs variable and preplanned condition_demands')
        variable = binding['variable']
        if (not isinstance(variable, str) or variable not in program['variables']
                or program['variables'][variable]['type'] != 'rule'):
            raise ProgramError('Rule binding variable must be declared as rule')
        demand = program['variables'][variable]['demand']
        conditions = binding['condition_demands']
        if (not isinstance(conditions, list) or not conditions
                or any(not isinstance(c, str) for c in conditions)
                or len(set(conditions)) != len(conditions)):
            raise ProgramError('Rule needs unique preplanned condition tasks')
        if any(c not in tasks or demand not in tasks[c]['inputs'] for c in conditions):
            raise ProgramError('Every condition task must declare its unbound rule input')
        if program.get('rules_complete', True):
            raise ProgramError('Unbound rules cannot be declared complete')
    # A symbolic result must reside in the terminal's dependency closure.
    def closure(n):
        return {n} | set().union(*(closure(p) for p in tasks[n]['inputs']))
    if any(program['variables'][v]['demand'] not in closure(final_node_id)
           for v in references(program['expression'])):
        raise ProgramError('Output expression is outside terminal DAG dependencies')
    if contract is not None:
        validate_program_contract(program, contract,
            allow_rule_placeholder=binding is not None,
            rule_completion_available=rule_completion_available or binding is not None)
    return deepcopy(program)
