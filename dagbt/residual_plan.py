"""One initial DAG plan, with typed bindings and a fixed output contract."""
from copy import deepcopy
from . import local_terminal
from .residual import validate_program, validate_output_contract, ProgramError, references
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
optional domain, open, unit. Variables may share a task for bounded multi-field reading.
Expressions ONLY use {op:const,value:...}, {op:var,id:...}, {op:and/or/not/eq/ne/lt/le/gt/ge/if/tuple,args:[...]},
or {op:record/failures/one_reason,fields:{name:expression}}. No code.
Use symbolic value mode for ordinary sourced facts/entity chains and comparisons.
For genuinely open advice use semantic mode, with the terminal text variable as expression.
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


def planner_contract(legacy_schema):
    schema = local_terminal.planner_contract(legacy_schema)
    schema['properties'].update(output_contract={'type':'object'}, program={'type':'object'})
    schema['required'] += ['output_contract','program']
    return schema


def validate_plan(value, legacy_validate):
    if not isinstance(value, dict) or set(value) != {'steps','final_node_id','output_contract','program'}:
        raise ProtocolError('Typed plan requires full DAG, terminal, output_contract and program')
    plan = local_terminal.validate_plan({k:value[k] for k in ('steps','final_node_id')}, legacy_validate)
    try:
        contract = validate_output_contract(value['output_contract'])
        program = validate_program(value['program'])
        validate_bindings(program, plan['steps'], plan['final_node_id'])
    except ProgramError as exc:
        raise ProtocolError(str(exc)) from exc
    return {**plan, 'output_contract':contract, 'program':program}


def validate_bindings(program, steps, final_node_id):
    tasks = {s['output_slot']:s for s in steps}
    for declaration in program['variables'].values():
        if set(declaration) - {'type','demand','domain','open','unit'}:
            raise ProgramError('Unknown variable declaration metadata')
        if declaration.get('demand') not in tasks:
            raise ProgramError('Variable demand must belong to the preplanned DAG')
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
        if variable not in program['variables'] or program['variables'][variable]['type'] != 'rule':
            raise ProgramError('Rule binding variable must be declared as rule')
        demand = program['variables'][variable]['demand']
        conditions = binding['condition_demands']
        if not isinstance(conditions, list) or not conditions or len(set(conditions)) != len(conditions):
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
    return deepcopy(program)
