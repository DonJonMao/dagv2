"""Pure, conservative evaluation of a small typed task language.

Certificates are conditional on validated semantic facts. This module neither
interprets documents nor equates retrieval absence with a negative observation.
Finite enumeration is all-or-nothing; abstract values are safe outer envelopes.
"""
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import math
import json

from .transport import digest


class ProgramError(ValueError):
    pass


class _Unknown:
    def __repr__(self):
        return 'UNKNOWN'


UNKNOWN = _Unknown()
OTHER_ENTITY = {'other_entity_class': True}
UNITS = {'ms': .001, 's': 1., 'seconds': 1., 'min': 60., 'minutes': 60.,
         'h': 3600., 'hours': 3600., 'm': 1., 'cm': .01, 'km': 1000.}
DIMENSIONS = {'ms': 'time', 's': 'time', 'seconds': 'time', 'min': 'time',
              'minutes': 'time', 'h': 'time', 'hours': 'time',
              'm': 'length', 'cm': 'length', 'km': 'length'}
COMPARISONS = {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    lower_closed: bool = True
    upper_closed: bool = True
    dimension: str = 'scalar'

    @property
    def empty(self):
        return self.lower > self.upper or (self.lower == self.upper and
                not (self.lower_closed and self.upper_closed))

    @property
    def singleton(self):
        return self.lower == self.upper and not self.empty


def quantity(value, unit=None):
    if isinstance(value, dict) and set(value) == {'number', 'unit'}:
        unit, value = value['unit'], value['number']
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ProgramError('Quantity requires a finite number, not a boolean')
    if unit is None:
        return Interval(value, value)
    if unit not in UNITS:
        raise ProgramError('Unknown unit: ' + str(unit))
    number = value * UNITS[unit]
    return Interval(number, number, dimension=DIMENSIONS[unit])


def interval(value, unit=None):
    if not isinstance(value, dict) or not {'lower', 'upper'} <= set(value):
        return quantity(value, unit)
    if set(value) - {'lower', 'upper', 'lower_closed', 'upper_closed', 'unit'}:
        raise ProgramError('Unknown interval fields')
    unit = value.get('unit', unit)
    if unit is not None and unit not in UNITS:
        raise ProgramError('Unknown interval unit')
    endpoints = []
    for key, default in (('lower', -math.inf), ('upper', math.inf)):
        item = value[key]
        if item is not None and (type(item) not in (int, float) or not math.isfinite(item)):
            raise ProgramError('Interval endpoint must be finite or null')
        endpoints.append(default if item is None else item * UNITS.get(unit, 1.))
    closed = [value.get(k, True) for k in ('lower_closed', 'upper_closed')]
    if any(type(v) is not bool for v in closed):
        raise ProgramError('Interval endpoint closure must be boolean')
    return Interval(*endpoints, *closed, DIMENSIONS.get(unit, 'scalar'))


def validate_value(value, declaration):
    try:
        json.dumps(value,allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise ProgramError('Fact values must be finite JSON') from exc
    kind = declaration['type']
    valid = (type(value) is bool if kind == 'bool' else
             isinstance(value, str) and bool(value.strip()) if kind in ('entity', 'text') else
             isinstance(value, list) if kind == 'set' else
             isinstance(value, dict) if kind in ('record', 'rule') else
             kind == 'number')
    if not valid or value == OTHER_ENTITY:
        raise ProgramError('Fact value does not match its declared type: ' + kind)
    if kind == 'number' and interval(value, declaration.get('unit')).empty:
        raise ProgramError('Supported numeric fact has an empty interval')
    domain = declaration.get('domain')
    if isinstance(domain, list) and not declaration.get('open', False):
        if digest(value) not in {digest(v) for v in domain}:
            raise ProgramError('Fact outside its finite declared domain')
    return deepcopy(value)


def references(expression):
    op = expression['op']
    if op == 'var':
        return {expression['id']}
    children = expression.get('args', [])
    if op in ('record', 'failures', 'one_reason'):
        children = expression['fields'].values()
    return set().union(*(references(c) for c in children)) if children else set()


def validate_program(program):
    if not isinstance(program, dict) or not {'variables', 'expression'} <= set(program):
        raise ProgramError('Program requires variables and expression')
    if set(program) - {'variables', 'expression', 'constraints', 'rules_complete', 'rule_binding', 'task_expressions'}:
        raise ProgramError('Unknown program field')
    variables = program['variables']
    if not isinstance(variables, dict) or any(not isinstance(k, str) or not k for k in variables):
        raise ProgramError('Variables must have stable string identities')
    for declaration in variables.values():
        if not isinstance(declaration, dict) or declaration.get('type') not in {
                'bool', 'entity', 'number', 'text', 'set', 'record', 'rule'}:
            raise ProgramError('Unsupported variable type')
        if 'domain' in declaration:
            domain = declaration['domain']
            if not isinstance(domain, (dict, list)):
                raise ProgramError('Domain must be finite values or an interval')
            if isinstance(domain, dict):
                if declaration['type'] != 'number':
                    raise ProgramError('Only numeric variables have interval domains')
                interval(domain, declaration.get('unit'))
            else:
                for value in domain:
                    validate_value(value, {k: v for k, v in declaration.items() if k != 'domain'})
                if len({digest(v) for v in domain}) != len(domain):
                    raise ProgramError('Duplicate finite domain value')
        if declaration.get('unit') is not None and declaration['unit'] not in UNITS:
            raise ProgramError('Unknown variable unit')
        if 'open' in declaration and type(declaration['open']) is not bool:
            raise ProgramError('open must be boolean')

    def check(expr):
        if not isinstance(expr, dict):
            raise ProgramError('Expression must be a JSON object')
        op = expr.get('op')
        if op == 'const' and set(expr) == {'op', 'value'}:
            try:
                json.dumps(expr['value'],allow_nan=False)
            except (TypeError,ValueError) as exc:
                raise ProgramError('Constants must be finite JSON') from exc
            return ('bool' if type(expr['value']) is bool else 'number'
                    if type(expr['value']) in (int, float) or isinstance(expr['value'], dict)
                    and set(expr['value']) == {'number', 'unit'} else 'entity'
                    if isinstance(expr['value'], str) else 'set' if isinstance(expr['value'], list) else 'record')
        if op == 'var' and set(expr) == {'op', 'id'} and expr['id'] in variables:
            return variables[expr['id']]['type']
        if op in ('record', 'failures', 'one_reason') and set(expr) == {'op', 'fields'}:
            if not isinstance(expr['fields'], dict) or not expr['fields']:
                raise ProgramError('Output fields must be a nonempty record')
            kinds = [check(v) for v in expr['fields'].values()]
            if op != 'record' and any(k != 'bool' for k in kinds):
                raise ProgramError('Failure outputs require boolean conditions')
            return 'set' if op == 'failures' else 'record'
        if set(expr) != {'op', 'args'} or not isinstance(expr.get('args'), list):
            raise ProgramError('Unsupported expression or fields: ' + str(op))
        args = expr['args']
        kinds = [check(a) for a in args]
        if op in ('and', 'or') and args and all(k == 'bool' for k in kinds):
            return 'bool'
        if op == 'not' and kinds == ['bool']:
            return 'bool'
        if op in COMPARISONS and len(args) == 2 and kinds[0] == kinds[1]:
            if op not in ('eq', 'ne') and kinds[0] != 'number':
                raise ProgramError('Ordered comparison requires numeric operands')
            return 'bool'
        if op == 'if' and len(args) == 3 and kinds[0] == 'bool' and kinds[1] == kinds[2]:
            return kinds[1]
        if op == 'tuple':
            return 'record'
        raise ProgramError('Invalid expression arity or operand types: ' + str(op))

    check(program['expression'])
    definitions=program.get('task_expressions',{})
    if not isinstance(definitions,dict):
        raise ProgramError('Task expressions must map declared variable IDs to expressions')
    for variable,expression in definitions.items():
        if variable not in variables or check(expression)!=variables[variable]['type']:
            raise ProgramError('Deterministic task expression type differs from its binding')
    for constraint in program.get('constraints', []):
        if check(constraint) != 'bool':
            raise ProgramError('Joint constraints must be boolean')
    if type(program.get('rules_complete', True)) is not bool:
        raise ProgramError('Rule completeness must be explicit boolean')
    return deepcopy(program)


def _compare(op, left, right):
    if left is UNKNOWN or right is UNKNOWN or left == OTHER_ENTITY or right == OTHER_ENTITY:
        return UNKNOWN
    numeric = (isinstance(left, Interval) or isinstance(right, Interval)
               or type(left) in (int, float) or type(right) in (int, float))
    if numeric:
        left = left if isinstance(left, Interval) else interval(left)
        right = right if isinstance(right, Interval) else interval(right)
        if left.dimension != right.dimension:
            raise ProgramError('Comparison mixes incompatible units')
        if op in ('gt', 'ge'):
            return _compare('lt' if op == 'gt' else 'le', right, left)
        strict_before = left.upper < right.lower or (left.upper == right.lower and
                        not (left.upper_closed and right.lower_closed))
        strict_after = left.lower > right.upper or (left.lower == right.upper and
                       not (left.lower_closed and right.upper_closed))
        if op == 'lt':
            return True if strict_before else False if left.lower >= right.upper else UNKNOWN
        if op == 'le':
            return True if left.upper <= right.lower else False if strict_after else UNKNOWN
        equal = (left.lower == right.lower if left.singleton and right.singleton else
                 False if strict_before or strict_after else UNKNOWN)
    else:
        equal = type(left) is type(right) and left == right
    return equal if op == 'eq' or equal is UNKNOWN else not equal


def evaluate(expression, world, variables):
    op = expression['op']
    if op == 'const':
        value = expression['value']
        return interval(value) if isinstance(value, dict) and (
            set(value) == {'number', 'unit'} or {'lower', 'upper'} <= set(value)) else value
    if op == 'var':
        value = world.get(expression['id'], UNKNOWN)
        if value is not UNKNOWN and variables[expression['id']]['type'] == 'number' and not isinstance(value, Interval):
            return interval(value, variables[expression['id']].get('unit'))
        return value
    if op in ('record', 'failures', 'one_reason'):
        values = {k: evaluate(v, world, variables) for k, v in expression['fields'].items()}
        if op == 'one_reason':
            reason = next((k for k, v in values.items() if v is False), None)
            if reason is not None:
                return {'decision': False, 'reason': reason}
            return UNKNOWN if any(v is UNKNOWN for v in values.values()) else {'decision': True, 'reason': None}
        if any(v is UNKNOWN for v in values.values()):
            return UNKNOWN
        return [k for k, v in values.items() if v is False] if op == 'failures' else values
    args = expression['args']
    if op in COMPARISONS and args[0] == args[1]:
        return op in ('eq', 'le', 'ge')
    values = [evaluate(a, world, variables) for a in args]
    if op == 'and':
        return False if any(v is False for v in values) else UNKNOWN if any(v is UNKNOWN for v in values) else True
    if op == 'or':
        return True if any(v is True for v in values) else UNKNOWN if any(v is UNKNOWN for v in values) else False
    if op == 'not':
        return UNKNOWN if values[0] is UNKNOWN else not values[0]
    if op == 'if':
        return values[1] if values[0] is True else values[2] if values[0] is False else (
            values[1] if values[1] is not UNKNOWN and values[1] == values[2] else UNKNOWN)
    if op in COMPARISONS:
        return _compare(op, *values)
    if op == 'tuple':
        return UNKNOWN if any(v is UNKNOWN for v in values) else values
    raise ProgramError('Unsupported operator')


def _public(value):
    if value is UNKNOWN or value == OTHER_ENTITY:
        return {'unknown': True}
    if isinstance(value, Interval):
        if value.singleton:
            return value.lower if value.dimension == 'scalar' else {'number': value.lower,
                'unit': 's' if value.dimension == 'time' else 'm'}
        return {'interval': {'lower': value.lower if math.isfinite(value.lower) else None,
            'upper': value.upper if math.isfinite(value.upper) else None,
            'lower_closed': value.lower_closed, 'upper_closed': value.upper_closed,
            'dimension': value.dimension}}
    if isinstance(value, dict):
        return {k: _public(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_public(v) for v in value]
    return value


def _exact(value):
    if value is UNKNOWN or value == OTHER_ENTITY:
        return False
    if isinstance(value, Interval):
        return value.singleton
    if isinstance(value, dict):
        return all(_exact(v) for v in value.values())
    if isinstance(value, list):
        return all(_exact(v) for v in value)
    return True


def reduce(program, valid_facts):
    """Substitute facts, then apply equivalences with actual proof premises."""
    program = validate_program(program)
    facts = {k: validate_value(v, program['variables'][k]) for k, v in valid_facts.items()}
    proofs = []

    def visit(expr):
        op = expr['op']
        if op == 'var':
            key = expr['id']
            if key not in facts:
                return deepcopy(expr), set()
            value = deepcopy(facts[key])
            declaration = program['variables'][key]
            if declaration['type'] == 'number' and declaration.get('unit'):
                if type(value) in (int, float):
                    value = {'number': value, 'unit': declaration['unit']}
                elif isinstance(value, dict) and 'lower' in value:
                    value.setdefault('unit', declaration['unit'])
            return {'op': 'const', 'value': value}, {key}
        if op == 'const':
            return deepcopy(expr), set()
        field_mode = op in ('record', 'failures', 'one_reason')
        keys = list(expr['fields']) if field_mode else list(range(len(expr['args'])))
        children = expr['fields'].values() if field_mode else expr['args']
        visited = [visit(v) for v in children]
        values = [evaluate(v, {}, program['variables']) for v, _ in visited]
        decisive = (False if op == 'and' else True if op == 'or' else None)
        if decisive is not None and any(v is decisive for v in values):
            index = next(i for i, v in enumerate(values) if v is decisive)
            proofs.append({'rule': op + '_short_circuit', 'used_variables': sorted(visited[index][1])})
            return {'op': 'const', 'value': decisive}, visited[index][1]
        if op == 'one_reason' and any(v is False for v in values):
            index = next(i for i, v in enumerate(values) if v is False)
            proofs.append({'rule': 'necessary_condition_false_witness', 'field': keys[index],
                           'used_variables': sorted(visited[index][1])})
            return {'op': 'const', 'value': {'decision': False, 'reason': keys[index]}}, visited[index][1]
        if op == 'if' and type(values[0]) is bool:
            selected = 1 if values[0] else 2
            proofs.append({'rule': 'if_bound_condition', 'selected_branch': selected})
            return visited[selected][0], visited[0][1] | visited[selected][1]
        reduced = {'op': op, 'fields': {k: v[0] for k, v in zip(keys, visited)}} if field_mode else {
            'op': op, 'args': [v[0] for v in visited]}
        used = set().union(*(v[1] for v in visited))
        value = evaluate(reduced, {}, program['variables'])
        if _exact(value):
            proofs.append({'rule': 'constant_' + op, 'used_variables': sorted(used)})
            return {'op': 'const', 'value': _public(value)}, used
        return reduced, used

    expression, used = visit(program['expression'])
    return {'expression': expression, 'used_variables': sorted(used), 'steps': proofs,
            'remaining_variables': sorted(references(expression))}


def _domain(declaration, value=UNKNOWN):
    if value is not UNKNOWN:
        return [interval(value, declaration.get('unit')) if declaration['type'] == 'number' else value]
    domain = declaration.get('domain')
    if isinstance(domain, list):
        values = deepcopy(domain)
    elif declaration['type'] == 'bool':
        values = [False, True]
    elif declaration['type'] == 'number':
        numeric = interval(domain, declaration.get('unit')) if isinstance(domain, dict) else Interval(
            -math.inf, math.inf, dimension=DIMENSIONS.get(declaration.get('unit'), 'scalar'))
        return [] if numeric.empty else [numeric]
    else:
        values = []
    if declaration.get('open', domain is None) and declaration['type'] in ('entity', 'text'):
        values.append(OTHER_ENTITY)
    elif domain is None and declaration['type'] not in ('bool', 'number'):
        values.append(UNKNOWN)
    return values


def possible_outputs(program, valid_facts, *, max_states=10000):
    program = validate_program(program)
    if type(max_states) is not int or max_states < 1:
        raise ProgramError('Enumeration limit must be positive')
    reduction = reduce(program, valid_facts)
    constraints = program.get('constraints', [])
    names = set(reduction['remaining_variables']) | set().union(*(references(c) for c in constraints))
    domains = {k: _domain(d, valid_facts.get(k, UNKNOWN)) for k, d in program['variables'].items()}
    if any(not d for d in domains.values()):
        return {'status': 'inconsistent', 'outputs': [], 'reduction': reduction, 'live_variables': [],
                'nonempty_proven': False, 'worlds_checked': 0}
    names = sorted(names)
    size = math.prod(len(domains[k]) for k in names)
    live = sorted(set(reduction['remaining_variables']) | set().union(*(references(c) for c in constraints)))
    if size > max_states:
        # No partial prefix of a product is ever treated as its full domain.
        value = evaluate(reduction['expression'], {}, program['variables'])
        exact = not constraints and _exact(value)
        return {'status': 'determined' if exact else 'indeterminate',
                'outputs': [_public(value)] if exact else [{'unknown': True}],
                'nonempty_proven': not constraints, 'worlds_checked': 0,
                'enumeration_skipped': size, 'reduction': reduction, 'live_variables': [] if exact else live}
    outputs, checked, admitted, uncertain, nonempty_proven = {}, 0, 0, False, False
    for values in product(*(domains[k] for k in names)):
        checked += 1
        world = {**valid_facts, **dict(zip(names, values))}
        guards = [evaluate(c, world, program['variables']) for c in constraints]
        if any(g is False for g in guards):
            continue
        admitted += 1
        # Unknown guards are admitted into the safe outer envelope. A separate
        # fully satisfied abstract world proves existence; uncertain extra
        # worlds do not invalidate a common exact output over the whole envelope.
        nonempty_proven |= all(g is True for g in guards)
        value = evaluate(reduction['expression'], world, program['variables'])
        uncertain |= not _exact(value)
        public = _public(value)
        outputs[digest(public)] = public
    determined = nonempty_proven and not uncertain and len(outputs) == 1
    return {'status': 'inconsistent' if not admitted else 'determined' if determined else 'indeterminate',
            'outputs': list(outputs.values()), 'nonempty_proven': nonempty_proven,
            'worlds_checked': checked, 'reduction': reduction, 'live_variables': [] if determined else live}


def live_demands(program, valid_facts, *, max_states=10000):
    outcome = possible_outputs(program, valid_facts, max_states=max_states)
    return list(dict.fromkeys(program['variables'][v].get('demand', v) for v in outcome['live_variables']))


def certify_output(program, valid_facts, contract, *, program_revision=0,
                   versions=None, sources=None, pending_variables=(), max_states=10000):
    validate_output_contract(contract)
    mode = contract['mode']
    if mode == 'semantic' or (mode in ('one_reason', 'all_failures') and
            program['expression']['op'] != ('one_reason' if mode == 'one_reason' else 'failures')):
        return None
    outcome = possible_outputs(program, valid_facts, max_states=max_states)
    used = set(outcome['reduction']['used_variables'])
    # Facts used by joint constraints also belong to the actual assumptions.
    used |= set(valid_facts) & set().union(*(references(c) for c in program.get('constraints', [])))
    if outcome['status'] != 'determined' or not outcome['nonempty_proven'] or used & set(pending_variables):
        return None
    result = outcome['outputs'][0]
    try:
        validate_value(result, {'type': contract['result_type']})
    except ProgramError:
        return None
    if contract.get('fields') and (not isinstance(result, dict) or set(result) != set(contract['fields'])):
        return None
    if not program.get('rules_complete', True):
        negative_witness = (program['expression']['op'] == 'one_reason' and result.get('decision') is False)
        negative_witness |= (program['expression']['op']=='and' and result is False and any(
            step['rule']=='and_short_circuit' for step in outcome['reduction']['steps']))
        if not negative_witness:
            return None
    eliminated = set(program['variables']) - set(outcome['live_variables']) - used
    return {'kind': 'certified_symbolic', 'contract_identity': digest(contract),
            'program_identity': digest(program), 'program_revision': program_revision,
            'output': result, 'fact_versions': {k: (versions or {}).get(k, 0) for k in sorted(used)},
            'fact_identities': {k: digest(valid_facts[k]) for k in sorted(used)},
            'source_ids': sorted({s for k in used for s in (sources or {}).get(k, [])}),
            'derivation': outcome['reduction']['steps'] + [{'rule': 'complete_conservative_world_envelope',
                'worlds_checked': outcome['worlds_checked'], 'constraints_identity': digest(program.get('constraints', [])),
                'outputs': outcome['outputs'], 'nonempty_proven': outcome['nonempty_proven']}],
            'eliminated_variables': sorted(eliminated),
            'conditional_on': 'validated semantic interpretation; not proof that all counterevidence was retrieved'}


def validate_output_contract(contract):
    required = {'mode', 'result_type', 'entity_scope', 'time_scope', 'explanation'}
    if not isinstance(contract, dict) or not required <= set(contract) or set(contract) - required - {'fields'}:
        raise ProgramError('Output contract requires mode, result type, scope and explanation')
    if contract['mode'] not in ('value', 'one_reason', 'all_failures', 'semantic'):
        raise ProgramError('Unknown output contract mode')
    if contract['result_type'] not in ('bool', 'entity', 'number', 'record', 'set', 'text'):
        raise ProgramError('Unknown output contract result type')
    if any(not isinstance(contract[k], str) for k in ('entity_scope', 'time_scope', 'explanation')):
        raise ProgramError('Output scopes and explanation must be explicit strings')
    if contract['explanation'] not in ('none', 'one', 'all', 'semantic'):
        raise ProgramError('Unknown explanation requirement')
    if contract['mode'] == 'one_reason' and contract['explanation'] != 'one':
        raise ProgramError('One-reason output must retain one explanation')
    if contract['mode'] == 'all_failures' and contract['explanation'] != 'all':
        raise ProgramError('All-failures output must retain completeness')
    if 'fields' in contract and (not isinstance(contract['fields'],list) or not contract['fields'] or
            any(not isinstance(f,str) or not f for f in contract['fields']) or len(set(contract['fields']))!=len(contract['fields'])):
        raise ProgramError('Contract fields must be unique nonempty string names')
    return deepcopy(contract)


def certificate_valid(certificate, program, valid_facts, contract, *, program_revision=0,
                      versions=None, sources=None, pending_variables=(), max_states=10000):
    current = certify_output(program, valid_facts, contract, program_revision=program_revision,
        versions=versions, sources=sources, pending_variables=pending_variables, max_states=max_states)
    return current is not None and current == certificate
