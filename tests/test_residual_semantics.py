"""Independent finite-world oracle and boundary checks for residual evaluation."""
from itertools import product
import random

import pytest

from dagbt.residual import (ProgramError, UNKNOWN, certificate_valid, certify_output,
    evaluate, live_demands, possible_outputs, reduce, validate_program, validate_value)


def var(name):
    return {'op': 'var', 'id': name}


def const(value):
    return {'op': 'const', 'value': value}


def operation(op, *args):
    return {'op': op, 'args': list(args)}


def program(expression, **extra):
    return {'variables': {k: {'type': 'bool', 'demand': k} for k in ('A', 'D', 'O')},
            'expression': expression, **extra}


def contract(mode='value', result_type='bool'):
    return {'mode': mode, 'result_type': result_type, 'entity_scope': 'current example',
            'time_scope': 'current', 'explanation': {'one_reason': 'one', 'all_failures': 'all'}.get(mode, 'none')}


def oracle(expr, world):
    op = expr['op']
    if op == 'const':
        return expr['value']
    if op == 'var':
        return world[expr['id']]
    values = [oracle(a, world) for a in expr['args']]
    if op == 'and':
        return all(values)
    if op == 'or':
        return any(values)
    if op == 'not':
        return not values[0]
    if op == 'if':
        return values[1] if values[0] else values[2]
    if op == 'eq':
        return values[0] == values[1]
    raise AssertionError(op)


def test_unknown_not_false_and_supported_falsy_values():
    p = program(var('D'))
    assert set(possible_outputs(p, {})['outputs']) == {False, True}
    assert certify_output(p, {}, contract()) is None
    assert certify_output(p, {'D': False}, contract())['output'] is False
    for value, kind in [(0, 'number'), (False, 'bool'), ([], 'set')]:
        assert validate_value(value, {'type': kind}) == value
        p = {'variables': {'x': {'type': kind}}, 'expression': var('x')}
        assert certify_output(p, {'x': value}, contract(result_type=kind))['output'] == value


@pytest.mark.parametrize('op,facts,remaining', [
    ('and', {'D': False}, []), ('or', {'D': True}, []),
    ('and', {'D': True}, ['A', 'O']), ('or', {'D': False}, ['A', 'O'])])
def test_boolean_short_circuits_remove_only_unneeded_demands(op, facts, remaining):
    p = program(operation(op, var('A'), var('D'), var('O')))
    assert reduce(p, facts)['remaining_variables'] == remaining
    assert live_demands(p, facts) == remaining
    if not remaining:
        c = certify_output(p, facts, contract(), versions={'D': 2}, sources={'D': ['s4']})
        assert c['fact_versions'] == {'D': 2}
        assert c['source_ids'] == ['s4']
        assert c['eliminated_variables'] == ['A', 'O']


def test_IF_keeps_condition_provenance_and_only_selected_branch():
    p = program(operation('if', var('D'), var('A'), var('O')))
    r = reduce(p, {'D': False})
    assert r['remaining_variables'] == ['O'] and r['used_variables'] == ['D']
    assert certify_output(p, {'D': False, 'O': True}, contract())['fact_versions'] == {'D': 0, 'O': 0}


def test_one_decisive_reason_differs_from_all_failures():
    fields = {k: var(k) for k in ('A', 'D', 'O')}
    one = program({'op': 'one_reason', 'fields': fields})
    all_ = program({'op': 'failures', 'fields': fields})
    c = certify_output(one, {'D': False}, contract('one_reason', 'record'))
    assert c['output'] == {'decision': False, 'reason': 'D'}
    assert c['fact_versions'] == {'D': 0}
    assert certify_output(all_, {'D': False}, contract('all_failures', 'set')) is None
    assert set(live_demands(all_, {'D': False})) == {'A', 'O'}
    assert certify_output(all_, {'A': True, 'D': False, 'O': False}, contract('all_failures', 'set'))['output'] == ['D', 'O']
    # A planner cannot shrink the fixed output contract by changing its operator.
    assert certify_output(one, {'D': False}, contract('all_failures', 'set')) is None


def test_incomplete_rule_never_certifies_positive_or_exhaustive_output():
    fields = {k: var(k) for k in ('A', 'D', 'O')}
    p = program({'op': 'one_reason', 'fields': fields}, rules_complete=False)
    assert certify_output(p, {'A': True, 'D': True, 'O': True}, contract('one_reason', 'record')) is None
    assert certify_output(p, {'D': False}, contract('one_reason', 'record')) is not None
    p['expression']['op'] = 'failures'
    assert certify_output(p, {'A': False, 'D': False, 'O': False}, contract('all_failures', 'set')) is None


def test_empty_worlds_are_inconsistent_not_vacuously_certified():
    p = program(operation('and', var('A'), operation('not', var('A'))),
                constraints=[operation('and', var('D'), operation('not', var('D')))])
    assert possible_outputs(p, {})['status'] == 'inconsistent'
    assert certify_output(p, {}, contract()) is None
    p = program(const(True))
    p['variables']['A']['domain'] = []
    assert possible_outputs(p, {})['status'] == 'inconsistent'


def test_limit_does_not_certify_partial_enumeration():
    p = program(var('A'), constraints=[operation('eq', var('A'), var('D'))])
    result = possible_outputs(p, {}, max_states=1)
    assert result['status'] == 'indeterminate' and result['worlds_checked'] == 0
    assert certify_output(p, {}, contract(), max_states=1) is None


def test_shared_variables_and_joint_constraints_preserve_correlation():
    p = program(operation('and', var('A'), operation('not', var('A'))))
    assert certify_output(p, {}, contract())['output'] is False
    p = program(operation('eq', var('A'), var('D')),
                constraints=[operation('eq', var('A'), var('D'))])
    assert certify_output(p, {}, contract())['output'] is True
    p['expression'] = var('A')
    assert set(possible_outputs(p, {})['outputs']) == {False, True}
    assert set(live_demands(p, {})) == {'A', 'D'}


def test_open_entity_domain_keeps_other_possibilities():
    p = {'variables': {'city': {'type': 'entity', 'domain': ['River City'], 'open': True}},
         'expression': var('city')}
    assert possible_outputs(p, {})['status'] == 'indeterminate'
    assert certify_output(p, {}, contract(result_type='entity')) is None
    assert certify_output(p, {'city': 'Other City'}, contract(result_type='entity'))['output'] == 'Other City'


@pytest.mark.parametrize('op,upper_closed,expected', [('lt', False, True), ('lt', True, None),
    ('le', False, True), ('le', True, True), ('gt', True, False), ('ge', False, False)])
def test_units_and_strict_interval_boundaries(op, upper_closed, expected):
    p = {'variables': {'delay': {'type': 'number', 'unit': 's',
            'domain': {'lower': 0, 'upper': 60, 'upper_closed': upper_closed}}},
         'expression': operation(op, var('delay'), const({'number': 1, 'unit': 'min'}))}
    c = certify_output(p, {}, contract())
    assert (None if c is None else c['output']) is expected
    if op == 'lt':
        assert certify_output(p, {'delay': 30}, contract())['output'] is True


def test_incompatible_units_are_rejected_and_zero_is_present():
    p = {'variables': {'time': {'type': 'number', 'unit': 's'}},
         'expression': operation('lt', var('time'), const({'number': 1, 'unit': 'm'}))}
    with pytest.raises(ProgramError, match='incompatible'):
        possible_outputs(p, {'time': 0})
    p['expression'] = operation('le', var('time'), const({'number': 0, 'unit': 'ms'}))
    assert certify_output(p, {'time': 0}, contract())['output'] is True


def test_certificate_binds_sources_versions_contract_and_program_revision():
    p = program(operation('and', var('A'), var('D'), var('O')))
    kw = {'versions': {'D': 2}, 'sources': {'D': ['s4']}, 'program_revision': 3}
    c = certify_output(p, {'D': False}, contract(), **kw)
    assert certificate_valid(c, p, {'D': False}, contract(), **kw)
    assert not certificate_valid(c, p, {}, contract(), **kw)
    assert live_demands(p, {}) == ['A', 'D', 'O']
    for change in ({'versions': {'D': 3}}, {'sources': {'D': ['s8']}}, {'program_revision': 4},
                   {'pending_variables': ['D']}):
        assert not certificate_valid(c, p, {'D': False}, contract(), **(kw | change))
    assert certificate_valid(c, p, {'D': False}, contract(), **(kw | {'pending_variables': ['A']}))


def test_fixed_seed_random_expressions_against_independent_finite_world_oracle():
    rng = random.Random(83171)
    def expression(depth):
        if not depth or rng.random() < .28:
            return var(rng.choice(['A', 'D', 'O'])) if rng.random() < .8 else const(rng.choice([True, False]))
        op = rng.choice(['and', 'or', 'not', 'if', 'eq'])
        arity = 1 if op == 'not' else 3 if op == 'if' else 2
        return operation(op, *(expression(depth - 1) for _ in range(arity)))
    for _ in range(300):
        expr = expression(4)
        constraints = [operation('eq', var('A'), var('D'))] if rng.random() < .3 else []
        p = program(expr, constraints=constraints)
        facts = {k: rng.choice([False, True]) for k in ('A', 'D', 'O') if rng.random() < .35}
        worlds = [dict(zip(('A', 'D', 'O'), values)) for values in product([False, True], repeat=3)]
        worlds = [w for w in worlds if all(w[k] == v for k, v in facts.items())
                  and all(oracle(c, w) for c in constraints)]
        expected = {oracle(expr, w) for w in worlds}
        result = possible_outputs(p, facts)
        assert set(result['outputs']) == expected
        assert result['status'] == ('inconsistent' if not worlds else 'determined' if len(expected) == 1 else 'indeterminate')
        for world in worlds:
            assert oracle(reduce(p, facts)['expression'], world) == oracle(expr, world)
        certificate = certify_output(p, facts, contract())
        assert (certificate is not None) == (len(expected) == 1)


@pytest.mark.parametrize('expr', [
    {'op': 'python', 'code': 'arbitrary()'}, operation('and', const(1), const(True)),
    operation('not', const(True), const(False)), var('absent')])
def test_whitelist_rejects_arbitrary_code_and_bad_types(expr):
    with pytest.raises(ProgramError):
        validate_program(program(expr))

def test_incomplete_necessary_conjunction_can_certify_negative_but_not_positive():
    p={'variables':{'a':{'type':'bool'},'b':{'type':'bool'}},
       'expression':{'op':'and','args':[{'op':'var','id':'a'},{'op':'var','id':'b'}]},'rules_complete':False}
    assert certify_output(p,{'a':False},contract(result_type='bool'))['output'] is False
    assert certify_output(p,{'a':True,'b':True},contract(result_type='bool')) is None

def test_abstract_extra_worlds_do_not_erase_proven_nonempty_common_output():
    p={'variables':{'x':{'type':'number','domain':{'lower':0,'upper':1}},'b':{'type':'bool'}},
       'expression':{'op':'const','value':False},
       'constraints':[{'op':'or','args':[{'op':'var','id':'b'},
          {'op':'eq','args':[{'op':'var','id':'x'},{'op':'const','value':0}]}]}]}
    # b=true supplies genuine feasible worlds; b=false has uncertain interval
    # equality and is conservatively retained. Every admitted world outputs false.
    result=possible_outputs(p,{})
    assert result['status']=='determined' and result['nonempty_proven']
    assert certify_output(p,{},contract(result_type='bool'))['output'] is False
    p['constraints']=[{'op':'eq','args':[{'op':'var','id':'x'},{'op':'const','value':0}]}]
    assert certify_output(p,{},contract(result_type='bool')) is None
