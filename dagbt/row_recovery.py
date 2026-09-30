"""Scoped v3 row correction that never turns global faults into partial success."""
from copy import deepcopy

from .budget import BudgetExceeded
from .reasoning import ProtocolError, OutputTruncated, RefusalError, InputOverflow
from .transport import digest


def _identity(field, row):
    fields = {
        'alternatives': ('source_span_ids', 'guard_span_ids', 'used_parent_ids', 'applicable_scope', 'semantic_status'),
        'conflicts': ('alternative_ids', 'span_ids', 'entity_scope', 'event_time'),
        'resolutions': ('conflict_id', 'resolution_span_ids', 'addressed_conflict_span_ids', 'resolution_kind'),
        'refinements': ('question', 'answer_type', 'inputs', 'source_span_ids'),
        'unresolved_guards': ('node_id', 'description'),
    }.get(field)
    value = ({key: row.get(key) for key in fields} if fields else
             {key: value for key, value in row.items() if key not in {'reason', 'explanation'}})
    normalized = {key: sorted(value) if isinstance(value, list) and all(isinstance(x, str) for x in value)
                  else ' '.join(value.split()) if isinstance(value, str) else value
                  for key, value in value.items()}
    return digest(normalized)


def _failure(exc, scope):
    category = (exc.category if isinstance(exc, ProtocolError) else
                'input_budget' if isinstance(exc, InputOverflow) else
                'call_budget' if isinstance(exc, BudgetExceeded) else 'service_or_backend')
    return {'scope': scope, 'error_type': type(exc).__name__, 'error': str(exc), 'failure_category': category}


def recover_rows(reasoner, operation, system, data, validate, fields, *, decode=lambda x: x,
                 repair_builder=None, validate_header=None, schema=None,
                 reserve=None, extra_reserve=0, reserve_repairs=None, allow_partial=True):
    """Return ``(result, diagnostics)`` after bounded, explicitly scoped recovery.

    repair_builder(state) receives copied original_data/fixed_header/retained_rows,
    pending_fields/pending_row_counts/failed_fields/failed_rows/repair_index and
    returns a payload. Recovery instructions are attached and actual wire input
    checked before sending. validate_header(header) checks hard constraints
    separately. Without it, a valid aggregate must establish header validity.
    """
    fields = tuple(fields)
    if not fields or len(set(fields)) != len(fields) or any(not isinstance(f, str) or not f for f in fields):
        raise ValueError('fields must contain distinct nonempty field names')
    retained = {field: [] for field in fields}
    needed = {field: 0 for field in fields}
    failed_fields = set()
    header, last_error, last_scope = None, None, None
    header_valid = False
    errors, failures, actions = [], [], []
    repaired = 0
    current = deepcopy(data)
    complete = False
    repair_reserve = reasoner.repair_reservation(operation, reserve_repairs)

    def record_failure(exc, scope):
        nonlocal last_error, last_scope
        last_error, last_scope = exc, scope
        failures.append(_failure(exc, scope))

    def diagnostic():
        return {'complete': complete, 'validation_complete': complete,
                'failed_rows': deepcopy(errors), 'pending_fields': [f for f in fields if needed[f] or f in failed_fields],
                'pending_row_counts': dict(needed), 'failed_fields': sorted(failed_fields),
                'retained_counts': {k: len(v) for k, v in retained.items()}, 'repairs': repaired,
                'error_type': type(last_error).__name__ if last_error else None,
                'error': str(last_error) if last_error else None,
                'failure_category': _failure(last_error, last_scope)['failure_category'] if last_error else None,
                'failure_scope': last_scope, 'failures': deepcopy(failures),
                'recovery_actions': deepcopy(actions), 'header_valid': header_valid,
                'reserved_repairs': repair_reserve}

    def emit_final():
        reasoner.event({'event': 'local_rows_complete', 'operation': operation, **diagnostic()})

    while True:
        try:
            value = reasoner.request(operation, system, current, schema=schema,
                                    reserve=reserve, extra_reserve=extra_reserve)
        except (OutputTruncated, RefusalError) as exc:
            record_failure(exc, 'global_response')
            errors = [{'field': 'response', **_failure(exc, 'global_response')}]
            emit_final()
            raise
        except (BudgetExceeded, InputOverflow) as exc:
            scope = 'row_budget' if last_scope == 'rows' else 'global_budget'
            record_failure(exc, scope)
            errors = [*errors, {'field': 'request', **_failure(exc, scope)}]
            break
        except ProtocolError as exc:
            record_failure(exc, 'global_response')
            errors = [{'field': 'response', **_failure(exc, 'global_response')}]
        except BaseException as exc:
            record_failure(exc, 'service_or_backend')
            errors = [*errors, {'field': 'request', **_failure(exc, 'service_or_backend')}]
            emit_final()
            raise
        else:
            try:
                proposed_header = {k: v for k, v in value.items() if k not in fields}
                if header is not None and proposed_header != header:
                    raise ProtocolError('Local repair changed the fixed response header', category='header')
                if validate_header is not None:
                    try:
                        validate_header(deepcopy(proposed_header))
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ProtocolError(str(exc), category='header') from exc
                if header is None:
                    header = deepcopy(proposed_header)
                if validate_header is not None:
                    header_valid = True
                errors = []
                for field in fields:
                    required = field == fields[0] or needed[field] > 0 or field in failed_fields
                    if field not in value and required:
                        failed_fields.add(field)
                        errors.append({'field': field, 'error': 'Required correction field was omitted'})
                        continue
                    rows = value.get(field, [])
                    if not isinstance(rows, list):
                        failed_fields.add(field)
                        errors.append({'field': field, 'error': 'Expected an explicit list'})
                        continue
                    failed_fields.discard(field)
                    added = invalid = 0
                    for index, row in enumerate(rows):
                        if not isinstance(row, dict):
                            invalid += 1
                            errors.append({'field': field, 'index': index, 'error': 'Row must be an object'})
                            continue
                        if _identity(field, row) in {_identity(field, x) for x in retained[field]}:
                            continue
                        candidate = {**deepcopy(header), **deepcopy(retained)}
                        candidate[field].append(row)
                        try:
                            validate(decode(candidate))
                        except (ValueError, KeyError, TypeError) as exc:
                            invalid += 1
                            errors.append({'field': field, 'index': index, 'error': str(exc)[:240]})
                        else:
                            retained[field].append(deepcopy(row))
                            header_valid = True
                            added += 1
                    missing = max(0, needed[field] - added - invalid)
                    if missing:
                        errors.append({'field': field, 'index': 'missing_replacement', 'missing_count': missing,
                                       'error': 'Failed rows still require new valid replacements'})
                    needed[field] = invalid + missing
                candidate = {**deepcopy(header), **deepcopy(retained)}
                try:
                    validate(decode(candidate))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ProtocolError(str(exc), category='header_or_aggregate') from exc
                header_valid = True
                if not errors:
                    complete = True
                    last_error = last_scope = None
                    break
                record_failure(ProtocolError('Some response rows remain invalid', category='row_annotation'), 'rows')
            except ProtocolError as exc:
                record_failure(exc, 'header_or_aggregate')
                errors = [*errors, {'field': 'header', **_failure(exc, 'header_or_aggregate')}]

        reasoner.event({'event': 'local_rows_retained', 'operation': operation, **diagnostic()})
        if not header_valid and not any(retained.values()):
            header = None
        if (repaired >= reasoner.settings.get('max_repairs_per_request', 2)
                or reasoner.ledger.remaining('json_repairs') <= repair_reserve):
            actions.append({'action': 'repair_allowance_exhausted', 'repairs': repaired,
                            'remaining_repairs': reasoner.ledger.remaining('json_repairs'),
                            'reserved_repairs': repair_reserve})
            break
        pending = [field for field in fields if needed[field] or field in failed_fields]
        state = {'original_data': deepcopy(data), 'fixed_header': deepcopy(header),
                 'retained_rows': deepcopy(retained), 'pending_fields': pending,
                 'pending_row_counts': dict(needed), 'failed_fields': sorted(failed_fields),
                 'failed_rows': deepcopy(errors), 'repair_index': repaired + 1}
        try:
            current = repair_builder(deepcopy(state)) if repair_builder is not None else deepcopy(data)
            if not isinstance(current, dict):
                raise TypeError('repair_builder must return a payload object')
            current = deepcopy(current)
            current['local_repair'] = {
                'instruction': 'Return new replacement rows for failed fields. Keep the fixed header exactly. '
                               'Do not repeat accepted rows. Explicitly return lists for requested fields; '
                               'omitting a failed field does not repair it.',
                'failed_rows': deepcopy(errors), 'fixed_header': deepcopy(header),
                'pending_fields': pending, 'pending_row_counts': dict(needed),
                'failed_fields': sorted(failed_fields),
                'accepted_row_counts': {k: len(v) for k, v in retained.items()},
            }
            count = reasoner.estimate(operation, system, current, schema)
            reasoner._preflight(operation, count, reserve=reserve, extra_reserve=extra_reserve)
        except (BudgetExceeded, InputOverflow) as exc:
            scope = 'row_budget' if last_scope == 'rows' else 'global_budget'
            record_failure(exc, scope)
            errors = [*errors, {'field': 'repair_input', **_failure(exc, scope)}]
            actions.append({'action': 'repair_preflight_failed', **_failure(exc, scope)})
            break
        except BaseException as exc:
            record_failure(exc, 'repair_builder')
            emit_final()
            raise
        actions.append({'action': 'repair_prepared', 'repair_index': repaired + 1,
                        'pending_fields': pending, 'input_tokens_local': count,
                        'scoped_builder': repair_builder is not None})
        reasoner.event({'event': 'local_rows_repair_prepared', 'operation': operation, **actions[-1]})
        reasoner.ledger.reserve('json_repairs', operation)
        repaired += 1

    if header is None or not header_valid or (not complete and
            (not allow_partial or last_scope not in {'rows', 'row_budget'})):
        emit_final()
        raise last_error or ProtocolError('No valid recovered response header', category='header')
    try:
        result = validate(decode({**header, **retained}))
    except (ValueError, KeyError, TypeError) as exc:
        record_failure(exc, 'header_or_aggregate')
        emit_final()
        raise
    emit_final()
    return result, diagnostic()
