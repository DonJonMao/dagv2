"""Keep independent valid support/audit rows during bounded local correction.

Mapping has its own unit-aware splitter. Here rows share a fixed conclusion
header: repairs may replace failed rows, never change an accepted conclusion.
"""
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
    # Reference order and whitespace-only paraphrases cannot create an
    # independent replacement for a failed row.
    normalized = {key: sorted(value) if isinstance(value, list) and all(isinstance(x, str) for x in value)
                  else ' '.join(value.split()) if isinstance(value, str) else value
                  for key, value in value.items()}
    return digest(normalized)


def recover_rows(reasoner, operation, system, data, validate, fields, *, decode=lambda x: x):
    retained = {field: [] for field in fields}
    header = None
    errors = []
    repaired = 0
    current = deepcopy(data)
    last_error = None
    complete = False
    needed = {field: 0 for field in fields}
    while True:
        try:
            value = reasoner.request(operation, system, current)
            proposed_header = {k: v for k, v in value.items() if k not in fields}
            if header is not None and proposed_header != header:
                raise ProtocolError('Local repair changed the fixed response header')
            if header is None:
                header = deepcopy(proposed_header)
            errors = []
            for field in fields:
                rows = value.get(field, None if field == fields[0] else [])
                if not isinstance(rows, list):
                    errors.append({'field': field, 'error': 'Expected a list'})
                    continue
                added = 0
                invalid = 0
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
                        added += 1
                missing = max(0, needed[field] - added - invalid)
                for _ in range(missing):
                    errors.append({'field': field, 'index': 'missing_replacement',
                                   'error': 'A failed row still requires a valid replacement'})
                needed[field] = invalid + missing
            candidate = {**deepcopy(header), **deepcopy(retained)}
            try:
                validate(decode(candidate))
            except (ValueError, KeyError, TypeError) as exc:
                errors.append({'field': 'header', 'error': str(exc)[:240]})
            if not errors:
                complete = True
                break
            last_error = ProtocolError('Some response rows remain invalid')
        except (OutputTruncated, RefusalError) as exc:
            last_error = exc
            errors = [{'field': 'response', 'error': exc.category}]
            break
        except (BudgetExceeded, InputOverflow) as exc:
            last_error = exc
            errors = [{'field': 'response', 'error': type(exc).__name__}]
            break
        except ProtocolError as exc:
            last_error = exc
            errors = [{'field': 'response', 'error': str(exc)[:240]}]
        reasoner.event({'event': 'local_rows_retained', 'operation': operation,
                        'retained_counts': {k: len(v) for k, v in retained.items()},
                        'failed_rows': errors, 'repairs': repaired})
        if not any(retained.values()):
            # There is no accepted semantic unit to freeze yet.
            header = None
        if (repaired >= reasoner.settings.get('max_repairs_per_request', 2)
                or reasoner.ledger.remaining('json_repairs') == 0):
            break
        # Retry only failed fields. Full raw broken output is never appended.
        current = deepcopy(data)
        current['local_repair'] = {
            'instruction': 'Return replacement rows for failed fields only; return [] for other row fields. '
                           'Do not repeat accepted rows. Keep the fixed header exactly.',
            'failed_rows': errors, 'fixed_header': header,
            'accepted_row_counts': {k: len(v) for k, v in retained.items()},
        }
        try:
            reasoner._preflight(operation, reasoner.estimate(operation, system, current))
        except (BudgetExceeded, InputOverflow) as exc:
            last_error = exc
            break
        reasoner.ledger.reserve('json_repairs', operation)
        repaired += 1
    reasoner.event({'event': 'local_rows_complete', 'operation': operation,
                    'complete': complete, 'retained_counts': {k: len(v) for k, v in retained.items()},
                    'failed_rows': errors, 'repairs': repaired})
    if header is None:
        raise last_error or ProtocolError('No complete response object')
    try:
        result = validate(decode({**header, **retained}))
    except (ValueError, KeyError, TypeError):
        raise last_error or ProtocolError('No structurally valid recovered response')
    return result, {'complete': complete, 'failed_rows': errors,
                    'retained_counts': {k: len(v) for k, v in retained.items()},
                    'repairs': repaired,
                    'error_type': type(last_error).__name__ if last_error else None}
