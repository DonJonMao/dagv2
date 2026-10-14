"""Lossless, input-only sharing of exact strings in registered review rows.

This changes only the request representation. Observation roles, permissions,
source identities, canonical state and response validation remain independent.
Byte savings select a candidate; the existing exact token admission still rules.
"""
from collections import Counter
from copy import deepcopy
import json


INSTRUCTION = '''
shared_review_texts stores exact literal strings. In supported_parents, a
review_value_ref replaces only value. In dictionary known_observations and
the observation rows of failed_batch_reviews or repair_scope (validated,
failed observation/slot/duplicate/replacement/required-counter observations,
and whole-value draft/known observations), review_value_ref and
review_reason_ref replace only value and reason with the string at that key.
Look up each reference once; never parse or interpret the resulting string.
All original objects, roles, order, sources, versions and permissions remain
separate and unchanged. These storage references are not observations,
citations, review permissions or evidence of agreement. Old observation
tables and raw invalid_response text are unchanged. Decode before reasoning.
References are input-only: return full typed values and reasons under the
unchanged response schema, never review_value_ref or review_reason_ref.
'''

POOL = 'shared_review_texts'
REF_FIELDS = {'value': 'review_value_ref', 'reason': 'review_reason_ref'}


def _serialized(value):
    # No sorting: dictionary order and Boolean/integer/float representations
    # are part of this layer's exact round-trip contract.
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _dict(value, label):
    if type(value) is not dict:
        raise ValueError('invalid ' + label)
    return value


def _list(value, label):
    if type(value) is not list:
        raise ValueError('invalid ' + label)
    return value


def _observation_list(container, field):
    if field in container:
        for row in _list(container[field], field):
            yield _dict(row, field + ' row'), ('value', 'reason')


def _review_rows(container):
    """Only producer-defined failure and whole-value review paths."""
    if 'failed' in container:
        for failure in _list(container['failed'], 'failed'):
            failure = _dict(failure, 'failure')
            # Both nullable fields are emitted by the failure producers when
            # a missing/unparsed response has no observation or slot draft.
            if failure.get('observation') is not None:
                yield _dict(failure['observation'], 'failed observation'), ('value', 'reason')
            if failure.get('slot') is not None:
                yield from _observation_list(_dict(failure['slot'], 'failed slot'), 'observations')
            for field in ('observations', 'replacement_observations', 'required_counter_observations'):
                yield from _observation_list(failure, field)
    if 'whole_value_reviews' in container:
        for group in _list(container['whole_value_reviews'], 'whole_value_reviews'):
            group = _dict(group, 'whole-value review')
            for field in ('draft_observations', 'known_observations'):
                yield from _observation_list(group, field)


def _rows(wire):
    """Yield mutable rows plus allowed fields, without entering typed values."""
    _dict(wire, 'wire')
    for row in _dict(wire.get('supported_parents', {}), 'supported_parents').values():
        yield _dict(row, 'supported parent'), ('value',)
    if 'known_observation_columns' in wire:
        # Preserve the preexisting physical table exactly; it is not this
        # codec's reference namespace and its cells are never traversed.
        columns = _list(wire['known_observation_columns'], 'known columns')
        if any(type(key) is not str for key in columns) or len(set(columns)) != len(columns):
            raise ValueError('invalid known columns')
        _list(wire.get('known_observation_values', []), 'known values')
        for row in _list(wire.get('known_observations'), 'known table'):
            if type(row) is not list or len(row) != len(columns):
                raise ValueError('invalid known table row')
    else:
        if 'known_observation_values' in wire:
            raise ValueError('orphan known values')
        yield from _observation_list(wire, 'known_observations')
    if 'failed_batch_reviews' in wire:
        for review in _list(wire['failed_batch_reviews'], 'failed_batch_reviews'):
            review = _dict(review, 'failed batch review')
            yield from _observation_list(review, 'validated_observations')
            yield from _review_rows(review)
    if 'repair_scope' in wire:
        yield from _review_rows(_dict(wire['repair_scope'], 'repair_scope'))


def _replace(row, replacements):
    """Rename fields at their original positions and keep every extra field."""
    items = [(replacements[key] if key in replacements else (key, value))
             for key, value in row.items()]
    row.clear()
    row.update(items)


def pack_with_status(wire):
    """Return an independent candidate and trusted applied flag, or fall back.

    Every row the decoder visits is checked before any replacement, including
    noncandidate rows. A preexisting reserved field never activates decoding.
    """
    original = deepcopy(wire)
    try:
        before = _serialized(original)
        rows = list(_rows(original))
        if POOL in original or any(set(REF_FIELDS.values()) & row.keys() for row, _ in rows):
            return original, False
        counts = Counter(row[field] for row, fields in rows for field in fields
                         if type(row.get(field)) is str)
        refs = {}
        for value, count in counts.items():
            if count < 2:
                continue
            ref = 'r' + str(len(refs))
            literal_bytes, ref_bytes = len(_serialized(value)), len(_serialized(ref))
            # Both registered field renames add eleven ASCII bytes. Each pool
            # entry pays its key, colon, exact literal and one separator byte.
            saved = count * (literal_bytes - ref_bytes - 11)
            if saved > ref_bytes + 1 + literal_bytes + 1:
                refs[value] = ref
        if not refs:
            return original, False
        out = deepcopy(original)
        for row, fields in _rows(out):
            replacements = {field: (REF_FIELDS[field], refs[row[field]])
                            for field in fields if type(row.get(field)) is str and row[field] in refs}
            _replace(row, replacements)
        out[POOL] = {ref: value for value, ref in refs.items()}
        if len(_serialized(out)) + len(INSTRUCTION.encode('utf-8')) >= len(before):
            return original, False
        if _serialized(unpack(out)) != before:
            raise AssertionError('review wire failed exact structural roundtrip')
        return out, True
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        return original, False


def unpack(wire):
    """Restore only this codec, once, with strict references and original order.

    No model response is passed through this inverse. Pool contents are opaque
    literal strings, even when they look like JSON or additional references.
    """
    out = deepcopy(wire)
    _serialized(out)
    rows = list(_rows(out))
    if POOL not in out:
        if any(set(REF_FIELDS.values()) & row.keys() for row, _ in rows):
            raise ValueError('review reference without pool')
        return out
    texts = out[POOL]
    if (type(texts) is not dict or not texts
            or any(type(key) is not str or type(value) is not str for key, value in texts.items())):
        raise ValueError('invalid review text pool')
    for row, fields in rows:
        replacements = {}
        for field, ref_field in REF_FIELDS.items():
            if ref_field not in row:
                continue
            if field not in fields or field in row:
                raise ValueError('invalid or ambiguous review reference field')
            ref = row[ref_field]
            if type(ref) is not str or ref not in texts:
                raise ValueError('unknown review text reference')
            replacements[ref_field] = (field, texts[ref])
        _replace(row, replacements)
    del out[POOL]
    return out
