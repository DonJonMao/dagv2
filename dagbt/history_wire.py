"""Lossless input-only sharing of repeated typed history text.

The deterministic byte check chooses a representation, not a token budget.
Canonical evidence and response schemas remain outside this wire transform.
"""
from copy import deepcopy
import json


INSTRUCTION = '''
shared_history_texts stores exact text values used more than once. In
supported_parents and known_observations, value_ref replaces only the value
field with the exact string at that key; all other fields remain unchanged.
If grounded_subquestion_parts is present, concatenate its literal strings and
the exact strings named by {"text_ref": key} to recover grounded_subquestion.
Decode these references before reasoning. They are storage references, not
new observations, citations, review permissions or evidence of agreement.
References are input-only. Return full typed values in output observations
under the unchanged response schema; never return value_ref or text_ref.
'''
RESERVED = {'shared_history_texts', 'grounded_subquestion_parts'}


def byte_size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8'))


def normalize_known_table(wire):
    """Decode the existing observation table exactly once, retaining literals."""
    out = deepcopy(wire)
    if 'known_observation_columns' not in out:
        if 'known_observation_values' in out:
            raise ValueError('orphan known values')
        return out
    columns = out.pop('known_observation_columns')
    values = out.pop('known_observation_values', [])
    if (not isinstance(columns, list) or any(type(k) is not str for k in columns)
            or len(set(columns)) != len(columns) or not isinstance(values, list)):
        raise ValueError('invalid old observation table')
    if not isinstance(out.get('known_observations'), list):
        raise ValueError('missing or invalid old observation rows')
    rows = []
    for row in out.get('known_observations', []):
        if not isinstance(row, list) or len(row) != len(columns):
            raise ValueError('invalid old observation row')
        decoded = {}
        for key, cell in zip(columns, row):
            if isinstance(cell, dict) and set(cell) == {'shared'}:
                index = cell['shared']
                if type(index) is not int or not 0 <= index < len(values):
                    raise ValueError('invalid old observation reference')
                cell = values[index]
            decoded[key] = deepcopy(cell)
        rows.append(decoded)
    out['known_observations'] = rows
    return out


def _rows(wire):
    parents = wire.get('supported_parents', {})
    known = wire.get('known_observations', [])
    if (not isinstance(parents, dict) or not isinstance(known, list)
            or any(not isinstance(v, dict) for v in parents.values())
            or any(not isinstance(v, dict) for v in known)):
        raise ValueError('invalid parent or observation rows')
    return list(parents.values()) + known


def _parts(text, refs):
    parts, cursor = [], 0
    while cursor < len(text):
        hits = [(at, -len(value), value, ref) for value, ref in refs.items()
                if (at := text.find(value, cursor)) >= 0]
        if not hits:
            parts.append(text[cursor:])
            break
        at, _, value, ref = min(hits)
        if at > cursor:
            parts.append(text[cursor:at])
        parts.append({'text_ref': ref})
        cursor = at + len(value)
    return parts


def pack(wire):
    """Return only the representation for pure transform callers."""
    return pack_with_status(wire)[0]


def pack_with_status(wire):
    """Return (wire, applied); collisions retain the exact old representation.

    Callers must use this status to select decoding instructions. A reserved
    field already present in an unchanged input never authorizes decoding.
    """
    original = deepcopy(wire)
    if RESERVED & set(wire):
        return original, False
    try:
        out = normalize_known_table(wire)
        rows = _rows(out)
    except (ValueError, TypeError, KeyError):
        return original, False
    if any('value_ref' in row for row in rows):
        return original, False
    query = out.get('grounded_subquestion')
    if type(query) is not str:
        return original, False
    eligible = dict.fromkeys(row['value'] for row in out.get('supported_parents', {}).values()
                             if row.get('type') == 'text' and type(row.get('value')) is str
                             and len(row['value'].encode('utf-8')) >= 256)
    eligible = [value for value in eligible
                if sum(type(row.get('value')) is str and row['value'] == value for row in rows)
                + query.count(value) >= 2]
    if not eligible:
        return original, False
    refs = {value: 't' + str(i) for i, value in enumerate(eligible)}
    out['shared_history_texts'] = {ref: value for value, ref in refs.items()}
    for row in rows:
        value = row.get('value')
        if type(value) is str and value in refs:
            row['value_ref'] = refs[value]
            del row['value']
    parts = _parts(query, refs)
    if any(isinstance(part, dict) for part in parts):
        out['grounded_subquestion_parts'] = parts
        del out['grounded_subquestion']
    if byte_size(out) + len(INSTRUCTION.encode('utf-8')) >= byte_size(original):
        return original, False
    def canonical(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(',', ':'))
    if canonical(unpack(out)) != canonical(normalize_known_table(original)):
        raise AssertionError('candidate failed exact structural roundtrip')
    return out, True


def unpack(wire):
    """Reconstruct the exact data behind this input-only wire format."""
    out = normalize_known_table(wire)
    if 'shared_history_texts' not in out:
        if 'grounded_subquestion_parts' in out:
            raise ValueError('query parts without reference table')
        return out
    texts = out.pop('shared_history_texts')
    if (not isinstance(texts, dict) or not texts
            or any(type(k) is not str or type(v) is not str for k, v in texts.items())):
        raise ValueError('invalid shared text table')

    def resolve(ref):
        if type(ref) is not str or ref not in texts:
            raise ValueError('unknown shared text reference')
        return texts[ref]

    for row in _rows(out):
        if 'value_ref' in row:
            if 'value' in row:
                raise ValueError('both value and value_ref')
            row['value'] = resolve(row.pop('value_ref'))
    if 'grounded_subquestion_parts' in out:
        if 'grounded_subquestion' in out:
            raise ValueError('both query and query parts')
        parts = out.pop('grounded_subquestion_parts')
        if not isinstance(parts, list):
            raise ValueError('invalid query parts')
        text = []
        for part in parts:
            if type(part) is str:
                text.append(part)
            elif isinstance(part, dict) and set(part) == {'text_ref'}:
                text.append(resolve(part['text_ref']))
            else:
                raise ValueError('invalid query part')
        out['grounded_subquestion'] = ''.join(text)
    return out
