"""Development-only answer-effect diagnostics.

The oracle is deliberately isolated from retrieval and protocol selection.  It
requires token log-probabilities; when a client does not expose them the
result is the explicit ``oracle_unavailable`` diagnostic rather than a guessed
score.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Mapping, Sequence

ORACLE_UNAVAILABLE = "oracle_unavailable"


def _label(value: Any) -> str:
    match = re.search(r"\(([a-z])\)|\b([a-z])\b", str(value or ""), re.IGNORECASE)
    if not match:
        return ""
    return f"({(match.group(1) or match.group(2)).lower()})"


def answer_effect_oracle(
    logprobs: Any = None,
    answer_options: Sequence[str] | None = None,
    *,
    response: Any = None,
    phase: str = "development",
) -> dict[str, Any]:
    """Summarize option effects from exposed log-probabilities.

    ``phase`` is checked defensively: confirmatory runs may not request this
    diagnostic.  Inputs can be a label->logprob mapping, a list of mappings,
    or OpenAI-style ``content`/``token_logprobs`` records.
    """
    if phase not in {"development", "development-seen", "diagnostic"}:
        return {"status": ORACLE_UNAVAILABLE, "reason": "confirmatory_oracle_forbidden"}
    raw = logprobs
    if raw is None and isinstance(response, Mapping):
        raw = response.get("logprobs")
        if raw is None:
            choice = response.get("choices", [{}])[0] if response.get("choices") else {}
            raw = choice.get("logprobs") if isinstance(choice, Mapping) else None
    if raw is None:
        return {"status": ORACLE_UNAVAILABLE, "reason": "no_logprob"}

    scores: dict[str, float] = {}
    if isinstance(raw, Mapping):
        # Direct label -> log probability form.
        if all(isinstance(value, (int, float)) for value in raw.values()):
            for key, value in raw.items():
                label = _label(key) or str(key)
                scores[label] = float(value)
        else:
            candidates = raw.get("top_logprobs", raw.get("content", raw.get("token_logprobs")))
            raw = candidates
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                token = item.get("token", item.get("text", item.get("label", "")))
                value = item.get("logprob", item.get("log_probability"))
                if value is None and item.get("top_logprobs") is not None:
                    top = item["top_logprobs"]
                    pairs = top.items() if isinstance(top, Mapping) else (
                        ((entry.get("token", entry.get("text", "")), entry.get("logprob", entry.get("log_probability")))
                         for entry in top if isinstance(entry, Mapping))
                        if isinstance(top, Sequence) and not isinstance(top, (str, bytes)) else ()
                    )
                    for key, probability in pairs:
                        label = _label(key)
                        if label and probability is not None:
                            scores[label] = max(scores.get(label, -math.inf), float(probability))
                    continue
                if value is not None:
                    label = _label(token)
                    if label:
                        scores[label] = max(scores.get(label, -math.inf), float(value))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                label = _label(item[0])
                if label:
                    scores[label] = max(scores.get(label, -math.inf), float(item[1]))
    if answer_options:
        if isinstance(answer_options, str):
            try:
                parsed_options = ast.literal_eval(answer_options)
            except (SyntaxError, ValueError):
                parsed_options = [answer_options]
            answer_options = parsed_options if isinstance(parsed_options, (list, tuple)) else [parsed_options]
        allowed = {_label(option) for option in answer_options}
        scores = {key: value for key, value in scores.items() if key in allowed}
    scores = {key: value for key, value in scores.items() if math.isfinite(value)}
    if not scores:
        return {"status": ORACLE_UNAVAILABLE, "reason": "unrecognized_logprob_schema"}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    maximum = ordered[0][1]
    normalizer = sum(math.exp(value - maximum) for _key, value in ordered)
    probabilities = {key: math.exp(value - maximum) / normalizer for key, value in ordered}
    return {
        "status": "available",
        "scores": scores,
        "probabilities": probabilities,
        "predicted_label": ordered[0][0],
        "margin": float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else None,
        "answer_effect": float(maximum),
    }


class AnswerEffectOracle:
    """Small callable wrapper used by development diagnostics."""

    def __init__(self, phase: str = "development") -> None:
        self.phase = phase

    def score(self, logprobs: Any = None, answer_options: Sequence[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        return answer_effect_oracle(logprobs, answer_options, phase=self.phase, **kwargs)

    __call__ = score


__all__ = ["ORACLE_UNAVAILABLE", "answer_effect_oracle", "AnswerEffectOracle"]
