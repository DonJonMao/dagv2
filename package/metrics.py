from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from collections.abc import Sequence


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, aliases: Sequence[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(alias) for alias in aliases))


def _f1(prediction: str, answer: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not predicted_tokens or not answer_tokens:
        return float(predicted_tokens == answer_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(answer_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, aliases: Sequence[str]) -> float:
    return max((_f1(prediction, alias) for alias in aliases), default=0.0)


def normalize_characters(text: str) -> list[str]:
    return [
        character.casefold()
        for character in str(text)
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    ]


def character_f1(prediction: str, aliases: Sequence[str]) -> float:
    predicted = normalize_characters(prediction)
    scores: list[float] = []
    for alias in aliases:
        answer = normalize_characters(alias)
        overlap = sum((Counter(predicted) & Counter(answer)).values())
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(answer) if answer else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return max(scores, default=0.0)


def support_recall(
    retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int = 5
) -> float:
    gold = set(map(str, gold_doc_ids))
    if not gold:
        return 0.0
    retrieved = set(map(str, retrieved_doc_ids[:k]))
    return len(retrieved & gold) / len(gold)
