from __future__ import annotations
import re
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def passage(self) -> str:
        if self.title and self.text:
            return f"{self.title}\n{self.text}"
        return self.title or self.text

def render_source_guide(spans: Sequence[GroundedSpan], required_count: int) -> str:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for span in spans:
        key = (span.source_doc_id, " ".join(span.source_span.split()).casefold())
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- source_doc_id: {span.source_doc_id}\n  source_span: {span.source_span}")
    grounded_count = len({span.demand_id for span in spans})
    return (
        f"status: {grounded_count} lineages grounded; {int(required_count)} demands required\n"
        "source_evidence:\n"
        + ("\n".join(lines) if lines else "(none)")
    )

def render_lineage_state(evidence: Sequence[LineageEvidence]) -> str:
    if not evidence:
        return "(none)"
    return "\n".join(
        f"- demand_id: {item.demand_id}\n"
        f"  state: {item.state_kind} {item.state_label} = {item.state_value}\n"
        f"  source_doc_id: {item.source_doc_id}\n"
        f"  exact_source_span: {item.source_span}"
        for item in evidence
    )

def reader_messages(
    *,
    question: str,
    selected_doc_ids: Sequence[str],
    documents: Mapping[str, Document],
    grounded_spans: Sequence[GroundedSpan],
    required_count: int,
    lineage_evidence: Sequence[LineageEvidence] = (),
) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[{rank}] source_doc_id={doc_id}\n{documents[doc_id].passage}"
        for rank, doc_id in enumerate(selected_doc_ids, start=1)
    )
    guide = render_source_guide(grounded_spans, required_count)
    lineage_state = render_lineage_state(lineage_evidence)
    user = (
        f"Context passages:\n{passages}\n\n"
        f"Committed demand-state evidence:\n{lineage_state}\n\n"
        f"Source-grounded evidence guide:\n{guide}\n\n"
        f"Question: {question}\n\n"
        "Answer with the shortest exact phrase supported by the context passages. "
        "When the question asks for a list or order, include every supported item. "
        "Return only answer values; never repeat the question wording. "
        "For who or name questions return only the person or role; for where return "
        "only the location; for when return only the time or event phrase; for why or "
        "how return only the cause or mechanism. Never attach the predicate copied "
        "from the question to an otherwise sufficient answer value. "
        "Use the source-grounded guide only to locate additional evidence; neither "
        "guide is a new source.\n"
        "Do not explain. Return exactly one line:\nAnswer: <short phrase>"
    )
    return [
        {
            "role": "system",
            "content": "You are a long-document QA reader. Give concise answers from the context.",
        },
        {"role": "user", "content": user},
    ]

def parse_answer(text: str) -> str:
    value = str(text).strip()
    match = re.search(r"(?:^|\n)\s*Answer\s*:\s*(.+)", value, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip()
    return value.splitlines()[0].strip() if value else ""
