# PersonaMem-v1 32k

This directory contains **all 589 questions** in the official 32k file, covering
20 personas and 37 shared contexts. It uses the same pinned source as the local
BridgeTree project:

- Dataset: `bowen-upenn/PersonaMem-v1`
- Revision: `fd7c30f071d5c2ee2a211506783be222d7b6002e`
- `questions_32k.csv`: SHA256 `cccd34cf53e0bc4d9536c04cff5ca045156d9a4e227e83327112482840bbc93c`
- `shared_contexts_32k.jsonl`: SHA256 `217247ebfec9e8442fc53570c795ab69f21aad08745f7de78d9beab51b122d4a`

The exact source URLs, generated public file hashes, and separate evaluation
file hash are recorded in `manifest.json`. Raw labels are not included anywhere
except `evaluation_only.json`. The raw source CSV is not needed at runtime.

`questions.jsonl` and `questions.json` contain the original request, every public
answer option, source question ID, persona ID, shared context ID, and exclusive
message cutoff. They contain no correct-answer label or supporting-evidence
annotations. Both experiment arms receive exactly the same formatted request
and four answer options.

`corpus.jsonl` is a deduplicated **storage pool**, not a globally searchable
corpus. For each distinct `(shared_context_id, end_index)`, the importer slices
the source messages with `messages[:end_index]` **before** calling the frozen BT
`messages_to_memories` function with `include_system_persona=True` and
`memory_granularity="user_assistant_pair"`. This produces 222 retrieval scopes
and 3,187 distinct memories. Memory IDs bind the context, message indices, and
exact text; a partially visible final user/assistant pair therefore has a
different identity from the later complete pair.

Each document's `text` remains the exact BT role-tagged memory text. Its `title`
exposes the zero-based source message range, for example
`Conversation message indices 12–13 (zero-based chronological observation order; not calendar time)`.
Both arms embed and read the same `title + newline + text` passage. This makes
recorded conversation order visible when preferences change; message indices
are observation order and are never presented as calendar dates or event times.
The title is derived only from that memory's visible source indices.

For every question, retrieval, reranking, navigation, evidence validation, and
final reading must use only the document IDs in
`scopes.json[question.scope_id]`. Neither another person's memories nor messages
at or after that question's cutoff are allowed. Precomputing embeddings for the
whole storage pool does not grant permission to retrieve from the whole pool.

Evaluation uses strict single-choice accuracy. Valid responses contain one
label, such as `(a)`, `A`, or `Answer: (b)`; multiple labels, prose explanations,
or out-of-range labels count as invalid and incorrect. This is intentionally
stricter than the vendored BT first-match label extractor. Persona-macro
accuracy should also be reported. This complete 589-question run is not a claim
to reproduce BT's persisted train/validation/test partition or its held-out
test score.

Rebuild from downloaded files using:

```bash
.venv/bin/python -m dagbt.personamem --raw-dir /path/to/personamem-v1
```

Both source checksums must match before importing. Runtime public validation
does not read or hash `evaluation_only.json`; evaluation validation must opt in
with `validate_dataset(..., include_labels=True)`.
