# Frozen latest BridgeTree working-tree snapshot

`bridgetree/` contains every Python source file from the current working tree of
`/Users/mao/projects/bridgetree_preference_rag/src/bridgetree`, copied on 2026-09-23.
The source includes the uncommitted **Evidence BridgeTree v1** revision. Git HEAD
alone does not identify that revision. `bridgetree_manifest.json` records each
file's byte length and SHA-256. No source file or initializer has been changed.

This snapshot is imported as `vendor.bridgetree`; the experiment has no runtime
dependency on the original project directory. Its full initializer and all 57
source modules are preserved to avoid reducing the research implementation to
a similarly named local approximation. Optional local-model and FAISS modules
are not needed by the supplied remote-model adapter.

Task-specific changes live only in `dagbt/bridge.py`:

- corpus passages replace personal memories in probe and set-score wrappers;
  raw source text is never rewritten;
- node queries and exact parent-source passages guide candidate discovery;
- the unchanged source four-set scorer uses the original overall question,
  explicitly as a relevance proxy, not semantic dependency verification;
- source multi-root search, four-set measurements, pair testing, pivots,
  speculation, scheduling, archive and source validation execute unchanged;
- one question-level ledger charges ANN attempts and newly seen scored sets;
  per-node fair shares and a reserved gap allowance prevent resetting the
  global budget at each DAG node;
- the initial-pool ANN allowance scales to 40% of each node's search share
  (floor, at least one), so smaller fair shares still leave calls for bridge
  search. The value and resulting actual source search allocation are logged;
- all discovered candidates are retained for DAG evidence mapping regardless
  of activation or target-retention scores;
- factual support alternatives, proof closure and the reader are implemented
  in the DAG fusion layer, separate from BT's navigation edges.

The adapter requires a real pointwise reranker service for bridge mode. No
missing-service or failed-reranker fallback to cosine, dense or constant scores
exists. Tests substitute only transports; they execute the production search.

The source's estimated reranker token budget is preserved and labeled as an
estimate. The reader's final budget must be checked with the real tokenizer by
the fusion engine. A service response declaring reranker truncation is rejected.
