"""Explicit offline HTTP response fixture; never selected as a service fallback."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path


class FixtureTransport:
    backend = "fixture"

    def __init__(self, record_path: str | Path | None = None, delay_seconds: float = 0):
        self.record_path = Path(record_path) if record_path else None
        self.delay_seconds = delay_seconds
        self.requests = []

    def __call__(self, url, payload, timeout, headers=None):
        # Record semantic payload, never Authorization. Only this boundary is
        # replaced; the caller still runs actual clients/search/reader/evaluator.
        request = {"backend": "fixture", "url": url, "payload": payload}
        self.requests.append(request)
        if self.record_path:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if "input" in payload:
            rows = []
            for i, text in enumerate(payload["input"]):
                vector = [0.01] * 16
                for word in re.findall(r"\w+", text.lower()):
                    vector[int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 16] += 1
                norm = math.sqrt(sum(x*x for x in vector))
                rows.append({"index": i, "embedding": [x/norm for x in vector]})
            return {"data": rows}
        if "documents" in payload:
            return {"results": [{"index": i, "relevance_score": 1 / (i + 2)}
                                for i in range(len(payload["documents"]))]}
        if payload.get("logprobs") or "response_format" in payload:
            public = json.loads(payload["messages"][1]["content"])
            records = public["records"]
            if "response_format" in payload:
                content = json.dumps({"text": "Use the stated personal constraints to answer the question.",
                                      "source_ids": [r["id"] for r in records]})
                return {"choices": [{"message": {"content": content}}]}
            label = "A" if records else "B"
            yes, no = (-.2, -1.7) if records else (-1.7, -.2)
            return {"choices": [{"message": {"content": label}, "logprobs": {"content": [
                {"token": label, "top_logprobs": [{"token": "A", "logprob": yes},
                                                 {"token": "B", "logprob": no}]}]}}],
                "usage": {"prompt_tokens": 32, "completion_tokens": 1}}
        return {"choices": [{"message": {"content": "(a) Fixture answer based on the supplied context."}}],
                "usage": {"prompt_tokens": 32, "completion_tokens": 12}}
