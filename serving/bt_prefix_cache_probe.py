#!/usr/bin/env python3
"""Inspect installed runtime, plan an isolated arm, or observe an existing proxy.

Uses the installed proxy and engine unchanged. Never edits/restarts production.
No assumptions about NVIDIA/Ascend kernels, pooling, or precision.
"""
from __future__ import annotations

import argparse
import contextvars
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_bt_reranker_prefix_cache import digest, file_hash, isolated_url, new_output, write, read_rows


PACKAGES = ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers")
IDENTITY_FIELDS = ("weights_sha256", "tokenizer_files_sha256", "proxy_source_sha256",
                   "chat_template_sha256", "default_instruction_sha256", "adapter", "dtype", "kv_dtype", "quantization",
                   "attention", "position_rope", "runner", "pooling", "engine_version",
                   "ascend_version", "cann_version", "device", "max_model_len", "block_size",
                   "score_contract", "logprobs_mode")


def template_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if isinstance(value, str) else digest(value)


def inspect_runtime(output):
    out = new_output(output)
    packages, evidence = {}, {}
    for name in PACKAGES:
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = dist.version
            matches = []
            # Installed source evidence only, not documentation for another tag.
            for f in dist.files or []:
                s = str(f)
                if s.endswith(("arg_utils.py", "cache.py", "completion/protocol.py", "config.json")):
                    path = Path(dist.locate_file(f))
                    if path.is_file():
                        matches.append({"path": s, "sha256": file_hash(path)})
                        if path.suffix == ".py":
                            text = path.read_text(errors="replace")
                            evidence[s] = [{"line": i, "text": line.strip()} for i, line in enumerate(text.splitlines(), 1)
                                           if any(k in line for k in ("enable-prefix-caching", "enable_prefix_caching", "cache_salt", "is_prefix_caching_supported"))][:80]
            packages[name] = {"version": dist.version, "source_files": matches}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    probes = {}
    for name, argv in {"vllm_help": ["vllm", "serve", "--help"], "vllm_full_help": ["vllm", "serve", "--help=all"], "npu": ["npu-smi", "info"],
                       "nvidia": ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]}.items():
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            probes[name] = {"returncode": r.returncode, "stdout": r.stdout[:80000]}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            probes[name] = None
    result = {"packages": packages, "probes": probes, "source_evidence": evidence,
              "production_configuration": None, "note": "Presence/help is capability evidence, not proof of scoring-path KV hits"}
    write(out / "runtime.json", result)
    return result


def build_arm(command, help_text, mode, port):
    if command[:2] != ["vllm", "serve"] or len(command) < 3:
        raise ValueError("Supply the observed existing vllm serve argv, preserving all model options")
    if any(x.startswith(("--api-key", "--token", "--hf-token")) for x in command):
        raise ValueError("Credential-bearing argv cannot be written; use existing secret environment injection")
    if port in (8002, 18002) or not 1024 <= port <= 65535:
        raise ValueError("Use a new isolated port")
    if "--enable-prefix-caching" not in help_text or "--no-enable-prefix-caching" not in help_text:
        raise ValueError("Installed --help does not verify both APC flag spellings")
    result = []
    i = 0
    while i < len(command):
        arg = command[i]
        if arg in ("--enable-prefix-caching", "--no-enable-prefix-caching"):
            i += 1
            if i < len(command) and command[i].lower() in ("true", "false"):
                raise ValueError("Unreviewed boolean CLI form")
            continue
        if arg.startswith(("--enable-prefix-caching=", "--no-enable-prefix-caching=")):
            raise ValueError("Unreviewed boolean CLI form")
        if arg in ("--port", "--host"):
            if i + 1 >= len(command): raise ValueError("Missing CLI value")
            i += 2
            continue
        if arg.startswith(("--port=", "--host=")):
            i += 1; continue
        result.append(arg); i += 1
    return result + ["--host", "127.0.0.1", "--port", str(port),
                     "--enable-prefix-caching" if mode == "on" else "--no-enable-prefix-caching"]


def plan(args):
    inventory = json.loads(Path(args.inventory).read_text())
    missing = [k for k in IDENTITY_FIELDS if inventory.get(k) is None]
    if missing:
        raise ValueError("Runtime identity incomplete: " + ", ".join(missing))
    if inventory.get("causal_prefix_supported") is not True:
        raise ValueError("Actual model/attention scoring-path capability not verified")
    if inventory.get("private_single_tenant_instance") is not True:
        raise ValueError("This tool requires the deployment's private single-tenant instance")
    if not inventory.get("tenant_namespace_hash"):
        raise ValueError("Stable existing tenant namespace required")
    command = build_arm(inventory["engine_argv"], Path(args.help).read_text(), args.mode, args.port)
    out = new_output(args.output)
    result = {"isolated": True, "mode": args.mode, "engine_argv": command,
              "source_inventory_sha256": file_hash(args.inventory),
              "computation_identity": digest({k: inventory[k] for k in IDENTITY_FIELDS}),
              "enable_prefix_caching": args.mode == "on", "backend_url": "http://127.0.0.1:" + str(args.port),
              "tenant_namespace_hash": inventory.get("tenant_namespace_hash"),
              "label_token_ids": inventory.get("label_token_ids"), "metrics": inventory.get("metrics", {}),
              "settings": {k: inventory[k] for k in IDENTITY_FIELDS},
              "launch_performed": False, "health_verified": False,
              "rollback": "Stop only the isolated PID/container you launched; production requires no rollback",
              "prerequisite": "Run under the existing Ascend launcher/image with identical environment/hotfixes on an authorized idle device"}
    write(out / "launch_plan.json", result)
    return result


def observe_proxy(args):
    """Load exact deployed proxy, wrap final post_json without changing payloads.

    Dedicated process/engine per private tenant avoids depending on unverified
    cache_salt support in a different API. No new result cache or model retry.
    """
    backend = isolated_url(args.backend)
    if args.port in (8002, 18002) or args.port < 1024:
        raise ValueError("Isolated proxy port required")
    inventory = json.loads(Path(args.inventory).read_text())
    if inventory.get("private_single_tenant_instance") is not True:
        raise ValueError("Private single-tenant engine isolation must be attested")
    if not inventory.get("tenant_namespace_hash"):
        raise ValueError("Stable private tenant namespace required")
    if file_hash(args.proxy_source) != inventory.get("proxy_source_sha256"):
        raise ValueError("Proxy source differs from observed production identity")
    for name in IDENTITY_FIELDS:
        if inventory.get(name) is None: raise ValueError("Missing identity: " + name)
    identity = digest({k: inventory[k] for k in IDENTITY_FIELDS})
    out = new_output(args.output)
    # Preserve source proxy model/template/environment settings, only move its backend.
    os.environ["VLLM_BASE_URL"] = backend
    spec = importlib.util.spec_from_file_location("bt_isolated_original_proxy", args.proxy_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "MAX_MODEL_LEN") and module.MAX_MODEL_LEN != inventory["max_model_len"]:
        raise ValueError("Proxy environment changed maximum input capacity")
    if hasattr(module, "LABEL_TOKEN_IDS") and module.LABEL_TOKEN_IDS != inventory.get("label_token_ids"):
        raise ValueError("Proxy tokenizer canonical labels differ from runtime inventory")
    if hasattr(module, "DEFAULT_INSTRUCTION") and template_hash(module.DEFAULT_INSTRUCTION) != inventory["default_instruction_sha256"]:
        raise ValueError("Proxy environment changed the default instruction")
    if hasattr(module, "tokenizer") and template_hash(module.tokenizer.chat_template) != inventory["chat_template_sha256"]:
        raise ValueError("Proxy tokenizer chat template differs from inventory")
    original_post, original_compute = module.post_json, module.compute_scores
    lock = threading.Lock()
    context = contextvars.ContextVar("bt_request", default=None)
    ordinal = 0
    source_rows = None
    if getattr(args, "frozen_trace", None):
        selected = None
        if getattr(args, "smoke_only", False):
            manifest = json.loads((Path(args.frozen_trace).parent / "manifest.json").read_text())
            if manifest["trace_sha256"] != file_hash(args.frozen_trace):
                raise ValueError("Smoke trace changed")
            selected = set(manifest["smoke_question_identities"])
        source_rows = (r for _, r in read_rows(args.frozen_trace) if r["success"] and
                       (selected is None or r["dataset"] + ":" + r["question_id"] in selected))
    fd = os.open(out / "engine_trace.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    trace = os.fdopen(fd, "w", buffering=1)

    def post(url, payload, *pos, **kw):
        nonlocal ordinal
        ctx = context.get()
        if ctx is None or not url.endswith("/v1/completions"):
            return original_post(url, payload, *pos, **kw)
        with lock:
            n = ordinal; ordinal += 1
        started = time.perf_counter()
        row = {"ordinal": n, "request_id": ctx["request_id"], "question_id": ctx.get("question_id", ctx["query_sha256"]),
               'algorithm_version':ctx.get('algorithm_version','dagbt_fusion_reliability_v3'),
               'scoring_context_id':ctx.get('scoring_context_id'),
               "dataset": ctx.get("dataset"), "source_ordinal": ctx.get("source_ordinal"),
               "rerank_payload": ctx["rerank_payload"], "batch_indices": list(range(len(payload["prompt"]))),
               "payload": payload, "payload_sha256": digest(payload), "observed_at": "proxy_post_json",
               "final_inputs": [{"index": i, "token_ids_sha256": digest(tokens), "input_length": len(tokens)}
                                for i, tokens in enumerate(payload["prompt"])],
               "computation_identity": identity, "cache_namespace": inventory["tenant_namespace_hash"],
               "started_unix": time.time(), "completed": False}
        try:
            body = original_post(url, payload, *pos, **kw)
            row.update(completed=True, response=body)
            return body
        finally:
            row["seconds"] = time.perf_counter() - started
            row["finished_unix"] = time.time()
            with lock:
                trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def compute(query, documents, instruction, *pos, **kw):
        # A sequence counter, not content hash, separates repeated physical calls.
        with lock:
            request_id = str(time.time_ns())
            extra = {}
            if source_rows is not None:
                expected = next(source_rows)
                p = expected["payload"]
                if p["query"] != query or p["documents"] != documents or p.get("instruction") != instruction:
                    raise ValueError("Frozen request order/input differs; no observation remapping")
                request_id = expected["request_id"] + ":" + str(expected["ordinal"])
                extra = {"question_id": expected["question_id"], "dataset": expected["dataset"],
                         'algorithm_version':expected.get('algorithm_version','dagbt_fusion_reliability_v3'),
                         'scoring_context_id':expected.get('scoring_context_id'),
                         "source_ordinal": expected["ordinal"]}
        token = context.set({"request_id": request_id, "query_sha256": digest(query), **extra,
                             "rerank_payload": {"query": query, "documents": documents, "instruction": instruction}})
        try:
            return original_compute(query, documents, instruction, *pos, **kw)
        finally:
            context.reset(token)
    module.post_json, module.compute_scores = post, compute
    write(out / "observer_manifest.json", {"proxy_source_sha256": file_hash(args.proxy_source),
          "computation_identity": identity, "backend_url": backend,
          "private_single_tenant_instance": True, "payload_modified": False,
          "frozen_trace_sha256": file_hash(args.frozen_trace) if getattr(args, "frozen_trace", None) else None,
          "concurrency_note": "Completion-ordered trace files must be sorted by ordinal before sequential analysis; concurrent overlap is not a hit"})
    import uvicorn
    # Existing ASGI app, schemas, scoring and error handling; new loopback port.
    uvicorn.run(module.app, host="127.0.0.1", port=args.port)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect"); p.add_argument("--output", required=True)
    p = sub.add_parser("plan"); p.add_argument("--inventory", required=True); p.add_argument("--help-file", dest="help", required=True); p.add_argument("--mode", choices=["off", "on"], required=True); p.add_argument("--port", type=int, required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("observe-proxy"); p.add_argument("--inventory", required=True); p.add_argument("--proxy-source", required=True); p.add_argument("--backend", required=True); p.add_argument("--port", type=int, required=True); p.add_argument("--output", required=True); p.add_argument("--frozen-trace"); p.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.command == "observe-proxy": observe_proxy(args)
    else:
        inspect_runtime(args.output) if args.command == "inspect" else plan(args)
        print(json.dumps({"command": args.command, "output": args.output, "launch_performed": False}))


if __name__ == "__main__": main()
