import json
import os
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        def _nvembed_get_usable_length(self, new_seq_length=None, layer_idx=0):
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = _nvembed_get_usable_length
except Exception:
    pass

HOST = os.environ.get("NV_EMBED_HOST", "127.0.0.1")
PORT = int(os.environ.get("NV_EMBED_PORT", "8019"))
MODEL_PATH = os.environ["NV_EMBED_MODEL_PATH"]
MODEL_NAME = os.environ.get("NV_EMBED_MODEL_NAME", "nvidia/NV-Embed-v2")
BATCH_SIZE = int(os.environ.get("NV_EMBED_BATCH_SIZE", "4"))
MAX_LENGTH = int(os.environ.get("NV_EMBED_MAX_LENGTH", "4096"))
INSTRUCTION = os.environ.get("NV_EMBED_INSTRUCTION", "")

_MODEL = None
_DEVICE = None
_LOAD_STARTED = time.time()
_READY_AT = None


def ts() -> str:
    return time.strftime("%F %T")


def load_model():
    global _MODEL, _DEVICE, _READY_AT
    if _MODEL is not None:
        return _MODEL
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{ts()}] loading {MODEL_PATH} on {_DEVICE}", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16 if _DEVICE == "cuda" else torch.float32,
    )
    model.eval()
    model.to(_DEVICE)
    with torch.inference_mode():
        warmup = model.encode(["warmup"], instruction=INSTRUCTION, max_length=min(MAX_LENGTH, 128))
        warmup = F.normalize(warmup, p=2, dim=1)
        print(f"[{ts()}] warmup dim={int(warmup.shape[-1])}", flush=True)
    _MODEL = model
    _READY_AT = time.time()
    print(f"[{ts()}] ready model={MODEL_NAME} port={PORT}", flush=True)
    return _MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = load_model()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), max(BATCH_SIZE, 1)):
            batch = texts[start:start + max(BATCH_SIZE, 1)]
            emb = model.encode(batch, instruction=INSTRUCTION, max_length=MAX_LENGTH)
            emb = F.normalize(emb, p=2, dim=1)
            vectors.extend(emb.detach().cpu().float().tolist())
    return vectors


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json({
                "ok": _MODEL is not None,
                "model": MODEL_NAME,
                "path": MODEL_PATH,
                "device": _DEVICE,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "ready_at": _READY_AT,
                "uptime_s": round(time.time() - _LOAD_STARTED, 3),
            })
            return
        if self.path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(_READY_AT or _LOAD_STARTED),
                    "owned_by": "local",
                    "root": MODEL_PATH,
                }],
            })
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        try:
            if self.path != "/v1/embeddings":
                self.send_error(404, "not found")
                return
            body = self.read_json()
            raw_input = body.get("input", [])
            if isinstance(raw_input, str):
                texts = [raw_input]
            else:
                texts = [str(item) if item is not None else " " for item in raw_input]
            texts = [text.replace("\n", " ") or " " for text in texts]
            vectors = embed_texts(texts)
            self.send_json({
                "object": "list",
                "model": body.get("model") or MODEL_NAME,
                "data": [
                    {"object": "embedding", "index": idx, "embedding": vector}
                    for idx, vector in enumerate(vectors)
                ],
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in texts),
                    "total_tokens": sum(len(text.split()) for text in texts),
                },
            })
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc)}, status=500)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        return json.loads(payload.decode("utf-8")) if payload else {}

    def send_json(self, value: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[nv-embed] {self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    load_model()
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
