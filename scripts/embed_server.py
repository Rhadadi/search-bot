#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""A small OpenAI-compatible /v1/embeddings server on the CPU, for machines without llama.cpp
(a cloud container, say). Uses fastembed (ONNX); pip install fastembed.

    .venv/bin/python scripts/embed_server.py                       # BAAI/bge-base-en-v1.5 on :8082
    .venv/bin/python scripts/embed_server.py --model BAAI/bge-small-en-v1.5 --port 8083

bge models expect a prefix on queries only, so run the engine with
    export SEARCHBOT_QUERY_INSTRUCT="Represent this sentence for searching relevant passages: "
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastembed import TextEmbedding


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    args = ap.parse_args()
    model = TextEmbedding(args.model)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [{"id": args.model, "object": "model"}]})
            if self.path.rstrip("/") in ("/health", ""):
                return self._send(200, {"status": "ok"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/embeddings", "/embeddings"):
                return self._send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            texts = req.get("input", [])
            texts = [texts] if isinstance(texts, str) else texts
            vecs = list(model.embed(texts, batch_size=32))
            self._send(200, {"object": "list", "model": args.model,
                             "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)]})

        def log_message(self, *a):
            pass

    print(f"embeddings: {args.model} on http://{args.host}:{args.port}/v1", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
