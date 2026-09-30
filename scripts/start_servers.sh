#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

# Start the local embedder the bot needs. The chat server is yours to run and choose.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${SEARCHBOT_LLAMA_BIN:-$HOME/.local/bin/llama}"
M="${SEARCHBOT_EMBED_GGUF:-$ROOT/models/embeddinggemma-300M-Q8_0.gguf}"
CHATPORT=${SEARCHBOT_CHAT_PORT:-8080}
if ! curl -s --max-time 2 http://127.0.0.1:$CHATPORT/health | grep -q ok; then
  echo "WARN: no chat server on :$CHATPORT — start any OpenAI-compatible server, or set SEARCHBOT_CHAT_URL"
fi
if curl -s --max-time 2 http://127.0.0.1:8082/health | grep -q ok; then
  echo "embedder :8082 already running"
else
  nohup "$BIN" server -m "$M" --embeddings --pooling mean -b 2048 -ub 2048 \
    --port 8082 --host 127.0.0.1 >| "$ROOT/data/embed8082.log" 2>&1 &
  echo "embedder :8082 starting (pid $!) — log: data/embed8082.log"
fi
