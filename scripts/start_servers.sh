#!/usr/bin/env bash
# Start the two local model servers the bot needs (LFM chat :8081 is expected already running).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${SEARCHBOT_LLAMA_BIN:-$HOME/.local/bin/llama}"
M="${SEARCHBOT_EMBED_MODEL:-$ROOT/models/embeddinggemma-300M-Q8_0.gguf}"
CHATPORT=${SEARCHBOT_CHAT_PORT:-8080}
if ! curl -s --max-time 2 http://127.0.0.1:$CHATPORT/health | grep -q ok; then
  echo "WARN: LFM chat server on :$CHATPORT is not up — start it yourself (the bot will not touch it)"
fi
if curl -s --max-time 2 http://127.0.0.1:8082/health | grep -q ok; then
  echo "embedder :8082 already running"
else
  nohup "$BIN" server -m "$M" --embeddings --pooling mean -b 2048 -ub 2048 \
    --port 8082 --host 127.0.0.1 >| "$ROOT/data/embed8082.log" 2>&1 &
  echo "embedder :8082 starting (pid $!) — log: data/embed8082.log"
fi
