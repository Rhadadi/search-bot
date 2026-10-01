# search-bot

A grounded scientific RAG bot: it answers **only from indexed research PDFs**, and
every claim is cited to an `[E#]` evidence entry.

The bot talks to **any OpenAI-compatible chat endpoint** — a small local model, a
large one, llama.cpp, vLLM, Ollama, or a hosted API. Nothing in the code assumes a
particular model or vendor. All conversational state lives **outside the model** in
SQLite and is re-injected on every call, so quality depends on the corpus and the
retrieval, not on how much the model can hold in context.
Retrieval is hybrid (vector + BM25 with RRF fusion), and when the local corpus is
too thin the agent can fetch and index more papers on its own.

By default everything runs locally: no cloud API is required, and no data leaves
the machine except outbound catalogue/PDF lookups.

---

## Architecture

```
┌─────────────┐   stdio MCP    ┌──────────────────────┐
│ MCP client  │◄──────────────►│ searchbot/mcp_server │
├─────────────┤   HTTP :8181   └──────────┬───────────┘
│ Web UI      │◄──────────────►            │
│ web/*.html  │                  ┌────────▼───────────┐
└─────────────┘                  │  RAG engine        │
                                 │  searchbot/*.py    │
                                 └──┬──────────┬──────┘
              chat endpoint ◄───────┘          └──► embeddings :8082
    any OpenAI-compatible server       (embeddinggemma-300M Q8, mean pooling)
     (you choose, local or hosted)              │
                       SQLite data/searchbot.db│ chunks + FTS5(BM25) + vec0(embedder dim)
                                              │
                                 libgen-mcp ◄─┘  (LibGen + Anna's Archive +
                                               arXiv / Crossref / PubMed / EuropePMC)
```

Two OpenAI-compatible endpoints back the engine:

| role | default | what it can be |
|---|---|---|
| chat / generation | `:8080` | **any** model behind any OpenAI-compatible `/v1/chat/completions` server |
| embeddings | `:8082` | any `/v1/embeddings` server (dev default: `embeddinggemma-300M-Q8_0` + `--embeddings --pooling mean`) |

The chat model name is **auto-detected** from `/v1/models`, so swapping models
needs no code change — or no config change, just point `SEARCHBOT_CHAT_URL`
somewhere else. Embedding dimension is read back from the first response, so a
different embedder size works too (re-index after switching).

## Pipeline per question

1. **Agentic gate** (`searchbot/agent.py`) — retrieve first. If evidence is thin
   (<3 hits, or top RRF score < 0.02), automatically: search the catalogue via
   libgen-mcp, apply a title-anchor relevance gate, dedup against the `acquired`
   ledger and already-indexed titles, download up to 3 PDFs into `search/{slug}/pdf/`,
   index them, and re-retrieve. Max 2 rounds, then answer from whatever exists.
   Disable per-call with `acquire:false`.
2. **Hybrid retrieval** (`searchbot/retriever.py`) — vec0 KNN (query wrapped in the
   embeddinggemma instruction prefix) + FTS5 BM25 → **RRF fusion** → top 8 chunks.
   Year bounds, recency and citation weighting are optional on top of that
   (see [Ranking signals](#ranking-signals)).
3. **Memory assembly** (`searchbot/memory.py`) — session summary + facts ledger +
   the last 8 raw turns injected as system blocks, so the model never has to
   "remember" anything.
4. **Grounded generation** — the model answers in English citing `[E#]`; `FACT:`
   lines are extracted into a per-topic facts table, then stripped from the display.
5. **Compaction** — roughly every 12 turns, older turns are summarized into the
   session summary.

## Ranking signals

RRF alone is relevance-only. Retrieval accepts four optional signals on top of it
— per request in the web API and the MCP `ask` tool, or as flags in the CLI:

| signal | what it does |
|---|---|
| `year_after` / `year_before` | drop candidates outside a publication-year range (a doc with no recoverable year is excluded when a bound is set, rather than passing as "recent enough") |
| `recency` | additive weight: `0.5 ** (age / SEARCHBOT_RECENCY_HALF_LIFE)` |
| `citations` | additive weight: OpenAlex `cited_by_count`, log-scaled against the strongest candidate |

Both weights default to `0.0`, so an untouched query stays pure RRF and the
ordering is exactly what it was.

```bash
.venv/bin/python scripts/ask.py ephedra --after 2015 --recency 0.2
.venv/bin/python scripts/ask.py ephedra --citations 0.3

curl -s 127.0.0.1:8181/api/ask -H 'Content-Type: application/json' \
     -d '{"slug":"ephedra","question":"…","year_after":2015,"recency":0.2}'
```

`citations` needs counts in the database, so backfill once per corpus (and
refresh occasionally — counts drift):

```bash
.venv/bin/python scripts/citations.py --dry-run     # coverage report, no requests
.venv/bin/python scripts/citations.py ephedra       # one folder
.venv/bin/python scripts/citations.py --all-again   # re-fetch everything
```

That writes `docs.citations` (and `docs.year`, when the metadata sidecar left it
blank) from OpenAlex: 40 identifiers per request, DOI first and PMID for whatever
DOI missed. Works OpenAlex does not know are stamped `openalex:not-found` so a
re-run does not re-query the same dead ends, while a failed request leaves rows
untouched and retries next time. Set `SEARCHBOT_MAILTO` to use OpenAlex's polite
pool; without it the anonymous rate limit applies.

## Evidence only: `research_section`

For when you (or another model) want to do the writing yourself: the MCP tool
`research_section` and `scripts/research.py` return the evidence and stop. No answer
is generated and the chat model is never called, so only an embeddings endpoint is
needed.

```bash
.venv/bin/python scripts/research.py epistemology \
    "What did Russell mean by knowledge by acquaintance?" \
    --query "acquaintance description Russell 1912" \
    --target gutenberg:5827 --target sep:knowledge-acquaindescrip --digest
```

Each evidence item carries the passage, its `locator` (page or section), authors,
year, venue, publisher, DOI/ISBN/PMID/PMCID, URL, licence, citation count, score and
retrieval lane; `works` lists the distinct works. When the evidence is thin (or when
`targets` names works), it acquires **open-access and public-domain** material only
(`searchbot/oa.py`: the Stanford Encyclopedia of Philosophy, Project Gutenberg, arXiv,
Europe PMC, Zenodo, CC-licensed DOIs), indexes it and retrieves again. A DOI with no
open copy is returned as a catalogue record.

For corpora of books, raise the per-document caps (`SEARCHBOT_MAX_CHUNKS=4000`,
`SEARCHBOT_MAX_PAGES=900`). Without llama.cpp, `scripts/embed_server.py` serves
embeddings on the CPU (`pip install fastembed`; set
`SEARCHBOT_QUERY_INSTRUCT="Represent this sentence for searching relevant passages: "`
for its default bge model).

## Agentic trace UI

Every question runs as a background **run** (`searchbot/trace.py`) emitting typed,
timestamped events that the UI renders live:

```
start → retrieve (hits + RRF score + lanes: vec/fts/both)
      → gate (which key terms are covered / missing)
      → acquire_plan → libgen_search → download → index_done (+docs/+chunks)
      → re-retrieve → prompt (evidence N, memory turns)
      → generate → answer streams token-by-token into the bubble
      → memory → done
```

Control tokens are scrubbed from the stream; a `reset` event clears the bubble if
the model needed a retry.

- `POST /api/ask` returns a `run_id` instantly.
- `GET /api/trace?id=&since=n` long-polls for new events (3s) until the run finishes.
- The MCP `ask` tool returns the final result plus the full `agent_events` list.
- `acquire:false` runs the same trace without any catalogue fetching.

## Layout

```
searchbot/            the engine (importable package)
  config.py           ports, budgets, paths
  db.py               schema: docs / chunks / fts5 / vec0 / sessions / turns / facts / jobs
  llm.py              chat client + embedder client (batched, retrying)
  indexer.py          PDF & XML extraction, chunking, embedding
  retriever.py        hybrid vec + BM25, RRF fusion, year/recency/citation ranking
  memory.py           summary compaction, facts ledger, memory block
  pipeline.py         the answer loop (evidence → grounded answer)
  agent.py            the gate/acquire loop
  citations.py        OpenAlex citation-count backfill
  libgen.py           stdio client for libgen-mcp + table parser + term widener
  trace.py            background runs and the event stream
  mcp_server.py       stdio MCP server (JSON-RPC 2.0)
  webserver.py        127.0.0.1:8181 JSON API + static UI
web/index.html        chat UI with evidence panel
scripts/              index.py, ask.py, citations.py, start_servers.sh
tests/                stdlib unittest suite — no model server, no corpus, no network
.github/workflows/    CI: the suite on 3.11 / 3.12 / 3.13
CHANGELOG.md          dated update log
searchbot_mcp.py      standalone MCP entry point for clients that scrub cwd/PYTHONPATH
web_run.py            starts the web UI
```

Not committed (see `.gitignore`): the PDF/XML corpus under `search/`, model weights
in `models/`, the `bin/libgen-mcp` binary, and all runtime state under `data/`.

## Getting started

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. point the bot at a chat model — any OpenAI-compatible server works.
#    e.g. llama.cpp:  llama server -m <your-model> --port 8080 --host 127.0.0.1
#    e.g. Ollama:    export SEARCHBOT_CHAT_URL=http://127.0.0.1:11434/v1
export SEARCHBOT_CHAT_URL=http://127.0.0.1:8080/v1     # optional; this is the default

# 2. bring up an embeddings server on :8082 (embeddinggemma is the reference config;
#    scripts/start_servers.sh is an example, not a requirement)
scripts/start_servers.sh

# 3. drop PDFs into a topic folder and index it
.venv/bin/python scripts/index.py ephedra

# 4. use it
.venv/bin/python web_run.py                   # UI at http://127.0.0.1:8181
.venv/bin/python scripts/ask.py ephedra       # CLI chat
.venv/bin/python searchbot_mcp.py             # MCP server
```

### Adding material

Each topic is `search/{slug}/`. Drop PDFs anywhere inside (`pdf/`, `fulltext_xml/`,
…) — the indexer walks `*.pdf` and `*.xml` recursively, joins metadata from any
`*.csv` sidecar (title / DOI / PMCID / year), chunks at ~1100 chars with overlap,
hard-splits oversized chunks, embeds, and stores. Re-running `index` on a folder
skips already-stored files, so it is safe to run repeatedly.

### Configuration

All via environment variables, no code edits needed:

| variable | default | meaning |
|---|---|---|
| `SEARCHBOT_CHAT_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible chat endpoint |
| `SEARCHBOT_CHAT_MODEL` | *(auto-detect)* | pin a model name instead of detecting from `/v1/models` |
| `SEARCHBOT_EMBED_URL` | `http://127.0.0.1:8082/v1` | OpenAI-compatible embeddings endpoint |
| `SEARCHBOT_EMBED_MODEL` | *(omitted from request)* | model name for multi-model embed servers |
| `SEARCHBOT_QUERY_INSTRUCT` | embeddinggemma `Instruct: …\nQuery: ` | retrieval prefix added to queries only |
| `SEARCHBOT_RECENCY_WEIGHT` | `0.0` | default recency boost for every query |
| `SEARCHBOT_RECENCY_HALF_LIFE` | `10` | years for a paper's recency score to halve |
| `SEARCHBOT_CITATION_WEIGHT` | `0.0` | default citation-count boost for every query |
| `SEARCHBOT_OPENALEX_URL` | `https://api.openalex.org/works` | citation-count source |
| `SEARCHBOT_OPENALEX_BATCH` | `40` | identifiers per request |
| `SEARCHBOT_MAILTO` | *(none)* | contact for OpenAlex's polite pool (higher rate limit) |
| `SEARCHBOT_LIBGEN_BIN` | `bin/libgen-mcp` | path to the libgen-mcp binary |
| `SEARCHBOT_SOCKS` | `socks5h://127.0.0.1:1090` | proxy for catalogue traffic only; `export SEARCHBOT_SOCKS=` to go direct |
| `SEARCHBOT_CHAT_PORT` | `8080` | health-check port used by `start_servers.sh` |
| `SEARCHBOT_LLAMA_BIN` | `~/.local/bin/llama` | llama.cpp binary used by `start_servers.sh` |
| `SEARCHBOT_EMBED_GGUF` | `models/embeddinggemma-300M-Q8_0.gguf` | weights used by `start_servers.sh` |

### Bring your own model

The chat and embedding endpoints are independent and both are plain
OpenAI-compatible HTTP, so nothing has to be local:

```bash
# one server serving both chat and embeddings (Ollama)
export SEARCHBOT_CHAT_URL=http://127.0.0.1:11434/v1
export SEARCHBOT_EMBED_URL=http://127.0.0.1:11434/v1
export SEARCHBOT_EMBED_MODEL=qwen3-embedding:0.6b
export SEARCHBOT_QUERY_INSTRUCT=          # no instruction prefix

# a hosted API
export SEARCHBOT_CHAT_URL=https://api.example.com/v1
export SEARCHBOT_CHAT_MODEL=their-model
export SEARCHBOT_EMBED_URL=https://api.example.com/v1
export SEARCHBOT_EMBED_MODEL=their-embedding-model
export SEARCHBOT_QUERY_INSTRUCT=
```

What the engine sends is the bare minimum of the spec: `{"input": [...]}`
(plus `model` only when you set it) to `/embeddings`, and a standard
`/chat/completions` payload. Vectors are L2-normalized client-side, and the
sqlite-vec column dimension is taken from the first response — so any embedder
size works, but the dimension is fixed per table, so **delete `data/searchbot.db`
and re-index after switching embedders**.

If your embedder wants its own query prefix (bge, e5, gte), put it in
`SEARCHBOT_QUERY_INSTRUCT`; it is applied to queries only, never to indexed passages.

MCP clients register the server by launching `python -m searchbot.mcp_server` from
the project root with a virtualenv that has the deps installed; it exposes
`list_searches`, `add_search`, `ask`, `index_status`, `libgen_search`,
`libgen_download`.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -t tests
```

The suite builds its own throwaway database in a temp directory, swaps the
embedder for a deterministic token-hash vectoriser and stubs `requests` wherever
an HTTP client is involved — so it needs no model server, no corpus and no
network. Coverage: the control-token scrubber and chat retry ladder, the
vec+BM25+RRF fusion and every ranking option, the acquire gate's term coverage,
the OpenAlex backfill (lanes, batching, dead-end stamping), schema migration, the
HTTP API's routing and trace, and the MCP tool schemas plus JSON-RPC loop.

What changed and when is in [CHANGELOG.md](CHANGELOG.md).

## Notes from building this

Measured with the reference setup (llama.cpp chat + embeddinggemma-300M-Q8_0):

- **Pooling matters.** With `embeddinggemma-300M-Q8_0`: `--pooling mean` gives
  0.73 similarity for related text vs 0.42 for unrelated. `rank`/`last` produce
  degenerate zero vectors; `cls` gives no separation (0.81/0.81). Use `mean`.
- **Embeddings need their own server.** A chat llama.cpp server returns 501 for
  `/v1/embeddings` unless launched with `--embeddings`, which is why `:8080` and
  `:8082` are separate. Not a problem if you use Ollama, vLLM, or a hosted API,
  which serve both from one endpoint.
- **sqlite-vec dimension is fixed per table.** It is read from the first embedding
  response, so switching embedders means deleting and re-indexing the corpus.
- **libgen-mcp returns markdown.** `search` output is a markdown table, parsed to
  JSON by `libgen.py`; `results_per_page` accepts only 25/50/100; `download` needs
  `md5`/`doi`/`isbn` plus a `path` confined by `LIBGEN_MCP_ALLOWED_DOWNLOAD_DIRS`.
  Anna's Archive is already federated into the results.
- Catalogue traffic is routed through a SOCKS proxy only in the libgen-mcp
  subprocess — never in the Python process.
- sqlite-vec KNN and FTS5 behaviour verified on Python 3.12.

## Caveats

- PDFs come from third-party catalogues. **You are responsible for the rights to
  whatever you download**, which is why the corpus and the binary are gitignored.
- The web UI binds to `127.0.0.1` only — it has no authentication. Don't expose it.
- Grounding is enforced by the pipeline (citation validation, fact extraction,
  retry/reset) rather than by the model's own judgment, which keeps weak models
  honest and strong models equally cited. `chat_llm()` retries on empty or
  control-token-garbage output; that path exists for tool-trained local GGUFs and
  is a no-op for well-behaved APIs.
- Answer quality tracks the corpus: expect a weak answer when nothing indexed has
  anything to say.

## License

GNU GPL-3.0-only — see [LICENSE](LICENSE). Every source file carries the matching
SPDX header. Downloaded papers, model weights and any other material you put under
`search/` or `models/` stay outside the repository and outside this license; you
are responsible for their terms.
