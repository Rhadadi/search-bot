# Update log

Change history for search-bot, newest first. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is not
version-tagged, so entries are dated.

## 2026-10-01 — evidence-only research, open-access acquisition, books

### Added
- **`research_section`** (MCP tool, `agent.research_section`, `scripts/research.py`) —
  raw evidence for a question, for the caller to synthesise: the retrieved passages
  with full metadata (authors, year, venue, publisher, DOI, ISBN, PMID/PMCID, URL,
  licence, locator, citation count, score, and whether vector, BM25 or both found
  it), at most `max_per_work` passages per work, and a list of the works used. It
  never calls the language model. `queries` adds further phrasings (retrieved and
  merged at each passage's best score); `targets` names works to fetch
  (`{doi}`, `{gutenberg}`, `{sep}`, `{ia}` for an Internet Archive item read through
  its OCR text, `{url}`). A DOI with no open copy comes back
  as a catalogue record, so the work can still be cited from verified details.
- **Open-access acquisition** (`searchbot/oa.py`) — used by `research_section`:
  Stanford Encyclopedia of Philosophy entries (cited by their archive edition),
  Project Gutenberg public-domain books (through its catalogue file), arXiv,
  Europe PMC open-access full texts, Zenodo open records and Creative
  Commons-licensed DOIs. Files go to `search/{slug}/oa/{source}/`, their metadata
  to `search/{slug}/oa_metadata.csv`. No shadow libraries.
- **Books and web pages** in the indexer: HTML (split by heading, with an
  encyclopedia entry's main text and bibliography picked out) and plain text
  (split by chapter, Gutenberg's licence header and footer removed), besides PDF
  and XML.
- **Locators** — each chunk records where it is in its work: `PDF p. 12` or
  `PDF pp. 12–13`, `§ 2.1 Intellectualism`, `Chapter V. Knowledge By Acquaintance…`
  (`chunks.locator`). New `docs.isbn` and `docs.publisher` columns, read from the
  metadata sidecar. Existing databases are migrated in place.
- `SEARCHBOT_MAX_CHUNKS` and `SEARCHBOT_MAX_PAGES` lift the 40-chunk / 60-page
  caps for corpora of books (defaults unchanged).
- `scripts/embed_server.py` — a CPU OpenAI-compatible `/v1/embeddings` server
  (fastembed, `BAAI/bge-base-en-v1.5` by default) for machines without llama.cpp.
- 17 tests (`tests/test_research.py`), offline as before: the evidence shape and
  the per-work cap, multi-query retrieval, when acquisition does and does not
  run, locators for books, web pages and PDFs, sidecar metadata, the SEP citation
  parser, catalogue-only DOIs, and that this path never reaches the language model
  or LibGen.

### Unchanged
- `ask` and its acquisition behave exactly as before.

## 2026-09-30 — license, test suite, ranking signals

### Added
- **Ranking signals** (`searchbot/retriever.py`) — `year_after` / `year_before`
  publication-year bounds, a half-life `recency` boost and a log-scaled
  `citations` boost, all additive on top of RRF. Set per request via the CLI
  (`--after/--before/--recency/--citations`), the `POST /api/ask` body or the MCP
  `ask` tool; defaults live in `SEARCHBOT_RECENCY_WEIGHT`,
  `SEARCHBOT_RECENCY_HALF_LIFE` and `SEARCHBOT_CITATION_WEIGHT`, all `0.0`, so an
  untouched query stays pure RRF and ordering is unchanged.
- **Citation backfill** (`searchbot/citations.py`, `scripts/citations.py`) —
  OpenAlex `cited_by_count` into `docs.citations`, plus `docs.year` when the
  metadata sidecar left it blank. 40 identifiers per request, DOI lane first and
  PMID for whatever DOI missed; unknown works are stamped `openalex:not-found` so
  re-runs skip them, while a failed request leaves rows unstamped and retries.
- `db.migrate()` adds the new `docs` columns to an existing corpus in place — no
  re-indexing required.
- **Test suite** (`tests/`) — 111 `unittest` tests with a throwaway database, a
  deterministic token-hash embedder and stubbed `requests`: no model server, no
  corpus, no network. Covers the control-token scrubber and chat retry ladder,
  RRF fusion and every ranking option, the acquire gate, the OpenAlex lanes and
  batching, schema migration, HTTP routing and trace, MCP schemas and the
  JSON-RPC loop.
- **CI** (`.github/workflows/ci.yml`) — the suite on Python 3.11, 3.12 and 3.13.
- `LICENSE` (GNU GPL-3.0-only) and a short SPDX notice in every source file.
- `requirements.txt`.
- README: ranking signals, citation backfill, configuration table for the new
  variables, development and license sections.

### Fixed
- `POST /api/ask` with `acquire:false` raised `NameError` on an unbound `events`
  list before the answer thread could start.
- The citation backfill query let `OR` precedence swallow the `slug` and
  `only_missing` filters, so a per-folder run considered the whole corpus.
- The not-found stamping `UPDATE` passed its third parameter outside the params
  tuple, so every unmatched doc crashed the run.
- Generic research-speak ("effects", "study", "main") leaked back into catalogue
  queries through the keyword fallback, which triggered downloads on noise.

## 2026-09-30 — first public release

### Added
- Grounded RAG engine over a local PDF/XML corpus: hybrid retrieval (sqlite-vec
  KNN + FTS5 BM25 fused with RRF), citation-forced `[E#]` answers, `FACT:`
  extraction into a per-topic ledger, session summary compaction.
- Agentic acquire loop: when evidence is thin, search the catalogue through
  libgen-mcp, gate by title relevance, dedup against the `acquired` ledger,
  download and index up to 3 papers, re-retrieve (max 2 rounds).
- Live agent trace (`searchbot/trace.py`) rendered by the web UI, and the same
  event list returned by the MCP `ask` tool.
- Web UI + JSON API on `127.0.0.1:8181`, CLI chat, and a stdio MCP server.
- `.gitignore` keeping the corpus, model weights, the libgen-mcp binary and all
  runtime state out of the repository.

### Changed
- Both model endpoints are swappable and vendor-neutral: the chat model name is
  auto-detected from `/v1/models` (or pinned with `SEARCHBOT_CHAT_MODEL`), the
  embeddings request is spec-minimal (`{"input": [...]}`, `model` only when
  configured), vectors are L2-normalized client-side, and the sqlite-vec column
  dimension is read from the first embedding response — so any
  OpenAI-compatible server, local or hosted, works.
