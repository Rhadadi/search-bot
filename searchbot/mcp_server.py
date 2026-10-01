# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Stdio MCP server exposing the search-bot RAG engine to any MCP client.

Tools:
  list_searches            - existing search folders + corpus stats
  add_search               - create search/{slug} folder and index every PDF/XML in it
  ask                      - grounded RAG answer for one search (uses external memory)
  research_section         - raw evidence for a question (passages + full metadata), no generated answer
  index_status             - indexing job progress
  libgen_search            - federated catalogue search (LibGen + Anna's Archive/arXiv/PubMed)
  libgen_download          - download a record's file into the search folder, then index it
"""
import json
import sys
import threading
from . import config, db, indexer, libgen, memory, pipeline, llm

TOOLS = [
    {"name": "list_searches", "description": "List all search folders and corpus stats.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "add_search", "description": "Create search/{slug} and index all PDF/XML files inside (embed + FTS). Idempotent.",
     "inputSchema": {"type": "object", "properties": {
         "slug": {"type": "string", "description": "folder name, e.g. ephedra"},
         "title": {"type": "string"}}, "required": ["slug"]}},
    {"name": "ask", "description": "Answer a scientific question grounded in the indexed corpus of one search folder. If evidence is thin, AUTOMATICALLY searches the catalogue (LibGen/Anna's Archive/PubMed), downloads and indexes the best papers, then answers. Cites evidence. Memory per session.",
     "inputSchema": {"type": "object", "properties": {
         "slug": {"type": "string"}, "question": {"type": "string"},
         "session_id": {"type": "string", "description": "stable id to keep conversation memory"},
         "acquire": {"type": "boolean", "description": "auto-download more papers when evidence is thin (default true)"},
         "year_after": {"type": "integer", "description": "ignore evidence published before this year"},
         "year_before": {"type": "integer", "description": "ignore evidence published after this year"},
         "recency": {"type": "number", "description": "weight of a half-life recency boost added to RRF (0 = off)"},
         "citations": {"type": "number", "description": "weight of a log-scaled OpenAlex citation-count boost (0 = off; run scripts/citations.py first)"}},
         "required": ["slug", "question"]}},
    {"name": "research_section", "description": "Return RAW scholarly evidence for a question, for the caller to synthesise: "
     "retrieved passages with full metadata (authors, year, venue, DOI/ISBN/PMID, locator such as page or section, "
     "retrieval score and lane), at most a few per work, plus a list of the works used. Never generates an answer. "
     "If evidence is thin and acquire=true, fetches OPEN-ACCESS / public-domain works (SEP, Project Gutenberg, arXiv, "
     "Europe PMC, Zenodo, CC-licensed DOIs), indexes them and retrieves again. Targets name specific works to fetch.",
     "inputSchema": {"type": "object", "properties": {
         "slug": {"type": "string", "description": "search folder, e.g. epistemology"},
         "question": {"type": "string"},
         "queries": {"type": "array", "items": {"type": "string"}, "description": "further phrasings to retrieve with"},
         "acquire": {"type": "boolean", "description": "fetch open-access works when evidence is thin (default true)"},
         "targets": {"type": "array", "items": {"type": "object"}, "description":
                     "works to fetch: {doi} | {gutenberg: number} | {sep: entry name} | {url, title, authors, year, license}"},
         "k": {"type": "integer", "description": "passages to return (default 15)"},
         "max_per_work": {"type": "integer", "description": "passages from any one work (default 3)"},
         "year_after": {"type": "integer"}, "year_before": {"type": "integer"},
         "recency": {"type": "number"}, "citations": {"type": "number"}},
         "required": ["slug", "question"]}},
    {"name": "index_status", "description": "Indexing jobs status for a slug.",
     "inputSchema": {"type": "object", "properties": {"slug": {"type": "string"}}}},
    {"name": "libgen_search", "description": "Search Library Genesis + Anna's Archive + arXiv/PubMed/Crossref catalogue.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}},
    {"name": "libgen_download", "description": "Download a catalogue record (md5/doi/isbn) into search/{slug}/pdf/ and index it.",
     "inputSchema": {"type": "object", "properties": {
         "slug": {"type": "string"}, "md5": {"type": "string"},
         "doi": {"type": "string"}, "isbn": {"type": "string"}},
         "required": ["slug"]}},
]


def _conn():
    c = db.connect()
    db.init(c, llm.embed_dim())
    return c


def _dispatch(name, args):
    c = _conn()
    if name == "list_searches":
        rows = c.execute("SELECT slug,title,path,created_at FROM searches ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            n_docs = c.execute("SELECT COUNT(*) n FROM docs WHERE slug=?", (r["slug"],)).fetchone()["n"]
            n_ch = c.execute("SELECT COUNT(*) n FROM chunks WHERE slug=?", (r["slug"],)).fetchone()["n"]
            out.append({**dict(r), "docs": n_docs, "chunks": n_ch})
        return out
    if name == "add_search":
        slug = args["slug"]
        d = config.SEARCH_DIR / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "pdf").mkdir(exist_ok=True)
        job = db.job_set(c, slug, "index", "queued")

        def run():
            try:
                stats = indexer.index_search(c, slug, job_id=job)
                db.job_update(c, job, "done", json.dumps(stats))
            except Exception as e:
                db.job_update(c, job, "error", str(e))
        threading.Thread(target=run, daemon=True).start()
        return {"slug": slug, "path": str(d), "job": job,
                "message": "indexing started in background; poll index_status"}
    if name == "ask":
        from . import agent, retriever
        events = []
        rank = retriever.rank_opts(args)
        res = agent.ask_agentic(c, args["slug"], args.get("session_id", "mcp-default"),
                                args["question"],
                                log=lambda m: events.append(m), rank=rank) \
            if args.get("acquire", True) \
            else pipeline.answer(c, args["slug"], args.get("session_id", "mcp-default"),
                                 args["question"], rank=rank)
        res["agent_events"] = events
        return res
    if name == "research_section":
        from . import agent, retriever
        return agent.research_section(c, args["slug"], args["question"], queries=args.get("queries"),
                                      k=int(args.get("k", agent.RESEARCH_K)), acquire=args.get("acquire", True),
                                      targets=args.get("targets"), max_per_work=int(args.get("max_per_work", agent.MAX_PER_WORK)),
                                      rank=retriever.rank_opts(args))
    if name == "index_status":
        q = "SELECT id,slug,kind,state,info,updated_at FROM jobs"
        p = ()
        if args.get("slug"):
            q += " WHERE slug=?"; p = (args["slug"],)
        return [dict(r) for r in c.execute(q + " ORDER BY id DESC LIMIT 5", p)]
    if name == "libgen_search":
        return libgen.get_client().search(args["query"], limit=int(args.get("limit", 10)))
    if name == "libgen_download":
        ident = {k: args.get(k) for k in ("md5", "doi", "isbn") if args.get(k)}
        dest = config.SEARCH_DIR / args["slug"] / "pdf"
        dest.mkdir(parents=True, exist_ok=True)
        dl = libgen.get_client().download(dest_dir=dest, **ident)
        path = dl.get("path") or dl.get("file") or dl.get("filename") or ""
        if path and not dl.get("error"):
            stats = indexer.index_search(c, args["slug"])
            return {"saved": str(path), "indexed": stats}
        return dl
    raise ValueError(f"unknown tool {name}")


def _send(msg):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid, method = req.get("id"), req.get("method")
        try:
            if method == "initialize":
                _send({"jsonrpc": "2.0", "id": rid, "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "searchbot-rag", "version": "1.0"}}})
            elif method == "tools/list":
                _send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
            elif method == "tools/call":
                try:
                    result = _dispatch(req["params"]["name"], req["params"].get("arguments", {}))
                    _send({"jsonrpc": "2.0", "id": rid, "result": {
                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=1)}]}})
                except Exception as e:
                    _send({"jsonrpc": "2.0", "id": rid, "result": {
                        "content": [{"type": "text", "text": f"ERROR: {e}"}], "isError": True}})
            elif method.startswith("notifications/"):
                pass
            elif rid is not None:
                _send({"jsonrpc": "2.0", "id": rid,
                       "error": {"code": -32601, "message": f"unhandled {method}"}})
        except BrokenPipeError:
            break


if __name__ == "__main__":
    serve()
