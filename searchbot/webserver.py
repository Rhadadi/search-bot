# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Local web server: tiny HTML chat UI + JSON API for the search-bot RAG engine.

Endpoints:
  GET  /                     - chat UI
  GET  /api/searches         - list search folders
  POST /api/searches         - {slug} create+index (background job)
  GET  /api/status?slug=     - index jobs
  POST /api/ask              - {slug, question, session_id} -> answer+evidence
  POST /api/libgen/search    - {query, slug?} catalogue search (LibGen/Anna's Archive/...)
  POST /api/libgen/get       - {slug, id} bring a record into the corpus + index
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from . import config, db, indexer, libgen, pipeline, llm, trace as trace_mod

WEB = config.ROOT / "web"
_conn = None
_conn_lock = threading.Lock()


def conn():
    global _conn
    with _conn_lock:
        if _conn is None:
            _conn = db.connect()
            db.init(_conn, llm.embed_dim())
        return _conn


def _slugify(s):
    s = re.sub(r"[^a-z0-9_-]+", "-", s.lower()).strip("-")
    return s[:48] or "untitled"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text, code=200):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._html((WEB / "index.html").read_text(encoding="utf-8"))
        elif u.path == "/api/searches":
            c = conn()
            rows = c.execute("SELECT slug,title,path,created_at FROM searches ORDER BY created_at DESC").fetchall()
            out = []
            for r in rows:
                n_docs = c.execute("SELECT COUNT(*) n FROM docs WHERE slug=?", (r["slug"],)).fetchone()["n"]
                n_ch = c.execute("SELECT COUNT(*) n FROM chunks WHERE slug=?", (r["slug"],)).fetchone()["n"]
                out.append({**dict(r), "docs": n_docs, "chunks": n_ch})
            self._json(out)
        elif u.path == "/api/status":
            slug = parse_qs(u.query).get("slug", [""])[0]
            c = conn()
            if slug:
                rows = c.execute("SELECT * FROM jobs WHERE slug=? ORDER BY id DESC LIMIT 3", (slug,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 5").fetchall()
            self._json([dict(r) for r in rows])
        elif u.path == "/api/history":
            q = parse_qs(u.query)
            sid = q.get("session", [""])[0]
            c = conn()
            rows = c.execute(
                "SELECT role,text,refs_json,created_at FROM turns WHERE session_id=? ORDER BY id LIMIT 200",
                (sid,)).fetchall()
            self._json([{**dict(r)} for r in rows])
        elif u.path == "/api/sessions":
            c = conn()
            rows = c.execute(
                "SELECT s.id, s.slug, s.summary, s.created_at, "
                " (SELECT MIN(t.id) FROM turns t WHERE t.session_id=s.id AND t.role='user') first_turn, "
                " (SELECT COUNT(*) FROM turns t WHERE t.session_id=s.id) turns, "
                " (SELECT COUNT(*) FROM facts f WHERE f.slug=s.slug) facts "
                "FROM sessions s ORDER BY "
                " (SELECT MAX(t.created_at) FROM turns t WHERE t.session_id=s.id) DESC NULLS LAST, "
                " s.created_at DESC LIMIT 50").fetchall()
            out = []
            for r in rows:
                title = ""
                if r["first_turn"]:
                    t = c.execute("SELECT text FROM turns WHERE id=?", (r["first_turn"],)).fetchone()
                    title = (t["text"][:60] + ("…" if len(t["text"]) > 60 else "")) if t else ""
                out.append({"id": r["id"], "slug": r["slug"], "title": title,
                            "turns": r["turns"], "facts": r["facts"],
                            "has_summary": bool(r["summary"]),
                            "created_at": r["created_at"]})
            self._json(out)
        elif u.path == "/api/trace":
            q = parse_qs(u.query)
            rid = q.get("id", [""])[0]
            since = int(q.get("since", ["0"])[0] or 0)
            tr = trace_mod.RUNS.get(rid)
            if not tr:
                self._json({"error": "unknown run"}, 404)
                return
            self._json(tr.wait(since, timeout=3.0))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        try:
            if u.path == "/api/searches":
                slug = _slugify(b.get("slug", ""))
                d = config.SEARCH_DIR / slug
                d.mkdir(parents=True, exist_ok=True)
                (d / "pdf").mkdir(exist_ok=True)
                c = conn()
                db.upsert_search(c, slug, b.get("title", slug), str(d))
                job = db.job_set(c, slug, "index", "queued")

                def run():
                    try:
                        stats = indexer.index_search(c, slug, job_id=job)
                        db.job_update(c, job, "done", json.dumps(stats))
                    except Exception as e:
                        db.job_update(c, job, "error", str(e))
                threading.Thread(target=run, daemon=True).start()
                self._json({"slug": slug, "path": str(d), "job": job})
            elif u.path == "/api/ask":
                from . import agent
                rid = b.get("run_id") or trace_mod.new_run().id
                tr = trace_mod.RUNS[rid]

                def work(tr=tr):
                    import time as _t
                    from . import retriever
                    t0 = _t.time()
                    events = []
                    rank = retriever.rank_opts(b)
                    try:
                        if b.get("acquire", True):
                            res = agent.ask_agentic(conn(), b["slug"],
                                                    b.get("session_id") or "web",
                                                    b["question"], topk=b.get("topk"),
                                                    log=lambda m: events.append(m),
                                                    trace=tr, rank=rank)
                            res["agent_events"] = events
                        else:
                            res = agent.ask_agentic(conn(), b["slug"],
                                                    b.get("session_id") or "web",
                                                    b["question"], topk=b.get("topk"),
                                                    log=lambda m: events.append(m),
                                                    trace=tr, acquire=False, rank=rank)
                            res["agent_events"] = events
                        tr.emit("done", "completed",
                                ms=int((_t.time() - t0) * 1000), count=len(tr.events))
                        tr.finish(res)
                    except Exception as e:
                        tr.emit("error", str(e)[:300])
                        tr.finish({"error": str(e)[:300]})
                threading.Thread(target=work, daemon=True).start()
                self._json({"run_id": rid, "status": "started"})
            elif u.path == "/api/libgen/search":
                self._json(libgen.get_client().search(b["query"], limit=int(b.get("limit", 10))))
            elif u.path == "/api/libgen/get":
                ident = {k: b.get(k) for k in ("md5", "doi", "isbn") if b.get(k)}
                if not ident:
                    self._json({"error": "need md5, doi or isbn"}, 400)
                    return
                slug = b["slug"]
                c = conn()
                job = db.job_set(c, slug, "libgen_get", "queued", str(ident))
                def run_fetch():
                    try:
                        dest = config.SEARCH_DIR / slug / "pdf"
                        dest.mkdir(parents=True, exist_ok=True)
                        dl = libgen.get_client().download(dest_dir=dest, **ident)
                        path = dl.get("path") or dl.get("file") or dl.get("filename") or ""
                        if path and not dl.get("error"):
                            db.job_update(c, job, "indexing", f"saved {Path(path).name}")
                            stats = indexer.index_search(c, slug, job_id=job)
                            db.job_update(c, job, "done", json.dumps(stats))
                        else:
                            db.job_update(c, job, "error", json.dumps(dl)[:300])
                    except Exception as e:
                        db.job_update(c, job, "error", str(e)[:300])
                threading.Thread(target=run_fetch, daemon=True).start()
                self._json({"job": job, "message": "download+index started; poll /api/status"})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)


def main(port=8181):
    print(f"search-bot web on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8181)
