# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""JSON-RPC stdio client for bin/libgen-mcp (search/get_details/download/read).

Real v2.0.1 tool schemas:
  search        : {query, topics[], search_in[], results_per_page, page, order, order_mode, extra_sources}
  get_details   : {md5 | id | doi, object, enrich}
  download      : {md5 | doi | isbn, path, filename, source, resolve_only}
  read          : {md5 | doi | path, source, pages}
Downloads are confined by the server to LIBGEN_MCP_ALLOWED_DOWNLOAD_DIRS (we set it).

Also used as a keyword widener: suggest_terms() asks libgen (which federates
Anna's Archive, arXiv, Crossref, PubMed, Europe PMC...) what the catalogue
calls the user's topic; those words improve BM25/vector recall.
"""
import json
import os
import subprocess
import threading
from . import config


class LibgenMCP:
    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()
        self._id = 0
        self.tools = None

    def start(self):
        if self._proc and self._proc.poll() is None:
            return
        config.LIBGEN_DOWNLOADS.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("LIBGEN_MCP_ALLOWED_DOWNLOAD_DIRS",
                       os.pathsep.join([str(config.DATA_DIR), str(config.SEARCH_DIR)]))
        # route the Go binary's outbound through smart-dns SOCKS (libgen mirrors
        # are ISP-filtered). Only this subprocess — never the Python side.
        env["HTTPS_PROXY"] = os.environ.get("SEARCHBOT_SOCKS", "socks5h://127.0.0.1:1090")
        env["HTTP_PROXY"] = env["HTTPS_PROXY"]
        env.setdefault("LIBGEN_MCP_EXTRA_SOURCES", "auto")
        self._proc = subprocess.Popen(
            [config.LIBGEN_BIN, "-download-dir", str(config.LIBGEN_DOWNLOADS),
             "-allow-private-addresses", "false"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env)
        self._rpc("initialize", {"protocolVersion": "2024-11-05",
                                 "capabilities": {},
                                 "clientInfo": {"name": "searchbot", "version": "1.0"}})
        self._notify("notifications/initialized", {})
        self.tools = [t["name"] for t in self._rpc("tools/list", {})["tools"]]

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    def _write(self, obj):
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _notify(self, method, params):
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                self.start()
            self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _rpc(self, method, params, timeout=120):
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                self.start()
            self._id += 1
            mid = self._id
            self._write({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("libgen-mcp died")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == mid:
                    if "error" in msg:
                        raise RuntimeError(msg["error"].get("message", "rpc error"))
                    return msg["result"]

    def call_tool(self, name, arguments, timeout=180):
        res = self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout)
        texts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
        payload = "\n".join(texts)
        try:
            parsed = json.loads(payload)
        except Exception:
            return {"raw": payload}
        if res.get("isError"):
            return {"error": parsed}
        return parsed

    def search(self, query, limit=10, topics=None):
        # server accepts only 25/50/100 per page; request smallest allowed, trim to limit
        rpp = 25 if int(limit) <= 25 else (50 if int(limit) <= 50 else 100)
        args = {"query": query, "results_per_page": rpp}
        if topics:
            args["topics"] = topics
        res = self.call_tool("search", args)
        items = _iter_results(res)
        if items:
            return {"results": items[:int(limit)], "count": len(items[:int(limit)]),
                    "summary": (res.get("raw") or "").splitlines()[0] if isinstance(res, dict) else ""}
        return res

    def get_details(self, **ident):
        return self.call_tool("get_details", ident)

    def download(self, dest_dir=None, **ident):
        """ident: md5=... | doi=... | isbn=... Returns dict with 'path' parsed
        from the server's markdown reply ('- **Path**: `file`')."""
        args = {k: v for k, v in ident.items() if v}
        args["path"] = str(dest_dir or config.LIBGEN_DOWNLOADS)
        res = self.call_tool("download", args)
        if isinstance(res, dict) and "raw" in res:
            raw = str(res["raw"])
            import re as _re
            m = _re.search(r"\*\*Path\*\*:\s*`?([^`\n]+)`?", raw)
            ok = "Downloaded" in raw and m
            return {"path": m.group(1).strip() if m else "",
                    "error": None if ok else raw[:300]}
        return res

    def read(self, pages="1-5", **ident):
        args = {k: v for k, v in ident.items() if v}
        args["pages"] = pages
        return self.call_tool("read", args)


_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            _client = LibgenMCP()
            _client.start()
    return _client


def _parse_md_table(text: str):
    """libgen-mcp returns search results as a markdown table; parse to dicts."""
    rows = []
    header = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = [c.lower() for c in cells]
            continue
        if set("".join(cells)) <= set("-: "):  # separator row
            continue
        if len(cells) != len(header):
            continue
        d = dict(zip(header, cells))
        title = d.get("title", "")
        ident = d.get("identifier", "")
        md5 = ""
        m = __import__("re").search(r"md5[:=]([0-9a-f]{32})", ident + " " + d.get("download links", ""))
        if m:
            md5 = m.group(1)
        rows.append({"title": title, "authors": d.get("authors", ""),
                     "year": d.get("year", ""), "ext": d.get("ext", ""),
                     "size": d.get("size", ""), "identifier": ident,
                     "md5": md5, "links": d.get("download links", "")})
    return rows


def _iter_results(res):
    """Normalize search results across catalogue/openalex shapes."""
    if isinstance(res, list):
        return res
    if not isinstance(res, dict):
        return []
    if "raw" in res and "|" in str(res.get("raw", "")):
        return _parse_md_table(str(res["raw"]))
    for key in ("results", "data", "items", "works", "records", "response"):
        v = res.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return _iter_results(v)
    return []


def suggest_terms(question: str) -> str:
    """Expand a natural-language question with catalogue terminology (best-effort)."""
    try:
        # catalogue search matches titles: query with content words, not the sentence
        import re as _re
        from .retriever import _STOP
        kw = [w for w in _re.sub(r"[^a-zA-Z0-9 -]", "", question.lower()).split()
              if len(w) > 3 and w not in _STOP]
        if not kw:
            return ""
        res = {}
        titles = []
        for n in (3, 2, 1):
            res = get_client().search(" ".join(kw[:n]), limit=5)
            for it in _iter_results(res)[:3]:
                t = it.get("title")
                if isinstance(t, dict):
                    t = t.get("value") or t.get("display_name") or ""
                if t:
                    titles.append(str(t))
            if titles:
                break
        if not titles:
            return ""
        import re
        from .retriever import _STOP
        seen, out = set(), []
        for w in " ".join(titles).lower().replace("&", " ").split():
            w = re.sub(r"[^a-z0-9-]", "", w)
            if len(w) > 3 and w not in _STOP and w not in seen:
                seen.add(w)
                out.append(w)
        return " ".join(out[:10])
    except Exception:
        return ""
