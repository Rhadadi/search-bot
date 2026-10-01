#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""research_section from the command line: raw evidence for a question, as JSON (or a readable digest).

    .venv/bin/python scripts/research.py epistemology "What is Ryle's regress argument against intellectualism?" \\
        --query "knowing how intellectualism Stanley Williamson" --target sep:knowledge-how --target gutenberg:5827
    .venv/bin/python scripts/research.py epistemology "..." --no-acquire --digest

Targets: doi:10.xxxx/yyy, gutenberg:NUMBER, sep:ENTRY-NAME, ia:IDENTIFIER (Internet Archive, public domain),
url:https://... (a known open copy); details may follow, e.g. gutenberg:5827|year=1912.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from searchbot import agent, db, llm  # noqa: E402


def target(s):
    """doi:..., gutenberg:5827, sep:entry, url:https://...; optional |year=1912|title=...|authors=A; B"""
    head, *extra = s.split("|")
    kind, _, value = head.partition(":")
    if kind not in ("doi", "gutenberg", "sep", "ia", "url") or not value:
        raise argparse.ArgumentTypeError(f"bad target {s!r}")
    t = {kind: value}
    for e in extra:
        k, _, v = e.partition("=")
        t[k.strip()] = [a.strip() for a in v.split(";")] if k.strip() == "authors" else v.strip()
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    ap.add_argument("question")
    ap.add_argument("--query", action="append", default=[], help="a further phrasing to retrieve with (repeatable)")
    ap.add_argument("--target", action="append", default=[], type=target, help="a work to fetch (repeatable)")
    ap.add_argument("--k", type=int, default=agent.RESEARCH_K)
    ap.add_argument("--max-per-work", type=int, default=agent.MAX_PER_WORK)
    ap.add_argument("--no-acquire", action="store_true")
    ap.add_argument("--digest", action="store_true", help="print a readable digest instead of JSON")
    a = ap.parse_args()
    c = db.connect()
    db.init(c, llm.embed_dim())
    res = agent.research_section(c, a.slug, a.question, queries=a.query, k=a.k, acquire=not a.no_acquire,
                                 targets=a.target, max_per_work=a.max_per_work,
                                 log=lambda m: print(m, file=sys.stderr))
    if not a.digest:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    for i, e in enumerate(res["evidence"], 1):
        who = "; ".join(e["authors"]) or "?"
        print(f"[{i}] chunk {e['chunk_id']} · {who} ({e['year']}), {e['title']} · {e['locator'] or '-'} · "
              f"{e['retrieval']} {e['score']}\n    {e['passage'][:700]}\n")
    for r in res["catalogue"]:
        print(f"[catalogue only] {r.get('title')} ({r.get('year')}) doi:{r.get('doi')}")


if __name__ == "__main__":
    main()
