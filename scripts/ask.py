#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""CLI chat with the search-bot RAG engine.
Usage: .venv/bin/python scripts/ask.py <slug> [session_id] [--after YEAR] [--before YEAR]
       [--recency W] [--citations W]
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from searchbot import db, llm, pipeline, retriever

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("slug", nargs="?", default="ephedra")
ap.add_argument("session", nargs="?", default="cli")
ap.add_argument("--after", type=int, dest="year_after", help="only evidence from this year on")
ap.add_argument("--before", type=int, dest="year_before", help="only evidence up to this year")
ap.add_argument("--recency", type=float, help="recency boost weight (default: config)")
ap.add_argument("--citations", type=float, help="citation-count boost weight (default: config)")
a = ap.parse_args()

slug, session = a.slug, a.session
rank = {k: v for k, v in (("year_after", a.year_after), ("year_before", a.year_before),
                          ("recency", a.recency), ("citations", a.citations)) if v is not None}
c = db.connect()
db.init(c, llm.embed_dim())
print(f"search-bot CLI — slug={slug} session={session} (ctrl-d to exit)")
if rank:
    print("   ranking:", ", ".join(f"{k}={v}" for k, v in rank.items()))
while True:
    try:
        q = input("\nYou> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        continue
    r = pipeline.answer(c, slug, session, q, rank=rank)
    print("Bot> " + r["answer"])
    if r.get("evidence"):
        print("   --- evidence ---")
        for i, e in enumerate(r["evidence"], 1):
            print(f"   [E{i}] {e['title'] or e['file']} ({e['year'] or ''} {e['journal'] or ''}) {e['pmcid'] or e['doi'] or ''}")
    if r.get("facts_stored"):
        print(f"   (+{r['facts_stored']} facts stored to memory)")
