#!/usr/bin/env python3
"""CLI chat with the search-bot RAG engine.
Usage: .venv/bin/python scripts/ask.py <slug> [session_id]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from searchbot import db, llm, pipeline

slug = sys.argv[1] if len(sys.argv) > 1 else "ephedra"
session = sys.argv[2] if len(sys.argv) > 2 else "cli"
c = db.connect()
db.init(c, llm.embed_dim())
print(f"search-bot CLI — slug={slug} session={session} (ctrl-d to exit)")
while True:
    try:
        q = input("\nYou> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        continue
    r = pipeline.answer(c, slug, session, q)
    print("Bot> " + r["answer"])
    if r.get("evidence"):
        print("   --- evidence ---")
        for i, e in enumerate(r["evidence"], 1):
            print(f"   [E{i}] {e['title'] or e['file']} ({e['year'] or ''} {e['journal'] or ''}) {e['pmcid'] or e['doi'] or ''}")
    if r.get("facts_stored"):
        print(f"   (+{r['facts_stored']} facts stored to memory)")
