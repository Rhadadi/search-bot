#!/usr/bin/env python3
"""Index one or all search folders: .venv/bin/python scripts/index.py [slug ...]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from searchbot import config, db, llm, indexer

c = db.connect()
db.init(c, llm.embed_dim())
slugs = sys.argv[1:]
if not slugs:
    slugs = [p.name for p in config.SEARCH_DIR.iterdir() if p.is_dir()]
for s in slugs:
    stats = indexer.index_search(c, s, log=lambda m: print(m, flush=True))
    print(f"[{s}] {stats}", flush=True)
