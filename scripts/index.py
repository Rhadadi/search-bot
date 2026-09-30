#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

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
