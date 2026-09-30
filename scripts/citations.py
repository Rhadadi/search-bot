#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Backfill citation counts (OpenAlex) so the `citations` ranking weight works.

Usage:
  .venv/bin/python scripts/citations.py                 # every doc, DOI/PMID known
  .venv/bin/python scripts/citations.py ephedra         # one search folder
  .venv/bin/python scripts/citations.py --all-again     # re-fetch (counts drift)
  .venv/bin/python scripts/citations.py --dry-run       # show coverage, no requests

Writes docs.citations / citations_updated / citations_source. Also fills docs.year
from OpenAlex when the sidecar CSV left it empty.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from searchbot import citations, config, db, llm

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("slug", nargs="?", help="only this search folder")
ap.add_argument("--all-again", action="store_true",
                help="re-fetch docs that already have an updated_at stamp")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

c = db.connect()
db.init(c, llm.embed_dim())

total = c.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
have_id = c.execute("SELECT COUNT(*) FROM docs WHERE COALESCE(doi,'')!='' OR COALESCE(pmid,'')!=''").fetchone()[0]
have_ct = c.execute("SELECT COUNT(*) FROM docs WHERE citations IS NOT NULL").fetchone()[0]
pending = c.execute("SELECT COUNT(*) FROM docs WHERE citations_updated IS NULL "
                    "AND (COALESCE(doi,'')!='' OR COALESCE(pmid,'')!='')"
                    + (" AND slug=?" if a.slug else ""),
                    ((a.slug,) if a.slug else ())).fetchone()[0]
print(f"{total} docs | {have_id} with DOI or PMID | {have_ct} already counted | {pending} to fetch")
if a.dry_run:
    for r in c.execute("SELECT id, year, citations, title FROM docs "
                       "WHERE citations IS NOT NULL ORDER BY citations DESC LIMIT 5"):
        print(f"   {r['citations']:>6} cites  {r['year'] or '?':>4}  {(r['title'] or '')[:60]}")
    sys.exit(0)
if not have_id:
    print("nothing to fetch: no doc has a DOI or PMID (index PDFs with a metadata CSV sidecar first)")
    sys.exit(1)

if not config.OPENALEX_MAILTO:
    print("note: SEARCHBOT_MAILTO is unset — OpenAlex polite pool unavailable, "
          "expect a lower rate limit")

stats = citations.fetch_counts(c, slug=a.slug, only_missing=not a.all_again, log=print)
print(f"updated={stats['updated']} not_found={stats['not_found']} "
      f"deferred={stats['deferred']} requests={stats['requests']}")
for r in c.execute("SELECT COUNT(*) n, MIN(citations) lo, MAX(citations) hi, AVG(citations) av "
                   "FROM docs WHERE citations IS NOT NULL"):
    if r["n"]:
        print(f"coverage now {r['n']}/{total} docs | min {r['lo']} max {r['hi']} mean {r['av']:.1f}")
