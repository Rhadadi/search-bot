# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Backfill OpenAlex citation counts into docs.citations.

OpenAlex is queried in batches of config.OPENALEX_BATCH identifiers, DOI first
and PMID for whatever DOI missed. `citations_updated` is stamped even when a work
is not found, so a re-run does not keep re-querying the same dead ends; unknown
counts stay NULL and simply contribute nothing to ranking.
"""
import datetime
import time
import requests
from . import config, db

_SELECT = "id,ids,doi,cited_by_count,publication_year"


def _headers():
    ua = "search-bot/1.0"
    if config.OPENALEX_MAILTO:
        ua += f" (mailto:{config.OPENALEX_MAILTO})"
    return {"User-Agent": ua}


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _query(field, values):
    """Return {value: work} for one batched filter request."""
    if not values:
        return {}
    r = requests.get(config.OPENALEX_URL,
                     params={"filter": f"{field}:" + "|".join(values),
                             "select": _SELECT, "per-page": max(len(values), 1)},
                     headers=_headers(), timeout=60)
    r.raise_for_status()
    out = {}
    for w in r.json().get("results", []):
        keys = set()
        doi = w.get("doi") or ""
        if doi:
            keys.add(doi.replace("https://doi.org/", "").lower())
        pmid = (w.get("ids") or {}).get("pmid") or ""
        if pmid:
            keys.add(pmid.rsplit("/", 1)[-1])
        for k in keys:
            out[k] = w
    return out


def _work_year(w):
    y = w.get("publication_year")
    return str(y) if y else ""


def fetch_counts(c, slug=None, only_missing=True, log=lambda m: None, sleep=0.1):
    """Fill docs.citations (+ year when missing). Returns stats dict."""
    sql = ("SELECT id, doi, pmid, year FROM docs "
           "WHERE ((doi IS NOT NULL AND doi != '') OR (pmid IS NOT NULL AND pmid != ''))")
    args = ()
    if slug:
        sql += " AND slug=?"
        args = (slug,)
    if only_missing:
        sql += " AND citations_updated IS NULL"
    rows = c.execute(sql, args).fetchall()

    stats = {"considered": len(rows), "updated": 0, "not_found": 0,
             "deferred": 0, "requests": 0}
    if not rows:
        return stats

    by_doi = {d["id"]: d for d in rows if (d["doi"] or "").strip()}
    by_pmid = {d["id"]: d for d in rows if not (d["doi"] or "").strip() and (d["pmid"] or "").strip()}

    found = {}
    # A failed batch is not a missing work: those docs stay unstamped so the
    # next run retries them instead of keeping a bogus "not-found" forever.
    deferred = set()
    for group, field, index in ((by_doi, "doi", lambda d: d["doi"].strip().lower()),
                                (by_pmid, "pmid", lambda d: str(d["pmid"]).strip())):
        ids = sorted(group, key=lambda i: index(group[i]))
        for batch in _chunks(ids, config.OPENALEX_BATCH):
            vals = sorted({index(group[i]) for i in batch})
            try:
                res = _query(field, vals)
                stats["requests"] += 1
            except Exception as e:
                log(f"openalex {field} batch failed: {e}")
                deferred.update(batch)
                continue
            for i in batch:
                w = res.get(index(group[i]))
                if w:
                    found[i] = w
            time.sleep(sleep)

    now = datetime.datetime.now().isoformat(timespec="seconds")
    for doc_id, d in list(by_doi.items()) + list(by_pmid.items()):
        if doc_id in deferred:
            stats["deferred"] += 1
            continue
        w = found.get(doc_id)
        if w:
            c.execute("UPDATE docs SET citations=?, citations_updated=?, citations_source=?, "
                      "year=COALESCE(NULLIF(year,''),?) WHERE id=?",
                      (int(w.get("cited_by_count") or 0), now, "openalex",
                       _work_year(w), doc_id))
            stats["updated"] += 1
        else:
            # stamp it so a later run does not re-query a known dead end
            c.execute("UPDATE docs SET citations_updated=?, citations_source=? WHERE id=?",
                      (now, "openalex:not-found", doc_id))
            stats["not_found"] += 1
    c.commit()
    log(f"citations: {stats['updated']} filled, {stats['not_found']} unmatched "
        f"in {stats['requests']} request(s)")
    return stats
