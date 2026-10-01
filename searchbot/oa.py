# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Open-access acquisition: find works that are free and legal to download, fetch them
into search/{slug}/oa/, record their metadata in search/{slug}/oa_metadata.csv, and index.

Sources, all of them public domain, open access, or free to read online:
  sep        Stanford Encyclopedia of Philosophy entries (cited by their archive edition)
  gutenberg  Project Gutenberg public-domain books (searched through its catalogue file)
  arxiv      arXiv preprints
  europepmc  Europe PMC open-access full texts
  zenodo     Zenodo records with open files
  crossref   no full texts; used to check a DOI's details, and for works with a Creative
             Commons licence and a full-text link

A request can also name targets explicitly (a DOI, a Gutenberg book number, an SEP entry, or a
URL the caller knows to be an open copy), which is how a researcher asks for a specific work.
A DOI with no open copy is not downloaded; its catalogue record is returned instead, so the
work can still be cited from its verified details.

Shadow libraries are never used here.
"""
import csv
import hashlib
import io
import re
import time
import urllib.parse
from pathlib import Path

import requests
from lxml import etree
from lxml import html as lhtml

from . import config

SEP = "https://plato.stanford.edu"
GUTENBERG_CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
CSV_FIELDS = ["file", "title", "authors", "year", "journal", "publisher", "doi", "isbn", "link", "license"]
STOP = set("""a an the of and or to in on for with by from what is are was were be been how why which who whom
whose does do did can could should would may might must this that these those its their his her our your
about into than then there here not no as at it any all some more most other such between within against""".split())

_session = None


def session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = config.OA_USER_AGENT
    return _session


def _get(url, **kw):
    kw.setdefault("timeout", config.OA_TIMEOUT_S)
    for attempt in range(3):
        r = session().get(url, **kw)
        if r.status_code != 429:
            return r
        time.sleep(2 + 3 * attempt)
    return r


def keywords(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", (text or "").lower())
    return [w for w in words if w not in STOP and len(w) > 2]


def relevant(cand, terms, need=1):
    """A candidate is relevant when its title (or, failing that, abstract) shares enough of the query's terms."""
    title = set(keywords(cand.get("title", "")))
    if len(title & set(terms)) >= need:
        return True
    return len(set(keywords(cand.get("abstract", ""))) & set(terms)) >= need + 1


# ----------------------------------------------------------------------------- sources

def sep_search(query, n=5):
    r = _get(f"{SEP}/search/searcher.py", params={"query": query})
    if r.status_code != 200:
        return []
    names = list(dict.fromkeys(re.findall(r"entry=/entries/([a-z0-9-]+)/", r.text)))[:n]
    return [{"source": "sep", "sep": name, "title": name.replace("-", " "), "url": f"{SEP}/entries/{name}/",
             "kind": "html", "license": "free-to-read"} for name in names]


def sep_details(name):
    """Author(s), title, edition and archive URL of an SEP entry, from the encyclopedia's own citation."""
    r = _get(f"{SEP}/cgi-bin/encyclopedia/archinfo.cgi", params={"entry": name})
    if r.status_code != 200:
        return {}
    text = re.sub(r"\s+", " ", lhtml.fromstring(r.text).text_content())
    m = re.search(r"cited via the earliest archive in which it appears:\s*(.+?), \"(.+?)\", The Stanford Encyclopedia of "
                  r"Philosophy\s*\((\w+ (\d{4})) Edition\),\s*(.+?)\s*\(eds?\.\),\s*URL = <([^>]+)>", text)
    if not m:
        return {}
    who, title, edition, year, eds, url = m.groups()
    authors = [a.strip() for a in re.split(r"\s+and\s+", who)]
    return {"title": title, "authors": authors, "year": year, "journal": "The Stanford Encyclopedia of Philosophy",
            "publisher": f"{edition} Edition, ed. {eds.replace('&', 'and')}", "link": url, "license": "free-to-read"}


def _gutenberg_catalog():
    path = config.DATA_DIR / "gutenberg_catalog.csv"
    if not path.exists() or time.time() - path.stat().st_mtime > 30 * 86400:
        r = _get(GUTENBERG_CATALOG, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def gutenberg_record(num, catalog=None):
    for row in catalog or _gutenberg_catalog():
        if row.get("Text#") == str(num):
            return _gutenberg_cand(row)
    return None


def _gutenberg_cand(row):
    num = row["Text#"]
    authors = [re.sub(r",\s*\d{3,4}\??-\d{0,4}\??$|,\s*\d{3,4}\s*BCE?.*$", "", a).strip()
               for a in (row.get("Authors") or "").split(";") if a.strip()]
    return {"source": "gutenberg", "gutenberg": num, "title": row.get("Title", "").replace("\n", " "),
            "authors": [a for a in authors if not re.search(r"\[(Translator|Editor|Contributor)", a)],
            "translators": [re.sub(r"\s*\[.*", "", a) for a in authors if "[Translator" in a],
            "url": f"https://www.gutenberg.org/cache/epub/{num}/pg{num}.txt", "kind": "txt",
            "link": f"https://www.gutenberg.org/ebooks/{num}", "license": "public-domain"}


def gutenberg_search(query, n=3):
    terms = set(keywords(query))
    scored = []
    for row in _gutenberg_catalog():
        if row.get("Type") != "Text" or row.get("Language") != "en":
            continue
        hay = set(keywords(row.get("Title", "") + " " + row.get("Authors", "")))
        hit = len(hay & terms)
        if hit >= max(2, (len(terms) + 1) // 2):
            scored.append((hit, row))
    scored.sort(key=lambda x: -x[0])
    return [_gutenberg_cand(row) for _, row in scored[:n]]


def arxiv_search(query, n=5):
    r = _get("https://export.arxiv.org/api/query", params={"search_query": f"all:{query}", "max_results": n})
    if r.status_code != 200:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    out = []
    for e in etree.fromstring(r.content).findall("a:entry", ns):
        pdf = next((l.get("href") for l in e.findall("a:link", ns) if l.get("title") == "pdf"), None)
        if not pdf:
            continue
        out.append({"source": "arxiv", "title": re.sub(r"\s+", " ", e.findtext("a:title", "", ns)).strip(),
                    "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip(),
                    "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
                    "year": (e.findtext("a:published", "", ns) or "")[:4], "doi": e.findtext("x:doi", "", ns),
                    "url": pdf, "kind": "pdf", "link": e.findtext("a:id", "", ns), "license": "open-access"})
    return out


def europepmc_search(query, n=5):
    r = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
             params={"query": f"({query}) AND OPEN_ACCESS:y", "format": "json", "pageSize": n, "resultType": "core"})
    if r.status_code != 200:
        return []
    out = []
    for it in r.json().get("resultList", {}).get("result", []):
        pmcid = it.get("pmcid")
        if not pmcid:
            continue
        out.append({"source": "europepmc", "title": it.get("title", ""), "abstract": it.get("abstractText", ""),
                    "authors": [a.get("fullName", "") for a in (it.get("authorList") or {}).get("author", [])],
                    "year": it.get("pubYear", ""), "doi": it.get("doi", ""), "pmcid": pmcid, "pmid": it.get("pmid", ""),
                    "journal": (it.get("journalInfo") or {}).get("journal", {}).get("title", ""),
                    "url": f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML", "kind": "xml",
                    "link": f"https://europepmc.org/article/PMC/{pmcid}", "license": "open-access"})
    return out


def zenodo_search(query, n=5):
    r = _get("https://zenodo.org/api/records", params={"q": query, "size": n, "access_right": "open"})
    if r.status_code != 200:
        return []
    out = []
    for it in r.json().get("hits", {}).get("hits", []):
        md = it.get("metadata", {})
        pdf = next((f.get("links", {}).get("self") for f in it.get("files", []) if f.get("key", "").lower().endswith(".pdf")), None)
        if not pdf or md.get("access_right") != "open":
            continue
        out.append({"source": "zenodo", "title": md.get("title", ""), "abstract": re.sub(r"<[^>]+>", " ", md.get("description", "")),
                    "authors": [c.get("name", "") for c in md.get("creators", [])], "year": (md.get("publication_date") or "")[:4],
                    "doi": it.get("doi", ""), "url": pdf, "kind": "pdf", "link": it.get("links", {}).get("html", ""),
                    "license": (md.get("license") or {}).get("id", "open-access")})
    return out


def crossref_work(doi):
    """A DOI's details from Crossref, and a full-text link when the work carries a Creative Commons licence."""
    r = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if r.status_code != 200:
        return None
    m = r.json().get("message", {})
    year = next(iter((m.get("issued") or {}).get("date-parts", [[None]])[0]), None)
    rec = {"source": "crossref", "doi": m.get("DOI", doi), "title": " ".join(m.get("title") or []),
           "authors": [", ".join(v for v in (a.get("family"), a.get("given")) if v) for a in m.get("author", [])],
           "year": str(year or ""), "journal": " ".join(m.get("container-title") or []),
           "publisher": m.get("publisher", ""), "volume": m.get("volume", ""), "issue": m.get("issue", ""),
           "pages": m.get("page", ""), "isbn": ", ".join(m.get("ISBN") or []), "type": m.get("type", ""),
           "link": f"https://doi.org/{m.get('DOI', doi)}", "license": "catalogue"}
    cc = any("creativecommons.org" in (l.get("URL") or "") for l in m.get("license", []))
    pdf = next((l.get("URL") for l in m.get("link", []) if "pdf" in (l.get("content-type") or "")), None)
    if cc and pdf:
        rec.update(url=pdf, kind="pdf", license="open-access")
    return rec


SEARCHERS = {"sep": sep_search, "gutenberg": gutenberg_search, "arxiv": arxiv_search,
             "europepmc": europepmc_search, "zenodo": zenodo_search}
DEFAULT_SOURCES = ("sep", "arxiv", "europepmc", "zenodo")


def resolve_target(t):
    """See _resolve; details the caller gives (title, authors, year, ...) override what the source says,
    for instance the first-publication year of a public-domain book."""
    cand, rec = _resolve(t)
    if cand:
        cand.update({k: t[k] for k in ("title", "authors", "year", "publisher", "journal", "isbn", "license") if t.get(k)})
        if isinstance(cand.get("year"), int):
            cand["year"] = str(cand["year"])
    return cand, rec


def _resolve(t):
    """A target is {"doi"|"gutenberg"|"sep"|"url": ...} (optionally with title/authors/year/kind/license).
    Returns (candidate to download or None, catalogue record or None)."""
    if t.get("sep"):
        return {"source": "sep", "sep": t["sep"], "title": t.get("title", t["sep"]), "url": f"{SEP}/entries/{t['sep']}/",
                "kind": "html", "license": "free-to-read"}, None
    if t.get("gutenberg"):
        return gutenberg_record(t["gutenberg"]), None
    if t.get("url"):
        kind = t.get("kind") or ("pdf" if t["url"].lower().endswith(".pdf") else "html")
        return {"source": "url", "title": t.get("title", ""), "authors": t.get("authors", []), "year": str(t.get("year", "")),
                "doi": t.get("doi", ""), "isbn": t.get("isbn", ""), "url": t["url"], "kind": kind, "link": t["url"],
                "license": t.get("license", "open-access")}, None
    if t.get("doi"):
        rec = crossref_work(t["doi"])
        if rec and rec.get("url"):
            return rec, rec
        for cand in arxiv_search(f'doi:"{t["doi"]}"', n=1) + europepmc_search(f'DOI:"{t["doi"]}"', n=1):
            if cand.get("doi", "").lower() == t["doi"].lower():
                return cand, rec
        return None, rec
    return None, None


# ----------------------------------------------------------------------------- fetching

def _safe(s, n=70):
    return re.sub(r"[^A-Za-z0-9]+", "_", s or "untitled").strip("_")[:n] or "untitled"


def already_have(c, slug, cand):
    url = cand.get("url", "")
    key = "oa:" + hashlib.sha1(url.encode()).hexdigest()
    if c.execute("SELECT 1 FROM acquired WHERE slug=? AND md5=?", (slug, key)).fetchone():
        return True
    if cand.get("doi") and c.execute("SELECT 1 FROM docs WHERE slug=? AND lower(doi)=lower(?)", (slug, cand["doi"])).fetchone():
        return True
    return False


def fetch(c, slug, cand, ev=None):
    """Download one candidate into search/{slug}/oa/{source}/ and record its metadata. Returns the path or None."""
    url = cand["url"]
    r = _get(url, stream=True)
    if r.status_code != 200:
        if ev:
            ev("oa_download_error", f"HTTP {r.status_code}: {url}", url=url)
        return None
    data = io.BytesIO()
    for block in r.iter_content(65536):
        data.write(block)
        if data.tell() > config.OA_MAX_BYTES:
            if ev:
                ev("oa_download_error", f"too large: {url}", url=url)
            return None
    body = data.getvalue()
    ctype = r.headers.get("content-type", "")
    kind = cand.get("kind") or ("pdf" if "pdf" in ctype else "xml" if "xml" in ctype else "txt" if "text/plain" in ctype else "html")
    if kind == "pdf" and not body.startswith(b"%PDF"):
        kind = "html" if b"<html" in body[:2000].lower() else kind
    meta = {k: cand.get(k, "") for k in ("title", "year", "journal", "publisher", "doi", "isbn", "link", "license")}
    meta["authors"] = "; ".join(cand.get("authors") or [])
    if cand.get("source") == "sep":
        d = sep_details(cand["sep"])
        if d:
            meta.update({k: v for k, v in d.items() if k != "authors"}, authors="; ".join(d["authors"]))
    if cand.get("translators"):
        meta["publisher"] = (meta["publisher"] + "; " if meta["publisher"] else "") + "translated by " + ", ".join(cand["translators"])
    meta["link"] = meta["link"] or url
    folder = config.SEARCH_DIR / slug / "oa" / cand.get("source", "web")
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{_safe(meta['title'])}_{hashlib.sha1(url.encode()).hexdigest()[:8]}.{kind}"
    path = folder / name
    path.write_bytes(body)
    sidecar = config.SEARCH_DIR / slug / "oa_metadata.csv"
    new = not sidecar.exists()
    with open(sidecar, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"file": name, **{k: meta.get(k, "") for k in CSV_FIELDS if k != "file"}})
    c.execute("INSERT OR IGNORE INTO acquired(slug,md5,title,path) VALUES(?,?,?,?)",
              (slug, "oa:" + hashlib.sha1(url.encode()).hexdigest(), meta["title"][:300], str(path)))
    c.commit()
    if ev:
        ev("oa_download_done", f"saved {name} ({len(body) // 1024} KB, {meta['license']})", path=str(path))
    return path


def acquire_round(c, slug, question, ev, targets=None, sources=DEFAULT_SOURCES, max_new=3):
    """Find, fetch and index open copies: the targets first, then whatever the sources return for the question.
    Returns (number of works added, catalogue records for targets with no open copy)."""
    from . import indexer
    got, catalogue = 0, []
    cands = []
    for t in targets or []:
        cand, rec = resolve_target(t)
        if cand:
            cands.append((cand, True))
        elif rec:
            ev("oa_catalogue", f"no open copy of {rec.get('title', t)[:80]}; catalogue record kept", record=rec)
            catalogue.append(rec)
        else:
            ev("oa_unresolved", f"could not resolve target {t}", target=t)
    terms = keywords(question)
    if question and got < max_new:
        for name in sources:
            try:
                found = SEARCHERS[name](" ".join(terms[:8]))
            except Exception as e:
                ev("oa_search_error", f"{name}: {e}")
                continue
            ev("oa_search", f"{name}: {len(found)} candidate(s)", source=name, titles=[x["title"][:80] for x in found[:5]])
            cands += [(x, False) for x in found if relevant(x, terms)]
    for cand, explicit in cands:
        if got >= max_new + len(targets or []):
            break
        if already_have(c, slug, cand):
            ev("oa_skip", f"already in the corpus: {cand.get('title', '')[:80]}")
            continue
        try:
            if fetch(c, slug, cand, ev):
                got += 1
        except Exception as e:
            ev("oa_download_error", f"{cand.get('url')}: {e}")
    if got:
        stats = indexer.index_search(c, slug, log=lambda m: None)
        ev("index_done", f"+{stats.get('new_docs', 0)} doc(s), +{stats.get('chunks', 0)} chunk(s)", **stats)
    return got, catalogue
