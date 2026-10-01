# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Corpus indexing: extract text from PDF, XML, HTML and plain-text files in search/{slug}/,
chunk, embed, store. Each chunk keeps a locator saying where in the work it is: the PDF page
("PDF p. 12"), or the section of an HTML page or a book ("§ 2.1 ...", "Chapter V. ...")."""
import csv
import hashlib
import re
from pathlib import Path
from pypdf import PdfReader
from lxml import etree
from lxml import html as lhtml
from . import config, db, llm

SUFFIXES = (".pdf", ".xml", ".html", ".htm", ".txt")


def extract_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        parts = []
        for p in reader.pages[:config.MAX_PDF_PAGES]:  # cap pathological docs
            parts.append(p.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        return ""


def extract_xml(path: Path) -> str:
    try:
        tree = etree.parse(str(path))
        for root in tree.iter("abstract", "sec", "p", "title", "body"):
            pass  # cheap parse warmup
        text = "".join(tree.getroot().itertext())
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def pdf_segments(path: Path):
    try:
        reader = PdfReader(str(path))
        return [(f"PDF p. {i + 1}", p.extract_text() or "") for i, p in enumerate(reader.pages[:config.MAX_PDF_PAGES])]
    except Exception:
        return []


_SKIP_HTML = {"script", "style", "nav", "header", "footer", "noscript", "form", "button", "svg"}
_SKIP_SECTIONS = re.compile(r"^(academic tools|other internet resources|related entries|acknowl?edge?ments|"
                            r"table of contents|contents|share|navigation)\b", re.I)


def html_segments(path: Path):
    """Text of an HTML page by section: each heading starts a new segment named after it.
    The main content is used when the page marks it (an encyclopedia entry's #main-text and
    #bibliography, an <article>, <main>), the whole body otherwise."""
    try:
        tree = lhtml.fromstring(path.read_bytes())
    except Exception:
        return []
    for bad in tree.xpath("//" + " | //".join(_SKIP_HTML)):
        bad.getparent().remove(bad) if bad.getparent() is not None else None
    roots = (tree.xpath('//*[@id="preamble"] | //*[@id="main-text"] | //*[@id="bibliography"]')
             or tree.xpath("//article") or tree.xpath("//main") or tree.xpath("//body") or [tree])
    segs, cur, buf, skipping = [], "", [], False
    for root in roots:
        cur = {"preamble": "§ Preamble", "bibliography": "§ Bibliography"}.get(root.get("id") or "", "")
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            if el.tag in ("h1", "h2", "h3", "h4"):
                if buf and not skipping:
                    segs.append((cur, " ".join(buf)))
                head = re.sub(r"\s+", " ", el.text_content()).strip()
                skipping = bool(_SKIP_SECTIONS.match(head))
                cur, buf = (f"§ {head}"[:90] if head else cur), []
            elif el.tag in ("p", "li", "blockquote", "dd", "dt", "td", "pre"):
                t = re.sub(r"\s+", " ", el.text_content()).strip()
                if t and not any(a.tag in ("p", "li", "blockquote", "dd", "td") for a in el.iterancestors() if a is not root):
                    buf.append(t)
        if buf and not skipping:
            segs.append((cur, " ".join(buf)))
        cur, buf, skipping = "", [], False
    return segs


_HEADING = re.compile(r"^\s*((?:CHAPTER|Chapter|BOOK|Book|PART|Part|LECTURE|Lecture|MEDITATION|Meditation|SECTION|Section|"
                      r"ESSAY|Essay|DIALOGUE)\b[^\n]{0,70}|[IVXLC]{1,7}\.\s+[A-Z][A-Z ,;:'-]{3,70})\s*$")


def _title(s):
    s = re.sub(r"\s+", " ", s).strip()
    if s.isupper():
        s = re.sub(r"\b([IVXLC]+)\b", lambda m: m.group(1).upper(), s.title(), flags=re.I)
    return s


def txt_segments(path: Path):
    """A plain-text book by chapter (Project Gutenberg's licence header and footer removed)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    if m:
        text = text[:m.start()]
    lines = text.splitlines()
    segs, cur, buf, i = [], "", [], 0
    while i < len(lines):
        line = lines[i]
        if _HEADING.match(line) and len(line.strip()) < 80:
            head = line.strip()
            nxt = next((l.strip() for l in lines[i + 1:i + 4] if l.strip()), "")
            if re.fullmatch(r"(CHAPTER|Chapter|LECTURE|Lecture|BOOK|Book|PART|Part)\s+[IVXLC\d]+\.?", head) and nxt.isupper():
                head += " " + nxt
            if buf:
                segs.append((cur, "\n".join(buf)))
            cur, buf = _title(head)[:90], []
        else:
            buf.append(line)
        i += 1
    if buf:
        segs.append((cur, "\n".join(buf)))
    return segs


def segments(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_segments(path)
    if suffix in (".html", ".htm"):
        return html_segments(path)
    if suffix == ".txt":
        return txt_segments(path)
    return [("", extract_xml(path))]


def chunk_segments(segs, merge_pages=False):
    """Chunks with their locators. PDF pages are not sections, so their text runs on across
    pages and a chunk is labelled with the page(s) it comes from; other segments are sections,
    chunked on their own."""
    out = []
    if merge_pages:
        text = " ".join(f"\u27e6{i}\u27e7 {t}" for i, (_, t) in enumerate(segs))  # page marks (not whitespace)
        page = 0
        for ch in chunk_text(text):
            marks = [int(m) for m in re.findall(r"\u27e6(\d+)\u27e7", ch)]
            lead = re.match(r"\s*\u27e6(\d+)\u27e7", ch)
            first = int(lead.group(1)) if lead else page
            last = marks[-1] if marks else first
            page = last
            loc = segs[first][0] if first == last else f"{segs[first][0]}–{segs[last][0].split()[-1]}".replace("p. ", "pp. ", 1)
            clean = re.sub(r"\s*\u27e6\d+\u27e7\s*", " ", ch).strip()
            if len(clean) >= 80:
                out.append((loc, clean))
    else:
        carry = ""
        for loc, t in segs:
            t = (carry + " " + t).strip() if carry else t
            chunks = chunk_text(t)
            if not chunks:
                carry = t if len(t) < 400 else ""
                continue
            carry = ""
            out += [(loc, ch) for ch in chunks]
    return out[:config.MAX_CHUNKS_PER_DOC]


def meta_from_csv(slug_dir: Path) -> dict:
    """filename -> metadata dict, from fulltext_index.csv / *_metadata.csv if present."""
    out = {}
    for csvf in slug_dir.glob("*.csv"):
        try:
            with open(csvf, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    fn = (row.get("file") or "").strip()
                    if fn:
                        out[Path(fn).name] = row
                        out[Path(fn).stem] = row
        except Exception:
            continue
    return out


def title_from_pdf_name(path: Path) -> str:
    # 2023_PMC10464275_Hemodynamic impact of ephedrine ....pdf
    m = re.match(r"^(\d{4})_((PMC|PMID)?\d+)?_?(.*)\.pdf$", path.name)
    if m:
        year, pmcid, _, rest = m.groups()
        return rest.replace("_", " ").strip() or path.stem
    return path.stem


def chunk_text(text: str):
    """Split on paragraph/sentence boundaries into ~CHUNK_CHARS pieces with overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 80:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    # hard-split any monster sentence (XML tables / no punctuation) into safe pieces
    safe = []
    for s in sentences:
        while len(s) > config.CHUNK_CHARS:
            safe.append(s[:config.CHUNK_CHARS])
            s = s[config.CHUNK_CHARS:]
        if s:
            safe.append(s)
    sentences = safe
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > config.CHUNK_CHARS and cur:
            chunks.append(cur.strip())
            cur = cur[-config.CHUNK_OVERLAP:] + " " + s if len(cur) > config.CHUNK_OVERLAP else s
        else:
            cur = (cur + " " + s).strip()
    if len(cur.strip()) >= 80:
        chunks.append(cur.strip())
    return chunks[:config.MAX_CHUNKS_PER_DOC]


def index_search(c, slug: str, job_id=None, log=print) -> dict:
    """(Re)index search/{slug}. Skips chunks already stored (content-hash per doc file)."""
    slug_dir = config.SEARCH_DIR / slug
    db.upsert_search(c, slug, path=str(slug_dir))
    files = sorted(p for p in slug_dir.rglob("*") if p.suffix.lower() in SUFFIXES and p.is_file())
    metas = meta_from_csv(slug_dir)
    stats = {"files": 0, "new_docs": 0, "chunks": 0, "errors": 0}
    dim = llm.embed_dim()
    db.init(c, dim)

    for fpath in files:
        stats["files"] += 1
        rel = str(fpath.relative_to(slug_dir))
        existing = c.execute("SELECT id FROM docs WHERE slug=? AND source_file=?",
                             (slug, rel)).fetchone()
        if existing:
            continue
        meta = metas.get(fpath.name) or metas.get(fpath.stem) or {}
        segs = segments(fpath)
        if sum(len(t.strip()) for _, t in segs) < 100:
            stats["errors"] += 1
            continue
        located = chunk_segments(segs, merge_pages=fpath.suffix.lower() == ".pdf")
        chunks = [ch for _, ch in located]
        if not chunks:
            stats["errors"] += 1
            continue
        title = (meta.get("title") or title_from_pdf_name(fpath))
        title = re.sub(r"&lt;i&gt;|&lt;/i&gt;|&lt;b&gt;|&lt;/b&gt;|&lt;[^>]+&gt;", "", title)
        c.execute("INSERT INTO docs(slug,kind,source_file,title,authors,year,journal,doi,pmcid,pmid,url,license,n_chunks,"
                  "isbn,publisher) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (slug, fpath.suffix.lower().lstrip(".").replace("htm", "html").replace("htmll", "html"), rel,
                   title.strip(), meta.get("authors", ""), meta.get("year", ""),
                   meta.get("journal", ""), meta.get("doi", ""), meta.get("pmcid", ""),
                   meta.get("pmid", ""), meta.get("link", ""), meta.get("license", ""), len(chunks),
                   meta.get("isbn", ""), meta.get("publisher", "")))
        doc_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        stats["new_docs"] += 1

        chunk_ids = []
        for ordinal, (loc, ch) in enumerate(located):
            c.execute("INSERT INTO chunks(slug,doc_id,ordinal,text,locator) VALUES(?,?,?,?,?)",
                      (slug, doc_id, ordinal, ch, loc))
            chunk_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            chunk_ids.append(chunk_id)
            c.execute("INSERT INTO chunks_fts(rowid,text,slug,doc_id,chunk_id) VALUES(?,?,?,?,?)",
                      (chunk_id, ch, slug, doc_id, chunk_id))
        vecs = llm.embed(chunks)
        c.executemany("INSERT INTO chunks_vec(chunk_id,slug,embedding) VALUES(?,?,?)",
                      [(cid, slug, db.ser(v)) for cid, v in zip(chunk_ids, vecs)])
        stats["chunks"] += len(chunks)
        c.commit()
        if job_id:
            db.job_update(c, job_id, "running", f"{rel} ({len(chunks)} chunks)")
        log(f"indexed {rel}: {len(chunks)} chunks")
    c.execute("UPDATE searches SET title=COALESCE(NULLIF(title,''),?) WHERE slug=?", (slug, slug))
    c.commit()
    return stats
