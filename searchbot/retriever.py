"""Hybrid retrieval: vec KNN + FTS5 BM25, fused with Reciprocal Rank Fusion."""
import math
from . import config, db, llm


def retrieve(c, query: str, slug: str = None, k: int = None):
    k = k or config.FINAL_K
    recall = config.RECALL_K
    qvec = llm.embed([query], is_query=True)[0]

    # --- vector lane ---
    vec_hits = {}
    if slug:
        rows = c.execute(
            "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? AND slug = ? ORDER BY distance",
            (db.ser(qvec), recall, slug)).fetchall()
    else:
        rows = c.execute(
            "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (db.ser(qvec), recall)).fetchall()
    for rank, r in enumerate(rows):
        vec_hits[r["chunk_id"]] = rank

    # --- BM25 lane ---
    fts_hits = {}
    terms = " ".join('"%s"' % t for t in _keywords(query))
    if terms:
        try:
            if slug:
                frows = c.execute(
                    "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? AND slug = ? ORDER BY s LIMIT ?",
                    (terms, slug, recall)).fetchall()
            else:
                frows = c.execute(
                    "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
                    (terms, recall)).fetchall()
            for rank, r in enumerate(frows):
                fts_hits[r["chunk_id"]] = rank
        except Exception:
            pass

    # --- RRF fusion ---
    scores = {}
    for cid, rank in vec_hits.items():
        scores[cid] = scores.get(cid, 0) + 1.0 / (60 + rank)
    for cid, rank in fts_hits.items():
        scores[cid] = scores.get(cid, 0) + 1.0 / (60 + rank)

    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    out = []
    for cid, score in top:
        row = c.execute(
            "SELECT ch.id, ch.text, d.title, d.year, d.journal, d.doi, d.pmcid, d.pmid, d.source_file, d.slug "
            "FROM chunks ch JOIN docs d ON d.id=ch.doc_id WHERE ch.id=?", (cid,)).fetchone()
        if row:
            out.append({"chunk_id": cid, "score": round(score, 5),
                        "in_vec": cid in vec_hits, "in_fts": cid in fts_hits, **dict(row)})
    return out


_STOP = set("what how why does do is are the of a an in on for and or to with by from was were be been being can could should would may might this that these those it its as at into about which who whom whose when where while after before between during under over more most other some such no nor not only own same than too very s t just now there here they them their".split())


def _keywords(query: str):
    words = [w.strip(".,?!;:'\"()[]") for w in query.lower().split()]
    return [w for w in words if len(w) > 2 and w not in _STOP][:12]
