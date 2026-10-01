# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""research_section (evidence only), the indexer's locators, and open-access acquisition.
No network: HTTP is stubbed, and the language model must never be called."""
import csv
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import support  # noqa: E402
from searchbot import agent, config, indexer, libgen, llm, mcp_server, oa  # noqa: E402


class NoModel:
    """Fail any test in which the evidence path reaches the language model or LibGen."""

    def setUp(self):
        super().setUp()
        self._chat, self._libgen = llm.chat_llm, libgen.get_client

        def boom(*a, **k):
            raise AssertionError("the evidence path must not call the language model or LibGen")
        llm.chat_llm = boom
        libgen.get_client = boom

    def tearDown(self):
        llm.chat_llm, libgen.get_client = self._chat, self._libgen
        super().tearDown()


def add(case, slug, title, chunks, authors="", doi="", isbn="", url="", locs=None):
    cur = case.c.execute("INSERT INTO docs(slug,kind,source_file,title,authors,year,doi,isbn,url,license) VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (slug, "html", f"{title}.html", title, authors, "2001", doi, isbn, url, "free-to-read"))
    doc = cur.lastrowid
    for i, text in enumerate(chunks):
        cid = case.c.execute("INSERT INTO chunks(slug,doc_id,ordinal,text,locator) VALUES(?,?,?,?,?)",
                             (slug, doc, i, text, (locs or [""] * len(chunks))[i])).lastrowid
        case.c.execute("INSERT INTO chunks_fts(rowid,text,slug,doc_id,chunk_id) VALUES(?,?,?,?,?)", (cid, text, slug, doc, cid))
        case.c.execute("INSERT INTO chunks_vec(chunk_id,slug,embedding) VALUES(?,?,?)",
                       (cid, slug, support.db.ser(support.embed_fake([text])[0])))
    case.c.commit()
    return doc


class ResearchSectionTest(NoModel, support.TempCase):
    def setUp(self):
        super().setUp()
        add(self, "t", "Ephedrine review", [f"ephedrine hypotension anesthesia spinal review part {i}" for i in range(6)],
            authors="Smith, Ann; Jones, Bo", doi="10.1/x", locs=[f"§ {i}" for i in range(6)])
        add(self, "t", "Metformin study", ["metformin glucose patients study", "metformin glucose blood pressure"],
            isbn="978-0-19-000000-0", url="https://example.org/m")

    def test_returns_passages_with_metadata_and_no_answer(self):
        res = agent.research_section(self.c, "t", "ephedrine hypotension", acquire=False)
        self.assertNotIn("answer", res)
        self.assertTrue(res["evidence"])
        e = res["evidence"][0]
        for f in ("chunk_id", "work_id", "passage", "locator", "title", "authors", "year", "journal", "publisher",
                  "doi", "isbn", "pmid", "pmcid", "url", "license", "source_file", "citations", "score", "retrieval"):
            self.assertIn(f, e)
        self.assertIn(e["retrieval"], ("vector", "bm25", "vector+bm25"))
        self.assertEqual(e["authors"], ["Smith, Ann", "Jones, Bo"])
        self.assertTrue(e["locator"].startswith("§"))

    def test_caps_passages_per_work_and_lists_the_works(self):
        res = agent.research_section(self.c, "t", "ephedrine hypotension", acquire=False, max_per_work=2, k=10)
        per = {}
        for e in res["evidence"]:
            per[e["work_id"]] = per.get(e["work_id"], 0) + 1
        self.assertLessEqual(max(per.values()), 2)
        self.assertEqual({w["work_id"] for w in res["works"]}, set(per))
        self.assertEqual(sum(w["passages"] for w in res["works"]), len(res["evidence"]))

    def test_further_phrasings_widen_retrieval(self):
        one = agent.research_section(self.c, "t", "ephedrine", acquire=False, k=20)
        both = agent.research_section(self.c, "t", "ephedrine", queries=["metformin glucose"], acquire=False, k=20)
        self.assertIn("Metformin study", {w["title"] for w in both["works"]})
        self.assertGreaterEqual(len(both["evidence"]), len(one["evidence"]))

    def test_isbn_reaches_the_evidence(self):
        res = agent.research_section(self.c, "t", "metformin glucose", acquire=False)
        m = next(e for e in res["evidence"] if e["title"] == "Metformin study")
        self.assertEqual(m["isbn"], "978-0-19-000000-0")


class GatherTest(NoModel, support.TempCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self._round = oa.acquire_round

        def fake_round(c, slug, question, ev, targets=None, sources=None, max_new=3):
            self.calls.append({"question": question, "targets": targets})
            return 0, []
        oa.acquire_round = fake_round

    def tearDown(self):
        oa.acquire_round = self._round
        super().tearDown()

    def test_targets_are_fetched_even_when_evidence_suffices(self):
        add(self, "t", "Ephedrine review", ["ephedrine hypotension anesthesia"] * 4)
        agent.gather_evidence(self.c, "t", "ephedrine", targets=[{"sep": "knowledge-how"}])
        self.assertEqual(self.calls[0]["targets"], [{"sep": "knowledge-how"}])

    def test_no_acquisition_when_evidence_suffices(self):
        add(self, "t", "Ephedrine review", ["ephedrine hypotension anesthesia"] * 4)
        agent.gather_evidence(self.c, "t", "ephedrine")
        self.assertEqual(self.calls, [])

    def test_thin_evidence_acquires_and_acquire_false_does_not(self):
        agent.gather_evidence(self.c, "t", "berberine")
        self.assertTrue(self.calls)
        self.calls.clear()
        agent.gather_evidence(self.c, "t", "berberine", acquire=False)
        self.assertEqual(self.calls, [])


class LocatorTest(support.TempCase):
    def test_book_chapters_become_locators(self):
        p = config.SEARCH_DIR / "b" / "book.txt"
        p.parent.mkdir(parents=True)
        p.write_text("*** START OF THE PROJECT GUTENBERG EBOOK X ***\nPreface text that is long enough to keep. " * 3 +
                     "\nCHAPTER V\n\nKNOWLEDGE BY ACQUAINTANCE\n\n" + "Ephedrine hypotension anesthesia sentence. " * 40 +
                     "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicence text")
        segs = indexer.txt_segments(p)
        self.assertIn("Chapter V Knowledge By Acquaintance", [s[0] for s in segs])
        self.assertFalse(any("Licence" in t for _, t in segs))

    def test_html_sections_become_locators_and_tools_are_dropped(self):
        p = config.SEARCH_DIR / "h" / "entry.html"
        p.parent.mkdir(parents=True)
        body = "<p>" + "Ephedrine hypotension anesthesia is discussed here at length. " * 5 + "</p>"
        p.write_text(f"<html><body><nav>menu</nav><div id='main-text'><h2>1. Intellectualism</h2>{body}"
                     f"<h3>1.1 Objections</h3>{body}</div><div id='academic-tools'><h2>Academic Tools</h2><p>cite this</p></div></body></html>")
        segs = indexer.html_segments(p)
        self.assertEqual([s[0] for s in segs], ["§ 1. Intellectualism", "§ 1.1 Objections"])
        self.assertNotIn("menu", " ".join(t for _, t in segs))

    def test_pdf_pages_label_chunks_with_page_ranges(self):
        segs = [(f"PDF p. {i + 1}", "Ephedrine hypotension anesthesia sentence number one. " * 12) for i in range(3)]
        located = indexer.chunk_segments(segs, merge_pages=True)
        self.assertTrue(all(loc.startswith("PDF p") for loc, _ in located))
        self.assertTrue(any("–" in loc for loc, _ in located))
        self.assertFalse(any("\u27e6" in t for _, t in located))

    def test_index_stores_locators_isbn_and_publisher(self):
        d = config.SEARCH_DIR / "x"
        (d / "oa").mkdir(parents=True)
        (d / "oa" / "b.txt").write_text("CHAPTER I. APPEARANCE AND REALITY\n" + "Ephedrine hypotension anesthesia. " * 60)
        with open(d / "oa_metadata.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=oa.CSV_FIELDS)
            w.writeheader()
            w.writerow({"file": "b.txt", "title": "A Book", "authors": "Russell, Bertrand", "year": "1912",
                        "isbn": "123", "publisher": "Williams and Norgate", "license": "public-domain"})
        indexer.index_search(self.c, "x", log=lambda m: None)
        row = self.c.execute("SELECT d.isbn, d.publisher, d.kind, ch.locator FROM chunks ch JOIN docs d ON d.id=ch.doc_id").fetchone()
        self.assertEqual((row["isbn"], row["publisher"], row["kind"]), ("123", "Williams and Norgate", "txt"))
        self.assertEqual(row["locator"], "Chapter I. Appearance And Reality")


class FakeResponse:
    def __init__(self, body=b"", status=200, ctype="text/html", data=None):
        self.content, self.status_code, self.headers, self._data = body, status, {"content-type": ctype}, data
        self.text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body

    def iter_content(self, n):
        yield self.content

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.seen = routes, {}, []

    def get(self, url, **kw):
        self.seen.append(url)
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                return resp
        return FakeResponse(status=404)


class OpenAccessTest(NoModel, support.TempCase):
    def setUp(self):
        super().setUp()
        self._session = oa._session

    def tearDown(self):
        oa._session = self._session
        super().tearDown()

    def test_relevance_needs_shared_terms(self):
        terms = oa.keywords("knowledge how intellectualism")
        self.assertTrue(oa.relevant({"title": "Knowledge How"}, terms))
        self.assertFalse(oa.relevant({"title": "Common Ground"}, terms))

    def test_sep_citation_is_parsed(self):
        page = (b"<html><body><p>may be cited via the earliest archive in which it appears:</p><p>Pavese, Carlotta, "
                b"\"Knowledge How\", The Stanford Encyclopedia of Philosophy (Fall 2022 Edition), Edward N. Zalta &amp; Uri "
                b"Nodelman (eds.), URL = &lt;https://plato.stanford.edu/archives/fall2022/entries/knowledge-how/&gt;.</p></body></html>")
        oa._session = FakeSession({"https://plato.stanford.edu/cgi-bin": FakeResponse(page)})
        d = oa.sep_details("knowledge-how")
        self.assertEqual(d["authors"], ["Pavese, Carlotta"])
        self.assertEqual((d["title"], d["year"]), ("Knowledge How", "2022"))
        self.assertTrue(d["link"].endswith("/archives/fall2022/entries/knowledge-how/"))

    def test_crossref_without_open_licence_is_catalogue_only(self):
        data = {"message": {"DOI": "10.2307/x", "title": ["Knowing How"], "author": [{"family": "Stanley", "given": "Jason"}],
                            "issued": {"date-parts": [[2001]]}, "container-title": ["J. Phil."], "license": [],
                            "link": [{"URL": "https://pub/x.pdf", "content-type": "application/pdf"}]}}
        oa._session = FakeSession({"https://api.crossref.org": FakeResponse(data=data)})
        rec = oa.crossref_work("10.2307/x")
        self.assertEqual(rec["license"], "catalogue")
        self.assertNotIn("url", rec)
        self.assertEqual(rec["authors"], ["Stanley, Jason"])

    def test_fetch_saves_records_and_indexes(self):
        body = ("<html><body><div id='main-text'><h2>1. Ephedrine</h2><p>" +
                "Ephedrine hypotension anesthesia is discussed here at length. " * 8 + "</p></div></body></html>").encode()
        oa._session = FakeSession({"https://example.org/open.html": FakeResponse(body)})
        got, cat = oa.acquire_round(self.c, "t", "", lambda *a, **k: None,
                                    targets=[{"url": "https://example.org/open.html", "title": "Open Paper",
                                              "authors": ["Doe, Jane"], "year": 2020, "license": "CC-BY-4.0"}])
        self.assertEqual((got, cat), (1, []))
        with open(config.SEARCH_DIR / "t" / "oa_metadata.csv") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual((rows[0]["title"], rows[0]["authors"], rows[0]["license"]), ("Open Paper", "Doe, Jane", "CC-BY-4.0"))
        doc = self.c.execute("SELECT title, authors, license FROM docs").fetchone()
        self.assertEqual(tuple(doc), ("Open Paper", "Doe, Jane", "CC-BY-4.0"))
        again, _ = oa.acquire_round(self.c, "t", "", lambda *a, **k: None, targets=[{"url": "https://example.org/open.html"}])
        self.assertEqual(again, 0)

    def test_doi_without_open_copy_returns_its_catalogue_record(self):
        data = {"message": {"DOI": "10.2307/x", "title": ["Knowing How"], "issued": {"date-parts": [[2001]]}}}
        oa._session = FakeSession({"https://api.crossref.org": FakeResponse(data=data),
                                   "https://export.arxiv.org": FakeResponse(b"<feed xmlns='http://www.w3.org/2005/Atom'/>"),
                                   "https://www.ebi.ac.uk": FakeResponse(data={"resultList": {"result": []}})})
        got, cat = oa.acquire_round(self.c, "t", "", lambda *a, **k: None, targets=[{"doi": "10.2307/x"}])
        self.assertEqual(got, 0)
        self.assertEqual(cat[0]["title"], "Knowing How")

    def test_internet_archive_target_reads_the_ocr_text(self):
        cand, cat = oa._resolve({"ia": "lecturesessaysby02clif", "title": "Lectures and Essays"})
        self.assertIsNone(cat)
        self.assertEqual(cand["url"], "https://archive.org/download/lecturesessaysby02clif/lecturesessaysby02clif_djvu.txt")
        self.assertEqual((cand["kind"], cand["license"]), ("txt", "public-domain"))


class DispatchResearchTest(NoModel, support.TempCase):
    def test_mcp_routes_research_section(self):
        add(self, "t", "Ephedrine review", ["ephedrine hypotension anesthesia"] * 4)
        real = mcp_server._conn
        mcp_server._conn = lambda: self.c
        try:
            res = mcp_server._dispatch("research_section", {"slug": "t", "question": "ephedrine", "acquire": False, "k": 2})
        finally:
            mcp_server._conn = real
        self.assertLessEqual(len(res["evidence"]), 2)
        self.assertNotIn("answer", res)
