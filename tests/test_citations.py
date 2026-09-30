# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""OpenAlex backfill: identifier lanes, batching, and dead-end stamping.

citations.requests is replaced wholesale, so no test here reaches the network.
"""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import support  # noqa: E402
from searchbot import citations  # noqa: E402


def work(doi=None, pmid=None, cites=7, year=2019):
    w = {"id": "https://openalex.org/W1", "cited_by_count": cites, "publication_year": year}
    if doi:
        w["doi"] = doi
    if pmid:
        w["ids"] = {"pmid": f"https://identifiers.org/pmid/{pmid}"}
    return w


class FakeOpenAlex:
    """Answers batched filter requests from a {key: work} table."""

    def __init__(self, table, found_year=None):
        self.table = table
        self.calls = []
        self.found_year = found_year or {}

    def get(self, url, params=None, headers=None, timeout=None, **kw):
        field, _, values = params["filter"].partition(":")
        vals = values.split("|")
        self.calls.append((field, list(vals), params["per-page"]))
        results = [self.table[v] for v in vals if v in self.table]
        return FakeJSON({"results": results})


class FakeJSON:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class CitationsCase(support.TempCase):
    def setUp(self):
        super().setUp()
        self._real_requests = citations.requests
        support.config.OPENALEX_BATCH = 40

    def tearDown(self):
        citations.requests = self._real_requests
        super().tearDown()

    def plug(self, table):
        fake = FakeOpenAlex(table)
        citations.requests = fake
        return fake

    def doc(self, **kw):
        return self.add_doc("ephedra", kw.pop("title", "t"), kw.pop("year", ""), **kw)

    def row(self, doc_id):
        return self.c.execute("SELECT * FROM docs WHERE id=?", (doc_id,)).fetchone()


class LaneTest(CitationsCase):
    def test_doi_hit_fills_count_and_stamps_source(self):
        self.doc(title="A study of ephedrine", year="", doi="10.1/AAA")
        self.plug({"10.1/aaa": work(doi="https://doi.org/10.1/AAA", cites=42)})
        stats = citations.fetch_counts(self.c, "ephedra", sleep=0)
        self.assertEqual(stats["updated"], 1)
        row = self.c.execute("SELECT * FROM docs").fetchone()
        self.assertEqual(row["citations"], 42)
        self.assertEqual(row["citations_source"], "openalex")
        self.assertTrue(row["citations_updated"])

    def test_year_from_openalex_fills_a_blank_but_never_overwrites(self):
        self.add_doc("ephedra", "blank", "", doi="10.1/B", text="ephedrine blood")
        self.add_doc("ephedra", "dated", "2001", doi="10.1/C", text="ephedrine glucose")
        self.plug({"10.1/b": work(doi="10.1/B", year=2019),
                   "10.1/c": work(doi="10.1/C", year=2019)})
        citations.fetch_counts(self.c, "ephedra", sleep=0)
        rows = {r["title"]: r["year"] for r in self.c.execute("SELECT title,year FROM docs")}
        self.assertEqual(rows["blank"], "2019")
        self.assertEqual(rows["dated"], "2001", "a parsed PDF year outranks the API")

    def test_pmid_only_docs_use_the_pmid_lane(self):
        self.doc(title="No doi here", year="", doi="", pmid="12345")
        fake = self.plug({"12345": work(pmid="12345", cites=3)})
        stats = citations.fetch_counts(self.c, "ephedra", sleep=0)
        self.assertEqual([c[0] for c in fake.calls], ["pmid"])
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(self.c.execute("SELECT citations FROM docs").fetchone()[0], 3)

    def test_doi_wins_so_a_doc_is_never_queried_twice(self):
        self.doc(title="Both ids", year="", doi="10.1/D", pmid="999")
        fake = self.plug({"10.1/d": work(doi="10.1/D", cites=5)})
        citations.fetch_counts(self.c, "ephedra", sleep=0)
        self.assertEqual([c[0] for c in fake.calls], ["doi"])


class StampingTest(CitationsCase):
    def test_not_found_is_stamped_so_reruns_skip_it(self):
        self.doc(title="Ghost", year="", doi="10.1/MISSING")
        fake = self.plug({})
        stats = citations.fetch_counts(self.c, "ephedra", sleep=0)
        self.assertEqual(stats["not_found"], 1)
        row = self.c.execute("SELECT * FROM docs").fetchone()
        self.assertIsNone(row["citations"], "unknown counts must stay NULL, not 0")
        self.assertEqual(row["citations_source"], "openalex:not-found")
        self.assertTrue(row["citations_updated"])

        fake2 = self.plug({})
        again = citations.fetch_counts(self.c, "ephedra", sleep=0)
        self.assertEqual(again["considered"], 0)
        self.assertEqual(fake2.calls, [], "a dead end is not re-queried")

    def test_only_missing_false_requeries_stamped_rows(self):
        self.doc(title="Stamp", year="", doi="10.1/E")
        self.plug({"10.1/e": work(doi="10.1/E", cites=1)})
        citations.fetch_counts(self.c, "ephedra", sleep=0)
        fake = self.plug({"10.1/e": work(doi="10.1/E", cites=80)})
        stats = citations.fetch_counts(self.c, "ephedra", only_missing=False, sleep=0)
        self.assertEqual(stats["considered"], 1)
        self.assertEqual(self.c.execute("SELECT citations FROM docs").fetchone()[0], 80)
        self.assertEqual(len(fake.calls), 1)

    def test_docs_without_any_identifier_are_left_alone(self):
        self.doc(title="No ids", year="1998", doi="", pmid="")
        fake = self.plug({})
        stats = citations.fetch_counts(self.c, "ephedra", sleep=0)
        self.assertEqual(stats["considered"], 0)
        self.assertEqual(fake.calls, [])


class BatchingTest(CitationsCase):
    def test_identifiers_are_split_at_the_batch_size(self):
        for i in range(100):
            self.doc(title=f"paper {i} ephedrine", year="", doi=f"10.1/P{i:03d}")
        fake = self.plug({})
        support.config.OPENALEX_BATCH = 40
        citations.fetch_counts(self.c, "ephedra", sleep=0)
        self.assertEqual([len(c[1]) for c in fake.calls], [40, 40, 20])
        self.assertTrue(all(c[2] >= len(c[1]) for c in fake.calls),
                        "per-page must never truncate the batch")

    def test_slug_scope_limits_the_query(self):
        self.doc(title="in scope", year="", doi="10.1/X")
        self.add_doc("metformin", "out of scope", "", doi="10.1/Y", text="metformin glucose")
        fake = self.plug({})
        stats = citations.fetch_counts(self.c, "ephedra", sleep=0)
        self.assertEqual(stats["considered"], 1)
        self.assertEqual(sorted(v for c in fake.calls for v in c[1]), ["10.1/x"])


class FailureTest(CitationsCase):
    def test_http_error_leaves_the_row_unstamped_for_a_later_run(self):
        """A 429 is not "this paper has no record" — stamping it would bury it."""
        self.doc(title="Flaky", year="", doi="10.1/F")

        class Boom:
            def get(self, *a, **kw):
                raise RuntimeError("429 slow down")
        citations.requests = Boom()
        msgs = []
        stats = citations.fetch_counts(self.c, "ephedra", log=msgs.append, sleep=0)
        self.assertEqual(stats["updated"], 0)
        self.assertEqual(stats["not_found"], 0)
        self.assertEqual(stats["deferred"], 1)
        self.assertTrue(msgs, "the failure is reported to the caller")
        self.assertIsNone(self.c.execute("SELECT citations_updated FROM docs").fetchone()[0])

        self.plug({"10.1/f": work(doi="10.1/F", cites=11)})
        recovered = citations.fetch_counts(self.c, "ephedra", sleep=0)
        self.assertEqual(recovered["updated"], 1, "the retry finds it")


class MailtoTest(CitationsCase):
    def test_user_agent_carries_the_mailto_when_configured(self):
        support.config.OPENALEX_MAILTO = "someone@example.test"
        try:
            self.assertIn("mailto:", citations._headers()["User-Agent"])
        finally:
            support.config.OPENALEX_MAILTO = ""
        self.assertNotIn("mailto:", citations._headers()["User-Agent"])
