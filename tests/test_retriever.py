# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Retrieval: RRF baseline, year bounds, recency and citation weights."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import support  # noqa: E402
from searchbot import config, retriever  # noqa: E402


class RetrieveTest(support.TempCase):
    def setUp(self):
        super().setUp()
        # Distinct text per doc (identical text would tie and be rank-order
        # dependent), all matching the query so every candidate is retrievable.
        self.old_low = self.add_doc("t", "Early ephedrine work", "1975", 3,
                                    text="ephedrine hypotension anesthesia spinal")
        self.mid = self.add_doc("t", "Mid ephedrine review", "2005", 40,
                                 text="ephedrine hypotension anesthesia review patients")
        self.new_low = self.add_doc("t", "New ephedrine study", "2024", 2,
                                     text="ephedrine hypotension anesthesia blood pressure")
        self.new_high = self.add_doc("t", "Big ephedrine trial", "2020", 4000,
                                      text="ephedrine hypotension anesthesia study case")

    def test_all_lanes_hit(self):
        hits = retriever.retrieve(self.c, self.QUERY)
        self.assertEqual(len(hits), 4)
        for h in hits:
            self.assertTrue(h["in_vec"] and h["in_fts"], "identical text should hit both lanes")

    def test_default_is_pure_rrf(self):
        """Zero weights must reproduce the pre-feature ordering exactly."""
        base = self.ids()
        self.assertEqual(base, self.ids(recency=0.0, citations=0.0))
        self.assertEqual(base, self.ids(recency=0))

    def test_rrf_sums_both_lanes(self):
        """A chunk ranked in both lanes outscores the same rank in one lane."""
        s = self.scores()
        hits = retriever.retrieve(self.c, self.QUERY)
        both = [h for h in hits if h["in_vec"] and h["in_fts"]]
        one = [h for h in hits if not (h["in_vec"] and h["in_fts"])]
        if both and one:
            self.assertGreater(min(h["score"] for h in both), max(h["score"] for h in one))

    def test_year_after_drops_older(self):
        got = set(self.ids(year_after=2010))
        self.assertEqual(got, {self.new_low, self.new_high})

    def test_year_before_drops_newer(self):
        got = set(self.ids(year_before=2006))
        self.assertEqual(got, {self.old_low, self.mid})

    def test_undated_doc_excluded_when_bound_given(self):
        undated = self.add_doc("t", "No year at all", "", 500)
        self.assertNotIn(undated, self.ids(year_after=2000))
        self.assertIn(undated, self.ids(), "without a bound an undated doc still ranks")

    def test_year_bound_only_filters_never_rescores(self):
        """With a bound that keeps every candidate, scores must be untouched."""
        self.assertEqual(self.scores(), self.scores(year_after=1900))
        self.assertEqual(self.scores(), self.scores(year_before=2100))

    def test_recency_weight_prefers_newer(self):
        base_old = self.scores()[self.old_low]
        boost_old = self.scores(recency=1.0)[self.old_low] - base_old
        boost_new = self.scores(recency=1.0)[self.new_low] - self.scores()[self.new_low]
        self.assertGreater(boost_new, boost_old)
        self.assertNotEqual(self.ids(recency=1.0), self.ids(), "a weight of 1.0 must move things")

    def test_recency_decay_respects_half_life(self):
        """Boost is 0.5 ** (age/half_life); with half_life=1 a 2020 doc in a
        2026-year run gets 0.5**age, which the code must match exactly."""
        import datetime
        config.RECENCY_HALF_LIFE_YEARS = 1.0
        age = datetime.datetime.now().year - 2020
        base = self.scores()[self.new_high]
        boosted = self.scores(recency=1.0)[self.new_high]
        self.assertAlmostEqual(base + 0.5 ** age, boosted, places=4)

    def test_recency_boost_shrinks_with_age(self):
        config.RECENCY_HALF_LIFE_YEARS = 5.0
        near = self.scores(recency=1.0)[self.new_high] - self.scores()[self.new_high]
        far = self.scores(recency=1.0)[self.old_low] - self.scores()[self.old_low]
        self.assertGreater(near, far)

    def test_citation_weight_prefers_cited(self):
        order = self.ids(citations=1.0)
        self.assertEqual(order[0], self.new_high)
        base = self.scores()[self.new_high]
        self.assertAlmostEqual(base + 1.0, self.scores(citations=1.0)[self.new_high], places=4,
                               msg="top-cited candidate gets exactly the full weight")

    def test_citation_log_scale_is_relative_to_candidates(self):
        """log1p over the candidate set: 4000 is the ceiling, so mid gets log1p(40)/log1p(4000)."""
        import math
        base, boosted = self.scores(), self.scores(citations=1.0)
        # scores are rounded to 5dp in retrieve(), so compare at 4dp
        self.assertAlmostEqual(base[self.mid] + math.log1p(40) / math.log1p(4000),
                              boosted[self.mid], places=4)

    def test_uncounted_docs_get_no_boost(self):
        noid = self.add_doc("t", "Untracked", "2021", None)
        self.assertEqual(self.scores()[noid], self.scores(citations=1.0)[noid])

    def test_slug_scope_limits_candidates(self):
        other = self.add_doc("u", "Elsewhere ephedrine", "2023", 900)
        self.assertNotIn(other, self.ids(slug="t"))
        self.assertIn(other, self.ids(slug="u"))

    def test_k_is_a_cap(self):
        self.assertEqual(len(retriever.retrieve(self.c, self.QUERY, k=2)), 2)

    def test_config_weights_apply_when_call_omits_them(self):
        baseline = self.scores()
        config.CITATION_WEIGHT = 1.0
        self.assertEqual(self.ids()[0], self.new_high,
                         "with no per-call weight the config default must drive ranking")
        self.assertEqual(self.scores(citations=0.0), baseline,
                         "a per-call 0.0 must override a non-zero config default")

    def test_vector_only_query_still_returns(self):
        """KNN always returns neighbours, so only the FTS lane can come up empty."""
        hits = retriever.retrieve(self.c, "zzzqqy nothinghere")
        self.assertTrue(hits, "vector lane still ranks the corpus")
        self.assertTrue(all(not h["in_fts"] for h in hits), "no FTS term should match")


class RankOptsTest(support.TempCase):
    def test_filters_to_known_keys(self):
        got = retriever.rank_opts({"year_after": "2015", "question": "x", "acquire": False,
                                   "recency": "0.2", "bogus": 1})
        self.assertEqual(got, {"year_after": 2015, "recency": 0.2})

    def test_drops_none_and_empty(self):
        self.assertEqual(retriever.rank_opts({"year_after": None, "citations": ""}), {})
        self.assertEqual(retriever.rank_opts(None), {})

    def test_zero_is_kept_not_dropped(self):
        """0.0 is a meaningful 'off' override, so it must survive."""
        self.assertEqual(retriever.rank_opts({"recency": 0}), {"recency": 0.0})

    def test_rejects_non_numeric(self):
        with self.assertRaises((ValueError, TypeError)):
            retriever.rank_opts({"year_after": "two thousand"})


class DocMetaTest(support.TempCase):
    def test_year_parsed_from_dirty_string(self):
        cid = self.add_doc("t", "Odd format", "c. 1998?", 5)
        self.assertIn(cid, self.ids(year_after=1990))
        self.assertNotIn(cid, self.ids(year_after=2000))

    def test_citation_boundaries(self):
        zero = self.add_doc("t", "Never cited", "2019", 0)
        self.assertEqual(self.scores()[zero], self.scores(citations=1.0)[zero],
                         "0 citations must not be treated as missing OR as a boost")
