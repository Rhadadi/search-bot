# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""The acquire gate: decide when the corpus genuinely cannot answer.

A wrong "yes" answers from thin evidence; a wrong "no" burns a download. These
tests pin both directions for the pure decision functions — no LLM, no network.
"""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import support  # noqa: E402
from searchbot import agent  # noqa: E402


def hit(score=0.05, title="", text="ephedrine raises blood pressure in patients"):
    return {"score": score, "title": title, "text": text}


def hits(n, **kw):
    return [hit(**kw) for _ in range(n)]


class SufficiencyTest(support.TempCase):
    def test_thin_corpus_is_insufficient_even_when_scores_are_high(self):
        ok, missing = agent.sufficiency(hits(agent.MIN_HITS - 1, score=0.9),
                                        "ephedrine berberine glucose")
        self.assertFalse(ok)
        self.assertTrue(missing, "a thin corpus must report what it lacks")

    def test_every_distinctive_term_present_is_sufficient(self):
        hs = hits(3, title="berberine reduces ephedrine hypotension",
                  text="berberine ephedrine hypotension glucose")
        ok, missing = agent.sufficiency(hs, "how does berberine affect ephedrine hypotension")
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_one_uncovered_term_blocks_the_answer(self):
        """The multi-part case: answering half a question is a wrong answer."""
        hs = hits(3, title="ephedrine hypotension", text="ephedrine hypotension blood pressure")
        ok, missing = agent.sufficiency(hs, "ephedrine hypotension and berberine")
        self.assertFalse(ok)
        self.assertIn("berberine", missing)

    def test_weak_scores_fail_without_a_question(self):
        self.assertEqual(agent.sufficiency(hits(3, score=0.001)), (False, []))
        self.assertEqual(agent.sufficiency(hits(3, score=agent.MIN_SCORE * 3)), (True, []))

    def test_no_hits_is_always_insufficient(self):
        ok, missing = agent.sufficiency([], "ephedrine")
        self.assertFalse(ok)
        self.assertEqual(missing, ["ephedrine"])

    def test_coverage_is_searched_in_titles_and_chunk_text(self):
        in_title = hits(3, title="spinal anesthesia hypotension", text="ephedrine hypotension")
        in_text = hits(3, title="ephedrine hypotension", text="spinal anesthesia effects")
        q = "ephedrine hypotension spinal anesthesia"
        self.assertTrue(agent.sufficiency(in_title, q)[0])
        self.assertTrue(agent.sufficiency(in_text, q)[0])

    def test_typo_in_the_question_still_counts_as_covered(self):
        """Gate false-negatives cost a whole download round; trigram tolerance avoids them."""
        hs = hits(3, title="ephedrine hypotension", text="ephedrine hypotension anesthesia")
        ok, _ = agent.sufficiency(hs, "ephedrin hypotension")
        self.assertTrue(ok)


class KeywordsTest(support.TempCase):
    def test_distinctive_terms_are_few_and_longest_first(self):
        terms = agent._distinctive_terms("how does berberine affect ephedrine hypotension")
        self.assertLessEqual(len(terms), 3)
        self.assertEqual(terms, sorted(terms, key=lambda w: (-len(w), w)))

    def test_generic_research_speak_never_becomes_a_catalogue_query(self):
        kws = agent._question_keywords("what are the main effects of this study on berberine")
        self.assertIn("berberine", kws)
        for junk in ("main", "effects", "study", "research", "based"):
            self.assertNotIn(junk, kws)

    def test_keyword_list_is_capped(self):
        long_q = "ephedrine hypotension anesthesia berberine metformin glucose spinal"
        self.assertLessEqual(len(agent._question_keywords(long_q)), 4)

    def test_a_vague_question_produces_nothing_to_acquire(self):
        """Better no catalogue query than a download triggered by 'main effects'."""
        self.assertEqual(agent._question_keywords("what are the main effects of this study"), [])
        self.assertEqual(agent._derive_folder("what are the main effects of this study"),
                         "general")

    def test_stopwords_and_short_words_are_dropped(self):
        self.assertEqual(agent._question_keywords("what is it"), [])

    def test_fuzzy_covered_accepts_exact_and_rejects_unrelated(self):
        words = {"hypotension", "ephedrine"}
        self.assertTrue(agent._fuzzy_covered("hypotension", words))
        self.assertFalse(agent._fuzzy_covered("metformin", words))

    def test_terms_too_short_to_judge_never_block_an_answer(self):
        """A 3-letter term has no trigram signal; failing it would loop forever."""
        self.assertTrue(agent._fuzzy_covered("abc", set()))
        self.assertTrue(agent._fuzzy_covered("ab", set()))

    def test_all_three_helpers_agree_on_a_real_question(self):
        q = "does ephedrine reduce spinal anesthesia hypotension in obstetric patients"
        kws = agent._question_keywords(q)
        terms = agent._distinctive_terms(q)
        self.assertTrue(terms)
        self.assertTrue(set(terms) <= set(kws))


class SlugTest(support.TempCase):
    def test_all_and_empty_mean_whole_corpus_but_memory_keeps_a_key(self):
        for raw in ("all", "", None):
            self.assertIsNone(agent.pipeline_normalize_slug(raw))
        self.assertEqual(agent.pipeline_normalize_slug("ephedra"), "ephedra")

    def test_folder_names_are_safe_for_paths(self):
        self.assertEqual(agent._folder_for("Spinal Anesthesia!", "spinal anesthesia"),
                         "spinalanesthesia")
        self.assertEqual(agent._folder_for("berberine", "q"), "berberine")
        self.assertTrue(len(agent._derive_folder("a" * 200)) <= 24)

    def test_rounds_and_acquisition_are_capped(self):
        """Unbounded acquire would turn one question into dozens of downloads."""
        self.assertLessEqual(agent.MAX_ROUNDS, 2)
        self.assertLessEqual(agent.MAX_ACQUIRE, 3)

    def test_merge_per_term_passes_ranking_options_through(self):
        """A year filter set by the user must survive the per-term top-up."""
        seen = {}
        real = support.retriever.retrieve

        def spy(c, q, slug=None, k=None, **kw):
            seen.update(kw)
            seen["queries"] = seen.get("queries", []) + [q]
            return []
        support.retriever.retrieve = spy
        try:
            c = self.c
            ev_calls = []
            agent._merge_per_term(c, [], "ephedrine berberine glucose", 8,
                                  lambda *a, **k: ev_calls.append(a),
                                  rank={"year_after": 2015, "recency": 0.3})
        finally:
            support.retriever.retrieve = real
        self.assertEqual(seen.get("year_after"), 2015)
        self.assertEqual(seen.get("recency"), 0.3)
        self.assertTrue(seen.get("queries"), "each missing term should get its own query")
