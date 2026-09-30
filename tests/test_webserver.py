# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""HTTP layer: routing, JSON shapes, and the wiring between the UI and the agent.

agent.ask_agentic is stubbed, so these pin the server's own behaviour — including
the acquire:false path, which used to die on an unbound `events` list before the
answer thread ever started.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import support  # noqa: E402
from searchbot import agent, trace as trace_mod, webserver  # noqa: E402


class ServerCase(support.TempCase):
    def setUp(self):
        super().setUp()
        self._saved_conn = webserver._conn
        webserver._conn = self.c                       # never open the user's DB
        self.calls = []
        self._real_ask = agent.ask_agentic
        agent.ask_agentic = self._stub_ask
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), webserver.H)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        agent.ask_agentic = self._real_ask
        webserver._conn = self._saved_conn
        super().tearDown()

    def _stub_ask(self, c, slug, session_id, question, **kw):
        self.calls.append({"slug": slug, "session_id": session_id,
                           "question": question, **kw})
        return {"answer": f"grounded answer for {question} [E1]", "evidence": [],
                "facts_stored": 0, "rounds": []}

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(self, path, obj):
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def run_ask(self, body):
        """POST /api/ask then follow the trace to its result."""
        code, started = self.post("/api/ask", body)
        self.assertEqual(code, 200)
        rid = started["run_id"]
        for _ in range(200):
            _, t = self.get(f"/api/trace?id={rid}")
            if t.get("done"):
                return t
            time.sleep(0.05)
        self.fail("ask never finished")


class AskEndpointTest(ServerCase):
    def test_acquire_false_returns_an_answer(self):
        """Regression: this path raised NameError before the answer thread ran."""
        trace = self.run_ask({"slug": "ephedra", "question": "ephedrine hypotension?",
                              "acquire": False})
        self.assertNotIn("error", trace["result"], json.dumps(trace["events"][-3:]))
        self.assertIn("[E1]", trace["result"]["answer"])
        self.assertEqual(self.calls[0]["acquire"], False)

    def test_acquire_defaults_to_true(self):
        """Omitting the field must not silently disable gathering."""
        self.run_ask({"slug": "ephedra", "question": "ephedrine hypotension?"})
        self.assertTrue(self.calls[0].get("acquire", True))

    def test_ranking_options_reach_the_agent_filtered(self):
        self.run_ask({"slug": "ephedra", "question": "ephedrine?", "acquire": False,
                      "year_after": "2015", "recency": 0.4, "session_id": "s1",
                      "topk": 5})
        rank = self.calls[0]["rank"]
        self.assertEqual(rank, {"year_after": 2015, "recency": 0.4})
        self.assertEqual(self.calls[0]["topk"], 5)
        self.assertEqual(self.calls[0]["session_id"], "s1")

    def test_trace_carries_the_phases_the_ui_renders(self):
        trace = self.run_ask({"slug": "ephedra", "question": "ephedrine?", "acquire": False})
        phases = [e["phase"] for e in trace["events"]]
        self.assertIn("done", phases)
        self.assertNotIn("error", phases)

    def test_agent_failure_becomes_an_error_result_not_a_hang(self):
        def boom(*a, **kw):
            raise RuntimeError("model server down")
        agent.ask_agentic = boom
        trace = self.run_ask({"slug": "ephedra", "question": "ephedrine?", "acquire": False})
        self.assertIn("error", trace["result"])
        self.assertIn("model server down", trace["result"]["error"])


class ReadEndpointsTest(ServerCase):
    def test_searches_listing_includes_doc_and_chunk_counts(self):
        from searchbot import db
        self.add_doc("ephedra", "Ephedrine and hypotension", "2001")
        db.upsert_search(self.c, "ephedra", "Ephedra studies", "")
        code, rows = self.get("/api/searches")
        self.assertEqual(code, 200)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["slug"], "ephedra")
        self.assertEqual((rows[0]["docs"], rows[0]["chunks"]), (1, 1))

    def test_creating_a_search_folder_slugifies_the_name(self):
        code, body = self.post("/api/searches", {"slug": "New Topic!", "title": "New Topic"})
        self.assertEqual(code, 200)
        self.assertEqual(body["slug"], "new-topic")
        _, rows = self.get("/api/searches")
        self.assertIn("new-topic", [r["slug"] for r in rows])

    def test_status_lists_jobs(self):
        from searchbot import db
        db.job_set(self.c, "ephedra", "index", "queued")
        code, rows = self.get("/api/status?slug=ephedra")
        self.assertEqual(code, 200)
        self.assertEqual(rows[0]["state"], "queued")

    def test_history_round_trips_turns(self):
        from searchbot import memory
        memory.add_turn(self.c, "sess1", "ephedra", "user", "ephedrine?")
        code, rows = self.get("/api/history?session=sess1")
        self.assertEqual(code, 200)
        self.assertEqual(rows[0]["role"], "user")
        self.assertEqual(rows[0]["text"], "ephedrine?")

    def test_unknown_run_id_is_404(self):
        code, body = self.get("/api/trace?id=nope")
        self.assertEqual(code, 404)
        self.assertIn("error", body)

    def test_unknown_path_is_404(self):
        self.assertEqual(self.get("/api/nope")[0], 404)


class SlugifyTest(support.TempCase):
    def test_slugs_cannot_escape_the_search_directory(self):
        for bad in ("../../etc/passwd", "a b c", "Ephedra!", "", "-----"):
            s = webserver._slugify(bad)
            self.assertNotIn("/", s)
            self.assertNotIn("..", s)
            self.assertTrue(s.replace("-", "").replace("_", "") or s == "untitled", s)
            self.assertLessEqual(len(s), 48)

    def test_blank_input_gets_a_usable_name(self):
        self.assertEqual(webserver._slugify("///"), "untitled")
