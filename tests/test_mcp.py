# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""The MCP surface: tool schemas, dispatch routing, and the JSON-RPC loop."""
import io
import json
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import support  # noqa: E402
from searchbot import agent, mcp_server, pipeline  # noqa: E402


class ToolSchemaTest(support.TempCase):
    def test_every_tool_is_well_formed(self):
        names = []
        for t in mcp_server.TOOLS:
            self.assertIn("name", t)
            self.assertIn("description", t)
            self.assertEqual(t["inputSchema"]["type"], "object", t["name"])
            self.assertNotIn(" ", t["name"])
            names.append(t["name"])
        self.assertEqual(len(names), len(set(names)), "duplicate tool name")

    def test_ask_publishes_the_ranking_options(self):
        ask = next(t for t in mcp_server.TOOLS if t["name"] == "ask")
        props = ask["inputSchema"]["properties"]
        for f in ("slug", "question", "acquire", "year_after", "year_before",
                  "recency", "citations"):
            self.assertIn(f, props)
        self.assertEqual(ask["inputSchema"]["required"], ["slug", "question"])

    def test_every_dispatched_tool_is_advertised(self):
        """A tool the server runs but never lists is invisible to clients."""
        listed = {t["name"] for t in mcp_server.TOOLS}
        self.assertEqual(listed, {"list_searches", "add_search", "ask", "research_section", "index_status",
                                  "libgen_search", "libgen_download"})

    def test_module_docstring_matches_the_tools(self):
        documented = set(re.findall(r"^  ([a-z_]+)\s+- ", mcp_server.__doc__, re.M))
        self.assertEqual(documented, {t["name"] for t in mcp_server.TOOLS})


class DispatchTest(support.TempCase):
    def setUp(self):
        super().setUp()
        self._real_agentic = agent.ask_agentic
        self._real_answer = pipeline.answer
        self.calls = []

    def tearDown(self):
        agent.ask_agentic = self._real_agentic
        pipeline.answer = self._real_answer
        super().tearDown()

    def stub(self, obj, name, key):
        def fake(c, slug, session_id, question, **kw):
            self.calls.append({"via": name, "slug": slug, "session": session_id,
                               "question": question, **kw})
            return {"answer": "grounded [E1]", "evidence": []}
        setattr(obj, key, fake)

    def test_acquire_true_routes_to_the_agent(self):
        self.stub(agent, "agent", "ask_agentic")
        out = mcp_server._dispatch("ask", {"slug": "ephedra", "question": "ephedrine?"})
        self.assertEqual(self.calls[0]["via"], "agent")
        self.assertIn("agent_events", out)

    def test_acquire_false_routes_to_the_plain_pipeline(self):
        self.stub(pipeline, "pipeline", "answer")
        mcp_server._dispatch("ask", {"slug": "ephedra", "question": "ephedrine?",
                                     "acquire": False})
        self.assertEqual(self.calls[0]["via"], "pipeline")

    def test_ranking_arguments_reach_retrieval_filtered(self):
        self.stub(pipeline, "pipeline", "answer")
        mcp_server._dispatch("ask", {"slug": "ephedra", "question": "q",
                                     "acquire": False, "year_after": 2010,
                                     "citations": 0.5, "session_id": "s",
                                     "bogus": "ignored"})
        self.assertEqual(self.calls[0]["rank"], {"year_after": 2010, "citations": 0.5})

    def test_session_id_defaults_so_two_clients_do_not_share_memory(self):
        self.stub(pipeline, "pipeline", "answer")
        mcp_server._dispatch("ask", {"slug": "ephedra", "question": "q", "acquire": False})
        self.assertEqual(self.calls[0]["session"], "mcp-default")

    def test_list_searches_reports_per_folder_counts(self):
        from searchbot import db
        self.add_doc("ephedra", "Ephedrine hypotension", "2001")
        db.upsert_search(self.c, "ephedra", "Ephedra", "")
        rows = mcp_server._dispatch("list_searches", {})
        self.assertEqual([(r["slug"], r["docs"], r["chunks"]) for r in rows],
                         [("ephedra", 1, 1)])
        self.assertEqual(mcp_server._dispatch("index_status", {"slug": "nothing"}), [])

    def test_unknown_tool_is_an_error_not_a_silent_none(self):
        with self.assertRaises(ValueError):
            mcp_server._dispatch("does_not_exist", {})


class ProtocolTest(support.TempCase):
    def rpc(self, requests, raw=""):
        stdin = io.StringIO(raw + "".join(json.dumps(r) + "\n" for r in requests))
        stdout = io.StringIO()
        real_in, real_out = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = stdin, stdout
        try:
            mcp_server.serve()
        finally:
            sys.stdin, sys.stdout = real_in, real_out
        return [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]

    def test_initialize_tools_list_and_call(self):
        self.stub_agentic()
        replies = self.rpc([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "ask", "arguments": {"slug": "ephedra", "question": "q"}}},
        ])
        self.assertEqual([r["id"] for r in replies], [1, 2, 3], "notifications get no reply")
        self.assertEqual(replies[0]["result"]["capabilities"], {"tools": {}})
        self.assertIn("tools", replies[1]["result"])
        text = replies[2]["result"]["content"][0]["text"]
        self.assertIn("grounded", text)

    def test_tool_failure_becomes_isError_content(self):
        replies = self.rpc([{"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                             "params": {"name": "ask", "arguments": {}}}])
        self.assertTrue(replies[0]["result"]["isError"])
        self.assertIn("ERROR", replies[0]["result"]["content"][0]["text"])

    def test_unknown_method_returns_jsonrpc_error(self):
        replies = self.rpc([{"jsonrpc": "2.0", "id": 4, "method": "tools/nope"}])
        self.assertEqual(replies[0]["error"]["code"], -32601)

    def test_garbage_and_blank_lines_do_not_kill_the_loop(self):
        """A client writing a stray keepalive must not take the server down."""
        replies = self.rpc([{"jsonrpc": "2.0", "id": 5, "method": "initialize"}],
                           raw="\nnot json at all\n")
        self.assertEqual([r["id"] for r in replies], [5])

    def stub_agentic(self):
        def fake(c, slug, session_id, question, **kw):
            return {"answer": "grounded [E1]", "evidence": []}
        self._real = agent.ask_agentic
        agent.ask_agentic = fake
        self.addCleanup(setattr, agent, "ask_agentic", self._real)
