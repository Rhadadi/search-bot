# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""External memory for the chat model (it may have weak retention — the bot stores
everything in SQLite and re-injects it every call):

  1. rolling turns  — last RECENT_TURNS raw turns always in the prompt
  2. session summary — compacted older history, refreshed by the model itself
  3. facts ledger   — durable per-search notes extracted after each Q&A
  4. search scoping — memory keyed by search slug
"""
import json
import re
from . import config, db, llm


def ensure_session(c, session_id: str, slug: str):
    if not c.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone():
        c.execute("INSERT INTO sessions(id,slug) VALUES(?,?)", (session_id, slug))
        c.commit()


def recent_turns(c, session_id: str):
    rows = c.execute(
        "SELECT role, text, refs_json FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, config.RECENT_TURNS)).fetchall()
    return [dict(r) for r in reversed(rows)]


def add_turn(c, session_id: str, slug: str, role: str, text: str, refs=None):
    c.execute("INSERT INTO turns(session_id,slug,role,text,refs_json) VALUES(?,?,?,?,?)",
              (session_id, slug, role, text, json.dumps(refs or [], ensure_ascii=False)))
    c.commit()
    maybe_compact(c, session_id)


def get_summary(c, session_id: str) -> str:
    row = c.execute("SELECT summary FROM sessions WHERE id=?", (session_id,)).fetchone()
    return row["summary"] if row else ""


def maybe_compact(c, session_id: str):
    """When >SUMMARY_TRIGGER turns sit beyond the summary watermark, compress them."""
    s = c.execute("SELECT summary, summary_upto FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not s:
        return
    upto = s["summary_upto"] or 0
    newest = c.execute("SELECT MAX(id) m FROM turns WHERE session_id=?",
                       (session_id,)).fetchone()["m"] or 0
    # turns older than the live window and not yet in the summary
    cutoff = newest - config.RECENT_TURNS
    if cutoff <= upto:
        return
    pending = c.execute(
        "SELECT id, role, text FROM turns WHERE session_id=? AND id>? AND id<=? ORDER BY id",
        (session_id, upto, cutoff)).fetchall()
    if len(pending) < config.SUMMARY_TRIGGER:
        return
    convo = "\n".join(f"{r['role']}: {r['text'][:600]}" for r in pending)
    instr = ("Summarize this research-chat conversation into AT MOST 8 bullet lines. "
             "Keep: the user's questions, key findings, numbers, paper citations, decisions. "
             "Old summary to merge:\n" + (s["summary"] or "(none)") + "\n\nConversation:\n" + convo)
    try:
        new_sum = llm.chat_llm([{"role": "user", "content": instr}], temperature=0.1, max_tokens=350)
    except Exception:
        new_sum = (s["summary"] or "") + "\n… " + convo[:400]
    new_sum = new_sum.strip()[:2000]
    c.execute("UPDATE sessions SET summary=?, summary_upto=? WHERE id=?",
              (new_sum, pending[-1]["id"], session_id))
    c.commit()


def build_memory_block(c, session_id: str) -> str:
    parts = []
    summ = get_summary(c, session_id)
    if summ:
        parts.append("## Conversation so far (summary)\n" + summ)
    facts = c.execute(
        "SELECT kind, text FROM facts WHERE slug=(SELECT slug FROM sessions WHERE id=?) "
        "ORDER BY id DESC LIMIT 12", (session_id,)).fetchall()
    if facts:
        parts.append("## Known facts about this topic\n" +
                     "\n".join(f"- [{f['kind']}] {f['text']}" for f in reversed(facts)))
    return "\n\n".join(parts)


_FACT_RE = re.compile(r"^\s*FACT:\s*(.+)", re.M | re.I)


def extract_facts(c, session_id: str, slug: str, answer: str):
    """The model was asked to emit FACT: lines; also store the last question as a fact."""
    n = 0
    for m in _FACT_RE.finditer(answer):
        t = m.group(1).strip()
        if 15 < len(t) < 400:
            c.execute("INSERT INTO facts(slug,session_id,kind,text) VALUES(?,?,?,?)",
                      (slug, session_id, "finding", t))
            n += 1
    c.commit()
    return n
