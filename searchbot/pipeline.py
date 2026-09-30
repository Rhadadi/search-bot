# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Answer pipeline: retrieve -> cite -> grounded answer (English), enforced."""
import re
from . import config, db, llm, memory, retriever, libgen

SYSTEM = """You are a scientific research assistant. Answer STRICTLY from the EVIDENCE block.

Rules:
- Answer in ENGLISH only.
- Every claim must cite evidence tags like [E1], [E3]. Never cite evidence that does not support the claim.
- If the evidence does not answer the question, say exactly what is missing and what to search next. Do NOT invent facts.
- Quote short passages from evidence when precision matters.
- State uncertainty honestly; numbers/units must be copied from evidence, never estimated.
- You have NO tools. NEVER output anything shaped like a tool call
  (e.g. "<|tool_call_start|>", "google(query=...)", "search(...)").
  Write the answer as plain English prose only.
- At the end, on separate lines, output up to 3 durable conclusions as:
FACT: <one-sentence finding>

FORMAT (mandatory): first a 3-6 sentence ANSWER paragraph with [E#] citations,
THEN the optional FACT: lines. NEVER answer with only FACT lines.
Be efficient: no preamble, no restating the question — the answer paragraph
must be complete within ~120 words.
"""

ASK_LIMIT = 2600  # chars of user question allowed into prompt


def format_evidence(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        src = h["title"] or h["source_file"]
        meta = ", ".join(x for x in [h["year"], h["journal"]] if x)
        ident = h["pmcid"] or h["doi"] or h["pmid"] or ""
        blocks.append(f"[E{i}] {src} ({meta}) {ident}\n{h['text']}")
    return "\n\n".join(blocks)


def answer(c, slug: str, session_id: str, question: str, topk=None, rank=None):
    search_slug = None if (not slug or slug == "all") else slug
    mem_slug = slug or "all"
    memory.ensure_session(c, session_id, mem_slug)
    question = question.strip()[:ASK_LIMIT]

    # 1) widen the question with web/libgen keywords when corpus recall is weak
    rank = rank or {}
    hits = retriever.retrieve(c, question, slug=search_slug, k=topk, **rank)
    if len(hits) < 3 or (hits and hits[0]["score"] < 0.025):
        terms = libgen.suggest_terms(question)
        if terms and terms != question:
            hits2 = retriever.retrieve(c, terms, slug=search_slug, k=topk, **rank)
            seen = {h["chunk_id"] for h in hits}
            hits = (hits + [h for h in hits2 if h["chunk_id"] not in seen])[: (topk or config.FINAL_K)]

    if not hits:
        msg = ("No indexed evidence for this question in this search folder yet. "
               "Use add_search / libgen_search to gather sources, or ask about what is indexed.")
        memory.add_turn(c, session_id, mem_slug, "user", question)
        memory.add_turn(c, session_id, mem_slug, "assistant", msg)
        return {"answer": msg, "evidence": [], "web_results": []}

    mem_block = memory.build_memory_block(c, session_id)
    turns = memory.recent_turns(c, session_id)

    messages = [{"role": "system", "content": SYSTEM}]
    if mem_block:
        messages.append({"role": "system", "content": "MEMORY (context from earlier)\n" + mem_block})
    for t in turns:
        messages.append({"role": t["role"], "content": t["text"][:1200]})
    messages.append({"role": "user", "content":
                     f"EVIDENCE:\n{format_evidence(hits)}\n\nQUESTION: {question}\n\n"
                     "Answer in English citing [E#] tags."})

    raw = llm.chat_llm(messages, temperature=0.2, max_tokens=800)
    facts = memory.extract_facts(c, session_id, slug, raw)
    # strip FACT lines from the displayed answer; if nothing remains, keep the raw text
    shown = re.sub(r"^\s*FACT:.*$", "", raw, flags=re.M | re.I).strip()
    if len(shown) < 40:
        shown = raw.strip()[:1500]
    if not shown:
        shown = ("The model returned no text for this query. Try again or "
                 "rephrase the question.")

    refs = [{"tag": f"E{i}", "title": h["title"], "year": h["year"], "journal": h["journal"],
             "pmcid": h["pmcid"], "doi": h["doi"], "file": h["source_file"], "slug": h["slug"]}
            for i, h in enumerate(hits, 1)]
    memory.add_turn(c, session_id, slug, "user", question)
    memory.add_turn(c, session_id, slug, "assistant", shown, refs)
    return {"answer": shown, "evidence": refs, "facts_stored": facts}
