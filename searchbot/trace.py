# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""In-memory run traces so the UI can show the agent thinking live
(Cursor/Windurf-style): every phase emits a typed event; clients
long-poll /api/trace?id=&since=n and get new events immediately."""
import threading
import time
import uuid


class Trace:
    def __init__(self, rid):
        self.id = rid
        self.events = []
        self.done = False
        self.result = None
        self.t0 = time.time()
        self._cv = threading.Condition()

    def emit(self, phase, msg="", **data):
        with self._cv:
            ev = {"id": len(self.events),
                  "t": int((time.time() - self.t0) * 1000),
                  "phase": phase, "msg": msg}
            ev.update({k: v for k, v in data.items() if v is not None})
            self.events.append(ev)
            self._cv.notify_all()
        return ev

    def finish(self, result):
        with self._cv:
            self.result = result
            self.done = True
            self._cv.notify_all()

    def wait(self, since, timeout=3.0):
        """Long-poll: return events[since:] as soon as one arrives (or after
        `timeout`). Client loops with since=next."""
        with self._cv:
            if len(self.events) <= since and not self.done:
                self._cv.wait(timeout)
            return {"events": self.events[since:],
                    "next": len(self.events),
                    "done": self.done,
                    "result": self.result if self.done else None}


RUNS = {}
_LOCK = threading.Lock()
KEEP_DONE = 30


def new_run(rid=None):
    with _LOCK:
        if rid and rid not in RUNS:
            tr = Trace(rid)
            RUNS[rid] = tr
            return tr
        rid = uuid.uuid4().hex[:12]
        tr = Trace(rid)
        RUNS[rid] = tr
        done = [k for k, v in RUNS.items() if v.done]
        for old in sorted(done, key=lambda k: RUNS[k].t0)[:-KEEP_DONE]:
            RUNS.pop(old, None)
    return tr


class Ev:
    """Fan out one agent step to (legacy log string sink) + (live Trace).
    Keeps the MCP's `agent_events` list working unchanged."""

    def __init__(self, log=None, trace=None):
        self.log = log
        self.trace = trace

    def __call__(self, phase, msg="", **data):
        if self.trace is not None:
            self.trace.emit(phase, msg, **data)
        if self.log is not None:
            self.log(f"{phase}: {msg}" if msg else phase)

    def delta(self, text):
        if self.trace is not None:
            self.trace.emit("delta", "", text=text)
