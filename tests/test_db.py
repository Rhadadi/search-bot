# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Schema upgrade path: a corpus indexed before the ranking columns existed."""
import sqlite3
import sys
import tempfile
import unittest
import pathlib

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import support  # noqa: E402
from searchbot import db  # noqa: E402

LEGACY_DOCS = """
CREATE TABLE docs(
  id INTEGER PRIMARY KEY, slug TEXT, kind TEXT, source_file TEXT,
  title TEXT, authors TEXT, year TEXT, journal TEXT, doi TEXT, pmcid TEXT, pmid TEXT,
  url TEXT, license TEXT, n_chunks INTEGER);
"""


class LegacyDB(unittest.TestCase):
    def setUp(self):
        self.path = pathlib.Path(tempfile.mkdtemp(prefix="searchbot-legacy-")) / "old.db"
        self.c = sqlite3.connect(str(self.path))
        self.c.row_factory = sqlite3.Row
        self.c.execute(LEGACY_DOCS)
        self.c.execute("INSERT INTO docs(slug,title,year) VALUES('ephedra','old paper','1998')")
        self.c.commit()

    def tearDown(self):
        self.c.close()

    def columns(self, table="docs"):
        return {r["name"] for r in self.c.execute(f"PRAGMA table_info({table})")}

    def test_migrate_adds_the_missing_columns(self):
        self.assertNotIn("citations", self.columns())
        db.migrate(self.c)
        self.assertTrue({"citations", "citations_updated", "citations_source"}
                        <= self.columns())
        row = self.c.execute("SELECT * FROM docs").fetchone()
        self.assertEqual(row["title"], "old paper", "existing rows survive untouched")
        self.assertIsNone(row["citations"], "un-backfilled counts stay NULL, not 0")

    def test_migrate_is_idempotent(self):
        db.migrate(self.c)
        before = self.columns()
        db.migrate(self.c)
        db.migrate(self.c)
        self.assertEqual(self.columns(), before)
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM docs").fetchone()[0], 1)

    def test_added_columns_are_writable_after_upgrade(self):
        db.migrate(self.c)
        self.c.execute("UPDATE docs SET citations=? WHERE slug='ephedra'", (12,))
        self.c.commit()
        self.assertEqual(self.c.execute("SELECT citations FROM docs").fetchone()[0], 12)


class FreshSchemaTest(support.TempCase):
    def test_init_creates_every_table_the_engine_needs(self):
        names = {r["name"] for r in self.c.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        for t in ("docs", "chunks", "chunks_fts", "chunks_vec", "sessions",
                  "turns", "facts", "acquired", "jobs", "searches"):
            self.assertIn(t, names)

    def test_init_on_an_existing_db_is_a_no_op(self):
        self.add_doc("ephedra", "keep me", "2001")
        db.init(self.c, support.DIM)
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM docs").fetchone()[0], 1)
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0], 1)

    def test_vec_table_is_sized_from_the_embedder(self):
        """Any embedder dimension works as long as index + DB agree on it."""
        cols = [r["name"] for r in self.c.execute("PRAGMA table_info(chunks_vec)")]
        self.assertIn("embedding", cols)
        self.assertEqual(len(db.ser([0.0] * support.DIM)), support.DIM * 4)
