import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from rag_system.ingestion import build_paper_chunks, persist_paper_records, load_chunk_records
from scripts.build_fts_index import sync_paper_fts
from rag_system import query_rag


class SharedIngestionTests(unittest.TestCase):
    def test_every_child_has_a_real_parent_and_retry_replaces_both_indexes(self):
        paper = {"paper_id": "test", "title": "Research", "authors": ["Author"]}
        children, parents = build_paper_chunks(" ".join(f"word{i}" for i in range(1800)), paper, 4)
        self.assertGreater(len(parents), 1)
        for child in children:
            self.assertIn(child["text"], parents[child["parent_id"]]["text"])
            self.assertLessEqual(child["word_count"], 150)
        with tempfile.TemporaryDirectory() as folder:
            persist_paper_records(paper, children, parents, folder)
            db = str(Path(folder) / "fts.db")
            sync_paper_fts(children, parents, db)
            smaller, smaller_parents = build_paper_chunks("revised research evidence", paper, 1)
            persist_paper_records(paper, smaller, smaller_parents, folder)
            sync_paper_fts(smaller, smaller_parents, db)
            self.assertEqual(len(load_chunk_records(str(Path(folder) / "paper_chunks.json"))), 1)
            with closing(sqlite3.connect(db)) as con:
                self.assertEqual(con.execute("select count(*) from chunks").fetchone()[0], 1)
                self.assertEqual(con.execute("select count(*) from parents").fetchone()[0], 1)
                self.assertEqual(con.execute("select count(*) from chunks_fts where chunks_fts match 'revised'").fetchone()[0], 1)
            found = query_rag.get_parent_contexts([smaller[0]["parent_id"]], db_path=db)
            self.assertEqual(found[smaller[0]["parent_id"]]["text"], "revised research evidence")
            sync_paper_fts(children, parents, db)
            self.assertEqual(len(query_rag.get_parent_contexts(list(parents), db_path=db)), len(parents))
