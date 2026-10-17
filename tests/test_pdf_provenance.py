import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf
from rag_system.pdf_extraction import extract_pdf
from rag_system.ingestion import build_paper_chunks, persist_paper_records, load_chunk_records
from rag_system import query_rag
from scripts.build_fts_index import build_index


class PdfProvenanceTests(unittest.TestCase):
    def create_pdf(self, path):
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((50, 50), "Page one discusses retrieval evidence and its limitations.")
            page = document.new_page()
            page.insert_text((50, 50), "Page two contains the benchmark table.")
            for y in (100, 130, 160):
                page.draw_line((50, y), (350, y))
            for x in (50, 200, 350):
                page.draw_line((x, 100), (x, 160))
            for x, y, text in ((60, 120, "Method"), (210, 120, "Score"), (60, 150, "Hybrid"), (210, 150, "0.90")):
                page.insert_text((x, y), text)
            document.save(path)

    def test_real_pdf_pages_tables_and_sparse_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "paper.pdf")
            self.create_pdf(path)
            pages = extract_pdf(path, visual=False, ocr=False)
            self.assertEqual([page["page_number"] for page in pages], [1, 2])
            self.assertTrue(pages[0]["blocks"])
            self.assertTrue(pages[1]["tables"])
            self.assertIn("|", pages[1]["text"])
            paper = {"paper_id": "test", "title": "Benchmark", "pdf_url": "https://arxiv.org/pdf/test"}
            children, parents = build_paper_chunks(pages, paper, 2)
            self.assertEqual({child["page_start"] for child in children}, {1, 2})
            self.assertTrue(all(parent["page_start"] == parent["page_end"] for parent in parents.values()))
            persist_paper_records(paper, children, parents, folder, pages=pages)
            self.assertFalse((Path(folder) / "paper_chunks.json").exists())
            self.assertEqual(len(load_chunk_records(str(Path(folder) / "paper_chunks.json"))), len(children))
            db = str(Path(folder) / "fts.db")
            self.assertTrue(build_index(str(Path(folder) / "paper_chunks.json"), db))
            candidates = query_rag.search_fts("Hybrid", db_path=db)
            self.assertTrue(candidates)
            self.assertEqual(candidates[0][0]["page_start"], 2)
            resolved = query_rag.get_parent_contexts([candidates[0][0]["parent_id"]], db_path=db)
            self.assertEqual(next(iter(resolved.values()))["page_start"], 2)

    def test_visual_mode_is_explicit_and_transcription_is_labelled(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "visual.pdf")
            self.create_pdf(path)
            reader = Mock(return_value="Table shows Hybrid score 0.90.")
            pages = extract_pdf(path, visual=True, visual_reader=reader)
            self.assertTrue(reader.called)
            self.assertIn("Model-assisted visual transcription", pages[1]["text"])
            self.assertTrue(reader.call_args.args[0].startswith(b"\x89PNG"))

    def test_unreadable_pages_warn_on_every_retrieved_context(self):
        pages = [{"page_number": 1, "text": "Readable evidence"}, {"page_number": 2, "text": ""}]
        children, parents = build_paper_chunks(pages, {"paper_id": "p"}, 2)
        self.assertEqual(len(children), 1)
        self.assertIn("Unreadable pages 2", children[0]["extraction_warning"])

    def test_staged_replacement_overrides_legacy_chunks_without_rewriting_them(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "paper_chunks.json"
            legacy = [{"paper_id": "p", "chunk_id": "p_old", "text": "old"}]
            path.write_text(json.dumps(legacy), encoding="utf-8")
            children, parents = build_paper_chunks("new evidence", {"paper_id": "p"}, 1)
            persist_paper_records({"paper_id": "p"}, children, parents, folder)
            self.assertEqual(json.loads(path.read_text()), legacy)
            self.assertEqual(load_chunk_records(str(path))[0]["text"], "new evidence")

    def test_dense_source_citation_links_to_the_actual_page(self):
        cache = Mock()
        cache.lookup.return_value = (False, None, [], 0, None)
        collection = Mock()
        collection.query.return_value = {
            "documents": [["Evidence on page seven"]], "ids": [["p_c0"]], "distances": [[0.1]],
            "metadatas": [[{"paper_id": "p", "title": "Paper", "page_start": 7, "page_end": 7,
                              "pdf_url": "https://arxiv.org/pdf/p"}]],
        }
        reranker = SimpleNamespace(is_active=True, rerank=lambda q, p: [dict(
            p[0], score=0.99, relevance_score=0.99, reranker_provider="flashrank")])
        with patch.object(query_rag, "get_semantic_cache", return_value=cache), patch.object(
            query_rag, "get_reranker", return_value=reranker
        ), patch.object(query_rag, "EMBEDDING_PROVIDER", "local"), patch.object(
            query_rag, "RAG_SPARSE_ONLY", False
        ), patch.object(query_rag.os.path, "exists", return_value=True), patch.object(
            query_rag, "search_fts", return_value=[]
        ), patch.object(query_rag, "get_knowledge_graph", return_value=SimpleNamespace(is_loaded=False)), patch.object(
            query_rag, "generate_content", return_value=SimpleNamespace(text="Answer [Source 1]")) as generate:
            _, sources = query_rag.query_rag(collection, "Explain research evidence", enable_mcp_fallback=False)
        self.assertEqual(sources[0]["pdf_url"], "https://arxiv.org/pdf/p#page=7")
        self.assertIn("Location: Page 7", generate.call_args.args[0])
