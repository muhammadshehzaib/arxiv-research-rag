import unittest
from unittest.mock import Mock, patch

from scripts import populate_db


class PopulationSyncTests(unittest.TestCase):
    def setUp(self):
        self.collection = Mock()
        self.collection.get.return_value = {"ids": []}
        client = Mock()
        client.get_or_create_collection.return_value = self.collection
        self.enterContext(patch.object(populate_db.chromadb, "PersistentClient", return_value=client))
        self.enterContext(patch.object(populate_db, "EMBEDDING_PROVIDER", "local"))
        self.enterContext(patch("chromadb.utils.embedding_functions.DefaultEmbeddingFunction", return_value=Mock()))
        self.chunks = self.enterContext(patch.object(populate_db, "load_chunks", return_value=[]))
        self.sync = self.enterContext(patch("scripts.build_fts_index.build_index", return_value=True))

    def test_population_syncs_even_when_no_new_vectors_exist(self):
        populate_db.populate_database()
        self.sync.assert_called_once_with()

    def test_false_sync_result_is_a_failure(self):
        self.sync.return_value = False
        with self.assertRaisesRegex(RuntimeError, "synchronization failed"):
            populate_db.populate_database()

    def test_sync_exception_is_not_swallowed(self):
        self.sync.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            populate_db.populate_database()

    def test_vector_failure_does_not_index_unstored_chunks(self):
        self.chunks.return_value = [{"chunk_id": "a", "text": "text", "authors": []}]
        self.collection.upsert.side_effect = OSError("vector store unavailable")
        with self.assertRaisesRegex(RuntimeError, "Vector ingestion failed"):
            populate_db.populate_database()
        self.sync.assert_not_called()
